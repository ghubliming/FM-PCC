from collections import namedtuple
import glob
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import cv2

from .normalization import LimitsNormalizer
# Gen16 — cameras and dims come from the task spec. This file names neither.
from ..models import visual_spec

Batch = namedtuple('Batch', 'trajectories conditions')


# ─── where the D3IL visual-avoiding data lives ────────────────────────────────
# 🔴 ONE definition, used by BOTH the dataset class below and the train script.
#
# These three strings must agree with each other — `DEFAULT_DATASET_PATH` is the episode
# LIST, and the state pickles and camera frames it names are resolved relative to
# `DATA_ROOT`. Gen9 spelled all three out at their use sites and the train script repeated
# the first one as a literal; Gen16 shipped that literal WRONG (dropped the `all_data/`
# segment, copied from the aligning layout where the list sits one level higher) and job
# 24857 died on FileNotFoundError after the config pkl had already been written.
#
# Defining them here and importing them is what makes that class of bug unrepresentable:
# there is no second place to get it wrong. Gate A6 pins the agreement.
DATA_ROOT            = 'environments/dataset/data/avoiding/all_data'
STATE_DIR            = f'{DATA_ROOT}/state'
DEFAULT_DATASET_PATH = f'{DATA_ROOT}/train_files.pkl'


# ─── 6D Visual Dataset for AVOIDING (Gen16, from Gen9 Epoch 2) ────────────────

class ParityAvoidingDataset(torch.utils.data.Dataset):
    """
    6D trajectory dataset for the visual D3IL avoiding task, shared by all four
    Gen16 ML engines (diffusion / fm / mf / af).

    🔴 ONE dataset for FOUR arms is the point: the engines differ only in their
    objective and sampler, so any difference in windows, normalizer fit or camera
    frames would confound every cross-arm number. `visual_spec` supplies the dims and
    the camera set, and the engine wrappers read the SAME module, so a spec change
    moves the data and the networks together or not at all.

    Trajectory layout (the 2-D analogue of aligning's 9-D):
        x[t] = [ dx   dy | des_x des_y | c_x  c_y ]
                 act(2)    des_xy(2)     c_xy(2)
                 idx 0-1   idx 2-3       idx 4-5

    Why 6D: the DPCC projector enforces Euler dynamics on the *actual* robot position
    (c_xy, indices 4-5). The avoiding task is planar — the robot moves in a plane, the
    obstacles are fixed 2-D discs, and z is held by the env, not part of the action space.

    Why NO obstacle positions in obs: D3IL avoiding's `get_obj_xy_list()` returns 6 fixed
    obstacle (x, y) pairs identical across every episode reset — environment constants,
    not state. They belong in the PLANNING config as `sphere_outside` projector
    constraints, not in the obs vector. Gen9 Epoch 2 plan §12 audit.

    Single camera: bp-cam only — the wrist/in-hand cam is irrelevant for avoiding
    because the robot never grasps anything; it dodges obstacles in a plane.

    Data source:
        - State (des_c_pos, c_pos, actions): loaded from the non-visual state pickles under
          `STATE_DIR`. Both `des_c_pos` and `c_pos` are (T+1, 3) but only the x, y
          indices [0:2] are used.
        - Images: bp-cam frames from `<DATA_ROOT>/images/bp-cam/<ep>/*`
          (collected by `collect_visual_avoiding_data.py`).

    Returns:
        Batch(trajectories: np.float32 (H, 6),
              conditions:   {0: np.float32 (4,),       <- 4D obs anchor
                             'primary_img': Tensor(C,H,W)})
              NOTE: 'wrist_img' is intentionally absent — see visual_spec.COND_IMG_KEYS.
    """

    ACTION_DIM = visual_spec.ACTION_DIM       # 2
    OBS_DIM    = visual_spec.OBS_DIM          # 4  — [des_xy(2), c_xy(2)]
    TRAJ_DIM   = visual_spec.TRANSITION_DIM   # 6

    # The camera folder each `visual_spec.COND_IMG_KEYS` entry is loaded from. Kept here
    # rather than in visual_spec because it is a property of the D3IL data layout on disk,
    # not of the network's observation spec.
    CAM_DIRS = {'primary_img': 'bp-cam'}

    # Re-exported on the class so a caller that already has the class does not need a second
    # import to learn where its data lives. The module-level names are the definition.
    DATA_ROOT            = DATA_ROOT
    DEFAULT_DATASET_PATH = DEFAULT_DATASET_PATH

    def __init__(self, dataset_path=DEFAULT_DATASET_PATH, horizon=8, max_n_episodes=1000):
        super().__init__()
        self.horizon = horizon

        # Bypass D3IL's Avoiding_Dataset: its fixed max_len_data buffer crashes on long
        # episodes (Gen6V4 fix_1 lesson). Loading pickles directly yields variable-length
        # arrays with no truncation.
        from agents.utils.sim_path import sim_framework_path

        _list_path = sim_framework_path(dataset_path)
        if not os.path.exists(_list_path):
            raise FileNotFoundError(
                f'[ ParityAvoidingDataset ] episode list not found:\n'
                f'    {_list_path}\n'
                f'  This is the D3IL visual-avoiding dataset collected by '
                f'`collect_visual_avoiding_data.py`. It is gitignored and lives only on the '
                f'cluster data volume. Check that {DATA_ROOT}/ exists and contains '
                f'train_files.pkl, state/ and images/bp-cam/.')
        state_files = np.load(_list_path, allow_pickle=True)
        rp_data_dir = sim_framework_path(STATE_DIR)
        data_dir    = sim_framework_path(DATA_ROOT)

        n_eps = min(len(state_files), max_n_episodes)

        all_obs_4d  = []
        all_actions = []

        for file in tqdm(state_files[:n_eps], desc='Loading states', mininterval=10.0):
            with open(os.path.join(rp_data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            # 2-D slice only — avoiding is a planar task
            robot_des_xy = env_state['robot']['des_c_pos'][:, :2]   # (T+1, 2)
            robot_c_xy   = env_state['robot']['c_pos'][:, :2]       # (T+1, 2)

            T = len(robot_des_xy) - 1
            obs_4d  = np.concatenate(
                [robot_des_xy[:T], robot_c_xy[:T]], axis=-1
            ).astype(np.float32)                                       # (T, 4)
            actions = (robot_des_xy[1:] - robot_des_xy[:-1]).astype(np.float32)  # (T, 2)

            all_obs_4d.append(obs_4d)
            all_actions.append(actions)

        self.n_episodes = n_eps

        valid_obs = np.concatenate(all_obs_4d,  axis=0)
        valid_act = np.concatenate(all_actions, axis=0)

        self.obs_normalizer = LimitsNormalizer(valid_obs)
        self.act_normalizer = LimitsNormalizer(valid_act)

        self._obs_4d   = all_obs_4d
        self._actions  = all_actions

        # ── Images, one list per camera in visual_spec.COND_IMG_KEYS order ────────
        # A dict keyed by the CONDITION key means __getitem__ never hard-codes a camera
        # name, so adding a second camera is a visual_spec + CAM_DIRS edit and nothing else.
        self.cam_imgs = {key: [] for key in visual_spec.COND_IMG_KEYS}
        for file in tqdm(state_files[:n_eps], desc='Loading images', mininterval=10.0):
            file_name = os.path.basename(file).split('.')[0]
            for key in visual_spec.COND_IMG_KEYS:
                self.cam_imgs[key].append(
                    self._load_images(data_dir, self.CAM_DIRS[key], file_name))

        self.indices = self._make_indices()
        print(f'[ ParityAvoidingDataset ] {n_eps} episodes, {len(self.indices)} windows '
              f'(horizon={horizon}, traj_dim={self.TRAJ_DIM}, '
              f'{visual_spec.N_CAMERAS} camera: {", ".join(visual_spec.COND_IMG_KEYS)})')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, start, end = self.indices[idx]

        obs_raw = self._obs_4d[ep][start:end]    # (H, 4)
        act_raw = self._actions[ep][start:end]   # (H, 2)

        obs_norm = self.obs_normalizer.normalize(obs_raw).astype(np.float32)  # (H, 4)
        act_norm = self.act_normalizer.normalize(act_raw).astype(np.float32)  # (H, 2)

        trajectories = np.concatenate([act_norm, obs_norm], axis=-1)   # (H, 6)

        conditions = {0: obs_norm[0]}   # (4,) float32 numpy
        for key in visual_spec.COND_IMG_KEYS:
            conditions[key] = self.cam_imgs[key][ep][start]   # (C, H, W) tensor
        return Batch(trajectories, conditions)

    def _make_indices(self):
        """(ep, start, end) tuples where the window fits within BOTH the state trajectory
        and every camera's frame count."""
        indices = []
        for ep in range(self.n_episodes):
            usable = len(self._obs_4d[ep])
            for key in visual_spec.COND_IMG_KEYS:
                usable = min(usable, len(self.cam_imgs[key][ep]))
            for start in range(usable - self.horizon + 1):
                indices.append((ep, start, start + self.horizon))
        return np.array(indices, dtype=np.int64)

    def episode_split(self, train_fraction):
        """Episode-level train/test split (no window leakage across the boundary).

        Both Gen16 trainers look for this method (`utils/training.py` and
        `utils/training_twotime.py`); without it they fall back to a random window split,
        under which overlapping windows from the SAME episode land on both sides and the
        test loss reads far too low. Gen9 U4 added it for exactly that reason.
        """
        n_train_eps = max(1, int(train_fraction * self.n_episodes))
        train_idx = [i for i, (ep, _, _) in enumerate(self.indices) if ep < n_train_eps]
        test_idx  = [i for i, (ep, _, _) in enumerate(self.indices) if ep >= n_train_eps]
        return train_idx, test_idx

    @staticmethod
    def _load_images(data_dir, cam_name, file_name):
        """Load all frames for one camera / one episode, sorted by frame index.
        Returns a CPU float32 tensor of shape (T_img, C, H, W) in [0,1].

        🔴 NO RESIZE — Gen9 verbatim. The collector already writes frames at the training
        resolution, and silently rescaling here would hide a data/​spec mismatch instead of
        surfacing it. The one-time shape check below is that surfacing.
        """
        pattern = os.path.join(data_dir, 'images', cam_name, file_name, '*')
        paths   = sorted(
            glob.glob(pattern),
            key=lambda p: int(os.path.basename(p).split('.')[0])
        )
        frames = []
        for path in paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0))
        if frames:
            out = torch.cat(frames, dim=0)
            expected = tuple(visual_spec.IMG_SHAPE)
            if tuple(out.shape[1:]) != expected:
                raise ValueError(
                    f"[ ParityAvoidingDataset ] {cam_name}/{file_name} frames are "
                    f"{tuple(out.shape[1:])} but visual_spec.IMG_SHAPE is {expected}. "
                    f"Re-collect the data at the spec resolution, or change the spec — "
                    f"do not resize here (the encoder was trained at one resolution).")
            return out
        return torch.zeros(0, *visual_spec.IMG_SHAPE)
