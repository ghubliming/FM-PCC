from collections import namedtuple
import glob
import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import cv2

from .normalization import LimitsNormalizer

Batch = namedtuple('Batch', 'trajectories conditions')


# ─── 6D Visual-DPCC Dataset for AVOIDING (Gen9 Epoch 2) ──────────────────────

class ParityAvoidingDataset(torch.utils.data.Dataset):
    """
    6D trajectory dataset for Visual-DPCC on the D3IL avoiding task.

    Trajectory layout (2-D analogue of aligning's 9-D):
        x[t] = [ dx   dy | des_x des_y | c_x  c_y ]
                 act(2)    des_xy(2)     c_xy(2)
                 idx 0-1   idx 2-3       idx 4-5

    Why 6D: DPCC projector enforces Euler dynamics on the *actual* robot
    position (c_xy, indices 4-5). The avoiding task is 2-D — the robot
    moves in a plane and the obstacles are fixed 2-D points; the z-axis
    is held by the env, not part of the action space.

    Why NO obstacle positions in obs: D3IL avoiding's `get_obj_xy_list()`
    returns 6 fixed obstacle (x, y) pairs that are identical across every
    episode reset — environment constants, not state. They belong in the
    PLANNING config as `sphere_outside` projector constraints, not in
    the obs vector. Per Gen9 Epoch 2 plan §12 audit.

    Single camera: bp-cam only — wrist/inhand cam is irrelevant for
    avoiding because the robot never grasps anything; it dodges
    obstacles in a 2-D plane.

    Data source:
        - State (des_c_pos, c_pos, actions): loaded from non-visual state
          pickles at `environments/dataset/data/avoiding/all_data/state/`.
          Both `des_c_pos` and `c_pos` are (T+1, 3) but only x, y indices
          [0:2] are used.
        - Images: bp-cam frames from
          `environments/dataset/data/avoiding/all_data/images/bp-cam/<ep>/*`
          (collected by `collect_visual_avoiding_data.py`).

    Returns:
        Batch(trajectories: np.float32 (H, 6),
              conditions:   {0: np.float32 (4,),       <- 4D obs anchor
                             'primary_img': Tensor(C,H,W)})
              NOTE: 'wrist_img' is intentionally absent for avoiding.
    """

    ACTION_DIM = 2
    OBS_DIM    = 4   # [des_xy(2), c_xy(2)]
    TRAJ_DIM   = 6   # ACTION_DIM + OBS_DIM

    def __init__(self, dataset_path, horizon=8, max_n_episodes=1000):
        super().__init__()
        self.horizon = horizon

        from agents.utils.sim_path import sim_framework_path

        state_files = np.load(sim_framework_path(dataset_path), allow_pickle=True)
        rp_data_dir = sim_framework_path("environments/dataset/data/avoiding/all_data/state")
        data_dir    = sim_framework_path("environments/dataset/data/avoiding/all_data")

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

        # Single camera — bp-cam only
        self.bp_cam_imgs = []
        for file in tqdm(state_files[:n_eps], desc='Loading images', mininterval=10.0):
            file_name = os.path.basename(file).split('.')[0]
            self.bp_cam_imgs.append(self._load_images(data_dir, 'bp-cam', file_name))

        self.indices = self._make_indices()
        print(f'[ ParityAvoidingDataset ] {n_eps} episodes, {len(self.indices)} windows '
              f'(horizon={horizon}, traj_dim={self.TRAJ_DIM}, single-camera)')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, start, end = self.indices[idx]

        obs_raw = self._obs_4d[ep][start:end]    # (H, 4)
        act_raw = self._actions[ep][start:end]   # (H, 2)

        obs_norm = self.obs_normalizer.normalize(obs_raw).astype(np.float32)  # (H, 4)
        act_norm = self.act_normalizer.normalize(act_raw).astype(np.float32)  # (H, 2)

        trajectories = np.concatenate([act_norm, obs_norm], axis=-1)   # (H, 6)

        conditions = {
            0:             obs_norm[0],                       # (4,) float32 numpy
            'primary_img': self.bp_cam_imgs[ep][start],      # (C, H, W) tensor
        }
        return Batch(trajectories, conditions)

    def _make_indices(self):
        indices = []
        for ep in range(self.n_episodes):
            T     = len(self._obs_4d[ep])
            n_img = len(self.bp_cam_imgs[ep])
            usable = min(T, n_img)
            for start in range(usable - self.horizon + 1):
                indices.append((ep, start, start + self.horizon))
        return np.array(indices, dtype=np.int64)

    def episode_split(self, train_fraction):
        """Episode-level train/test split (no window leakage across the boundary)."""
        n_train_eps = max(1, int(train_fraction * self.n_episodes))
        train_idx = [i for i, (ep, _, _) in enumerate(self.indices) if ep < n_train_eps]
        test_idx  = [i for i, (ep, _, _) in enumerate(self.indices) if ep >= n_train_eps]
        return train_idx, test_idx

    @staticmethod
    def _load_images(data_dir, cam_name, file_name):
        """Load all frames for one camera / one episode, sorted by frame index."""
        pattern = os.path.join(data_dir, 'images', cam_name, file_name, '*')
        paths   = sorted(
            glob.glob(pattern),
            key=lambda x: int(os.path.basename(x).split('.')[0]),
        )
        frames = []
        for p in paths:
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            frames.append(torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0))
        if frames:
            return torch.cat(frames, dim=0)
        return torch.zeros(0, 3, 96, 96)
