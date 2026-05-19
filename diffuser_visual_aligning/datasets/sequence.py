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


# ─── 9D Visual-DPCC Dataset ───────────────────────────────────────────────────

class ParityAligningDataset(torch.utils.data.Dataset):
    """
    9D trajectory dataset for Visual-DPCC (Gen6V4).

    Trajectory layout:
        x[t] = [ dx   dy   dz | des_x des_y des_z | x    y    z  ]
                  act(3D)       des_c_pos(3D)         c_pos(3D)
                 indices 0-2   indices 3-5            indices 6-8

    Why 9D: DPCC projector enforces Euler dynamics on the *actual* robot position
    (c_pos, indices 6-8).  des_c_pos alone (6D) would project on command targets
    instead of real end-effector positions, violating the DPCC physical contract.

    Data source:
        - State (des_c_pos, c_pos, actions): loaded directly from state pickle files.
          NOTE: Aligning_Dataset has max_len_data=256 (hardcoded) and crashes on
          episodes longer than 256 steps (fix_1 lesson). We bypass it entirely.
        - Images: loaded directly from the image directory without the [:3] cap
          that Aligning_Img_Dataset hardcodes.

    State pickle layout (per timestep t in [0, T-1]):
        robot_des_pos   → des_c_pos  (commanded EE position)
        robot_c_pos     → c_pos      (actual EE position)
        actions         = des_c_pos[t+1] - des_c_pos[t]  (velocity)

    Returns:
        Batch(trajectories: np.float32 (H,9),
              conditions:   {0: np.float32 (6,),      <- 6D obs anchor for apply_conditioning
                             'primary_img': Tensor(C,H,W),
                             'wrist_img':   Tensor(C,H,W)})
    """

    ACTION_DIM = 3
    OBS_DIM    = 6   # [des_c_pos(3), c_pos(3)]
    TRAJ_DIM   = 9   # ACTION_DIM + OBS_DIM

    def __init__(self, dataset_path, horizon=8, max_n_episodes=1000):
        super().__init__()
        self.horizon = horizon

        # ── 1. Load state data directly from pickle files ──────────────────────
        # Bypass Aligning_Dataset: its max_len_data=256 buffer crashes for episodes
        # longer than 256 steps (ValueError in __init__, fix_1). Loading pickles
        # directly yields variable-length arrays with no truncation.
        from agents.utils.sim_path import sim_framework_path

        state_files = np.load(sim_framework_path(dataset_path), allow_pickle=True)
        rp_data_dir = sim_framework_path("environments/dataset/data/aligning/all_data/state")
        data_dir    = sim_framework_path("environments/dataset/data/aligning/all_data")

        n_eps = min(len(state_files), max_n_episodes)

        all_obs_6d  = []   # list of (T_i, 6) float32 arrays — variable length per episode
        all_actions = []   # list of (T_i, 3) float32 arrays

        for file in tqdm(state_files[:n_eps], desc='Loading states', mininterval=10.0):
            with open(os.path.join(rp_data_dir, file), 'rb') as f:
                env_state = pickle.load(f)

            robot_des_pos = env_state['robot']['des_c_pos']   # (T+1, 3)
            robot_c_pos   = env_state['robot']['c_pos']       # (T+1, 3)

            T = len(robot_des_pos) - 1
            obs_6d  = np.concatenate(
                [robot_des_pos[:T], robot_c_pos[:T]], axis=-1
            ).astype(np.float32)                                     # (T, 6)
            actions = (robot_des_pos[1:] - robot_des_pos[:-1]).astype(np.float32)  # (T, 3)

            all_obs_6d.append(obs_6d)
            all_actions.append(actions)

        self.n_episodes = n_eps

        # ── 2. Fit LimitsNormalizer on all valid timesteps ──────────────────────
        valid_obs = np.concatenate(all_obs_6d,  axis=0)   # (sum(T_i), 6)
        valid_act = np.concatenate(all_actions, axis=0)   # (sum(T_i), 3)

        self.obs_normalizer = LimitsNormalizer(valid_obs)
        self.act_normalizer = LimitsNormalizer(valid_act)

        # ── 3. Store raw state data (variable-length lists, not padded arrays) ──
        self._obs_6d  = all_obs_6d    # list of (T_i, 6) arrays
        self._actions = all_actions   # list of (T_i, 3) arrays

        # ── 4. Load images directly (avoids Aligning_Img_Dataset[:3] stub) ───
        self.bp_cam_imgs     = []   # list of (T_img_i, C, H, W) tensors
        self.inhand_cam_imgs = []

        for file in tqdm(state_files[:n_eps], desc='Loading images', mininterval=10.0):
            file_name = os.path.basename(file).split('.')[0]
            self.bp_cam_imgs.append(self._load_images(data_dir, 'bp-cam',     file_name))
            self.inhand_cam_imgs.append(self._load_images(data_dir, 'inhand-cam', file_name))

        # ── 5. Build sliding window indices ──────────────────────────────────
        self.indices = self._make_indices()
        print(f'[ ParityAligningDataset ] {n_eps} episodes, {len(self.indices)} windows '
              f'(horizon={horizon}, traj_dim={self.TRAJ_DIM})')

    # ── dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, start, end = self.indices[idx]

        obs_raw = self._obs_6d[ep][start:end]    # (H, 6) — list index then slice
        act_raw = self._actions[ep][start:end]   # (H, 3)

        obs_norm = self.obs_normalizer.normalize(obs_raw).astype(np.float32)  # (H, 6)
        act_norm = self.act_normalizer.normalize(act_raw).astype(np.float32)  # (H, 3)

        # [act(3) | obs(6)] → (H, 9)
        trajectories = np.concatenate([act_norm, obs_norm], axis=-1)

        conditions = {
            0:             obs_norm[0],                       # (6,) float32 numpy
            'primary_img': self.bp_cam_imgs[ep][start],      # (C, H, W) tensor
            'wrist_img':   self.inhand_cam_imgs[ep][start],  # (C, H, W) tensor
        }
        return Batch(trajectories, conditions)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _make_indices(self):
        """Build (ep, start, end) tuples where the full window fits within both
        the state trajectory length and the image tensor length."""
        indices = []
        for ep in range(self.n_episodes):
            T     = len(self._obs_6d[ep])
            n_img = len(self.bp_cam_imgs[ep])
            usable = min(T, n_img)
            for start in range(usable - self.horizon + 1):
                indices.append((ep, start, start + self.horizon))
        return np.array(indices, dtype=np.int64)

    @staticmethod
    def _load_images(data_dir, cam_name, file_name):
        """
        Load all frames for one camera / one episode, sorted by frame index.
        Returns a CPU float32 tensor of shape (T_img, C, H, W) in [0,1].
        """
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
            return torch.cat(frames, dim=0)   # (T_img, C, H, W)
        return torch.zeros(0, 3, 96, 96)
