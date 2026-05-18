from collections import namedtuple
import glob
import os
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
        - State (des_c_pos, c_pos, actions, masks): Aligning_Dataset (20D obs).
          Aligning_Dataset loads ALL episodes; Aligning_Img_Dataset only loads
          the first 3 (hardcoded [:3] stub) and is therefore unusable for training.
        - Images: loaded directly from the image directory, same as Aligning_Img_Dataset
          but without the [:3] cap.

    Aligning_Dataset.observations layout (20D, per timestep t in [0, T-2]):
        obs[..., 0:3]  = des_c_pos   (commanded EE position)
        obs[..., 3:6]  = c_pos       (actual EE position)
        obs[..., 6:9]  = push_box_pos
        obs[..., 9:13] = push_box_quat
        obs[..., 13:16]= target_box_pos
        obs[..., 16:20]= target_box_quat

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

        # ── 1. Load state data via Aligning_Dataset (loads all episodes) ──────
        # Aligning_Dataset.observations is (N, max_len_data, 20).
        # First 6 dims: [des_c_pos(3) | c_pos(3)] — exactly our 6D obs.
        # Actions: velocity = des_c_pos[t+1] - des_c_pos[t], shape (N, max_len_data, 3).
        from d3il.environments.dataset.aligning_dataset import Aligning_Dataset

        base = Aligning_Dataset(
            data_directory=dataset_path,
            obs_dim=20,        # full 20D obs: [des_c_pos(3)|c_pos(3)|box(14)]
            action_dim=3,      # velocity = des_c_pos[t+1] - des_c_pos[t]
            window_size=1,     # raw episode storage; slicing done here
            device='cpu',
        )

        n_eps  = min(len(base.observations), max_n_episodes)
        max_len = base.observations.shape[1]  # max_len_data (padded episode length)

        obs_full_np = base.observations[:n_eps].cpu().numpy()   # (N, T, 20)
        obs_6d_np   = obs_full_np[..., :6]                      # (N, T, 6) [des+c_pos]
        actions_np  = base.actions[:n_eps].cpu().numpy()        # (N, T, 3)
        masks_np    = base.masks[:n_eps].cpu().numpy()          # (N, T)

        # ── 2. Fit LimitsNormalizer on valid (non-padded) timesteps only ──────
        # Fitting on padded zeros would collapse per-dim min to 0 and distort ±1 range.
        valid_mask = masks_np > 0                                           # (N,T) bool
        valid_obs  = obs_6d_np[valid_mask].reshape(-1, self.OBS_DIM)        # (M, 6)
        valid_act  = actions_np[valid_mask].reshape(-1, self.ACTION_DIM)    # (M, 3)

        self.obs_normalizer = LimitsNormalizer(valid_obs)
        self.act_normalizer = LimitsNormalizer(valid_act)

        # ── 3. Store raw state data ───────────────────────────────────────────
        self._obs_6d   = obs_6d_np    # (N, T, 6)
        self._actions  = actions_np   # (N, T, 3)
        self._masks    = masks_np     # (N, T)
        self.n_episodes = n_eps
        self.max_path_length = max_len

        # ── 4. Load images directly (avoids Aligning_Img_Dataset[:3] stub) ───
        # Aligning_Img_Dataset hardcodes [:3] in its episode loop — training on
        # only 3 episodes. We replicate its per-image loading over all n_eps files.
        from agents.utils.sim_path import sim_framework_path

        state_files = np.load(sim_framework_path(dataset_path), allow_pickle=True)
        data_dir    = sim_framework_path("environments/dataset/data/aligning/all_data")

        self.bp_cam_imgs     = []   # list of (T_img, C, H, W) tensors
        self.inhand_cam_imgs = []

        for file in tqdm(state_files[:n_eps], desc='Loading images'):
            file_name = os.path.basename(file).split('.')[0]

            bp_imgs_tensor    = self._load_images(data_dir, 'bp-cam',     file_name)
            inhand_imgs_tensor = self._load_images(data_dir, 'inhand-cam', file_name)

            self.bp_cam_imgs.append(bp_imgs_tensor)
            self.inhand_cam_imgs.append(inhand_imgs_tensor)

        # ── 5. Build sliding window indices ──────────────────────────────────
        # Only windows fully within the valid (unpadded) episode range,
        # and within the image count for that episode.
        self.indices = self._make_indices()
        print(f'[ ParityAligningDataset ] {n_eps} episodes, {len(self.indices)} windows '
              f'(horizon={horizon}, traj_dim={self.TRAJ_DIM})')

    # ── dataset protocol ──────────────────────────────────────────────────────

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep, start, end = self.indices[idx]

        obs_raw = self._obs_6d[ep, start:end]    # (H, 6) float32
        act_raw = self._actions[ep, start:end]   # (H, 3) float32

        obs_norm = self.obs_normalizer.normalize(obs_raw).astype(np.float32)  # (H, 6)
        act_norm = self.act_normalizer.normalize(act_raw).astype(np.float32)  # (H, 3)

        # [act(3) | obs(6)] → (H, 9)
        trajectories = np.concatenate([act_norm, obs_norm], axis=-1)

        # conditions:
        #   0             → 6D obs anchor at t=0 for apply_conditioning snap
        #   'primary_img' → agentview camera frame at timestep `start`
        #   'wrist_img'   → wrist camera frame at timestep `start`
        conditions = {
            0:             obs_norm[0],                       # (6,) float32 numpy
            'primary_img': self.bp_cam_imgs[ep][start],      # (C, H, W) tensor
            'wrist_img':   self.inhand_cam_imgs[ep][start],  # (C, H, W) tensor
        }
        return Batch(trajectories, conditions)

    # ── internal helpers ──────────────────────────────────────────────────────

    def _make_indices(self):
        """Build (ep, start, end) tuples where the full window is within valid data
        and within the image tensor length for that episode."""
        indices = []
        for ep in range(self.n_episodes):
            valid_len = int(self._masks[ep].sum())
            n_imgs    = len(self.bp_cam_imgs[ep])
            # window end must not exceed either state-valid length or image count
            usable = min(valid_len, n_imgs)
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
