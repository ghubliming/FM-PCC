import torch
import torch.nn as nn
import os
import sys

# Gen16 — the task's observation spec (cameras, latent width, trajectory dims).
# Nothing in this file names a camera or a dimension; see visual_spec.py's header.
from mix_visual_avoiding.models import visual_spec

sys.path.append(os.path.abspath('d3il'))

class VisualUNet(nn.Module):
    """
    Vision encoder + 1D temporal U-Net backbone for Visual-DPCC on AVOIDING (Gen16).

    Trajectory dimension comes from `visual_spec`, never from config.obs_dim — that
    field can be a stale placeholder (Gen6V4 fix_5 lesson, preserved here).

    Backbone: mix_visual_avoiding.models.unet1d_temporal_cond.UNet1DTemporalCondModel
    Vision:   MultiImageObsEncoder (single ResNet, bp-cam) → 64D latent → FiLM
    """

    # Gen16 — mirrored from visual_spec so `VisualUNet.TRANSITION_DIM` keeps working for
    # every caller that reads it off the class (Gen14 parity). visual_spec is the source.
    TRANSITION_DIM = visual_spec.TRANSITION_DIM
    LATENT_DIM     = visual_spec.LATENT_DIM

    def __init__(self, config):
        super().__init__()
        self.device     = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.if_vision  = getattr(config, 'if_vision', True)

        # ── 1. Vision encoder (Gen16: built from visual_spec) ─────────────────
        if self.if_vision:
            self.obs_encoder = visual_spec.build_obs_encoder(self.device)
            latent_dim = self.LATENT_DIM
            print(f'[ VisualUNet ] MultiImageObsEncoder initialized — '
                  f'LATENT_DIM={self.LATENT_DIM} ({visual_spec.N_CAMERAS} cam x '
                  f'{visual_spec.RGB_OUTPUT_SIZE}), imagenet_norm=True | {visual_spec.LAYOUT}')
        else:
            self.obs_encoder = None
            latent_dim = 0

        # ── 2. Temporal U-Net backbone ────────────────────────────────────────
        # film_mode selects the conditioning backbone (default 'v1' = current behavior):
        #   'v1' → UNet1DTemporalCondModel  (Fake FiLM: additive bias via time-embed concat)
        #   'v2' → UNet1DTemporalFiLMModel  (True FiLM: per-block γ scale + β shift, opt-in)
        # Absence of the key defaults to 'v1', so all existing configs/checkpoints
        # run byte-identically. v2 is only constructed when explicitly requested.
        self.film_mode = getattr(config, 'film_mode', 'v1')

        self.target_horizon  = config.horizon
        # U-Net needs temporal dim divisible by 8 (3 levels of stride-2 downsampling)
        self.padded_horizon  = ((self.target_horizon + 7) // 8) * 8

        # The visual transition_dim comes from visual_spec. config.obs_dim is
        # intentionally ignored: legacy configs often set it to a placeholder (e.g. 128)
        # that would produce the wrong backbone input channel count.
        if self.if_vision:
            transition_dim = self.TRANSITION_DIM
        else:
            obs_dim = getattr(config, 'obs_dim', visual_spec.STATE_ONLY_OBS_DIM)
            transition_dim = config.action_dim + obs_dim

        backbone_kwargs = dict(
            horizon=self.padded_horizon,
            transition_dim=transition_dim,
            cond_dim=latent_dim,
            dim=getattr(config, 'dim', 128),
            dim_mults=getattr(config, 'dim_mults', (1, 2, 4, 8)),
            returns_condition=getattr(config, 'returns_condition', False),
            condition_dropout=getattr(config, 'condition_dropout', 0.1),
            use_cond_projection=self.if_vision,   # conditioning enabled for visual mode
        )
        if self.film_mode == 'v2':
            from mix_visual_avoiding.models.unet1d_temporal_film import UNet1DTemporalFiLMModel
            self.backbone = UNet1DTemporalFiLMModel(**backbone_kwargs).to(self.device)
            print('[ VisualUNet ] film_mode=v2 — TRUE FiLM backbone (per-block γ scale + β shift) ACTIVE')
        else:
            from mix_visual_avoiding.models.unet1d_temporal_cond import UNet1DTemporalCondModel
            self.backbone = UNet1DTemporalCondModel(**backbone_kwargs).to(self.device)

        # Expose action_dim so diffusion engine can reference it
        self.action_dim = getattr(config, 'action_dim', visual_spec.ACTION_DIM)

    # ── forward helpers ───────────────────────────────────────────────────────

    def encode_visual(self, *cam_imgs):
        """
        cam_imgs: one (B, T_win, C, H, W) tensor per camera, in visual_spec.CAMERA_KEYS
                  order. This task has visual_spec.N_CAMERAS of them.
        Returns:  (B, LATENT_DIM) — mean-pooled over the T_win window
        """
        B, T, C, H, W = cam_imgs[0].shape
        obs_dict = visual_spec.build_obs_dict(
            [img.reshape(B * T, C, H, W) for img in cam_imgs])
        features = self.obs_encoder(obs_dict)          # (B*T, LATENT_DIM)
        return features.view(B, T, -1).mean(dim=1)     # (B, LATENT_DIM)

    def forward(self, x, cond, t, returns=None, use_dropout=True, force_dropout=False):
        """
        x:    (B, T, TRANSITION_DIM)  — noisy trajectory
        cond: dict with 'visual': (*camera_imgs, obs_seq)   — see visual_spec.pack_visual
        t:    (B,) diffusion timestep indices
        """
        # Pool visual embeddings over the window BEFORE trajectory padding so that
        # zero-padded frames never dilute the FiLM conditioning signal.
        visual_cond = None
        if self.if_vision and isinstance(cond, dict) and 'visual' in cond:
            cam_imgs, _ = visual_spec.split_visual(cond['visual'])
            visual_cond = self.encode_visual(*cam_imgs)   # (B, LATENT_DIM)

        B, T, D = x.shape
        if T < self.padded_horizon:
            pad = self.padded_horizon - T
            x = torch.cat([x, x.new_zeros(B, pad, D)], dim=1)

        out = self.backbone(
            x, visual_cond, t,
            returns=returns,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )
        return out[:, :T, :]
