import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
import os
import sys

sys.path.append(os.path.abspath('d3il'))

class VisualUNet(nn.Module):
    """
    Vision encoder + 1D temporal U-Net backbone for Visual-DPCC on AVOIDING (Gen9 Ep 2).

    Trajectory dimension is hardcoded to 6D (act=2, obs=4) for the visual avoiding
    path. Never reads config.obs_dim — that field can be a stale placeholder
    (Gen6V4 fix_5 lesson, preserved here).

    Backbone: diffuser_visual_avoiding.models.unet1d_temporal_cond.UNet1DTemporalCondModel
    Vision:   MultiImageObsEncoder (SINGLE ResNet, agentview only) → 64D latent → FiLM

    Why single camera: the avoiding task has no grasping; the wrist/inhand cam
    adds no information. Only bp-cam sees the obstacle field. Per Gen9 Ep 2
    plan §6.
    """

    # 6D = act(2) + [des_xy(2) + c_xy(2)]
    TRANSITION_DIM = 6
    LATENT_DIM     = 64    # single ResNet-64 output (no concatenation)

    def __init__(self, config):
        super().__init__()
        self.device     = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.if_vision  = getattr(config, 'if_vision', True)

        # ── 1. Vision encoder (SINGLE camera) ─────────────────────────────────
        if self.if_vision:
            shape_meta = {
                'obs': {
                    'agentview_image': {'shape': [3, 96, 96], 'type': 'rgb'},
                    # in_hand_image removed for avoiding — single camera
                }
            }
            obs_encoder_cfg = OmegaConf.create({
                '_target_': 'agents.models.vision.multi_image_obs_encoder.MultiImageObsEncoder',
                'shape_meta': shape_meta,
                'rgb_model': {
                    '_target_': 'agents.models.vision.model_getter.get_resnet',
                    'input_shape': [3, 96, 96],
                    'output_size': 64,
                },
                'resize_shape':    None,
                'random_crop':     False,
                'use_group_norm':  True,
                'share_rgb_model': False,
                'imagenet_norm':   True,
            })
            self.obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(self.device)
            latent_dim = self.LATENT_DIM
            print(f'[ VisualUNet ] MultiImageObsEncoder initialized (SINGLE-CAM) — '
                  f'LATENT_DIM={self.LATENT_DIM}, imagenet_norm=True')
        else:
            self.obs_encoder = None
            latent_dim = 0

        # ── 2. Temporal U-Net backbone ────────────────────────────────────────
        from diffuser_visual_avoiding.models.unet1d_temporal_cond import UNet1DTemporalCondModel

        self.target_horizon  = config.horizon
        self.padded_horizon  = ((self.target_horizon + 7) // 8) * 8

        # 6D is hardcoded for visual avoiding mode. config.obs_dim is intentionally
        # ignored to defeat stale-placeholder configs (Gen6V4 fix_5 lesson).
        if self.if_vision:
            transition_dim = self.TRANSITION_DIM   # 6
        else:
            obs_dim = getattr(config, 'obs_dim', 4)
            transition_dim = config.action_dim + obs_dim

        self.backbone = UNet1DTemporalCondModel(
            horizon=self.padded_horizon,
            transition_dim=transition_dim,
            cond_dim=latent_dim,
            dim=getattr(config, 'dim', 128),
            dim_mults=getattr(config, 'dim_mults', (1, 2, 4, 8)),
            returns_condition=getattr(config, 'returns_condition', False),
            condition_dropout=getattr(config, 'condition_dropout', 0.1),
            use_cond_projection=self.if_vision,
        ).to(self.device)

        self.action_dim = getattr(config, 'action_dim', 2)

    # ── forward helpers ───────────────────────────────────────────────────────

    def encode_visual(self, bp_imgs):
        """
        bp_imgs: (B, T_win, C, H, W) — single camera
        Returns: (B, LATENT_DIM=64) — mean-pooled over the T_win window
        """
        B, T, C, H, W = bp_imgs.shape
        obs_dict = {
            'agentview_image': bp_imgs.reshape(B * T, C, H, W),
        }
        features = self.obs_encoder(obs_dict)          # (B*T, 64)
        return features.view(B, T, -1).mean(dim=1)     # (B, 64)

    def forward(self, x, cond, t, returns=None, use_dropout=True, force_dropout=False):
        """
        x:    (B, T, 6)  — noisy trajectory
        cond: dict with 'visual': (bp_imgs, obs_seq)   — single-cam tuple, NO inhand
        t:    (B,) diffusion timestep indices
        """
        visual_cond = None
        if self.if_vision and isinstance(cond, dict) and 'visual' in cond:
            visual_payload = cond['visual']
            # Accept either (bp_imgs, obs_seq) or (bp_imgs,) for forward-compat
            bp_imgs = visual_payload[0]
            visual_cond = self.encode_visual(bp_imgs)   # (B, 64)

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
