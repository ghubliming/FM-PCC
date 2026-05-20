import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
import os
import sys

sys.path.append(os.path.abspath('d3il'))

class VisualUNet(nn.Module):
    """
    Vision encoder + 1D temporal U-Net backbone for Visual-DPCC (Gen6V4).

    Trajectory dimension is hardcoded to 9D (act=3, obs=6) for the visual path.
    Never reads config.obs_dim — that field can be a stale placeholder (fix_5 lesson).

    Backbone: fm_visual_aligning.models.unet1d_temporal_cond.UNet1DTemporalCondModel
    Vision:   MultiImageObsEncoder (dual ResNet, agentview + wrist) → 128D latent → FiLM
    """

    # 9D = act(3) + [des_c_pos(3) + c_pos(3)]
    TRANSITION_DIM = 9
    LATENT_DIM     = 128   # dual ResNet-64 concatenated

    def __init__(self, config):
        super().__init__()
        self.device     = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.if_vision  = getattr(config, 'if_vision', True)

        # ── 1. Vision encoder ─────────────────────────────────────────────────
        if self.if_vision:
            shape_meta = {
                'obs': {
                    'agentview_image': {'shape': [3, 96, 96], 'type': 'rgb'},
                    'in_hand_image':   {'shape': [3, 96, 96], 'type': 'rgb'},
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
            print(f'[ VisualUNet ] MultiImageObsEncoder initialized — '
                  f'LATENT_DIM={self.LATENT_DIM}, imagenet_norm=True, share_rgb_model=False')
        else:
            self.obs_encoder = None
            latent_dim = 0

        # ── 2. Temporal U-Net backbone ────────────────────────────────────────
        from fm_visual_aligning.models.unet1d_temporal_cond import UNet1DTemporalCondModel

        self.target_horizon  = config.horizon
        # U-Net needs temporal dim divisible by 8 (3 levels of stride-2 downsampling)
        self.padded_horizon  = ((self.target_horizon + 7) // 8) * 8

        # 9D is hardcoded for visual mode. config.obs_dim is intentionally ignored:
        # legacy configs often set it to a placeholder (e.g. 128) that would
        # produce the wrong backbone input channel count.
        if self.if_vision:
            transition_dim = self.TRANSITION_DIM   # 9
        else:
            obs_dim = getattr(config, 'obs_dim', 20)
            transition_dim = config.action_dim + obs_dim

        self.backbone = UNet1DTemporalCondModel(
            horizon=self.padded_horizon,
            transition_dim=transition_dim,
            cond_dim=latent_dim,
            dim=getattr(config, 'dim', 128),
            dim_mults=getattr(config, 'dim_mults', (1, 2, 4, 8)),
            returns_condition=getattr(config, 'returns_condition', False),
            condition_dropout=getattr(config, 'condition_dropout', 0.1),
            use_cond_projection=self.if_vision,   # FiLM gates enabled for visual mode
        ).to(self.device)

        # Expose action_dim so diffusion engine can reference it
        self.action_dim = getattr(config, 'action_dim', 3)

    # ── forward helpers ───────────────────────────────────────────────────────

    def encode_visual(self, bp_imgs, inhand_imgs):
        """
        bp_imgs, inhand_imgs: (B, T_win, C, H, W)
        Returns: (B, LATENT_DIM) — mean-pooled over the T_win window
        """
        B, T, C, H, W = bp_imgs.shape
        obs_dict = {
            'agentview_image': bp_imgs.reshape(B * T, C, H, W),
            'in_hand_image':   inhand_imgs.reshape(B * T, C, H, W),
        }
        features = self.obs_encoder(obs_dict)          # (B*T, 128)
        return features.view(B, T, -1).mean(dim=1)     # (B, 128)

    def forward(self, x, cond, t, returns=None, use_dropout=True, force_dropout=False):
        """
        x:    (B, T, 9)  — noisy trajectory
        cond: dict with 'visual': (bp_imgs, inhand_imgs, obs_seq)
        t:    (B,) diffusion timestep indices
        """
        # Pool visual embeddings over the window BEFORE trajectory padding so that
        # zero-padded frames never dilute the FiLM conditioning signal.
        visual_cond = None
        if self.if_vision and isinstance(cond, dict) and 'visual' in cond:
            bp_imgs, inhand_imgs, _ = cond['visual']
            visual_cond = self.encode_visual(bp_imgs, inhand_imgs)  # (B, 128)

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
