"""Gen14 — VisualUNet twin for the TWO-TIME arms (mf / af).

Body copied from `fm_visual_aligning/models/visual_unet.py` (Gen7). Three deltas,
all additive, each marked `Gen14`:

  1. Backbone is a two-time one (h_mlp + visual conditioning) instead of Gen7's
     h-less pair. `film_mode` picks which:
       'v1' -> `unet1d_twotime_cond.Flow_matcher_U_Net_v2`      (Gen7 v1 + h_mlp)
       'v2' -> `unet1d_twotime_film.Flow_matcher_U_Net_v2_FiLM` (Gen7 v2 + h_mlp, U5)
  2. `forward()` accepts and forwards the two-time surface:
     `h`, `omega`, `t_min`, `t_max`, `return_v` — so `MFTrajectoryModel` /
     `AFTrajectoryModel` can call it exactly like they call the state-only UNet.
  3. 🔴 The `visual_latent` SHORT-CIRCUIT — the reason this file exists.
     See the JVP note below.

Gen7's own `visual_unet.py` is untouched; the ddpm/fm arms still import it.

────────────────────────────────────────────────────────────────────────────────
🔴 WHY THE SHORT-CIRCUIT EXISTS (PLAN §6.1)

`MeanFlowODE._p_losses_meanflow` differentiates the network with a forward-mode
JVP:

    u_pred, du_dr = jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))

`_u_of` closes over `cond`. If `cond` still carries raw IMAGES, the two ResNet-18
encoders run INSIDE the JVP — ~2x compute and activation memory for a derivative
that is identically zero (the image latent does not depend on z, r or h), and
forward-mode AD through torchvision ResNet + GroupNorm may not even be
implemented.

So `VisualMeanFlow.loss()` / `VisualAlphaFlow.loss()` call `encode_visual()` ONCE
up front and pass the resulting tensor down as `cond['visual_latent']`. Inside the
JVP that tensor is a captured CONSTANT, so its tangent is zero BY CONSTRUCTION
rather than by hope, and only the 1-D U-Net is differentiated.

⚠️ The pre-encode is NOT wrapped in no_grad by default: the vision encoder is
trained end-to-end in Gen6V4/Gen7 (it is in the optimizer's parameter list).
Capturing the tensor already zeroes the JVP tangent; freezing it would change what
is learned. `mf_freeze_vision_encoder` is the explicit opt-in ablation.
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
import os
import sys

sys.path.append(os.path.abspath('d3il'))


class VisualUNetTwoTime(nn.Module):
    """
    Vision encoder + two-time 1-D temporal U-Net backbone (Gen14 mf/af arms).

    Trajectory dimension is hardcoded to 9D (act=3, obs=6) for the visual path.
    Never reads config.obs_dim — that field can be a stale placeholder (fix_5 lesson).

    Backbone: mix_visual_aligning.models.unet1d_twotime_cond.Flow_matcher_U_Net_v2
    Vision:   MultiImageObsEncoder (dual ResNet, agentview + wrist) -> 128D latent -> FiLM
    """

    # 9D = act(3) + [des_c_pos(3) + c_pos(3)]
    TRANSITION_DIM = 9
    LATENT_DIM     = 128   # dual ResNet-64 concatenated

    def __init__(self, config, dual_head=False, interval_cfg=False):
        super().__init__()
        self.device     = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.if_vision  = getattr(config, 'if_vision', True)

        # ── 1. Vision encoder (Gen7 verbatim) ─────────────────────────────────
        if self.if_vision:
            shape_meta = {
                'obs': {
                    'agentview_image': {'shape': [3, 96, 96], 'type': 'rgb'},
                    'in_hand_image':   {'shape': [3, 96, 96], 'type': 'rgb'},
                }
            }
            # ── Gen14 U9 ── ImageNet initialisation of the ResNet-18 trunk.
            # Default False == every pre-U9 run, byte-identical. Only the WEIGHTS change:
            # architecture, LATENT_DIM, use_group_norm, imagenet_norm and share_rgb_model are
            # all untouched, so cond_dim never moves and every U8 gate stays valid as written.
            # 🔴 The encoder spec below is BYTE-IDENTICAL between visual_unet_twotime.py and
            # visual_dit_twotime.py by design — drift silently breaks the U-Net-vs-DiT
            # comparison this generation exists to make. Gate G-B8 asserts it on every run.
            _vis_pretrained = bool(getattr(config, 'vis_pretrained', False))
            obs_encoder_cfg = OmegaConf.create({
                '_target_': 'agents.models.vision.multi_image_obs_encoder.MultiImageObsEncoder',
                'shape_meta': shape_meta,
                'rgb_model': {
                    '_target_': 'agents.models.vision.model_getter.get_resnet',
                    'input_shape': [3, 96, 96],
                    'output_size': 64,
                    'pretrained': _vis_pretrained,
                },
                'resize_shape':    None,
                'random_crop':     False,
                'use_group_norm':  True,
                'share_rgb_model': False,
                'imagenet_norm':   True,
            })
            self.obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(self.device)
            latent_dim = self.LATENT_DIM
            print(f'[ VisualUNetTwoTime ] MultiImageObsEncoder initialized — '
                  f'LATENT_DIM={self.LATENT_DIM}, imagenet_norm=True, share_rgb_model=False')
            # ── Gen14 U9 ── say it out loud: a run whose encoder was ImageNet-initialised
            # must be identifiable from the log alone, not only from the path key.
            print(f'[ VisualUNetTwoTime ] vis_pretrained={_vis_pretrained}  (ImageNet ResNet-18 init)' if _vis_pretrained else
                  f'[ VisualUNetTwoTime ] vis_pretrained=False  (random init — pre-U9 behaviour)')
        else:
            self.obs_encoder = None
            latent_dim = 0

        # ── 2. Two-time temporal U-Net backbone (Gen14) ───────────────────────
        # film_mode selects how the visual latent reaches the residual blocks:
        #   'v1' → Flow_matcher_U_Net_v2       (Fake FiLM: additive bias via time-embed concat)
        #   'v2' → Flow_matcher_U_Net_v2_FiLM  (True FiLM: per-block γ scale + β shift)
        # U5: v2 used to raise here, because Gen7's v2 file has no h_mlp and would have
        # dropped the MeanFlow/alpha-Flow h-conditioning. unet1d_twotime_film.py now
        # carries BOTH — see its JVP-safety note. Absence of the key still means 'v1',
        # so every existing config and checkpoint is unaffected.
        self.film_mode = getattr(config, 'film_mode', 'v1') or 'v1'
        if self.film_mode not in ('v1', 'v2'):
            raise ValueError(
                f"[ VisualUNetTwoTime ] film_mode='{self.film_mode}' is not a known mode "
                "(want 'v1' or 'v2')."
            )

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

        backbone_kwargs = dict(
            horizon=self.padded_horizon,
            transition_dim=transition_dim,
            cond_dim=latent_dim,
            dim=getattr(config, 'dim', 128),
            dim_mults=getattr(config, 'dim_mults', (1, 2, 4, 8)),
            returns_condition=getattr(config, 'returns_condition', False),
            condition_dropout=getattr(config, 'condition_dropout', 0.1),
            dual_head=dual_head,
            interval_cfg=interval_cfg,
            # Visual conditioning enabled for visual mode. Under v1 this concatenates
            # into the time embedding; under v2 it feeds the per-block γ/β heads.
            use_cond_projection=self.if_vision,
        )
        if self.film_mode == 'v2':
            from mix_visual_aligning.models.unet1d_twotime_film import Flow_matcher_U_Net_v2_FiLM
            self.backbone = Flow_matcher_U_Net_v2_FiLM(**backbone_kwargs).to(self.device)
            print('[ VisualUNetTwoTime ] film_mode=v2 — TRUE FiLM backbone '
                  '(per-block γ scale + β shift) ACTIVE, h_mlp retained')
        else:
            from mix_visual_aligning.models.unet1d_twotime_cond import Flow_matcher_U_Net_v2
            self.backbone = Flow_matcher_U_Net_v2(**backbone_kwargs).to(self.device)

        # Expose action_dim so the diffusion engine can reference it
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

    def resolve_visual_cond(self, cond):
        """Gen14 — the SHORT-CIRCUIT. Returns the (B, LATENT_DIM) tensor or None.

        Preference order:
          1. cond['visual_latent'] — ALREADY ENCODED upstream. Used inside the JVP
             and the alpha-Flow bootstrap, where re-running the ResNets would be
             both wasteful and (for forward-mode AD) possibly unimplemented.
          2. cond['visual']        — raw images; encode here. This is the eval /
             closed-loop path, where there is no JVP and one encode is correct.
        """
        if not self.if_vision or not isinstance(cond, dict):
            return None
        if 'visual_latent' in cond and cond['visual_latent'] is not None:
            return cond['visual_latent']
        if 'visual' in cond:
            bp_imgs, inhand_imgs, _ = cond['visual']
            return self.encode_visual(bp_imgs, inhand_imgs)
        return None

    def forward(self, x, cond, t, returns=None, use_dropout=True, force_dropout=False,
                h=None, omega=None, t_min=None, t_max=None, return_v=False):
        """
        x:    (B, T, 9)  — noisy trajectory
        cond: dict with 'visual_latent': (B,128)  or  'visual': (bp, inhand, obs_seq)
        t:    (B,) continuous time (tau)
        h:    (B,) two-time interval — Gen14, threaded to the backbone's h_mlp
        """
        # Pool visual embeddings over the window BEFORE trajectory padding so that
        # zero-padded frames never dilute the FiLM conditioning signal.
        visual_cond = self.resolve_visual_cond(cond)

        B, T, D = x.shape
        if T < self.padded_horizon:
            pad = self.padded_horizon - T
            x = torch.cat([x, x.new_zeros(B, pad, D)], dim=1)

        out = self.backbone(
            x, visual_cond, t,
            returns=returns,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
            h=h,
            omega=omega,
            t_min=t_min,
            t_max=t_max,
            return_v=return_v,
        )
        # dual_head / return_v ⇒ the backbone returns (u, v); crop both.
        if isinstance(out, tuple):
            u, v = out
            return u[:, :T, :], v[:, :T, :]
        return out[:, :T, :]
