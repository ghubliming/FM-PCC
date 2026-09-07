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
import os
import sys

# Gen16 — the task's observation spec (cameras, latent width, trajectory dims).
# Nothing in this file names a camera or a dimension; see visual_spec.py's header.
from mix_visual_avoiding.models import visual_spec

sys.path.append(os.path.abspath('d3il'))


class VisualUNetTwoTime(nn.Module):
    """
    Vision encoder + two-time 1-D temporal U-Net backbone (Gen14 mf/af arms).

    Trajectory dimension comes from `visual_spec`, never from config.obs_dim — that
    field can be a stale placeholder (fix_5 lesson).

    Backbone: mix_visual_avoiding.models.unet1d_twotime_cond.Flow_matcher_U_Net_v2
    Vision:   MultiImageObsEncoder (single ResNet, bp-cam) -> 64D latent -> FiLM
    """

    # Gen16 — from visual_spec, the single source of truth. Mirrored onto the class so
    # `VisualUNetTwoTime.TRANSITION_DIM` keeps working for callers that read it off the class.
    TRANSITION_DIM = visual_spec.TRANSITION_DIM
    LATENT_DIM     = visual_spec.LATENT_DIM

    def __init__(self, config, dual_head=False, interval_cfg=False):
        super().__init__()
        self.device     = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.if_vision  = getattr(config, 'if_vision', True)

        # ── 1. Vision encoder (Gen7 verbatim) ─────────────────────────────────
        if self.if_vision:
            self.obs_encoder = visual_spec.build_obs_encoder(self.device)
            latent_dim = self.LATENT_DIM
            print(f'[ VisualUNetTwoTime ] MultiImageObsEncoder initialized — '
                  f'LATENT_DIM={self.LATENT_DIM} ({visual_spec.N_CAMERAS} cam x '
                  f'{visual_spec.RGB_OUTPUT_SIZE}), imagenet_norm=True | {visual_spec.LAYOUT}')
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
            dual_head=dual_head,
            interval_cfg=interval_cfg,
            # Visual conditioning enabled for visual mode. Under v1 this concatenates
            # into the time embedding; under v2 it feeds the per-block γ/β heads.
            use_cond_projection=self.if_vision,
        )
        if self.film_mode == 'v2':
            from mix_visual_avoiding.models.unet1d_twotime_film import Flow_matcher_U_Net_v2_FiLM
            self.backbone = Flow_matcher_U_Net_v2_FiLM(**backbone_kwargs).to(self.device)
            print('[ VisualUNetTwoTime ] film_mode=v2 — TRUE FiLM backbone '
                  '(per-block γ scale + β shift) ACTIVE, h_mlp retained')
        else:
            from mix_visual_avoiding.models.unet1d_twotime_cond import Flow_matcher_U_Net_v2
            self.backbone = Flow_matcher_U_Net_v2(**backbone_kwargs).to(self.device)

        # Expose action_dim so the diffusion engine can reference it
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
            cam_imgs, _ = visual_spec.split_visual(cond['visual'])
            return self.encode_visual(*cam_imgs)
        return None

    def forward(self, x, cond, t, returns=None, use_dropout=True, force_dropout=False,
                h=None, omega=None, t_min=None, t_max=None, return_v=False):
        """
        x:    (B, T, TRANSITION_DIM)  — noisy trajectory
        cond: dict with 'visual_latent': (B, LATENT_DIM)
              or 'visual': (*camera_imgs, obs_seq)   — see visual_spec.pack_visual
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
