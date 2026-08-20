"""Gen14 U8 — `VisualDiTTwoTime`: the DiT/SiT twin of `VisualUNetTwoTime`.

Sibling of `visual_unet_twotime.py`, same contract, different bone. Whatever
`MFTrajectoryModel` / `AFTrajectoryModel` build as `velocity_net`, the engines above them
(`VisualMeanFlow`, `VisualAlphaFlow`) reach it the same way:

    self.model.model.velocity_net.encode_visual(...)      # visual_mf_diffusion.py:41

so this class exists to satisfy exactly that surface while delegating to a transformer.

────────────────────────────────────────────────────────────────────────────────
WHY A TOKEN AND NOT adaLN  (DECISION_Gen14_U8_injection_choice.md)

`diffusion_policy` is the upstream of THIS repo's vision encoder — D3IL's
`multi_image_obs_encoder.py` is their file, verbatim but for import paths. Their
`TransformerForDiffusion` ingests the obs latent as TOKENS (`cond_obs_emb`, one per obs
step) and reserves per-channel modulation (FiLM) for their `ConditionalUnet1D`. adaLN
appears nowhere in their transformer.

Our `VisualUNet` already occupies the modulation design point (`film_mode` v1/v2). Putting
adaLN on a DiT would re-ask the same conditioning question with a different trunk; a token
asks a new one. And `MFDiTTrajectory` / `AFDiTTrajectory` have no adaLN pathway at all
(their blocks are `forward(x, cos, sin)`), so the token is the only mechanism that spans
all four bones.

At `window_size=1` — a dataset-level lock, not a tuning choice — diffusion_policy's
`T_cond = 1 + n_obs_steps` collapses to ONE visual token, so this is their conditioning
stack at our settings rather than an approximation of it.
────────────────────────────────────────────────────────────────────────────────
🔴 THE JVP SHORT-CIRCUIT — inherited, not reimplemented

`VisualMeanFlow.loss()` encodes ONCE up front and passes `cond['visual_latent']` down as a
tensor. Inside `_p_losses_meanflow`'s `jvp` closure that tensor is a captured CONSTANT, so
its forward-mode tangent is zero BY CONSTRUCTION and the two ResNets never enter the
differentiated function. That mechanism lives in the ENGINE, above the bone — this class
only has to honour the same `visual_latent` → `visual` preference order, which
`resolve_visual_cond` does, copied from `visual_unet_twotime.py`.
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
import os
import sys

sys.path.append(os.path.abspath('d3il'))


# bone key -> (module, class). The four ports keep their own provenance; this table is the
# only place a Gen14 visual run names one.
_BONES = {
    'mf_dit': ('mix_visual_aligning.models.mf_dit_official_trajectory', 'MFDiTOfficialTrajectory'),
    'sit':    ('mix_visual_aligning.models.af_sit_trajectory',          'AFSiTTrajectory'),
    'dit_mf': ('mix_visual_aligning.models.mf_dit_trajectory',          'MFDiTTrajectory'),
    'dit_af': ('mix_visual_aligning.models.af_dit_trajectory',          'AFDiTTrajectory'),
}


class VisualDiTTwoTime(nn.Module):
    """Vision encoder + two-time transformer bone (Gen14 U8, mf/af arms).

    Trajectory dimension is hardcoded to 9D (act=3, obs=6) for the visual path, exactly as
    `VisualUNetTwoTime`. Never reads config.obs_dim — that field can be a stale placeholder
    (fix_5 lesson).

    Vision: MultiImageObsEncoder (dual ResNet, agentview + wrist) -> 128D latent -> ONE token.
    """

    # 9D = act(3) + [des_c_pos(3) + c_pos(3)]  — must match VisualUNetTwoTime
    TRANSITION_DIM = 9
    LATENT_DIM     = 128   # dual ResNet-64 concatenated

    def __init__(self, config, variant, dual_head=False, interval_cfg=False, **dit_kwargs):
        """`dit_kwargs` (dit_hidden_size / dit_depth / ...) come from the trajectory model,
        which received them from the engine. They WIN over the same-named attributes on
        `config`, so the sizing has exactly ONE source of truth even though both objects
        carry it. Anything not passed falls back to `config`, then to the parameter-matched
        default."""
        super().__init__()
        self.device    = getattr(config, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.if_vision = getattr(config, 'if_vision', True)
        self.variant   = variant

        if variant not in _BONES:
            raise ValueError(
                f"[ VisualDiTTwoTime ] variant='{variant}' is not a known bone "
                f"(want one of {sorted(_BONES)}).")

        # ── 1. Vision encoder ─────────────────────────────────────────────────
        # 🔴 BYTE-IDENTICAL to visual_unet_twotime.py:76-96. Any drift here silently breaks
        # the U-Net-vs-DiT comparison this whole unit exists to make.
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
            print(f'[ VisualDiTTwoTime ] MultiImageObsEncoder initialized — '
                  f'LATENT_DIM={self.LATENT_DIM}, imagenet_norm=True, share_rgb_model=False')
        else:
            self.obs_encoder = None
            latent_dim = 0

        # ── 2. Transformer bone ───────────────────────────────────────────────
        self.target_horizon = config.horizon
        # 🔴 NO PADDING. The U-Net pads to a multiple of 8 for its three stride-2 levels; a
        # DiT at patch_size=1 takes H=8 as 8 tokens directly. Dropping the pad also drops the
        # crop-back, so nothing here can mis-crop the output.
        def _knob(name, default):
            v = dit_kwargs.get(name, None)
            return default if v is None else v

        patch_size = int(_knob('dit_patch_size', getattr(config, 'dit_patch_size', 1)))
        if self.target_horizon % patch_size != 0:
            raise ValueError(
                f"[ VisualDiTTwoTime ] horizon={self.target_horizon} is not divisible by "
                f"dit_patch_size={patch_size}.")

        if self.if_vision:
            transition_dim = self.TRANSITION_DIM   # 9
        else:
            obs_dim = getattr(config, 'obs_dim', 20)
            transition_dim = config.action_dim + obs_dim

        module_path, cls_name = _BONES[variant]
        import importlib
        BoneCls = getattr(importlib.import_module(module_path), cls_name)

        # 🔴 dit_hidden_size defaults to 160, NOT the state-only 256. At depth 8 that is
        # ~3.9 M params against the visual U-Net's ~4.0 M (`dim=32`). 256 would be ~9.9 M —
        # 2.5x — and an unmatched A/B is precisely the Fix_8 defect that already forced one
        # retraction (see PLAN §1.2(c), §8). Do not raise this "to give the DiT a fair shot".
        hidden_size = int(_knob('dit_hidden_size', getattr(config, 'dit_hidden_size', 160)))
        depth       = int(_knob('dit_depth',       getattr(config, 'dit_depth', 8)))
        num_heads   = int(_knob('dit_num_heads',   getattr(config, 'dit_num_heads', 4)))

        bone_kwargs = dict(
            horizon=self.target_horizon,
            transition_dim=transition_dim,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
            patch_size=patch_size,
            # Gen14 U8 — this is the switch that turns the visual token on.
            cond_dim=latent_dim,
        )
        if variant in ('dit_mf', 'dit_af'):
            # iMF DiT sizing knobs the adaLN pair does not have.
            bone_kwargs.update(
                aux_head_depth=int(_knob('dit_aux_head_depth',
                                         getattr(config, 'dit_aux_head_depth', 2))),
                condition_dropout=float(getattr(config, 'condition_dropout', 0.1)),
                condition_on_t=bool(_knob('dit_condition_on_t',
                                          getattr(config, 'dit_condition_on_t', False))),
            )

        # 🔴 PARAMETER-MATCH GUARD. The ENGINE's default is 256 — the STATE-ONLY width, kept
        # for Gen3v6/v7 parity. On a visual run the matched width is 160 (~3.9 M vs the visual
        # U-Net's ~4.0 M); 256 gives ~9.9 M, i.e. 2.5x. An unmatched backbone A/B is exactly
        # the Fix_8 defect that already forced one published retraction (bb_unet_ablation
        # 2026-07-25, retracted by the 2026-08-19 STUDY). The config sets 160 explicitly; this
        # catches the case where that plumbing silently stops reaching us.
        if self.if_vision and hidden_size == 256:
            print('[ VisualDiTTwoTime ] 🔴 WARNING: dit_hidden_size=256 is the STATE-ONLY '
                  'default, NOT the parameter-matched visual width (160). This bone will be '
                  '~2.5x the U-Net and any U-Net-vs-DiT comparison from it is CONFOUNDED. '
                  'Set dit_hidden_size=160 (config/_mix_bone_keys) unless you mean this. '
                  'Gate G-B2 fails on it.')

        self.backbone = BoneCls(**bone_kwargs).to(self.device)
        print(f'[ VisualDiTTwoTime ] bone={variant} ({cls_name})  hidden={hidden_size} '
              f'depth={depth} heads={num_heads} patch={patch_size}  cond_dim={latent_dim} '
              f'(visual token {"ON" if latent_dim else "OFF"})')

        # dual_head / interval_cfg are structural on the U-Net; on every transformer bone the
        # twin u/v FinalLayers are native and (omega, t_min, t_max) are always accepted, so
        # both flags are inert here. Recorded so the wrapper stays self-describing.
        self.dual_head    = dual_head
        self.interval_cfg = interval_cfg

        # Expose action_dim so the diffusion engine can reference it (VisualUNetTwoTime parity)
        self.action_dim = getattr(config, 'action_dim', 3)

    # ── forward helpers (contract-critical: copied from VisualUNetTwoTime) ────────

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
        """Returns the (B, LATENT_DIM) tensor or None.

        Preference order:
          1. cond['visual_latent'] — ALREADY ENCODED upstream. Used inside the JVP and the
             alpha-Flow bootstrap, where re-running the ResNets would be both wasteful and
             (for forward-mode AD) possibly unimplemented.
          2. cond['visual']        — raw images; encode here. The eval / closed-loop path.
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
        h:    (B,) two-time interval, threaded to the bone's h/r conditioning
        """
        visual_cond = self.resolve_visual_cond(cond)

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
        # No crop-back: the bone emits exactly `horizon` steps (no padding was applied).
        return out
