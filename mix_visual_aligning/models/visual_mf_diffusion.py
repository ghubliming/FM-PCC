"""Gen14 — `VisualMeanFlow`: the MeanFlow (Gen3v6) engine on the visual-aligning task.

Shape copied from `fm_visual_aligning/models/visual_gaussian_diffusion.py` (Gen7's
`VisualFlowMatching`): a thin subclass that only repacks the condition dict at the two
boundaries (trainer -> loss, eval wrapper -> forward) and delegates everything else.

The MeanFlow objective, the JVP, the adaptive loss, the (t, r) sampling and the sampler
in `mf_diffusion.MeanFlowODE` are inherited UNCHANGED. Nothing in this file touches the
math.

🔴 The one substantive difference from Gen7's subclass — PLAN §6.1:
   the visual embedding is encoded ONCE here and handed down as a TENSOR
   (`cond['visual_latent']`), never as raw images. See the long note in
   `visual_unet_twotime.py` for why the JVP makes this mandatory rather than merely
   an optimisation.

Trajectory: 9D = [act(0:3) | des_c_pos(3:6) | c_pos(6:9)]
"""

import torch

from mix_visual_aligning.models.mf_diffusion import MeanFlowODE


class VisualMeanFlow(MeanFlowODE):
    """MeanFlow engine for Visual-PCC (Gen14, engine=mf)."""

    def __init__(self, *args, if_vision: bool = True,
                 mf_freeze_vision_encoder: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_vision = if_vision
        # Ablation knob only. Default OFF: the vision encoder is trained end-to-end in
        # Gen6V4/Gen7 and freezing it silently changes what is learned. Capturing the
        # latent tensor is what zeroes the JVP tangent — freezing is NOT needed for that.
        self.mf_freeze_vision_encoder = bool(mf_freeze_vision_encoder)

    # ── internals ─────────────────────────────────────────────────────────────

    def _visual_backbone(self):
        """VisualUNetTwoTime instance: MeanFlowODE -> MeanFlowEngine -> MFTrajectoryModel."""
        return self.model.model.velocity_net

    def _encode_once(self, primary_img, wrist_img):
        """Gen14 (PLAN §6.1) — encode the dual-cam window ONCE, up front.

        Returns a (B, 128) tensor. Downstream this is a captured CONSTANT inside
        `_p_losses_meanflow`'s JVP closure, so its forward-mode tangent is zero by
        construction and only the 1-D U-Net is differentiated.
        """
        backbone = self._visual_backbone()
        if self.mf_freeze_vision_encoder:
            with torch.no_grad():
                return backbone.encode_visual(primary_img, wrist_img)
        return backbone.encode_visual(primary_img, wrist_img)

    # ── training ──────────────────────────────────────────────────────────────

    def loss(self, trajectories, conditions):
        """
        Called as self.model.loss(*batch) where batch is Batch(trajectories, conditions).

        Visual (if_vision=True):
            trajectories: (B, H, 9)   — [act(3) | des_pos(3) | c_pos(3)]
            conditions:   {0: (B,6), 'primary_img': (B,C,H,W), 'wrist_img': (B,C,H,W)}

        Non-visual (if_vision=False):
            trajectories: (B, H, 23)  — [act(3) | obs_20D]
            conditions:   {0: (B,20)} — no image keys
        """
        if not self.if_vision:
            # Non-visual: the cond dict is already in MeanFlowODE's native format.
            # apply_conditioning uses cond[0] to pin obs at step 0.
            return super().loss(trajectories, conditions)

        # Visual path — pre-encode, then hand down a TENSOR (never images).
        primary_img = conditions['primary_img'].unsqueeze(1)   # (B, 1, C, H, W)
        wrist_img   = conditions['wrist_img'].unsqueeze(1)     # (B, 1, C, H, W)
        obs_0       = conditions[0]                            # (B, 6) — snap anchor

        cond = {
            0:                obs_0,
            'visual_latent':  self._encode_once(primary_img, wrist_img),   # (B, 128)
        }

        # NOTE: no `t` is sampled here. MeanFlow draws its OWN (t, r) pair from two
        # independent logit-normals inside _p_losses_meanflow — there is no p_losses()
        # hop and no Beta draw, unlike Gen7's VisualFlowMatching.loss(). Do not add one.
        return super().loss(trajectories, cond)

    # ── inference ─────────────────────────────────────────────────────────────

    def forward(self, cond, *args, **kwargs):
        """
        Closed-loop inference entry point.

        Expected cond format from VisualAgentWrapper:
            cond = {0: (bp_image_seq, inhand_image_seq, obs_6d_seq)}
        where each is (B, window_size, ...).

        Transforms to the internal format used by p_sample_loop:
            {0: obs_6d_at_last_step,     <- apply_conditioning anchor
             'visual_latent': (B, 128)}  <- encoded ONCE, reused across all K ODE steps

        Encoding here instead of inside the sampler is numerically identical (the encoder
        is deterministic: GroupNorm, no dropout, and the images are constant across the
        ODE loop) and cuts K ResNet passes per replan down to 1.
        """
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            bp_imgs, inhand_imgs, obs_seq = cond[0]
            snap_obs = obs_seq[:, -1]   # (B, 6) — most recent step
            new_cond = {
                0:               snap_obs,
                'visual_latent': self._encode_once(bp_imgs, inhand_imgs),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
