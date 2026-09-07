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

Trajectory: 6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)]   (2-D avoiding plane)
Single camera: bp-cam only — no wrist cam. See `visual_spec.py`.
"""

import torch

# Gen16 — cameras and dims come from the task spec, never from a literal here.
from mix_visual_avoiding.models import visual_spec
from mix_visual_avoiding.models.mf_diffusion import MeanFlowODE


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
        """The visual bone: MeanFlowODE -> MeanFlowEngine -> MFTrajectoryModel.velocity_net.
        Either `VisualUNetTwoTime` or `VisualDiTTwoTime` (Gen14 U8) — both expose the same
        `encode_visual(*cam_imgs)` surface, so this file never learns which is live."""
        return self.model.model.velocity_net

    def _encode_once(self, *cam_imgs):
        """Gen14 (PLAN §6.1) — encode the camera window ONCE, up front.

        Returns a (B, LATENT_DIM) tensor. Downstream this is a captured CONSTANT inside
        `_p_losses_meanflow`'s JVP closure, so its forward-mode tangent is zero by
        construction and only the 1-D U-Net is differentiated.

        Gen16: variadic over `visual_spec.CAMERA_KEYS`, so a camera-count change is a
        one-line edit in visual_spec rather than a signature change in four files.
        """
        backbone = self._visual_backbone()
        if self.mf_freeze_vision_encoder:
            with torch.no_grad():
                return backbone.encode_visual(*cam_imgs)
        return backbone.encode_visual(*cam_imgs)

    # ── training ──────────────────────────────────────────────────────────────

    def loss(self, trajectories, conditions):
        """
        Called as self.model.loss(*batch) where batch is Batch(trajectories, conditions).

        Visual (if_vision=True):
            trajectories: (B, H, 6)   — [act(2) | des_xy(2) | c_xy(2)]
            conditions:   {0: (B,4), 'primary_img': (B,C,H,W)}
                          NOTE: no 'wrist_img' — avoiding is single-camera (visual_spec).

        Non-visual (if_vision=False):
            trajectories: (B, H, action_dim + obs_dim)
            conditions:   {0: (B, obs_dim)} — no image keys
        """
        if not self.if_vision:
            # Non-visual: the cond dict is already in MeanFlowODE's native format.
            # apply_conditioning uses cond[0] to pin obs at step 0.
            return super().loss(trajectories, conditions)

        # Visual path — pre-encode, then hand down a TENSOR (never images).
        cam_imgs = visual_spec.images_from_conditions(conditions)  # N x (B, 1, C, H, W)
        obs_0    = conditions[0]                                   # (B, OBS_DIM) snap anchor

        cond = {
            0:                obs_0,
            'visual_latent':  self._encode_once(*cam_imgs),   # (B, LATENT_DIM)
        }

        # NOTE: no `t` is sampled here. MeanFlow draws its OWN (t, r) pair from two
        # independent logit-normals inside _p_losses_meanflow — there is no p_losses()
        # hop and no Beta draw, unlike Gen7's VisualFlowMatching.loss(). Do not add one.
        return super().loss(trajectories, cond)

    # ── inference ─────────────────────────────────────────────────────────────

    def forward(self, cond, *args, **kwargs):
        """
        Closed-loop inference entry point.

        Expected cond format from the eval policy:
            cond = {0: (*camera_image_seqs, obs_seq)}
        where each is (B, window_size, ...) and there are visual_spec.N_CAMERAS image
        sequences (Gen16 avoiding: one, bp-cam).

        Transforms to the internal format used by p_sample_loop:
            {0: obs_at_last_step,               <- apply_conditioning anchor
             'visual_latent': (B, LATENT_DIM)}  <- encoded ONCE, reused across all K steps

        Encoding here instead of inside the sampler is numerically identical (the encoder
        is deterministic: GroupNorm, no dropout, and the images are constant across the
        ODE loop) and cuts K ResNet passes per replan down to 1.
        """
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            cam_imgs, obs_seq = visual_spec.split_visual(cond[0])
            snap_obs = obs_seq[:, -1]   # (B, OBS_DIM) — most recent step
            new_cond = {
                0:               snap_obs,
                'visual_latent': self._encode_once(*cam_imgs),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
