"""Gen14 — `VisualAlphaFlow`: the alpha-Flow (Gen3v7) engine on the visual-aligning task.

Deliberate near-duplicate of `visual_mf_diffusion.VisualMeanFlow` — same shape, same two
boundary repacks, different base class. Per the Gen14 plan (§3.1) the two are kept as
separate files rather than factored into a shared mixin: a shared file drifts under
whichever arm last touched it, which is exactly how Gen8 died.

The alpha schedule, the bootstrapped target, the target clamp and the sampler in
`af_diffusion.AlphaFlowODE` are inherited UNCHANGED. `set_train_step()` is inherited too,
so `utils/training_twotime.py` drives the alpha anneal with no extra wiring here.

🔴 The pre-encoded latent (PLAN §6.1/§6.2b) matters MORE for alpha-Flow than for MeanFlow:
   `compute_u_target` evaluates the network a SECOND time per step at the shifted point
   `(z_r + dt*v, r+dt, h-dt)`. Without the cached latent that is a second full ResNet pass
   on top of the JVP's. See the note in `visual_unet_twotime.py`.

Trajectory: 6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)]   (2-D avoiding plane)
Single camera: bp-cam only — no wrist cam. See `visual_spec.py`.
"""

import torch

# Gen16 — cameras and dims come from the task spec, never from a literal here.
from mix_visual_avoiding.models import visual_spec
from mix_visual_avoiding.models.af_diffusion import AlphaFlowODE


class VisualAlphaFlow(AlphaFlowODE):
    """alpha-Flow engine for Visual-PCC (Gen14, engine=af)."""

    def __init__(self, *args, if_vision: bool = True,
                 mf_freeze_vision_encoder: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_vision = if_vision
        # Ablation knob only — see VisualMeanFlow. Key name is shared with the mf arm on
        # purpose so one config key covers both.
        self.mf_freeze_vision_encoder = bool(mf_freeze_vision_encoder)

    # ── internals ─────────────────────────────────────────────────────────────

    def _visual_backbone(self):
        """The visual bone: AlphaFlowODE -> AlphaFlowEngine -> AFTrajectoryModel.velocity_net.
        Either `VisualUNetTwoTime` or `VisualDiTTwoTime` (Gen14 U8) — both expose the same
        `encode_visual(*cam_imgs)` surface, so this file never learns which is live."""
        return self.model.model.velocity_net

    def _encode_once(self, *cam_imgs):
        """Gen14 (PLAN §6.1) — encode the camera window ONCE, up front.

        Gen16: variadic over `visual_spec.CAMERA_KEYS` — see the twin in
        `visual_mf_diffusion.py` for the full JVP rationale.
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
            return super().loss(trajectories, conditions)

        cam_imgs = visual_spec.images_from_conditions(conditions)  # N x (B, 1, C, H, W)
        obs_0    = conditions[0]                                   # (B, OBS_DIM) snap anchor

        cond = {
            0:                obs_0,
            'visual_latent':  self._encode_once(*cam_imgs),   # (B, LATENT_DIM)
        }

        # alpha-Flow, like MeanFlow, samples its own (t, r) internally. No Beta draw here.
        return super().loss(trajectories, cond)

    # ── inference ─────────────────────────────────────────────────────────────

    def forward(self, cond, *args, **kwargs):
        """
        Closed-loop inference entry point. Same contract as VisualMeanFlow.forward —
        see that docstring for why the encode happens here rather than in the sampler.
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
