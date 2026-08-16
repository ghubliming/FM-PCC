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
   `(z_r + dt*v, r+dt, h-dt)`. Without the cached latent that is a second full ResNet pair
   on top of the JVP's. See the note in `visual_unet_twotime.py`.

Trajectory: 9D = [act(0:3) | des_c_pos(3:6) | c_pos(6:9)]
"""

import torch

from mix_visual_aligning.models.af_diffusion import AlphaFlowODE


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
        """VisualUNetTwoTime instance: AlphaFlowODE -> AlphaFlowEngine -> AFTrajectoryModel."""
        return self.model.model.velocity_net

    def _encode_once(self, primary_img, wrist_img):
        """Gen14 (PLAN §6.1) — encode the dual-cam window ONCE, up front."""
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
            return super().loss(trajectories, conditions)

        primary_img = conditions['primary_img'].unsqueeze(1)   # (B, 1, C, H, W)
        wrist_img   = conditions['wrist_img'].unsqueeze(1)     # (B, 1, C, H, W)
        obs_0       = conditions[0]                            # (B, 6) — snap anchor

        cond = {
            0:                obs_0,
            'visual_latent':  self._encode_once(primary_img, wrist_img),   # (B, 128)
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
            bp_imgs, inhand_imgs, obs_seq = cond[0]
            snap_obs = obs_seq[:, -1]   # (B, 6) — most recent step
            new_cond = {
                0:               snap_obs,
                'visual_latent': self._encode_once(bp_imgs, inhand_imgs),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
