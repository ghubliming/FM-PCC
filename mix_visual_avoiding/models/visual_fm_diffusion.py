import torch
from mix_visual_avoiding.models.fm_diffusion import FlowMatchingODE
from mix_visual_avoiding.models.helpers import apply_conditioning
# Gen16 — cameras and dims come from the task spec, never from a literal here.
from mix_visual_avoiding.models import visual_spec


class VisualFlowMatching(FlowMatchingODE):
    """
    FM engine for Visual-DPCC (Gen6V4).

    Extends FlowMatchingODE with:
    - Explicit loss(trajectories, conditions) — matches Batch namedtuple unpacking
      by Trainer.train_epoch():  loss, infos = self.model.loss(*batch)
      *batch unpacks Batch(trajectories, conditions) → loss(trajectories, conditions)
    - Selective action-only clamp in p_mean_variance (avoids over-clipping obs)
    - Vision-conditioned forward() for closed-loop inference

    Trajectory: 6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)]   (2-D avoiding plane)
    Single camera: bp-cam only — no wrist cam (visual_spec).
    """

    # ── initialization ────────────────────────────────────────────────────────

    def __init__(self, *args,
                 ode_solver_backend_v3='legacy_euler',
                 ode_solver_method_v3='euler',
                 ode_solver_rtol_v3=None,
                 ode_solver_atol_v3=None,
                 ode_solver_step_size_v3=None,
                 **kwargs):
        # Intercept all ODE solver params so they don't cause TypeError in the
        # base FlowMatchingODE.__init__ (which has no **kwargs).
        super().__init__(*args, **kwargs)

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
        if not self.model.if_vision:
            # Non-visual: no image keys — route directly to base p_losses.
            # apply_conditioning uses cond[0] (20D obs anchor) to pin obs at step 0.
            x = trajectories
            batch_size = len(x)
            alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
            beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
            t = 1.0 - torch.distributions.Beta(alpha, beta).sample((batch_size,))
            return self.p_losses(x, conditions, t)

        # Visual path — Gen16: the camera set is whatever visual_spec declares.
        cam_imgs = visual_spec.images_from_conditions(conditions)  # N x (B, 1, C, H, W)
        obs_0    = conditions[0]                                   # (B, OBS_DIM) snap anchor
        obs_seq  = trajectories[..., self.action_dim:]             # (B, H, OBS_DIM)

        cond = {
            'visual': visual_spec.pack_visual(cam_imgs, obs_seq),
            0: obs_0,
        }

        x = trajectories
        batch_size = len(x)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
        beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
        beta_dist = torch.distributions.Beta(alpha, beta)
        t = beta_dist.sample((batch_size,))
        t = 1.0 - t
        return self.p_losses(x, cond, t)



    # ── inference ─────────────────────────────────────────────────────────────

    def forward(self, cond, *args, **kwargs):
        """
        Closed-loop inference entry point.

        Expected cond format from the eval policy:
            cond = {0: (*camera_image_seqs, obs_seq)}
        where each is (B, window_size, ...) and there are visual_spec.N_CAMERAS
        image sequences (Gen16 avoiding: one, bp-cam).

        Transforms to the internal format used by p_sample_loop:
            {0: obs_at_last_step,   ← snapping anchor
             'visual': (*camera_imgs, obs_seq)}
        """
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            cam_imgs, obs_seq = visual_spec.split_visual(cond[0])
            # Use the most recent obs as the apply_conditioning anchor
            snap_obs = obs_seq[:, -1]   # (B, OBS_DIM)
            new_cond = {
                0:        snap_obs,
                'visual': visual_spec.pack_visual(cam_imgs, obs_seq),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
