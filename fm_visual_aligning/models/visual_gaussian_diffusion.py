import torch
from fm_visual_aligning.models.diffusion import GaussianDiffusion
from fm_visual_aligning.models.helpers import apply_conditioning


class VisualGaussianDiffusion(GaussianDiffusion):
    """
    DDPM engine for Visual-DPCC (Gen6V4).

    Extends GaussianDiffusion with:
    - Explicit loss(trajectories, conditions) — matches Batch namedtuple unpacking
      by Trainer.train_epoch():  loss, infos = self.model.loss(*batch)
      *batch unpacks Batch(trajectories, conditions) → loss(trajectories, conditions)
    - Selective action-only clamp in p_mean_variance (avoids over-clipping obs)
    - Vision-conditioned forward() for closed-loop inference

    Trajectory: 9D = [act(0:3) | des_c_pos(3:6) | c_pos(6:9)]
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
        # base GaussianDiffusion.__init__ (which has no **kwargs).
        super().__init__(*args, **kwargs)

    # ── training ──────────────────────────────────────────────────────────────

    def loss(self, trajectories, conditions):
        """
        Called as self.model.loss(*batch) where batch is Batch(trajectories, conditions).

        trajectories: (B, H, 9)   — [act(3) | des_pos(3) | c_pos(3)] normalized
        conditions:   dict {
            0:             (B, 6)   — obs anchor for apply_conditioning at t=0
            'primary_img': (B,C,H,W) — agentview camera
            'wrist_img':   (B,C,H,W) — wrist camera
        }
        """
        # unsqueeze to window_size=1 for MultiImageObsEncoder: (B,C,H,W) → (B,1,C,H,W)
        primary_img = conditions['primary_img'].unsqueeze(1)   # (B, 1, C, H, W)
        wrist_img   = conditions['wrist_img'].unsqueeze(1)     # (B, 1, C, H, W)
        obs_0       = conditions[0]                             # (B, 6) — snap anchor
        obs_seq     = trajectories[..., self.action_dim:]      # (B, H, 6) — proprio context

        cond = {
            'visual': (primary_img, wrist_img, obs_seq),
            # 0 key used by apply_conditioning in p_sample_loop to snap x[:,0,action_dim:]
            0: obs_0,
        }

        x = trajectories                                        # (B, H, 9)
        batch_size = len(x)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
        beta = torch.tensor(self.time_beta_beta_v3, device=x.device)
        beta_dist = torch.distributions.Beta(alpha, beta)
        t = beta_dist.sample((batch_size,))
        t = 1.0 - t
        return self.p_losses(x, cond, t)



    # ── inference ─────────────────────────────────────────────────────────────

    def forward(self, cond, *args, **kwargs):
        """
        Closed-loop inference entry point.

        Expected cond format from VisualAgentWrapper:
            cond = {0: (bp_image_seq, inhand_image_seq, obs_6d_seq)}
        where each is (B, window_size, ...).

        Transforms to the internal format used by p_sample_loop:
            {0: obs_6d_at_last_step,   ← snapping anchor
             'visual': (bp_imgs, inhand_imgs, obs_6d_seq)}
        """
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            bp_imgs, inhand_imgs, obs_seq = cond[0]
            # obs_seq: (B, window_size, 6)
            # Use the most recent obs as the apply_conditioning anchor
            snap_obs = obs_seq[:, -1]   # (B, 6)
            new_cond = {
                0:        snap_obs,
                'visual': (bp_imgs, inhand_imgs, obs_seq),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
