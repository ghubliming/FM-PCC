import torch
from fm_visual_avoiding.models.diffusion import FlowMatchingODE
from fm_visual_avoiding.models.helpers import apply_conditioning


class VisualFlowMatching(FlowMatchingODE):
    """
    FM engine for Visual-DPCC on AVOIDING (Gen9 Epoch 2).

    Trajectory: 6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)]   (2-D avoiding plane)
    Single camera: bp-cam only — no inhand/wrist cam for avoiding.
    """

    def __init__(self, *args,
                 ode_solver_backend_v3='legacy_euler',
                 ode_solver_method_v3='euler',
                 ode_solver_rtol_v3=None,
                 ode_solver_atol_v3=None,
                 ode_solver_step_size_v3=None,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def loss(self, trajectories, conditions):
        """
        Visual (if_vision=True):
            trajectories: (B, H, 6)   — [act(2) | des_xy(2) | c_xy(2)]
            conditions:   {0: (B,4), 'primary_img': (B,C,H,W)}   ← NO wrist_img

        Non-visual fallback:
            trajectories: (B, H, traj_dim) — routed to base p_losses
        """
        if not self.model.if_vision:
            x = trajectories
            batch_size = len(x)
            alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
            beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
            t = 1.0 - torch.distributions.Beta(alpha, beta).sample((batch_size,))
            return self.p_losses(x, conditions, t)

        # Visual path — single camera (avoiding)
        primary_img = conditions['primary_img'].unsqueeze(1)   # (B, 1, C, H, W)
        obs_0       = conditions[0]                             # (B, 4) — snap anchor
        obs_seq     = trajectories[..., self.action_dim:]      # (B, H, 4)

        cond = {
            'visual': (primary_img, obs_seq),                  # single-cam tuple
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

    def forward(self, cond, *args, **kwargs):
        """
        Expected cond format from VisualAgentWrapper (single-cam avoiding):
            cond = {0: (bp_image_seq, obs_4d_seq)}

        Transforms to internal format used by p_sample_loop:
            {0: obs_4d_at_last_step,           ← snapping anchor (B, 4)
             'visual': (bp_imgs, obs_4d_seq)}
        """
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            payload = cond[0]
            bp_imgs = payload[0]
            obs_seq = payload[-1]
            snap_obs = obs_seq[:, -1]   # (B, 4)
            new_cond = {
                0:        snap_obs,
                'visual': (bp_imgs, obs_seq),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
