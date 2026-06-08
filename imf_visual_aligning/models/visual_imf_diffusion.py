import torch
from imf_visual_aligning.models.imf_diffusion import iMeanFlowODE
from imf_visual_aligning.models.helpers import apply_conditioning


class VisualIMF(iMeanFlowODE):
    """
    iMF engine for Visual-PCC (Gen8).

    Extends iMeanFlowODE with:
    - Visual loss(trajectories, conditions) — FiLM image conditioning via VisualUNet
    - Visual forward() for closed-loop inference
    - All iMF features inherited: h-conditioning, u/v dual heads, mean-flow target

    Trajectory: 9D = [act(0:3) | des_c_pos(3:6) | c_pos(6:9)]
    """

    def __init__(self, *args, if_vision: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.if_vision = if_vision

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
            # Non-visual: route directly to iMF p_losses with the raw condition dict.
            x = trajectories
            batch_size = len(x)
            alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
            beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
            t = 1.0 - torch.distributions.Beta(alpha, beta).sample((batch_size,))
            return self.p_losses(x, conditions, t)

        # Visual path — build FiLM-compatible cond dict from batch conditions.
        primary_img = conditions['primary_img'].unsqueeze(1)   # (B, 1, C, H, W)
        wrist_img   = conditions['wrist_img'].unsqueeze(1)     # (B, 1, C, H, W)
        obs_0       = conditions[0]                             # (B, 6) — snap anchor
        obs_seq     = trajectories[..., self.action_dim:]      # (B, H, 6)

        cond = {
            'visual': (primary_img, wrist_img, obs_seq),
            0: obs_0,
        }

        x = trajectories
        batch_size = len(x)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
        beta  = torch.tensor(self.time_beta_beta_v3, device=x.device)
        t = 1.0 - torch.distributions.Beta(alpha, beta).sample((batch_size,))
        return self.p_losses(x, cond, t)

    # ── inference ─────────────────────────────────────────────────────────────

    def forward(self, cond, *args, **kwargs):
        """
        Closed-loop inference entry point.

        Expected cond format from VisualAgentWrapper:
            cond = {0: (bp_image_seq, inhand_image_seq, obs_6d_seq)}
        where each is (B, window_size, ...).

        Transforms to the internal format used by p_sample_loop:
            {0: obs_6d_at_last_step,   ← apply_conditioning anchor
             'visual': (bp_imgs, inhand_imgs, obs_6d_seq)}
        """
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            bp_imgs, inhand_imgs, obs_seq = cond[0]
            snap_obs = obs_seq[:, -1]   # (B, 6) — most recent step
            new_cond = {
                0:        snap_obs,
                'visual': (bp_imgs, inhand_imgs, obs_seq),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
