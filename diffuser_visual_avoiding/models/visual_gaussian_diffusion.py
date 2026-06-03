import torch
from diffuser_visual_avoiding.models.diffusion import GaussianDiffusion
from diffuser_visual_avoiding.models.helpers import apply_conditioning


class VisualGaussianDiffusion(GaussianDiffusion):
    """
    DDPM engine for Visual-DPCC on AVOIDING (Gen9 Epoch 2).

    Extends GaussianDiffusion with:
    - Explicit loss(trajectories, conditions) — matches Batch namedtuple unpacking
      by Trainer.train_epoch():  loss, infos = self.model.loss(*batch)
    - Selective action-only clamp in p_mean_variance (avoids over-clipping obs)
    - Vision-conditioned forward() for closed-loop inference

    Trajectory: 6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)]   (2-D avoiding plane)
    Single camera: bp-cam only — no inhand/wrist cam for avoiding.
    """

    # ── training ──────────────────────────────────────────────────────────────

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
            batch_size = x.shape[0]
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
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
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    # ── override p_mean_variance for selective clamping ───────────────────────

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        if self.returns_condition:
            epsilon_cond   = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t_int = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t_int, noise=epsilon)

        if self.clip_denoised:
            x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t_int)

        if projector is not None and projector.gradient:
            if self.goal_dim > 0:
                grad = projector.compute_gradient(x_recon[:, :, :-self.goal_dim], constraints)
            else:
                grad = projector.compute_gradient(x_recon, constraints)
            model_mean = model_mean + grad

        return model_mean, posterior_variance, posterior_log_variance

    # ── inference ─────────────────────────────────────────────────────────────

    def forward(self, cond, *args, **kwargs):
        """
        Closed-loop inference entry point.

        Expected cond format from VisualAgentWrapper (single-cam avoiding):
            cond = {0: (bp_image_seq, obs_4d_seq)}
        where each is (B, window_size, ...).

        Transforms to internal format used by p_sample_loop:
            {0: obs_4d_at_last_step,           ← snapping anchor (B, 4)
             'visual': (bp_imgs, obs_4d_seq)}
        """
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            payload = cond[0]
            # Accept (bp_imgs, obs_seq) for single-cam avoiding
            bp_imgs = payload[0]
            obs_seq = payload[-1]   # last element is always obs
            snap_obs = obs_seq[:, -1]   # (B, 4)
            new_cond = {
                0:        snap_obs,
                'visual': (bp_imgs, obs_seq),
            }
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
