import torch
from diffuser_visual_aligning.models.diffusion import GaussianDiffusion
from diffuser_visual_aligning.models.helpers import apply_conditioning


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
        batch_size = x.shape[0]
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    # ── override p_mean_variance for selective clamping ───────────────────────

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        """
        Override to clamp only action dims (not obs dims).

        Base class does x_recon.clamp_(-1, 1) on the entire trajectory.
        That's too aggressive for action velocities which can temporarily
        exceed ±1 in normalized space before SLSQP projection snaps them back.
        We clamp actions to ±5 (generous safe range) and leave obs unclamped.
        """
        if self.returns_condition:
            epsilon_cond   = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w * (epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t_int = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t_int, noise=epsilon)

        if self.clip_denoised:
            # Clamp action dims only — obs dims stay as predicted.
            # Wide ±5 range: avoids over-clipping high-velocity actions in early
            # denoising steps while keeping gradients stable.
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
