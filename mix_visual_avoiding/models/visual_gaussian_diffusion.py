import torch
from mix_visual_avoiding.models.diffusion import GaussianDiffusion
from mix_visual_avoiding.models.helpers import apply_conditioning
# Gen16 — cameras and dims come from the task spec, never from a literal here.
from mix_visual_avoiding.models import visual_spec


class VisualGaussianDiffusion(GaussianDiffusion):
    """
    DDPM engine for Visual-DPCC (Gen6V4).

    Extends GaussianDiffusion with:
    - Explicit loss(trajectories, conditions) — matches Batch namedtuple unpacking
      by Trainer.train_epoch():  loss, infos = self.model.loss(*batch)
      *batch unpacks Batch(trajectories, conditions) → loss(trajectories, conditions)
    - Selective action-only clamp in p_mean_variance (avoids over-clipping obs)
    - Vision-conditioned forward() for closed-loop inference

    Trajectory: 6D = [act(0:2) | des_xy(2:4) | c_xy(4:6)]   (2-D avoiding plane)
    Single camera: bp-cam only — no wrist cam (visual_spec).
    """

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
            x = trajectories
            batch_size = x.shape[0]
            t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
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
