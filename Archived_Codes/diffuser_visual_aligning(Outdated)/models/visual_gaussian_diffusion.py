import torch
from diffuser_visual_aligning.models.diffusion import GaussianDiffusion
from diffuser_visual_aligning.models.helpers import apply_conditioning

class VisualGaussianDiffusion(GaussianDiffusion):
    """
    Overrides diffuser_visual_aligning GaussianDiffusion to handle vision-specific batch format and conditioning
    for the standard DDPM (Noise Prediction) baseline.
    """
    
    def loss(self, *args):
        """
        Supports both vision (5 arguments) and state-only (3 arguments) batches.
        """
        if getattr(self.model, 'if_vision', True):
            # Vision mode batch: bp_imgs, inhand_imgs, obs, act, mask
            bp_imgs, inhand_imgs, obs, act, mask = args
            x = torch.cat([act, obs], dim=-1)
            cond = {
                'visual': (bp_imgs, inhand_imgs, obs),
                0: obs[:, 0] # First-frame state for snapping
            }
        else:
            # State-only mode batch: obs, act, mask
            obs, act, mask = args
            x = torch.cat([act, obs], dim=-1)
            cond = {
                0: obs[:, 0] # First-frame state for snapping
            }
        
        # Standard DDPM training: sample t and calculate p_losses
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def p_mean_variance(self, x, cond, t, returns=None, projector=None, constraints=None):
        """
        Overridden to support safe z-score action clamping and eliminate RuntimeError crashes.
        """
        if self.returns_condition:
            epsilon_cond = self.model(x, cond, t, returns, use_dropout=False)
            epsilon_uncond = self.model(x, cond, t, returns, force_dropout=True)
            epsilon = epsilon_uncond + self.condition_guidance_w*(epsilon_cond - epsilon_uncond)
        else:
            epsilon = self.model(x, cond, t)

        t = t.detach().to(torch.int64)
        x_recon = self.predict_start_from_noise(x, t=t, noise=epsilon)

        if self.clip_denoised:
            # --- D3IL DDPM-ACT COMPATIBILITY CLAMP ---
            # We ONLY clamp the predicted action dimensions (first self.action_dim columns)
            # to a safe wide range, and NEVER clamp the observation/proprioceptive channels.
            x_recon[..., :self.action_dim].clamp_(-5.0, 5.0)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
                x_start=x_recon, x_t=x, t=t)

        if projector is not None and projector.gradient:
            if self.goal_dim > 0:
                grad = projector.compute_gradient(x_recon[:,:,:-self.goal_dim], constraints)
            else:
                grad = projector.compute_gradient(x_recon, constraints)
            model_mean = model_mean + grad

        return model_mean, posterior_variance, posterior_log_variance

    def forward(self, cond, *args, **kwargs):
        """
        Inference: Triggers the stochastic DDPM denoising loop (p_sample_loop).
        Handles vision-specific tuple unpacking and batch repetition.
        """
        if getattr(self.model, 'if_vision', True):
            # 1. Handle vision-specific cond unpacking
            if 0 in cond and isinstance(cond[0], tuple):
                # Extract: (bp_imgs, inhand_imgs, pos)
                bp_imgs, inhand_imgs, pos = cond[0]
                
                # Policy usually repeats tensors for batch_size, but doesn't handle tuples.
                # Create a clean 'visual' cond for VisualUNet
                visual_cond = (bp_imgs, inhand_imgs, pos)
                # Create a clean 'state' cond for apply_conditioning (snapping t=0)
                # pos is the context window [B, window_size, 3]. 
                # In training, obs[:, 0] was the current state. 
                # In eval, pos[:, -1] is the current state.
                snapping_cond = {0: pos[:, -1]} 
                
                new_cond = snapping_cond.copy()
                new_cond['visual'] = visual_cond
            else:
                new_cond = cond
        else:
            # 2. State-only cond unpacking: cond is {0: obs_seq}
            if 0 in cond and isinstance(cond[0], torch.Tensor):
                obs_seq = cond[0]
                new_cond = {0: obs_seq[:, -1]}
            else:
                new_cond = cond

        # Calls self.conditional_sample -> self.p_sample_loop
        return super().forward(new_cond, *args, **kwargs)
