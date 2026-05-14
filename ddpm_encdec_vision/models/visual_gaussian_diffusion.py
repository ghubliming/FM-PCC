import torch
from diffuser.models.diffusion import GaussianDiffusion
from diffuser.models.helpers import apply_conditioning

class VisualGaussianDiffusion(GaussianDiffusion):
    """
    Overrides GaussianDiffusion to handle vision-specific batch format and conditioning
    for the standard DDPM (Noise Prediction) baseline.
    """
    
    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        """
        Training loss: (bp_imgs, inhand_imgs, obs, act, mask)
        Standard DDPM loss (predict_epsilon).
        """
        # Trajectory x: [batch, horizon, transition_dim]
        # act: [B, T, 3], obs: [B, T, 3]
        x = torch.cat([act, obs], dim=-1)
        
        # Condition dict for VisualUNet
        cond = {
            'visual': (bp_imgs, inhand_imgs, obs),
            0: obs[:, 0] # First-frame state for snapping
        }
        
        # Standard DDPM training: sample t and calculate p_losses
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        """
        Inference: Triggers the stochastic DDPM denoising loop (p_sample_loop).
        Handles vision-specific tuple unpacking and batch repetition.
        """
        # 1. Handle vision-specific cond unpacking
        if 0 in cond and isinstance(cond[0], tuple):
            # Extract: (bp_imgs, inhand_imgs, pos)
            bp_imgs, inhand_imgs, pos = cond[0]
            
            # Policy usually repeats tensors for batch_size, but doesn't handle tuples.
            # Create a clean 'visual' cond for VisualUNet
            visual_cond = (bp_imgs, inhand_imgs, pos)
            # Create a clean 'state' cond for apply_conditioning (snapping t=0)
            snapping_cond = {0: pos[:, 0]} 
            
            new_cond = snapping_cond.copy()
            new_cond['visual'] = visual_cond
        else:
            new_cond = cond

        # Calls self.conditional_sample -> self.p_sample_loop
        return super().forward(new_cond, *args, **kwargs)
