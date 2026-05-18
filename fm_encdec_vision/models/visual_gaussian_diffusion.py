import torch
from flow_matcher_v3_ode_selectable.models.diffusion import GaussianDiffusion
from diffuser.models.helpers import apply_conditioning

class VisualGaussianDiffusion(GaussianDiffusion):
    """
    Overrides Flow Matching GaussianDiffusion to handle vision-specific batch format,
    continuous-time Beta distribution sampling, and image token conditioning.
    """
    
    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        """
        Flow Matching continuous-time training loss:
        Linear interpolation path x_t = (1-t)*x_base + t*x_start.
        """
        # Trajectory x: [batch, horizon, transition_dim] (act: 3D, obs: 3D)
        x = torch.cat([act, obs], dim=-1)
        
        # Condition dict containing image tokens and snapping key
        cond = {
            'visual': (bp_imgs, inhand_imgs, obs),
            0: obs[:, 0]  # Snapping boundaries
        }
        
        # Draw continuous time t from Beta(alpha, beta) distribution
        batch_size = len(x)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
        beta = torch.tensor(self.time_beta_beta_v3, device=x.device)
        beta_dist = torch.distributions.Beta(alpha, beta)
        t = beta_dist.sample((batch_size,))
        t = 1.0 - t  # D3IL standard continuous shift
        
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        """
        Inference: Triggers the Flow Matching ODE/Euler integration loop (p_sample_loop).
        Handles vision-specific tuple unpacking and batch repetition.
        """
        # 1. Unpack tuples passed from Aligning_Sim.test_agent()
        if 0 in cond and isinstance(cond[0], tuple):
            bp_imgs, inhand_imgs, pos = cond[0]
            
            # Create a clean 'visual' cond for VisualUNet
            visual_cond = (bp_imgs, inhand_imgs, pos)
            # Snapping context at t=0 (last frame in history)
            snapping_cond = {0: pos[:, -1]} 
            
            new_cond = snapping_cond.copy()
            new_cond['visual'] = visual_cond
        else:
            new_cond = cond

        # Forward integration: t=0 → t=1 (noise → action plan)
        return super().forward(new_cond, *args, **kwargs)
