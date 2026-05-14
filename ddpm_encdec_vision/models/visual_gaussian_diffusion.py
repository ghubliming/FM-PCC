import torch
from flow_matcher_v3_ode_selectable.models.diffusion import GaussianDiffusion

class VisualGaussianDiffusion(GaussianDiffusion):
    """
    Overrides GaussianDiffusion to handle vision-specific batch format:
    (bp_imgs, inhand_imgs, obs, act, mask)
    """
    
    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        """
        Entrypoint for Trainer.
        batch = (bp_imgs, inhand_imgs, obs, act, mask)
        """
        # x is the trajectory to be modeled: [batch, horizon, transition_dim]
        # In Aligning_Img_Dataset, obs is robot_pos [B, T, 3], act is velocity [B, T, 3].
        # We concatenate them as the target trajectory.
        x = torch.cat([act, obs], dim=-1)
        
        # cond for VisualUNet is (bp_imgs, inhand_imgs, obs)
        cond = (bp_imgs, inhand_imgs, obs)
        
        # Call base class loss(x, cond)
        return super().loss(x, cond)

    def forward(self, cond, *args, **kwargs):
        """
        Inference entrypoint.
        cond in Policy is usually a dict {0: obs_state}.
        For vision, we need to adapt this.
        """
        return super().forward(cond, *args, **kwargs)
