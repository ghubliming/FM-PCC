import torch
from flow_matcher_v3_ode_selectable.models.diffusion import GaussianDiffusion
from flow_matcher_v3_ode_selectable.models.helpers import apply_conditioning

class VisualGaussianDiffusion(GaussianDiffusion):
    """
    Overrides GaussianDiffusion to handle vision-specific batch format and conditioning.
    """
    
    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        """
        Training loss: (bp_imgs, inhand_imgs, obs, act, mask)
        """
        # Trajectory x: [batch, horizon, transition_dim]
        # Aligning_Img_Dataset: act is velocity [B, T, 3], obs is robot_pos [B, T, 3]
        x = torch.cat([act, obs], dim=-1)
        
        # In FM-PCC, cond is passed to the model during loss calculation.
        # We pass a dict so VisualUNet can extract the visual components.
        cond = {
            'visual': (bp_imgs, inhand_imgs, obs),
            0: obs[:, 0] # Include first-frame state for potential snapping-based loss
        }
        
        return super().loss(x, cond)

    def forward(self, cond, *args, **kwargs):
        """
        Inference: cond is usually a dict {0: state_vector} from Policy.
        For vision, the state_vector is actually a tuple (bp, inhand, pos).
        """
        # 1. Extract visual components and state for snapping
        if 0 in cond and isinstance(cond[0], tuple):
            bp_imgs, inhand_imgs, pos = cond[0]
            
            # Since Policy's utils.apply_dict doesn't recurse into tuples for einops.repeat,
            # we must handle the batch repetition here if it's not already repeated.
            # (Policy usually passes batch_size=1 for sim, but we should be robust).
            
            # Create a clean 'visual' cond for the model
            visual_cond = (bp_imgs, inhand_imgs, pos)
            # Create a clean 'state' cond for apply_conditioning (snapping t=0)
            snapping_cond = {0: pos[:, 0]} # pos is [B, 1, 3]
            
            # Combine them for the rest of the engine
            new_cond = snapping_cond.copy()
            new_cond['visual'] = visual_cond
        else:
            new_cond = cond

        return super().forward(new_cond, *args, **kwargs)
