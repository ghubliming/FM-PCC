import torch
import torch.nn as nn
from .imf_velocity import TimeConditionedDualVelocity
from ..utils.imf_training import (
    ImfTrainingWrapper,
    DualVelocityScheduler,
    compute_trajectory_targets,
)

class ImfDiffusion(nn.Module):
    """
    Diffusion-style wrapper for iMeanFlow models.
    Provides a .loss() method compatible with the standard Trainer.
    """
    
    def __init__(
        self,
        model,
        horizon,
        observation_dim,
        action_dim,
        goal_dim=0,
        u_loss_weight=0.5,
        v_loss_weight=0.5,
        jvp_weight=0.0,
        loss_type='mse',
        loss_schedule='u_first',
        n_train_steps=1e5,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        
        # Initialize training wrapper and scheduler
        self.scheduler = DualVelocityScheduler(
            mode=loss_schedule,
            total_steps=int(n_train_steps),
        )
        self.wrapper = ImfTrainingWrapper(
            loss_weights={'u': u_loss_weight, 'v': v_loss_weight, 'jvp': jvp_weight},
            scheduler=self.scheduler,
            loss_type=loss_type
        )
        
    def loss(self, x, cond, returns=None):
        """
        Compute iMeanFlow training loss.
        
        Args:
            x: (B, T, state_dim) trajectory
            cond: (B, cond_dim) conditioning
            returns: (B, 1) optional returns
            
        Returns:
            loss: Total loss
            info: Loss components
        """
        batch_size, horizon, state_dim = x.shape
        
        # 1. Compute trajectory targets (u and v)
        u_target, v_target = compute_trajectory_targets(x)
        
        # 2. Sample time steps (matching FMv3 style or using linear spacing for iMF)
        # iMF usually uses the whole trajectory or sampled points. 
        # Here we use the full sequence as in the original iMF implementation.
        t = torch.linspace(0, 1, horizon, device=x.device).unsqueeze(0).expand(batch_size, horizon)
        
        # 3. Forward pass
        u_pred, v_pred = self.model(x, t)
        
        # 4. Compute loss
        loss, info = self.wrapper.compute_training_loss(
            u_pred=u_pred,
            u_target=u_target,
            v_pred=v_pred,
            v_target=v_target,
        )
        
        # Step the scheduler
        self.wrapper.step()
        
        return loss, info

    def forward(self, *args, **kwargs):
        # Forward pass usually delegates to model
        return self.model(*args, **kwargs)
