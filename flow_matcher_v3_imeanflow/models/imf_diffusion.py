"""
iMeanFlow Diffusion: Integration wrapper for iMF engine into FM-PCC training pipeline.

Adapts the iMeanFlowEngine to work with existing FM-PCC Trainer/Config system.
Key differences from FMv3ODE:
- Dual velocity outputs (u, v) instead of single velocity
- Curriculum loss scheduling (u_first)
- Reuses FMv3ODE's data loading and trainer loop
"""

import torch
from torch import nn
from typing import Dict, Tuple, Optional
import numpy as np

from .imf_engine import iMeanFlowEngine
from .imf_losses import iMFTrainingLoss
from .helpers import apply_conditioning


class iMFDiffusion(nn.Module):
    """
    iMeanFlow Diffusion wrapper for FM-PCC integration.
    
    Wraps iMeanFlowEngine to provide:
    - Loss computation with dual u/v losses
    - Sampling interface matching FMv3ODE
    - Config-driven training (u_weight, v_weight, loss_schedule)
    """
    
    def __init__(
        self,
        model: iMeanFlowEngine,
        horizon: int,
        observation_dim: int,
        action_dim: int,
        goal_dim: int = 0,
        n_timesteps: int = 1000,
        loss_type: str = "l2",
        clip_denoised: bool = False,
        predict_epsilon: bool = True,
        action_weight: float = 1.0,
        loss_discount: float = 1.0,
        loss_weights: Optional[Dict] = None,
        returns_condition: bool = False,
        condition_guidance_w: float = 0.1,
        # iMF-specific parameters
        u_loss_weight: float = 0.5,
        v_loss_weight: float = 0.5,
        loss_schedule: str = "u_first",
        warmup_epochs: int = 30,
        transition_epochs: int = 30,
        ode_inference_steps_v3: int = 50,
    ):
        """
        Args:
            model: iMeanFlowEngine instance
            horizon: Trajectory horizon
            observation_dim: State dimension
            action_dim: Action dimension
            goal_dim: Goal dimension (optional)
            n_timesteps: Number of training timesteps
            loss_type: Loss function type ('l1' or 'l2')
            clip_denoised: Whether to clip predictions
            predict_epsilon: Compatibility flag
            action_weight: Weight for action component in loss
            loss_discount: Discount factor for loss
            loss_weights: Custom loss weights
            returns_condition: Whether to use returns conditioning
            condition_guidance_w: Conditioning guidance weight
            # iMF-specific
            u_loss_weight: Weight for u (mean velocity) loss
            v_loss_weight: Weight for v (deviation) loss
            loss_schedule: 'u_first' (curriculum) or 'balanced'
            warmup_epochs: Epochs for warmup phase
            transition_epochs: Epochs for transition phase
            ode_inference_steps_v3: ODE steps for inference
        """
        super().__init__()
        self.model = model
        self.horizon = horizon
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.goal_dim = goal_dim
        self.transition_dim = observation_dim + action_dim
        self.returns_condition = returns_condition
        self.condition_guidance_w = condition_guidance_w
        
        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon
        self.loss_type = loss_type
        self.ode_inference_steps_v3 = int(ode_inference_steps_v3)
        
        # iMF loss handler
        self.imf_loss = iMFTrainingLoss(
            state_dim=self.transition_dim,
            u_loss_weight=u_loss_weight,
            v_loss_weight=v_loss_weight,
            loss_schedule=loss_schedule,
            warmup_epochs=warmup_epochs,
            transition_epochs=transition_epochs,
        )
        
        # Dummy buffers for interface compatibility with FMv3ODE
        self.register_buffer('betas', torch.linspace(1.0, 0.0, n_timesteps, dtype=torch.float32))
        self.register_buffer('alphas_cumprod', torch.ones(n_timesteps, dtype=torch.float32))
        
    def forward_train(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[Dict] = None,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward pass: compute iMF dual losses.
        
        Args:
            x: Noisy trajectory [batch, horizon, transition_dim]
            t: Timestep [batch]
            cond: Conditioning dict (optional)
            epoch: Current training epoch (for curriculum scheduling)
            
        Returns:
            dict with keys: 'loss', 'u_loss', 'v_loss', 'u_weight', 'v_weight'
        """
        # Apply conditioning if provided
        if cond is not None:
            x = apply_conditioning(x, conditions=cond)
        
        # iMF forward: get (u, v) predictions
        u_pred, v_pred = self.model.forward_train(x, t, cond)
        
        # Compute iMF dual losses with curriculum
        loss, metrics = self.imf_loss.forward(
            u_pred, v_pred,
            target_trajectory=x,  # Simplified: predict on full trajectory
            current_epoch=epoch,
        )
        
        # Return loss + logging metrics
        return {
            'loss': loss,
            'u_loss': metrics['u_loss'],
            'v_loss': metrics['v_loss'],
            'u_weight': metrics['u_scale'],
            'v_weight': metrics['v_scale'],
        }
    
    def sample(
        self,
        batch_size: int,
        returns: Optional[torch.Tensor] = None,
        conditions: Optional[Dict] = None,
        returns_condition: Optional[bool] = None,
        guidance_weight: Optional[float] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Sample trajectories using iMF's mean flow algorithm.
        
        API compatibility with FMv3ODE's sample method.
        
        Args:
            batch_size: Number of trajectories
            returns: Optional returns for conditioning
            conditions: Optional conditioning dict
            returns_condition: Override for returns conditioning
            guidance_weight: Classifier-free guidance weight (unused for iMF)
            num_steps: Number of ODE steps (default: ode_inference_steps_v3)
            
        Returns:
            sampled trajectory [batch_size, horizon, transition_dim]
        """
        num_steps = num_steps or self.ode_inference_steps_v3
        
        # iMF sampling: use dual velocity with balanced weights
        # (In inference, use u+v equally, unlike training's curriculum)
        sampled = self.model.sample(
            batch_size=batch_size,
            num_steps=num_steps,
            t_schedule="linear",
            u_weight=0.5,  # Balanced for inference
            v_weight=0.5,
            schedule="balanced",
            seed=0,
        )
        
        return sampled

    def loss(
        self,
        x: torch.Tensor,
        cond: Dict,
        returns: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict]:
        """Trainer entrypoint matching FM-PCC's expected `model.loss(*batch)` contract."""
        batch_size = x.shape[0]
        t = torch.rand(batch_size, device=x.device)
        return self.p_losses(x, t, returns=returns, conditions=cond)
    
    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        conditions: Optional[Dict] = None,
        epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute losses (matches FMv3ODE signature).
        
        Args:
            x_start: Clean trajectory [batch, horizon, transition_dim]
            t: Timestep [batch]
            returns: Optional returns
            conditions: Conditioning dict
            epoch: Training epoch
            
        Returns:
            (loss_tensor, metrics_dict)
        """
        # Add noise (iMF uses Flow Matching continuous time)
        # For simplicity, scale x_start by t
        noise = torch.randn_like(x_start)
        x_noisy = (1 - t[:, None, None]) * x_start + t[:, None, None] * noise
        
        # Compute iMF loss
        loss_dict = self.forward_train(x_noisy, t, conditions, epoch)
        
        return loss_dict['loss'], loss_dict
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict (compatibility)."""
        return self.model.load_state_dict(state_dict, strict=strict)
    
    def state_dict(self, destination=None, prefix='', keep_vars=False):
        """Get state dict (compatibility)."""
        return self.model.state_dict(destination, prefix, keep_vars)
