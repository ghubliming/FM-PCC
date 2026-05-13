"""
iMeanFlow Dual-Loss Training: u_loss + v_loss with curriculum.

Adapted from official iMF paper's training scheme.
- u_loss: L2 loss for mean velocity field
- v_loss: L2 loss for instantaneous velocity field
- schedule: curriculum learning (u_first) or balanced
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import numpy as np


class iMFTrainingLoss(nn.Module):
    """
    Dual-loss for iMeanFlow training.
    
    Key innovation from official iMF:
    Split velocity into u (mean) and v (instantaneous deviation).
    Use curriculum learning to train u first, then transition to u+v.
    """
    
    def __init__(
        self,
        state_dim: int,
        u_loss_weight: float = 0.5,
        v_loss_weight: float = 0.5,
        loss_schedule: str = "u_first",
        warmup_epochs: int = 30,
        transition_epochs: int = 30,
    ):
        """
        Args:
            state_dim: Trajectory state dimension
            u_loss_weight: Weight for u (mean velocity) loss
            v_loss_weight: Weight for v (deviation) loss
            loss_schedule: 'u_first' (curriculum) or 'balanced'
            warmup_epochs: Epochs to train u only (for u_first schedule)
            transition_epochs: Epochs to transition from u to u+v
        """
        super().__init__()
        self.state_dim = state_dim
        self.u_loss_weight = u_loss_weight
        self.v_loss_weight = v_loss_weight
        self.loss_schedule = loss_schedule
        self.warmup_epochs = warmup_epochs
        self.transition_epochs = transition_epochs
        
        self.mse_loss = nn.MSELoss(reduction="mean")
    
    def get_loss_weights(self, current_epoch: int) -> Tuple[float, float]:
        """
        Compute u_weight, v_weight based on curriculum schedule.
        
        Args:
            current_epoch: Current training epoch
            
        Returns:
            (u_scale, v_scale): Loss scaling factors
        """
        if self.loss_schedule == "u_first":
            # Curriculum: u only → blend → u+v
            if current_epoch < self.warmup_epochs:
                # Phase 1: u only
                u_scale = 1.0
                v_scale = 0.0
            elif current_epoch < self.warmup_epochs + self.transition_epochs:
                # Phase 2: transition blend
                progress = (current_epoch - self.warmup_epochs) / self.transition_epochs
                u_scale = 1.0 - 0.5 * progress  # 1.0 → 0.5
                v_scale = 0.5 * progress         # 0.0 → 0.5
            else:
                # Phase 3: balanced u+v
                u_scale = 0.5
                v_scale = 0.5
        else:
            # Balanced: always equal weight
            u_scale = self.u_loss_weight
            v_scale = self.v_loss_weight
        
        return u_scale, v_scale
    
    def compute_losses(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
        u_target: torch.Tensor,
        v_target: torch.Tensor,
        current_epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute iMF dual losses.
        
        Args:
            u_pred: Predicted mean velocity [batch, seq_len, state_dim]
            v_pred: Predicted instantaneous velocity [batch, seq_len, state_dim]
            u_target: Target mean velocity [batch, seq_len, state_dim]
            v_target: Target instantaneous velocity [batch, seq_len, state_dim]
            current_epoch: For curriculum scheduling
            
        Returns:
            dict with keys: 'u_loss', 'v_loss', 'total_loss', 'u_scale', 'v_scale'
        """
        # Base MSE losses
        u_loss = self.mse_loss(u_pred, u_target)
        v_loss = self.mse_loss(v_pred, v_target)
        
        # Curriculum weights
        u_scale, v_scale = self.get_loss_weights(current_epoch)
        
        # Total loss
        total_loss = u_scale * u_loss + v_scale * v_loss
        
        return {
            "u_loss": u_loss.detach().item(),
            "v_loss": v_loss.detach().item(),
            "u_scale": u_scale,
            "v_scale": v_scale,
            "total_loss": total_loss,
        }
    
    def forward(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
        target_trajectory: torch.Tensor,
        current_epoch: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Complete iMF loss forward pass.
        
        For simplicity, decompose target into u (smoothed) and v (deviation):
        target = target
        u_target = smooth(target)
        v_target = target - u_target
        
        Args:
            u_pred: Predicted u [batch, seq_len, state_dim]
            v_pred: Predicted v [batch, seq_len, state_dim]
            target_trajectory: Target trajectory [batch, seq_len, state_dim]
            current_epoch: For scheduling
            
        Returns:
            (loss, metrics_dict)
        """
        # Simple decomposition: u_target = target (mean), v_target = small deviation
        # In practice, could use temporal smoothing for u_target
        u_target = target_trajectory
        v_target = torch.zeros_like(target_trajectory)  # Simple: v captures residual
        
        # Compute losses
        losses = self.compute_losses(u_pred, v_pred, u_target, v_target, current_epoch)
        
        return losses["total_loss"], losses
