"""Stable iMeanFlow loss helper.

This file keeps the iMF loss helper API available, but the rebuilt training
path now uses a stable FM-style main velocity loss with a small auxiliary
residual regularizer. The old curriculum-based dual-target formulation is
preserved here only as a compatibility surface for older imports.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn


class iMFTrainingLoss(nn.Module):
    """Compatibility helper for main velocity loss + small auxiliary residual."""
    
    def __init__(
        self,
        state_dim: int,
        u_loss_weight: float = 1.0,
        v_loss_weight: float = 0.1,
        loss_schedule: str = "balanced",
        warmup_epochs: int = 0,
        transition_epochs: int = 0,
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

        total_weight = float(u_loss_weight) + float(v_loss_weight)
        if total_weight <= 0:
            self.u_scale = 1.0
            self.v_scale = 0.1
        else:
            self.u_scale = float(u_loss_weight) / total_weight
            self.v_scale = float(v_loss_weight) / total_weight
        self.aux_loss_weight = max(0.01, 0.1 * float(v_loss_weight))
        
        self.mse_loss = nn.MSELoss(reduction="mean")
    
    def get_loss_weights(self, current_epoch: int) -> Tuple[float, float]:
        """Return stable compatibility weights for logging."""
        return self.u_scale, self.v_scale
    
    def compute_losses(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
        u_target: torch.Tensor,
        v_target: torch.Tensor,
        current_epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        """Compute stable main-loss + auxiliary-residual metrics."""
        u_loss = self.mse_loss(u_pred, u_target)
        v_loss = self.mse_loss(v_pred, v_target)

        u_scale, v_scale = self.get_loss_weights(current_epoch)
        total_loss = u_scale * u_loss + self.aux_loss_weight * v_loss
        
        return {
            "u_loss": u_loss,
            "v_loss": v_loss,
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
        """Compatibility forward pass used by older code paths."""
        u_target = target_trajectory
        v_target = torch.zeros_like(target_trajectory)
        losses = self.compute_losses(u_pred, v_pred, u_target, v_target, current_epoch)

        return losses["total_loss"], losses
