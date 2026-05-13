"""
Improved Mean Flows Training Utilities

Implements dual-velocity loss, training wrapper, and optimization utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


class DualVelocityLoss(nn.Module):
    """
    Combined loss for dual-velocity (u + v) training.
    
    L_total = α·L_u + β·L_v + γ·L_jvp
    
    Args:
        weight_u: Weight for u (average velocity) loss (default: 0.5)
        weight_v: Weight for v (instantaneous velocity) loss (default: 0.5)
        weight_jvp: Weight for JVP (constraint) loss (default: 0.0)
        loss_type: 'mse' (default) or 'l1'
    """
    
    def __init__(
        self,
        weight_u: float = 0.5,
        weight_v: float = 0.5,
        weight_jvp: float = 0.0,
        loss_type: str = 'mse',
    ):
        super().__init__()
        self.weight_u = weight_u
        self.weight_v = weight_v
        self.weight_jvp = weight_jvp
        self.loss_type = loss_type
    
    def forward(
        self,
        u_pred: torch.Tensor,
        u_target: torch.Tensor,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        jvp_penalty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined dual-velocity loss.
        
        Args:
            u_pred: Predicted average velocity (B, state_dim) or (B, T, state_dim)
            u_target: Target average velocity
            v_pred: Predicted instantaneous velocity
            v_target: Target instantaneous velocity
            jvp_penalty: Optional JVP constraint penalty (B,)
        
        Returns:
            loss: Total loss scalar
            loss_dict: Dictionary of individual loss components
        """
        # Average velocity loss
        if self.loss_type == 'mse':
            loss_u = F.mse_loss(u_pred, u_target)
            loss_v = F.mse_loss(v_pred, v_target)
        elif self.loss_type == 'l1':
            loss_u = F.l1_loss(u_pred, u_target)
            loss_v = F.l1_loss(v_pred, v_target)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # JVP constraint loss
        loss_jvp = torch.tensor(0.0, device=u_pred.device)
        if jvp_penalty is not None and self.weight_jvp > 0:
            loss_jvp = jvp_penalty.mean()
        
        # Combined loss
        total_loss = (
            self.weight_u * loss_u +
            self.weight_v * loss_v +
            self.weight_jvp * loss_jvp
        )
        
        loss_dict = {
            'loss_u': loss_u.item(),
            'loss_v': loss_v.item(),
            'loss_jvp': loss_jvp.item() if isinstance(loss_jvp, torch.Tensor) else 0.0,
            'loss_total': total_loss.item(),
        }
        
        return total_loss, loss_dict


class DualVelocityScheduler:
    """
    Scheduler for controlling loss weights during training.
    
    Supports different schedules to stabilize training:
    - balanced: Equal weights throughout
    - u_first: Strong u weight early, then balance
    - curriculum: Gradually increase v weight
    
    Args:
        mode: 'balanced', 'u_first', 'curriculum'
        total_steps: Total training steps
        weight_u_start: Starting u weight
        weight_v_start: Starting v weight
        weight_u_end: Final u weight
        weight_v_end: Final v weight
    """
    
    def __init__(
        self,
        mode: str = 'balanced',
        total_steps: int = 100000,
        weight_u_start: float = 0.5,
        weight_v_start: float = 0.01,
        weight_u_end: float = 0.5,
        weight_v_end: float = 0.5,
    ):
        self.mode = mode
        self.total_steps = total_steps
        self.weight_u_start = weight_u_start
        self.weight_v_start = weight_v_start
        self.weight_u_end = weight_u_end
        self.weight_v_end = weight_v_end
        self.step = 0
    
    def get_weights(self) -> Tuple[float, float]:
        """Get current loss weights."""
        progress = self.step / max(self.total_steps, 1)
        
        if self.mode == 'balanced':
            return self.weight_u_end, self.weight_v_end
        
        elif self.mode == 'u_first':
            # First 20% of training: strong u
            if progress < 0.2:
                return self.weight_u_start, 0.0
            else:
                # Fade in v weight linearly
                fade_progress = (progress - 0.2) / 0.8
                return (
                    self.weight_u_end,
                    self.weight_v_start + (self.weight_v_end - self.weight_v_start) * fade_progress
                )
        
        elif self.mode == 'curriculum':
            # Gradually increase v weight
            w_u = self.weight_u_start + (self.weight_u_end - self.weight_u_start) * progress
            w_v = self.weight_v_start + (self.weight_v_end - self.weight_v_start) * progress
            return w_u, w_v
        
        else:
            return self.weight_u_end, self.weight_v_end
    
    def step_forward(self) -> None:
        """Advance scheduler by one step."""
        self.step += 1


class ImfTrainingWrapper:
    """
    End-to-end training wrapper for iMF trajectory learning.
    
    Manages dual-velocity loss computation and training state.
    
    Args:
        loss_weights: Dictionary with 'u', 'v', 'jvp' weights
        scheduler: DualVelocityScheduler instance
        loss_type: 'mse' or 'l1'
    """
    
    def __init__(
        self,
        loss_weights: Optional[Dict[str, float]] = None,
        scheduler: Optional[DualVelocityScheduler] = None,
        loss_type: str = 'mse',
    ):
        self.loss_weights = loss_weights or {'u': 0.5, 'v': 0.5, 'jvp': 0.0}
        self.scheduler = scheduler or DualVelocityScheduler(mode='u_first')
        self.loss_type = loss_type
        
        self.loss_fn = DualVelocityLoss(
            weight_u=self.loss_weights['u'],
            weight_v=self.loss_weights['v'],
            weight_jvp=self.loss_weights['jvp'],
            loss_type=loss_type,
        )
    
    def compute_training_loss(
        self,
        u_pred: torch.Tensor,
        u_target: torch.Tensor,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
        jvp_penalty: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute total training loss with current schedule.
        
        Args:
            u_pred: Predicted average velocity
            u_target: Target average velocity
            v_pred: Predicted instantaneous velocity
            v_target: Target instantaneous velocity
            jvp_penalty: Optional JVP penalty
        
        Returns:
            loss: Total loss
            loss_dict: Loss components
        """
        # Get current schedule weights
        w_u, w_v = self.scheduler.get_weights()
        w_jvp = self.loss_weights.get('jvp', 0.0)
        
        # Update loss function weights
        self.loss_fn.weight_u = w_u
        self.loss_fn.weight_v = w_v
        self.loss_fn.weight_jvp = w_jvp
        
        # Compute loss
        loss, loss_dict = self.loss_fn(u_pred, u_target, v_pred, v_target, jvp_penalty)
        
        # Add schedule info to dict
        loss_dict['weight_u'] = w_u
        loss_dict['weight_v'] = w_v
        
        return loss, loss_dict
    
    def step(self) -> None:
        """Advance training scheduler."""
        self.scheduler.step_forward()


def extract_average_velocity(trajectory: torch.Tensor) -> torch.Tensor:
    """
    Extract average velocity from trajectory sequence.
    
    Approach: mean velocity across time dimension.
    
    Args:
        trajectory: (B, T, state_dim) trajectory sequence
    
    Returns:
        u_target: (B, T, state_dim) average velocity
    """
    if trajectory.dim() != 3:
        raise ValueError(f"Expected (B, T, state_dim), got {trajectory.shape}")
    
    # Mean velocity across trajectory
    u_mean = trajectory.mean(dim=1, keepdim=True)  # (B, 1, state_dim)
    u_target = u_mean.expand_as(trajectory)  # (B, T, state_dim)
    
    return u_target


def extract_instantaneous_velocity(trajectory: torch.Tensor) -> torch.Tensor:
    """
    Extract instantaneous velocity from trajectory sequence.
    
    Approach: finite differences at each timestep.
    
    Args:
        trajectory: (B, T, state_dim) trajectory sequence
    
    Returns:
        v_target: (B, T, state_dim) instantaneous velocity
    """
    if trajectory.dim() != 3:
        raise ValueError(f"Expected (B, T, state_dim), got {trajectory.shape}")
    
    B, T, D = trajectory.shape
    v_target = torch.zeros_like(trajectory)
    
    if T < 3:
        # Not enough steps for central difference
        if T >= 2:
            v_target[:, :-1] = trajectory[:, 1:] - trajectory[:, :-1]
        return v_target
    
    # Central difference for interior points: v_t = (x_{t+1} - x_{t-1}) / 2
    v_target[:, 1:-1] = (trajectory[:, 2:] - trajectory[:, :-2]) / 2.0
    
    # Forward difference for first point
    v_target[:, 0] = trajectory[:, 1] - trajectory[:, 0]
    
    # Backward difference for last point
    v_target[:, -1] = trajectory[:, -1] - trajectory[:, -2]
    
    return v_target


def compute_trajectory_targets(
    trajectory: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute both u and v targets from trajectory.
    
    Args:
        trajectory: (B, T, state_dim) trajectory
    
    Returns:
        u_target: (B, T, state_dim) average velocity
        v_target: (B, T, state_dim) instantaneous velocity
    """
    u_target = extract_average_velocity(trajectory)
    v_target = extract_instantaneous_velocity(trajectory)
    return u_target, v_target


if __name__ == '__main__':
    # Test dual-velocity loss
    batch_size = 4
    state_dim = 28
    seq_len = 10
    
    u_pred = torch.randn(batch_size, seq_len, state_dim)
    u_target = torch.randn(batch_size, seq_len, state_dim)
    v_pred = torch.randn(batch_size, seq_len, state_dim)
    v_target = torch.randn(batch_size, seq_len, state_dim)
    
    # Test loss function
    loss_fn = DualVelocityLoss(weight_u=0.5, weight_v=0.5)
    loss, loss_dict = loss_fn(u_pred, u_target, v_pred, v_target)
    print(f"Loss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    # Test scheduler
    scheduler = DualVelocityScheduler(mode='u_first', total_steps=1000)
    for step in range(0, 1000, 200):
        scheduler.step = step
        w_u, w_v = scheduler.get_weights()
        print(f"Step {step}: w_u={w_u:.3f}, w_v={w_v:.3f}")
    
    # Test training wrapper
    wrapper = ImfTrainingWrapper(scheduler=DualVelocityScheduler(mode='u_first'))
    for i in range(5):
        loss, loss_dict = wrapper.compute_training_loss(
            u_pred, u_target, v_pred, v_target
        )
        print(f"Step {i}: loss={loss.item():.4f}, w_u={loss_dict['weight_u']:.3f}")
        wrapper.step()
    
    # Test velocity extraction
    traj = torch.randn(batch_size, seq_len, state_dim)
    u_target, v_target = compute_trajectory_targets(traj)
    print(f"Trajectory: {traj.shape}")
    print(f"u_target: {u_target.shape}")
    print(f"v_target: {v_target.shape}")
