"""
Trajectory iMeanFlow Model: Dual-velocity decomposition for robotics.

Core idea from official iMF repo:
- u: mean velocity field (global trend)
- v: instantaneous velocity field (local deviation)
- Model outputs both (u, v) at each timestep
- Training: separate losses for u and v with curriculum (u_first schedule)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .unet1d_temporal_cond import Flow_matcher_U_Net_v2


class iMFTrajectoryModel(nn.Module):
    """
    Trajectory version of iMeanFlow with dual-velocity heads.
    
    Reuses FMv3ODE's U-Net backbone and adds:
    - v_head: auxiliary head for instantaneous velocity
    - Dual-output (u, v) prediction
    """
    
    def __init__(
        self,
        state_dim: int,
        seq_len: int,
        freq_dim: int = 256,
        depth: int = 8,
        num_heads: int = 4,
        mlp_dim: int = 256,
        time_dim: int = 256,
        dropout_rate: float = 0.1,
        device: str = "cuda",
    ):
        """
        Args:
            state_dim: Dimension of trajectory states
            seq_len: Trajectory horizon
            freq_dim: Feature dimension (from FMv3ODE)
            depth: Number of U-Net blocks
            num_heads: Attention heads
            mlp_dim: MLP hidden dim
            time_dim: Time embedding dimension
            dropout_rate: Dropout for regularization
            device: Device to place model on
        """
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.freq_dim = freq_dim
        self.depth = depth
        self.device = device
        
        # Reuse FMv3ODE U-Net backbone for u prediction
        self.u_net = Flow_matcher_U_Net_v2(
            horizon=seq_len,
            transition_dim=state_dim,
            cond_dim=state_dim,
            dim=freq_dim,
            dim_mults=(1, 2, 4, 8),
            returns_condition=False,
            condition_dropout=dropout_rate,
        )
        
        # Auxiliary v-head: instantaneous velocity (from official iMF)
        # Lightweight head that branches off u_net features
        self.v_head = nn.Sequential(
            nn.Linear(freq_dim, freq_dim),
            nn.ReLU(),
            nn.Linear(freq_dim, state_dim),
        )
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict dual velocity components (u, v).
        
        # u prediction via backbone
        u = self.u_net(x, cond, t)
            t: Timestep [batch] or [batch, 1]
            cond: Optional conditioning [batch, cond_dim]
            
        Returns:
            (u, v): Dual velocity predictions
                u [batch, seq_len, state_dim] - mean velocity (FMv3ODE baseline)
                v [batch, seq_len, state_dim] - instantaneous velocity (iMF deviation)
        """
        # u prediction via backbone
        u = self.u_net(x, t, cond)
        
        # v prediction via auxiliary head
        # For simplicity: v shares intermediate features with u
        # In practice, could extract pre-output features from u_net
        v = self.v_head(u)  # Simple additive decomposition
        
        return u, v

    def forward_train(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training entrypoint expected by iMFDiffusion."""
        return self.forward(x_noisy, t, cond)
    
    def sample_trajectory(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int,
        t_steps: torch.Tensor,
        schedule: str = "u_first",
        u_weight: float = 0.5,
        v_weight: float = 0.5,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Sample trajectory using iMF sampling (from official repo pattern).
        
        Args:
            batch_size: Number of trajectories
            seq_len: Length of trajectory
            num_steps: Number of sampling steps
            t_steps: Time schedule [num_steps+1]
            schedule: 'u_first' (curriculum) or 'balanced'
            u_weight: Weight for u component
            v_weight: Weight for v component
            device: Device
            
        Returns:
            sampled trajectory [batch_size, seq_len, state_dim]
        """
        # Initialize from Gaussian noise
        z_t = torch.randn(batch_size, seq_len, self.state_dim, device=device)
        
        t_steps = t_steps.to(device)
        
        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            h = t - r  # Time difference
            
            # Get dual velocity at current step
            u, v = self.forward(z_t, t.expand(batch_size))
            
            # Weighted combination based on schedule
            if schedule == "u_first":
                # Curriculum: early epochs use u only, transition to u+v
                # (controlled by u_weight, v_weight in training config)
                velocity = u_weight * u + v_weight * v
            elif schedule == "balanced":
                # Always balance
                velocity = 0.5 * u + 0.5 * v
            else:
                # Default: just u (fallback to FMv3ODE)
                velocity = u
            
            # ODE step (Euler integration, matching iMF's official sampling)
            z_t = z_t - h[:, None, None] * velocity
        
        return z_t

    def sample(
        self,
        batch_size: int,
        num_steps: int = 1,
        t_schedule: str = "linear",
        u_weight: float = 0.5,
        v_weight: float = 0.5,
        schedule: str = "u_first",
        seed: int = 0,
    ) -> torch.Tensor:
        """Sampling entrypoint expected by iMFDiffusion."""
        t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device)
        return self.sample_trajectory(
            batch_size=batch_size,
            seq_len=self.seq_len,
            num_steps=num_steps,
            t_steps=t_steps,
            schedule=schedule,
            u_weight=u_weight,
            v_weight=v_weight,
            device=self.device,
        )
