"""
Improved Mean Flows (iMF) Velocity Field Module

Implements dual-velocity prediction for trajectory generation:
- u: Average velocity (global trajectory direction)
- v: Instantaneous velocity (local refinement with JVP guidance)

Based on "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
(https://arxiv.org/abs/2512.02012)

Formula: z_t = z_r + ∫_r^t u(τ) dτ + ∫_r^t v(τ) dτ
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MLP(nn.Module):
    """Simple MLP for velocity component prediction."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 3,
        activation: str = 'relu',
    ):
        super().__init__()
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'gelu':
            layers.append(nn.GELU())
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualVelocityField(nn.Module):
    """
    Dual-velocity prediction network.
    
    Learns separate u (average) and v (instantaneous) velocity components.
    Optionally includes JVP guidance for constraint satisfaction.
    
    Args:
        state_dim: Dimension of trajectory state (e.g., 28 for 7-DOF arm)
        hidden_dim: Hidden layer dimension (default: 256)
        num_layers: Number of MLP layers (default: 3)
        include_jvp: Whether to include JVP guidance module (default: False)
        activation: Activation function ('relu' or 'gelu', default: 'relu')
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        include_jvp: bool = False,
        activation: str = 'relu',
    ):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.include_jvp = include_jvp
        
        # Average velocity network (global trajectory direction)
        self.u_net = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_layers=num_layers,
            activation=activation,
        )
        
        # Instantaneous velocity network (local refinement)
        self.v_net = MLP(
            input_dim=state_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_layers=num_layers,
            activation=activation,
        )
        
        # Optional JVP encoder for constraint-aware refinement
        if include_jvp:
            self.jvp_encoder = MLP(
                input_dim=state_dim * 2,  # [x, v_base]
                hidden_dim=hidden_dim // 2,
                output_dim=state_dim,
                num_layers=2,
                activation=activation,
            )
        
        # Layer normalization for stability
        self.u_norm = nn.LayerNorm(state_dim)
        self.v_norm = nn.LayerNorm(state_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict dual velocities.
        
        Args:
            x: Trajectory state (B, state_dim) or (B, T, state_dim)
            t: Time steps (optional, B)
            cond: Conditioning information (optional, B, cond_dim)
        
        Returns:
            u: Average velocity, same shape as x
            v: Instantaneous velocity, same shape as x
        """
        # For now, simple version: predict u and v directly from x
        # (Future: condition on t and cond)
        
        # Average velocity (global direction)
        u = self.u_net(x)
        u = self.u_norm(u)
        
        # Instantaneous velocity (local refinement)
        v_base = self.v_net(x)
        
        # Optional JVP refinement
        if self.include_jvp:
            x_v_cat = torch.cat([x, v_base], dim=-1)
            v_jvp = self.jvp_encoder(x_v_cat)
            v = self.v_norm(v_base + v_jvp)
        else:
            v = self.v_norm(v_base)
        
        return u, v
    
    def compute_u_target(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract average velocity from trajectory.
        Simple approach: use first-order finite difference on smoothed trajectory.
        
        Args:
            trajectory: (B, T, state_dim) trajectory sequence
        
        Returns:
            u_target: (B, T, state_dim) average velocity
        """
        # Smooth trajectory with simple moving average
        if trajectory.shape[1] < 3:
            return torch.zeros_like(trajectory)
        
        # Average velocity as mean of all steps
        u_target = trajectory.mean(dim=1, keepdim=True).expand_as(trajectory)
        return u_target
    
    def compute_v_target(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract instantaneous velocity from trajectory.
        Approach: finite difference at each step.
        
        Args:
            trajectory: (B, T, state_dim) trajectory sequence
        
        Returns:
            v_target: (B, T, state_dim) instantaneous velocity
        """
        # Finite difference: v_t = (x_{t+1} - x_{t-1}) / 2
        T = trajectory.shape[1]
        if T < 3:
            return torch.zeros_like(trajectory)
        
        v_target = torch.zeros_like(trajectory)
        v_target[:, 1:-1] = (trajectory[:, 2:] - trajectory[:, :-2]) / 2.0
        
        # Handle boundaries with forward/backward difference
        v_target[:, 0] = trajectory[:, 1] - trajectory[:, 0]
        v_target[:, -1] = trajectory[:, -1] - trajectory[:, -2]
        
        return v_target


class TimeConditionedDualVelocity(DualVelocityField):
    """
    Enhanced dual velocity field with explicit time conditioning.
    
    Incorporates time step t into velocity prediction for better
    temporal structure learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        time_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 3,
        include_jvp: bool = False,
        activation: str = 'relu',
    ):
        super().__init__(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            include_jvp=include_jvp,
            activation=activation,
        )
        self.time_dim = time_dim
        
        # Time embedding network
        self.time_embedding = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        
        # Redefine u and v networks to accept time embedding
        self.u_net = MLP(
            input_dim=state_dim + time_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_layers=num_layers,
            activation=activation,
        )
        
        self.v_net = MLP(
            input_dim=state_dim + time_dim,
            hidden_dim=hidden_dim,
            output_dim=state_dim,
            num_layers=num_layers,
            activation=activation,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict dual velocities with time conditioning.
        
        Args:
            x: Trajectory state (B, state_dim) or (B, T, state_dim)
            t: Time steps (B,) in [0, 1]
            cond: Conditioning information (optional, B, cond_dim)
        
        Returns:
            u: Average velocity
            v: Instantaneous velocity
        """
        batch_size = x.shape[0]
        
        # Time embedding
        if t is None:
            t = torch.zeros(batch_size, device=x.device)
        
        t_embed = self.time_embedding(t.unsqueeze(-1))  # (B, time_dim)
        
        # Expand t_embed to match x shape if needed
        if x.dim() == 3:  # (B, T, state_dim)
            t_embed = t_embed.unsqueeze(1).expand(batch_size, x.shape[1], -1)
        
        # Concatenate state with time embedding
        x_t_concat = torch.cat([x, t_embed], dim=-1)
        
        # Predict velocities
        u = self.u_net(x_t_concat)
        u = self.u_norm(u)
        
        v_base = self.v_net(x_t_concat)
        
        if self.include_jvp:
            x_v_cat = torch.cat([x_t_concat, v_base], dim=-1)
            v_jvp = self.jvp_encoder(x_v_cat)
            v = self.v_norm(v_base + v_jvp)
        else:
            v = self.v_norm(v_base)
        
        return u, v


if __name__ == '__main__':
    # Test dual velocity field
    batch_size = 4
    state_dim = 28
    seq_len = 10
    
    # Create model
    model = DualVelocityField(state_dim=state_dim, hidden_dim=128)
    
    # Test single state input
    x_single = torch.randn(batch_size, state_dim)
    u, v = model(x_single)
    print(f"Single state: u={u.shape}, v={v.shape}")
    
    # Test trajectory input
    traj = torch.randn(batch_size, seq_len, state_dim)
    u_traj, v_traj = model(traj)
    print(f"Trajectory: u={u_traj.shape}, v={v_traj.shape}")
    
    # Test velocity target extraction
    u_target = model.compute_u_target(traj)
    v_target = model.compute_v_target(traj)
    print(f"Targets: u_target={u_target.shape}, v_target={v_target.shape}")
    
    # Test time-conditioned version
    model_t = TimeConditionedDualVelocity(state_dim=state_dim, time_dim=64)
    t = torch.linspace(0, 1, batch_size)
    u_t, v_t = model_t(x_single, t=t)
    print(f"Time-conditioned: u={u_t.shape}, v={v_t.shape}")
