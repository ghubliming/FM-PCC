"""
Trajectory iMeanFlow Model: dual-velocity decomposition for robotics.

This is a minimal, consistent wrapper around the existing FMv3-style U-Net.
It exposes the API expected by iMFDiffusion:
- forward_train(x, t, cond)
- sample(batch_size, ...)
- forward(x, t, cond)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .unet1d_temporal_cond import Flow_matcher_U_Net_v2


class iMFTrajectoryModel(nn.Module):
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
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.freq_dim = freq_dim
        self.depth = depth
        self.device = device

        self.u_net = Flow_matcher_U_Net_v2(
            horizon=seq_len,
            transition_dim=state_dim,
            cond_dim=state_dim,
            dim=freq_dim,
            dim_mults=(1, 2, 4, 8),
            returns_condition=False,
            condition_dropout=dropout_rate,
        )

        self.v_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.ReLU(),
            nn.Linear(state_dim, state_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict dual velocity components (u, v)."""
        u = self.u_net(x, cond, t)
        v = self.v_head(u)
        return u, v

    def forward_train(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Sample a trajectory with a simple Euler ODE loop."""
        device = device or self.device
        z_t = torch.randn(batch_size, seq_len, self.state_dim, device=device)
        t_steps = t_steps.to(device)

        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            h = t - r
            u, v = self.forward(z_t, t.expand(batch_size))

            if schedule == "u_first":
                velocity = u_weight * u + v_weight * v
            elif schedule == "balanced":
                velocity = 0.5 * u + 0.5 * v
            else:
                velocity = u

            z_t = z_t - h * velocity

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
        torch.manual_seed(seed)

        if t_schedule == "quadratic":
            t_steps = torch.linspace(1.0, 0.0, num_steps + 1, device=self.device) ** 2
        else:
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
