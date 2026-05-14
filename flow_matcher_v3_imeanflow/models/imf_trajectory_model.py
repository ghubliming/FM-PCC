"""Trajectory iMeanFlow model built on the FMv3-style U-Net.

The backbone predicts the FM-style flow velocity. A small auxiliary residual
head remains to preserve the iMF split, but it is intentionally kept near zero
so it cannot destabilize training or sampling.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

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

        self.velocity_net = Flow_matcher_U_Net_v2(
            horizon=seq_len,
            transition_dim=state_dim,
            cond_dim=state_dim,
            dim=freq_dim,
            dim_mults=(1, 2, 4, 8),
            returns_condition=False,
            condition_dropout=dropout_rate,
        )

        self.aux_head = nn.Sequential(
            nn.Linear(state_dim, state_dim),
            nn.SiLU(),
            nn.Linear(state_dim, state_dim),
        )

        nn.init.zeros_(self.aux_head[-1].weight)
        nn.init.zeros_(self.aux_head[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the main flow velocity and a small auxiliary residual."""
        velocity = self.velocity_net(x, cond, t)
        aux = self.aux_head(velocity)
        return velocity, aux

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
        schedule: str = "balanced",
        u_weight: float = 1.0,
        v_weight: float = 0.1,
        cond: Optional[torch.Tensor] = None,
        device: Optional[str] = None,
    ) -> torch.Tensor:
        """Sample a trajectory using explicit Euler integration."""
        device = device or self.device
        z_t = torch.randn(batch_size, seq_len, self.state_dim, device=device)
        t_steps = t_steps.to(device)

        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            h = t - r
            velocity, aux = self.forward(z_t, t.expand(batch_size), cond)

            if schedule == "u_first":
                combined = u_weight * velocity + 0.1 * v_weight * aux
            elif schedule == "balanced":
                combined = velocity + 0.1 * v_weight * aux
            else:
                combined = velocity

            z_t = z_t - h * combined

        return z_t

    def sample(
        self,
        batch_size: int,
        num_steps: int = 1,
        t_schedule: str = "linear",
        u_weight: float = 1.0,
        v_weight: float = 0.1,
        schedule: str = "balanced",
        seed: int = 0,
        cond: Optional[torch.Tensor] = None,
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
            cond=cond,
            device=self.device,
        )
