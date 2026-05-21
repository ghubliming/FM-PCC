"""iMeanFlow engine wrapper for trajectory prediction.

The engine preserves the iMF naming surface while delegating the actual
learning signal to the FMv3-style velocity backbone.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .imf_trajectory_model import iMFTrajectoryModel


class iMeanFlowEngine(nn.Module):
    """
    iMeanFlow inference/training engine for trajectories.
    
    Direct adaptation of official iMF repo's iMeanFlow class,
    but for trajectory prediction instead of image generation.
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
        dtype: torch.dtype = torch.float32,
    ):
        """
        Args:
            state_dim: Trajectory state dimension
            seq_len: Sequence length
            freq_dim: Feature dimension
            depth: U-Net depth
            num_heads: Attention heads
            mlp_dim: MLP dimension
            time_dim: Time embedding dimension
            dropout_rate: Dropout
            device: Device
            dtype: Data type
        """
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.device = device
        self.dtype = dtype
        
        self.model = iMFTrajectoryModel(
            state_dim=state_dim,
            seq_len=seq_len,
            freq_dim=freq_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            time_dim=time_dim,
            dropout_rate=dropout_rate,
            device=device,
        )
        self.to(dtype).to(device)
    
    def u_fn(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict mean flow velocity u and instantaneous deviation v."""
        return self.model(x, t, h=h, cond=cond)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard nn.Module forward alias for velocity prediction."""
        return self.model(x, t, h=h, cond=cond)
    
    @torch.no_grad()
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
        """
        Generate trajectories via forward Euler 0→1 with h-conditioning.

        Uses DATA-AT-1 convention (t=0 noise, t=1 data) to match the training
        objective in iMFDiffusion.p_losses. Each step passes the interval size h
        to the model so it can adapt its mean-flow prediction accordingly.

        Args:
            batch_size: Number of trajectories
            num_steps: Number of ODE steps (NFE)
            t_schedule: Time schedule ('linear', 'quadratic')
            u_weight: Weight for the main velocity component
            v_weight: Weight for the auxiliary residual component
            schedule: 'u_first' or 'balanced'
            seed: Random seed

        Returns:
            sampled_trajectories [batch_size, seq_len, state_dim]
        """
        torch.manual_seed(seed)

        if t_schedule == "linear":
            t_steps = torch.linspace(0.0, 1.0, num_steps + 1, dtype=self.dtype, device=self.device)
        elif t_schedule == "quadratic":
            t_steps = torch.linspace(0.0, 1.0, num_steps + 1, dtype=self.dtype, device=self.device) ** 2
        else:
            t_steps = torch.linspace(0.0, 1.0, num_steps + 1, dtype=self.dtype, device=self.device)

        z_t = torch.randn(batch_size, self.seq_len, self.state_dim, dtype=self.dtype, device=self.device)

        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            h = t_next - t_cur  # forward step size > 0

            velocity, aux = self.model(z_t, t_cur.expand(batch_size).to(self.dtype), h=h, cond=cond)
            velocity = u_weight * velocity + 0.1 * v_weight * aux

            z_t = z_t + h * velocity  # forward integration 0→1

        return z_t
    
    def forward_train(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        force_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training: return (u, v) predictions.

        Args:
            x_noisy: Noisy trajectory [batch, seq_len, state_dim]
            t: Timestep [batch]
            h: Step-size conditioning [batch] (iMF mean-flow interval)
            cond: Conditioning (optional)
            force_dropout: Force condition dropout for CFG

        Returns:
            (u, v): Mean flow and instantaneous velocity predictions
        """
        return self.model(x_noisy, t, h=h, cond=cond, force_dropout=force_dropout)
