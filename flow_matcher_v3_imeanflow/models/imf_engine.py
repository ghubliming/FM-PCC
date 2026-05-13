"""
iMeanFlow Engine: Wrapper around dual-velocity model.

Reuses official iMF repo's conceptual pattern:
- u_fn: predict mean velocity
- v_fn: predict instantaneous velocity (auxiliary)
- Sampling: weighted combination of (u, v)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
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
        """
        Predict dual velocity: (u, v).
        
        API matches official iMF repo's u_fn signature (adapted for trajectories).
        
        Args:
            x: Noisy trajectory [batch, seq_len, state_dim]
            t: Timestep [batch]
            h: Time difference (unused in inference, kept for API compatibility)
            cond: Conditioning [batch, cond_dim] (optional)
            
        Returns:
            u [batch, seq_len, state_dim], v [batch, seq_len, state_dim]
        """
        with torch.no_grad():
            u, v = self.model(x, t, cond)
        return u, v
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        num_steps: int = 1,
        t_schedule: str = "linear",
        u_weight: float = 1.0,
        v_weight: float = 0.0,
        schedule: str = "balanced",
        seed: int = 0,
    ) -> torch.Tensor:
        """
        Generate trajectories via iMF sampling (matches official repo pattern).
        
        Args:
            batch_size: Number of trajectories
            num_steps: Number of ODE steps (NFE)
            t_schedule: Time schedule ('linear', 'quadratic')
            u_weight: Weight for u component
            v_weight: Weight for v component
            schedule: 'u_first' (curriculum) or 'balanced'
            seed: Random seed
            
        Returns:
            sampled_trajectories [batch_size, seq_len, state_dim]
        """
        torch.manual_seed(seed)
        
        # Generate time schedule
        if t_schedule == "linear":
            t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=self.dtype, device=self.device)
        elif t_schedule == "quadratic":
            t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=self.dtype, device=self.device) ** 2
        else:
            t_steps = torch.linspace(1.0, 0.0, num_steps + 1, dtype=self.dtype, device=self.device)
        
        # Sample trajectory using iMF's sampling loop (official repo pattern)
        z_t = torch.randn(batch_size, self.seq_len, self.state_dim, dtype=self.dtype, device=self.device)
        
        for i in range(num_steps):
            t = t_steps[i]
            r = t_steps[i + 1]
            h = t - r
            
            # Predict (u, v)
            u, v = self.model(z_t, t.expand(batch_size).to(self.dtype))
            
            # Weighted combination (from official iMF)
            velocity = u_weight * u + v_weight * v
            
            # ODE step
            z_t = z_t - h * velocity
        
        return z_t
    
    def forward_train(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training: return (u, v) predictions.
        
        Args:
            x_noisy: Noisy trajectory [batch, seq_len, state_dim]
            t: Timestep [batch]
            cond: Conditioning (optional)
            
        Returns:
            (u, v): Dual velocity predictions for loss computation
        """
        u, v = self.model(x_noisy, t, cond)
        return u, v
