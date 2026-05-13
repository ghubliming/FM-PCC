"""
Drift-Augmented U-Net for FM-D

Extends the base U-Net with drift-aware conditioning streams.
Concatenates trajectory encoding and drift metrics as additional context channels.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DriftConditioner(nn.Module):
    """
    Computes drift-aware conditioning embeddings from trajectory state.
    
    Encodes:
    1. Trajectory history (recent states)
    2. Goal state
    3. Drift metrics (deviation from reference distribution)
    """
    
    def __init__(
        self,
        state_dim: int,
        cond_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        """
        Args:
            state_dim: Dimension of state vector
            cond_dim: Output conditioning dimension
            hidden_dim: Hidden layer width
            num_layers: Number of MLP layers
        """
        super().__init__()
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        
        # MLP encoder: state -> embedding
        layers = []
        in_dim = state_dim
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else cond_dim
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.ReLU() if i < num_layers - 1 else nn.Identity(),
            ])
            in_dim = out_dim
        
        self.encoder = nn.Sequential(*layers)
        self.norm = nn.LayerNorm(cond_dim)

    def forward(
        self,
        trajectory: torch.Tensor,
        drift_metrics: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute drift-aware conditioning.
        
        Args:
            trajectory: (B, T, state_dim) or (B, state_dim) trajectory snippet
            drift_metrics: (B, metric_dim) optional drift quality metrics
            
        Returns:
            cond: (B, cond_dim) conditioning embedding
        """
        if trajectory.dim() == 3:
            # Average over time dimension
            traj_embedding = trajectory.mean(dim=1)  # (B, state_dim)
        else:
            traj_embedding = trajectory
        
        # Encode trajectory
        cond = self.encoder(traj_embedding)  # (B, cond_dim)
        
        # Optionally fuse drift metrics
        if drift_metrics is not None:
            # Project drift metrics to cond_dim and add
            drift_emb = nn.Linear(drift_metrics.shape[-1], self.cond_dim)(drift_metrics)
            cond = cond + 0.1 * drift_emb
        
        return self.norm(cond)


class DriftAugmentedUNet1D(nn.Module):
    """
    1D U-Net with drift augmentation for trajectory generation.
    
    Extends base U-Net by:
    - Injecting drift-conditioned embeddings
    - Maintaining compatibility with standard U-Net interface
    - Supporting both FM and diffusion paradigms
    """
    
    def __init__(
        self,
        base_unet: nn.Module,
        state_dim: int,
        cond_dim: int = 64,
        drift_cond_dim: int = 64,
        enable_drift_conditioning: bool = True,
    ):
        """
        Args:
            base_unet: Base 1D U-Net model (from models/unet1d_temporal_cond.py)
            state_dim: Dimension of trajectory state
            cond_dim: Original conditioning dimension
            drift_cond_dim: Drift conditioning dimension (appended)
            enable_drift_conditioning: Whether to include drift stream
        """
        super().__init__()
        self.base_unet = base_unet
        self.state_dim = state_dim
        self.cond_dim = cond_dim
        self.drift_cond_dim = drift_cond_dim if enable_drift_conditioning else 0
        self.enable_drift_conditioning = enable_drift_conditioning
        
        # Drift conditioner
        if enable_drift_conditioning:
            self.drift_conditioner = DriftConditioner(
                state_dim=state_dim,
                cond_dim=drift_cond_dim,
                hidden_dim=128,
                num_layers=2,
            )
        else:
            self.drift_conditioner = None
        
        # Optional fusion layer if base_unet expects different cond format
        total_cond_dim = cond_dim + self.drift_cond_dim
        if hasattr(base_unet, 'cond_dim') and base_unet.cond_dim != total_cond_dim:
            self.cond_fusion = nn.Linear(total_cond_dim, base_unet.cond_dim)
        else:
            self.cond_fusion = None

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
        trajectory: Optional[torch.Tensor] = None,
        drift_metrics: Optional[torch.Tensor] = None,
        use_dropout: bool = False,
        force_dropout: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with optional drift conditioning.
        
        Args:
            x: State tensor (B, T, state_dim) or (B, state_dim)
            cond: Original conditioning (B, cond_dim)
            t: Time step (scalar or B,)
            returns: Optional return-to-go (B, 1)
            trajectory: Trajectory history for drift conditioning (B, T', state_dim)
            drift_metrics: Drift quality metrics (B, metric_dim)
            use_dropout: Use dropout (for classifier-free guidance)
            force_dropout: Force dropout even during inference
            
        Returns:
            output: Velocity field estimate, shape same as x
        """
        
        # Build augmented conditioning
        if self.enable_drift_conditioning and trajectory is not None:
            drift_cond = self.drift_conditioner(trajectory, drift_metrics)
            # Concatenate with original conditioning
            if cond.dim() == 2:
                augmented_cond = torch.cat([cond, drift_cond], dim=-1)
            else:
                # Handle case where cond might be 1D
                shape = list(cond.shape)
                shape[-1] += self.drift_cond_dim
                augmented_cond = torch.cat(
                    [cond.unsqueeze(0), drift_cond.unsqueeze(0)], dim=-1
                ).squeeze(0)
        else:
            augmented_cond = cond
        
        # Fuse conditioning if needed
        if self.cond_fusion is not None:
            augmented_cond = self.cond_fusion(augmented_cond)
        
        # Call base U-Net with augmented conditioning
        return self.base_unet(
            x,
            augmented_cond,
            t,
            returns=returns,
            use_dropout=use_dropout,
            force_dropout=force_dropout,
        )

    def wrap_unet(base_unet: nn.Module, **kwargs) -> "DriftAugmentedUNet1D":
        """
        Convenience factory: wrap existing U-Net with drift augmentation.
        
        Args:
            base_unet: Existing U-Net model
            **kwargs: Passed to __init__
            
        Returns:
            DriftAugmentedUNet1D wrapping base_unet
        """
        return DriftAugmentedUNet1D(base_unet, **kwargs)
