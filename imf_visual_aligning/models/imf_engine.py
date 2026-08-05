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
        freq_dim: int = 32,          # 🔴 FIX_8_UNET_WIDTH — UNet CHANNEL width (was 256 => 253 M params);
                                     # ignored by the DiT/SiT backbones. See logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/.
        depth: int = 8,
        num_heads: int = 4,
        mlp_dim: int = 256,
        time_dim: int = 256,
        dropout_rate: float = 0.1,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        if_vision: bool = False,
        vis_config=None,
    ):
        """
        Args:
            state_dim: Trajectory state dimension (overridden by VisualUNet.TRANSITION_DIM=9 if if_vision)
            seq_len: Sequence length
            freq_dim: Feature dimension (ignored when if_vision=True, VisualUNet uses its own dim)
            depth: U-Net depth
            num_heads: Attention heads
            mlp_dim: MLP dimension
            time_dim: Time embedding dimension
            dropout_rate: Dropout
            device: Device
            dtype: Data type
            if_vision: If True, velocity_net → VisualUNet (FiLM-conditioned dual-cam)
            vis_config: args/config object forwarded to VisualUNet.__init__ (only used when if_vision=True)
        """
        super().__init__()
        self.if_vision = if_vision
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
            if_vision=if_vision,
            vis_config=vis_config,
        )
        self.state_dim = self.model.state_dim  # may be updated to 9 by VisualUNet branch
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
