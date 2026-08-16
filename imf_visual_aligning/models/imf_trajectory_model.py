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
        freq_dim: int = 32,          # 🔴 FIX_8_UNET_WIDTH — UNet CHANNEL width (was 256 => 253 M params);
                                     # ignored by the DiT/SiT backbones. See logs_in_develop/Gen3v6_MeanFlow/Fix_8_Unet/.
        depth: int = 8,
        num_heads: int = 4,
        mlp_dim: int = 256,
        time_dim: int = 256,
        dropout_rate: float = 0.1,
        device: str = "cuda",
        if_vision: bool = False,
        vis_config=None,
    ):
        super().__init__()
        self.if_vision = if_vision
        self.seq_len = seq_len
        self.freq_dim = freq_dim
        self.depth = depth
        self.device = device

        if if_vision:
            # Gen8: VisualUNet (FiLM-conditioned dual-cam ResNet + h-conditioned U-Net)
            # replaces the plain Flow_matcher_U_Net_v2 as the u-velocity backbone.
            from .visual_unet import VisualUNet
            self.velocity_net = VisualUNet(vis_config)
            # state_dim is fixed to VisualUNet.TRANSITION_DIM (9) for the aux head
            state_dim = VisualUNet.TRANSITION_DIM
        else:
            self.velocity_net = Flow_matcher_U_Net_v2(
                horizon=seq_len,
                transition_dim=state_dim,
                cond_dim=state_dim,
                # 🔴 FIX_8_UNET_WIDTH — `dim` is BOTH the channel width and the time-embed
                # width (unet1d_temporal_cond.py:106,110). `freq_dim` is this repo's
                # only source for it, so its value IS the backbone size: 32 => 3.97 M,
                # 256 => 253.0 M. Never raise freq_dim to "improve the embedding".
                dim=freq_dim,
                dim_mults=(1, 2, 4, 8),
                returns_condition=False,
                condition_dropout=dropout_rate,
            )

        # 🔴 FIX_8_UNET_WIDTH — announce the backbone size at BUILD time. A width defect is
        # otherwise invisible: the model builds, trains, and logs plausible losses at any
        # width, and nothing in the train log states a parameter count. One line here
        # would have caught the 253 M UNet on run 1 instead of ~3 months later.
        # 32 => 3.97 M (DPCC/FMv3ODE baseline) | 256 => 253.0 M. UNet arm only; the
        # DiT/SiT arms size from dit_hidden_size and are unaffected by freq_dim.
        _n_params = sum(p.numel() for p in self.velocity_net.parameters())
        print(f'[ iMFTrajectoryModel ] vision={self.if_vision}  unet_width(freq_dim)={freq_dim}  '
              f'params={_n_params / 1e6:.1f}M')

        self.state_dim = state_dim

        # Aux v-head: plain MLP on raw x (no image processing).
        # For visual mode state_dim=9 (transition_dim); shape (B, H, 9) → linear over last dim.
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
        h: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        force_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the mean flow velocity u and instantaneous deviation v."""
        velocity = self.velocity_net(x, cond, t, h=h, force_dropout=force_dropout)
        aux = self.aux_head(x)  # independent head on input x, not on velocity
        return velocity, aux

    def forward_train(
        self,
        x_noisy: torch.Tensor,
        t: torch.Tensor,
        h: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        force_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward(x_noisy, t, h=h, cond=cond, force_dropout=force_dropout)

