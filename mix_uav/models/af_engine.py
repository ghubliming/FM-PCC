"""MeanFlow engine wrapper for trajectory prediction (Gen3v6).

The engine is a thin two-time `(u, v)` surface over the velocity backbone (UNet or DiT).
Inherited unchanged from Gen3v4 so the backbone is byte-identical and the MeanFlow-vs-iMF
A/B is architecture-controlled; only the naming and the (unused) CFG arguments differ.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .af_trajectory_model import AFTrajectoryModel


class AlphaFlowEngine(nn.Module):
    """
    MeanFlow inference/training engine for trajectories.

    Two-time contract: forward(x, t, h, cond) -> (u, v), where u is the average-velocity
    field (deployed) and v is the instantaneous velocity (dropped at sampling).
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
        # U5 Phase 1 — real-iMF flags (default OFF), forwarded to the backbone.
        dual_head: bool = False,
        interval_cfg: bool = False,
        # U6 — backbone selector + DiT sizing (forwarded to AFTrajectoryModel).
        # 🔴 FIX_8_BACKBONE_DEFAULT — this generation's OWN backbone, not 'unet'. The UNet fallback was a
        # Gen3v4-era leftover: it is the one backbone whose every run is confounded by the
        # freq_dim width defect (see FIX_8_UNET_WIDTH), so a missing config key used to
        # silently select the known-bad arm. Config always passes this key; the default
        # only matters when it doesn't, which is exactly when a wrong default hurts.
        imf_backbone: str = 'sit',
        dit_depth: int = 8,
        dit_hidden_size: int = 256,
        dit_num_heads: int = 4,
        dit_aux_head_depth: int = 2,
        dit_patch_size: int = 1,
        dit_condition_on_t: bool = False,
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
        
        self.dual_head = dual_head
        self.interval_cfg = interval_cfg
        self.imf_backbone = imf_backbone
        self.model = AFTrajectoryModel(
            state_dim=state_dim,
            seq_len=seq_len,
            freq_dim=freq_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            time_dim=time_dim,
            dropout_rate=dropout_rate,
            device=device,
            dual_head=dual_head,
            interval_cfg=interval_cfg,
            imf_backbone=imf_backbone,
            dit_depth=dit_depth,
            dit_hidden_size=dit_hidden_size,
            dit_num_heads=dit_num_heads,
            dit_aux_head_depth=dit_aux_head_depth,
            dit_patch_size=dit_patch_size,
            dit_condition_on_t=dit_condition_on_t,
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
        objective in AlphaFlowODE._p_losses_meanflow. Each step passes the interval size h
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

        Gen3v6 has NO interval-CFG, so the backbone's (omega, t_min, t_max) inputs are never
        supplied — they fall back to their constant default (guidance off) at both train and
        sample time and are therefore inert. The arguments are left on the backbone so the
        DiT's parameter set stays identical to Gen3v4's.

        Args:
            x_noisy: Noisy trajectory [batch, seq_len, state_dim]
            t: Timestep [batch] (τ of the anchor; DATA-AT-1)
            h: Step-size conditioning [batch] (mean-flow interval h = t − r)
            cond: Conditioning (optional)
            force_dropout: Force condition dropout (returns-CFG path only; inert here)

        Returns:
            (u, v): Mean flow and instantaneous velocity predictions
        """
        return self.model(x_noisy, t, h=h, cond=cond, force_dropout=force_dropout)
