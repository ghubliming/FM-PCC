"""Trajectory MeanFlow backbone wrapper — the single backbone swap point (Gen3v6).

Contract:

    forward(x, t, h, cond, omega, t_min, t_max) -> (u, v)

where `u` is the average-velocity field (deployed) and `v` is the instantaneous
velocity (auxiliary, dropped at sampling).

Inherited from Gen3v4 UNCHANGED on purpose: Gen3v6 differs from Gen3v4 only in the
training objective, so keeping the backbone identical is what makes the MeanFlow-vs-iMF
A/B architecture-controlled. Gen3v6 never supplies (omega, t_min, t_max) — it has no
interval-CFG — so those inputs sit at their constant default in both training and sampling.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from .unet1d_twotime_cond import Flow_matcher_U_Net_v2   # Gen14: two-time + FiLM graft
from .af_dit_trajectory import AFDiTTrajectory
from .af_sit_trajectory import AFSiTTrajectory


class AFTrajectoryModel(nn.Module):
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
        # U5 Phase 1 — real-iMF flags (default OFF ⇒ legacy behaviour unchanged).
        dual_head: bool = False,     # v shares the backbone (vs the legacy orphan aux MLP)
        interval_cfg: bool = False,  # condition the backbone on (omega, t_min, t_max)
        # U6 — backbone selector. 'unet' (default) keeps the UNet; 'dit' swaps in the
        # faithful official-iMF transformer (AFDiTTrajectory); 'sit' (U2) swaps in α-Flow's
        # OWN backbone, the SiT (AFSiTTrajectory: LayerNorm, qk_norm=False, adaLN-zero, t+r
        # conditioning). All three satisfy the same velocity_net forward contract, so the
        # objective/JVP/sampler are unchanged.
        imf_backbone: str = 'unet',
        dit_depth: int = 8,
        dit_hidden_size: int = 256,
        dit_num_heads: int = 4,
        dit_aux_head_depth: int = 2,
        dit_patch_size: int = 1,
        dit_condition_on_t: bool = False,
        # ── Gen14 GRAFT — visual arm (identical to the mf_trajectory_model graft) ──
        # if_vision=True routes velocity_net to VisualUNetTwoTime, which owns the
        # dual-cam ResNet encoder and the FiLM-conditioned two-time U-Net. The
        # objective, bootstrap and sampler are UNCHANGED — this is a backbone swap.
        if_vision: bool = False,
        vis_config=None,   # args/config object forwarded to VisualUNetTwoTime
    ):
        super().__init__()
        self.state_dim = state_dim
        self.seq_len = seq_len
        self.freq_dim = freq_dim
        self.depth = depth
        self.device = device
        self.dual_head = dual_head
        self.interval_cfg = interval_cfg
        self.imf_backbone = imf_backbone
        self.if_vision = if_vision

        if if_vision:
            # ── Gen14 GRAFT ───────────────────────────────────────────────
            # Visual backbone wins over imf_backbone: the DiT/SiT variants have
            # no visual-conditioning input, so silently honouring them here
            # would train an image-blind model that still reports if_vision.
            if imf_backbone not in ('unet',):
                raise ValueError(
                    f"[ AFTrajectoryModel ] if_vision=True requires imf_backbone='unet' "
                    f"(got '{imf_backbone}'). The DiT/SiT backbones have no visual "
                    f"conditioning path and would train image-blind."
                )
            from .visual_unet_twotime import VisualUNetTwoTime
            self.velocity_net = VisualUNetTwoTime(
                vis_config, dual_head=dual_head, interval_cfg=interval_cfg)
            # state_dim is pinned to the visual transition dim for the aux head below
            state_dim = VisualUNetTwoTime.TRANSITION_DIM
            self.state_dim = state_dim
        elif imf_backbone == 'dit':
            # The DiT always carries its dual heads + interval conditioning natively;
            # the dual_head/interval_cfg flags are honoured by always returning (u, v)
            # and always conditioning on (omega, t_min, t_max) when supplied.
            self.velocity_net = AFDiTTrajectory(
                horizon=seq_len,
                transition_dim=state_dim,
                hidden_size=dit_hidden_size,
                depth=dit_depth,
                num_heads=dit_num_heads,
                aux_head_depth=dit_aux_head_depth,
                patch_size=dit_patch_size,
                condition_dropout=dropout_rate,
                condition_on_t=dit_condition_on_t,
            )
        elif imf_backbone == 'sit':
            # U2 — α-Flow's OWN backbone: the SiT (adaLN-zero, LayerNorm affine-off run in fp32,
            # qk_norm=False, GELU(tanh) mlp_ratio=4, frozen sin-cos pos-embed, two time
            # embedders t+r). Reuses the dit_* sizing knobs; SiT has no shared-trunk/head split
            # and no h-only conditioning switch, so dit_aux_head_depth / dit_condition_on_t are
            # N/A. α-Flow's SiT is single-head (u); AFSiTTrajectory adds a twin v FinalLayer
            # ONLY to feed this lineage's v aux-loss (dropped at inference) — see its docstring (B).
            self.velocity_net = AFSiTTrajectory(
                horizon=seq_len,
                transition_dim=state_dim,
                hidden_size=dit_hidden_size,
                depth=dit_depth,
                num_heads=dit_num_heads,
                patch_size=dit_patch_size,
            )
        elif imf_backbone == 'unet':
            self.velocity_net = Flow_matcher_U_Net_v2(
                horizon=seq_len,
                transition_dim=state_dim,
                cond_dim=state_dim,
                dim=freq_dim,
                dim_mults=(1, 2, 4, 8),
                returns_condition=False,
                condition_dropout=dropout_rate,
                dual_head=dual_head,
                interval_cfg=interval_cfg,
            )
        else:
            raise ValueError(f"Unknown imf_backbone '{imf_backbone}' (expected 'unet', 'dit' or 'sit')")

        # Legacy orphan aux head — kept ONLY for dual_head=False back-compat (does not
        # share the backbone). When dual_head=True, v comes from velocity_net's v-head.
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
        omega: Optional[torch.Tensor] = None,
        t_min: Optional[torch.Tensor] = None,
        t_max: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict the mean-flow velocity u and instantaneous velocity v → (u, v)."""
        if self.dual_head or self.imf_backbone in ('dit', 'sit'):
            # Shared-backbone u + v (official split). Both transformer backbones carry native
            # v-heads, so they always use this path. CFG knobs are constant w.r.t. the JVP.
            u, v = self.velocity_net(
                x, cond, t, h=h, force_dropout=force_dropout,
                omega=omega, t_min=t_min, t_max=t_max, return_v=True,
            )
            return u, v
        # Legacy path: single u-head + orphan aux MLP on raw x.
        velocity = self.velocity_net(
            x, cond, t, h=h, force_dropout=force_dropout,
            omega=omega, t_min=t_min, t_max=t_max,
        )
        aux = self.aux_head(x)
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
        """Sample via forward Euler 0→1 (noise→data) with h-conditioning."""
        device = device or self.device
        z_t = torch.randn(batch_size, seq_len, self.state_dim, device=device)  # sigma=1.0
        t_steps = t_steps.to(device)

        for i in range(num_steps):
            t_cur = t_steps[i]
            t_next = t_steps[i + 1]
            h = t_next - t_cur  # forward step size > 0
            velocity, aux = self.forward(z_t, t_cur.expand(batch_size), h=h, cond=cond)

            if schedule == "u_first":
                combined = u_weight * velocity + 0.1 * v_weight * aux
            elif schedule == "balanced":
                combined = velocity + 0.1 * v_weight * aux
            else:
                combined = velocity

            z_t = z_t + h * combined  # forward integration 0→1

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
        """Sampling entrypoint: forward Euler 0→1 with h-conditioning."""
        torch.manual_seed(seed)

        if t_schedule == "quadratic":
            t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device) ** 2
        else:
            t_steps = torch.linspace(0.0, 1.0, num_steps + 1, device=self.device)

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
