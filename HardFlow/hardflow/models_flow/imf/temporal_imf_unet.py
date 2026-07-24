"""Gen13 — TemporalImfUnet: HardFlow's TemporalUnet extended for iMF.

Additive sibling of hardflow/models_flow/unet.py:TemporalUnet (which is imported,
never modified). Two changes vs the FM backbone (Gen13 plan decision D2):

1. TWO-TIME conditioning: the net takes (x, tau, h) — interval start tau AND
   interval width h — each with its own sinusoidal embedding MLP; embeddings are
   summed into the single conditioning vector the ResidualTemporalBlocks expect.
   (h = 0 recovers an instantaneous-velocity query.)
2. DUAL HEADS: the final conv emits 2 * transition_dim channels, split into
   (u, v): u = average velocity over [tau, tau+h], v = auxiliary instantaneous
   velocity head (needed for the iMF JVP loss and the predicted-v tangent).

Convention: HardFlow native (tau=0 noise -> tau=1 data). See convention.py.
"""

import einops
import torch
import torch.nn as nn

from hardflow.models_flow.unet import (
    Conv1dBlock,
    Downsample1d,
    LinearAttention,
    PreNorm,
    Residual,
    ResidualTemporalBlock,
    SinusoidalPosEmb,
    Upsample1d,
)


class TemporalImfUnet(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f"[ models/temporal_imf ] Channel dimensions: {in_out}")

        time_dim = dim
        # tau (interval start) embedding — identical structure to the FM backbone
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )
        # h (interval width) embedding — the iMF second time argument
        self.h_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_in, dim_out, embed_dim=time_dim, horizon=horizon
                        ),
                        ResidualTemporalBlock(
                            dim_out, dim_out, embed_dim=time_dim, horizon=horizon
                        ),
                        (
                            Residual(PreNorm(dim_out, LinearAttention(dim_out)))
                            if attention
                            else nn.Identity()
                        ),
                        Downsample1d(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon
        )
        self.mid_attn = (
            Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
            if attention
            else nn.Identity()
        )
        self.mid_block2 = ResidualTemporalBlock(
            mid_dim, mid_dim, embed_dim=time_dim, horizon=horizon
        )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResidualTemporalBlock(
                            dim_out * 2, dim_in, embed_dim=time_dim, horizon=horizon
                        ),
                        ResidualTemporalBlock(
                            dim_in, dim_in, embed_dim=time_dim, horizon=horizon
                        ),
                        (
                            Residual(PreNorm(dim_in, LinearAttention(dim_in)))
                            if attention
                            else nn.Identity()
                        ),
                        Upsample1d(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                horizon = horizon * 2

        # dual-head output: 2 * transition_dim channels -> chunk into (u, v)
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, 2 * transition_dim, 1),
        )

    @staticmethod
    def _as_batch_time(val, b, ref):
        """Normalize a time-like input to a (B,) tensor (mirrors TemporalUnet)."""
        if not torch.is_tensor(val):
            val = torch.tensor(val, device=ref.device, dtype=ref.dtype)
        while val.dim() > 1:
            val = val[..., 0]
        if val.dim() == 0:
            val = val.repeat(b)
        return val

    def forward(self, x, tau, h):
        """
        x:   (batch_size, planning_horizon, transition_dim)
        tau: interval start in [0, 1]   (scalar, 0-d or (B,))
        h:   interval width in [0, 1-tau]  (scalar, 0-d or (B,))
        Returns (u, v), each (batch_size, planning_horizon, transition_dim).
        """
        b, _, _ = x.shape
        tau = self._as_batch_time(tau, b, x)
        h = self._as_batch_time(h, b, x)

        x = einops.rearrange(x, "b h t -> b t h")

        t = self.time_mlp(tau) + self.h_mlp(h)
        skips = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, skips.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b t h -> b h t")
        u, v = torch.chunk(x, 2, dim=-1)
        return u, v
