"""
FiLM_V2 — True FiLM (scale + shift) temporal U-Net backbone.

OPT-IN ADDITIVE MODULE. This file is NEW and does not modify or import the
existing `unet1d_temporal_cond.py`. It is only constructed when the config
sets `film_mode: 'v2'` (see VisualUNet). When `film_mode` is absent or 'v1',
the original `UNet1DTemporalCondModel` is used and this file is never touched.

Difference vs v1 ("Fake FiLM", additive-bias via time-embedding concat):

    v1 per block:  out = Conv(x) + time_mlp([t ‖ cond_emb])
    v2 per block:  out = (1 + γ(v)) · (Conv(x) + time_mlp(t)) + β(v)

In v2 the visual latent `v` is routed SEPARATELY from the time embedding and
produces a per-channel scale (γ) and shift (β) inside every ResidualTemporalBlock
— this is real FiLM (Perez et al. 2018). γ/β are ZERO-INITIALISED so that at
step 0 the block computes (1+0)·x + 0 = x (identity), keeping early training
stable. The forward() / get_pred() signatures are IDENTICAL to v1 so the
VisualUNet wrapper needs no branching in its forward path.
"""

import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.distributions import Bernoulli

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)


class FiLMResidualTemporalBlock(nn.Module):
    """ResidualTemporalBlock with a True-FiLM head (per-channel scale + shift).

    Identical to the v1 ResidualTemporalBlock except for the optional `film_proj`
    head. When `cond_dim == 0` (or `cond is None` at forward) it degrades exactly
    to the v1 block: out = Conv(x) + time_mlp(t) + residual.
    """

    def __init__(self, inp_channels, out_channels, embed_dim, horizon,
                 kernel_size=5, cond_dim=0):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        # Time path — UNCHANGED. embed_dim is TIME-ONLY in v2 (not widened by cond).
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # ── True FiLM head: visual latent → (gamma, beta), zero-initialised ──
        self.use_film = cond_dim > 0
        if self.use_film:
            self.film_proj = nn.Sequential(
                nn.Mish(),
                nn.Linear(cond_dim, out_channels * 2),   # γ ‖ β
            )
            # Identity init: (1 + 0) * x + 0 = x  → block starts as if no FiLM.
            nn.init.zeros_(self.film_proj[-1].weight)
            nn.init.zeros_(self.film_proj[-1].bias)

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, cond=None):
        '''
            x    : [ batch x inp_channels x horizon ]
            t    : [ batch x embed_dim ]   (time-only embedding)
            cond : [ batch x cond_dim ] or None  (projected visual latent)
            returns: [ batch x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)

        if self.use_film and cond is not None:
            gamma, beta = self.film_proj(cond).chunk(2, dim=-1)   # each [B, out_ch]
            out = out * (1 + gamma.unsqueeze(-1)) + beta.unsqueeze(-1)

        out = self.blocks[1](out)
        return out + self.residual_conv(x)


class UNet1DTemporalFiLMModel(ModelMixin, ConfigMixin):
    """True-FiLM variant of UNet1DTemporalCondModel.

    Drop-in replacement: same constructor kwargs, same forward()/get_pred()
    signatures. The only architectural differences vs v1:
      1. embed_dim = dim  (time-only; NOT widened by cond)
      2. ResidualTemporalBlock → FiLMResidualTemporalBlock(..., cond_dim=film_cond_dim)
      3. forward(): cond_emb is passed SEPARATELY to each block (not concat with t)
      4. get_pred(): same separate cond routing
    """

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        returns_condition=False,
        condition_dropout=0.1,
        calc_energy=False,
        kernel_size=5,
        use_cond_projection=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_dim = dim
        self.returns_dim = dim
        self.cond_dim = cond_dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # ── Visual conditioning projection (kept from v1) ──────────────────
        # Projects the external visual latent (cond_dim → dim). In v2 the result
        # is NOT concatenated into t; it is delivered to every block's FiLM head.
        if use_cond_projection and cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )
            film_cond_dim = dim   # γ/β are produced from a dim-wide vector
        else:
            self.cond_mlp = None
            film_cond_dim = 0

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        # embed_dim = TIME-ONLY (+ returns if enabled). Visual no longer widens it.
        embed_dim = dim
        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        nn.Mish(),
                        nn.Linear(dim, dim * 4),
                        nn.Mish(),
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim += dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                FiLMResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, cond_dim=film_cond_dim),
                FiLMResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, cond_dim=film_cond_dim),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = FiLMResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, cond_dim=film_cond_dim)
        self.mid_block2 = FiLMResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, cond_dim=film_cond_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                FiLMResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, cond_dim=film_cond_dim),
                FiLMResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size, cond_dim=film_cond_dim),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def _project_cond(self, cond):
        """Pool (if needed) and project the visual latent → [B, dim], or None."""
        if self.cond_mlp is not None and cond is not None and isinstance(cond, torch.Tensor):
            if len(cond.shape) == 3:
                cond_pooled = cond.mean(dim=1)   # [B, T, cond_dim] → [B, cond_dim]
            else:
                cond_pooled = cond               # [B, cond_dim]
            return self.cond_mlp(cond_pooled)    # [B, dim]
        return None

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x    : [ batch x horizon x transition ]
            cond : [ batch x cond_seq_len x cond_dim ] or [ batch x cond_dim ] or None
                   Visual latent. In v2 it is projected and delivered as per-block
                   FiLM scale/shift (NOT concatenated with the time embedding).
            returns : [batch x horizon]
        '''
        if self.calc_energy:
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        timesteps = time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t = self.time_mlp(timesteps)

        # Visual conditioning → separate FiLM signal (NOT concat with t).
        cond_emb = self._project_cond(cond)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, cond=cond_emb)
        x = self.mid_block2(x, t, cond=cond_emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        if self.calc_energy:
            energy = ((x - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]
        else:
            return x

    def get_pred(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        # Visual conditioning → separate FiLM signal.
        cond_emb = self._project_cond(cond)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        h = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, cond=cond_emb)
        x = self.mid_block2(x, t, cond=cond_emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
