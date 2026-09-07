"""
Gen14 U5 — TWO-TIME + TRUE-FiLM backbone.  The `film_mode: 'v2'` variant of
`unet1d_twotime_cond.Flow_matcher_U_Net_v2`, for the mf / af arms.

Before U5 the two-time arms could only run `film_mode: 'v1'`; `VisualUNetTwoTime`
raised on anything else, because Gen7's v2 file (`unet1d_temporal_film.py`) has no
`h_mlp` and would have silently dropped the MeanFlow / alpha-Flow h-conditioning.
This file closes that gap: v2's conditioning route + the two-time surface.

────────────────────────────────────────────────────────────────────────────────
WHAT CHANGES vs. THE v1 TWO-TIME BACKBONE

    v1 per block:  out = Conv(x) + time_mlp([ t(τ,h) ‖ cond_emb ])
    v2 per block:  out = (1 + γ(cond)) · ( Conv(x) + time_mlp(t(τ,h)) ) + β(cond)

The visual latent stops riding inside the time embedding and becomes a per-channel
scale/shift in every residual block. Everything on the TIME side is untouched:
`h_mlp` and the interval-CFG embeddings are still ADDED to `t` while `t` is [B, dim],
exactly as in v1. In v2 that is easier, not harder — `embed_dim` is time-only, so the
cond concat that forced the ORDER MATTERS rule in `unet1d_twotime_cond.py` is gone.

The FiLM block itself is IMPORTED, not reimplemented:
`FiLMResidualTemporalBlock` comes from `unet1d_temporal_film.py`, which is a G0
verbatim copy of Gen7. So "is mf's v2 the same FiLM as fm's v2?" has a one-line
answer — it is the same class object. Only the surrounding U-Net differs.

────────────────────────────────────────────────────────────────────────────────
🔴 WHY THIS IS JVP-SAFE (the reason the old guard existed)

`MeanFlowODE._p_losses_meanflow` differentiates the network in forward mode:

    u_pred, du_dr = jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))   # mf_diffusion.py:454

The tangents are carried by z, r and h ONLY. `cond` is closed over by `_u_of` as a
captured constant (`visual_latent`, pre-encoded once — see VisualUNetTwoTime), so
γ and β are constants with an identically-zero tangent. The FiLM update is then

    d/dr [ (1+γ)·f + β ]  =  (1+γ) · df/dr

i.e. a per-channel RESCALING of the same directional derivative v1 computes. No new
term enters the MeanFlow identity, and forward-mode AD only has to push a dual
number through `mul` and `add`, both of which have forward-AD formulas. The zero-init
on γ/β additionally makes step 0 numerically identical to no-FiLM.

⚠️ DO NOT make γ/β depend on `h` or `τ`. It is tempting — "condition the gate on the
interval too" — and it is a different model: `film_proj` would then sit INSIDE the
differentiated path, du/dr would pick up an extra `∂γ/∂h · f` term, and the MeanFlow
target would no longer be the Gen3v6 one. That is a research change, not a port.
────────────────────────────────────────────────────────────────────────────────
"""

import torch
import torch.nn as nn
import einops
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from torch.distributions import Bernoulli

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)
# G0-verbatim Gen7 block — imported so the two v2 arms share one FiLM implementation.
from .unet1d_temporal_film import FiLMResidualTemporalBlock


class Flow_matcher_U_Net_v2_FiLM(ModelMixin, ConfigMixin):
    """True-FiLM variant of `Flow_matcher_U_Net_v2`.

    Drop-in: same constructor kwargs and the same `forward()` / `get_pred()`
    signatures, including the two-time surface (`h`, `omega`, `t_min`, `t_max`,
    `return_v`), so `VisualUNetTwoTime` needs only a construction-time branch.

    Architectural deltas vs. the v1 two-time backbone:
      1. embed_dim = dim            (time-only; the visual latent no longer widens it)
      2. ResidualTemporalBlock      -> FiLMResidualTemporalBlock(cond_dim=film_cond_dim)
      3. cond_emb is handed to every block instead of being concatenated into t
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
        dual_head=False,     # v-head sharing the backbone trunk (official u_heads/v_heads split)
        interval_cfg=False,  # condition on (omega, t_min, t_max) for interval-CFG
        use_cond_projection=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # Two-time h-conditioning — the whole reason this file exists rather than
        # reusing Gen7's UNet1DTemporalFiLMModel directly.
        self.h_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # Visual conditioning projection. Same shape as v1's; the result is routed to
        # the per-block FiLM heads instead of being concatenated with t.
        if use_cond_projection and cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )
            film_cond_dim = dim
        else:
            self.cond_mlp = None
            film_cond_dim = 0

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        # TIME-ONLY. In v1 this was `dim + cond_embed_dim`; the visual latent no
        # longer travels through the time embedding, so it no longer widens it.
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

        def _block(dim_in, dim_out, horizon_):
            return FiLMResidualTemporalBlock(
                dim_in, dim_out, embed_dim=embed_dim, horizon=horizon_,
                kernel_size=kernel_size, cond_dim=film_cond_dim)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                _block(dim_in, dim_out, horizon),
                _block(dim_out, dim_out, horizon),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = _block(mid_dim, mid_dim, horizon)
        self.mid_block2 = _block(mid_dim, mid_dim, horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                _block(dim_out * 2, dim_in, horizon),
                _block(dim_in, dim_in, horizon),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

        # Shared-backbone v-head (official imfDiT u_heads/v_heads split). Mirrors
        # final_conv and reads the SAME post-up trunk feature.
        self.dual_head = dual_head
        if dual_head:
            self.v_final_conv = nn.Sequential(
                Conv1dBlock(dim, dim, kernel_size=kernel_size),
                nn.Conv1d(dim, transition_dim, 1),
            )

        # Interval-CFG conditioning (omega, t_min, t_max), summed into the time/h
        # embedding exactly like h_mlp. Off by default.
        self.interval_cfg = interval_cfg
        if interval_cfg:
            def _scalar_mlp():
                return nn.Sequential(
                    SinusoidalPosEmb(dim), nn.Linear(dim, dim * 4), nn.Mish(), nn.Linear(dim * 4, dim),
                )
            self.omega_mlp = _scalar_mlp()
            self.tmin_mlp = _scalar_mlp()
            self.tmax_mlp = _scalar_mlp()

    def _embed_scalar(self, mlp, val, x):
        """Broadcast a scalar/[B] conditioning value through `mlp` to a [B, dim] embedding."""
        if not torch.is_tensor(val):
            val = torch.tensor([val], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(val) and len(val.shape) == 0:
            val = val[None].to(x.device)
        val = val.float() * torch.ones(x.shape[0], dtype=torch.float32, device=x.device)
        return mlp(val)

    def _project_cond(self, cond):
        """Pool (if needed) and project the visual latent -> [B, dim], or None.

        Same contract as v1's graft: only fires for TENSOR cond. The state-based
        pipeline passes a dict {0: state} for inpainting and must not reach here.
        """
        if self.cond_mlp is not None and cond is not None and isinstance(cond, torch.Tensor):
            if len(cond.shape) == 3:
                cond_pooled = cond.mean(dim=1)   # [B, T, cond_dim] -> [B, cond_dim]
            else:
                cond_pooled = cond               # [B, cond_dim]
            return self.cond_mlp(cond_pooled)    # [B, dim]
        return None

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False, h=None,
                omega=None, t_min=None, t_max=None, return_v=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
            h : step-size conditioning scalar or [batch] tensor (iMF h-conditioning)
        '''
        if self.calc_energy:
            x_inp = x

        x = einops.rearrange(x, 'b h t -> b t h')

        timesteps = time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.float()
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        t = self.time_mlp(timesteps)

        if h is not None:
            if not torch.is_tensor(h):
                h = torch.tensor([h], dtype=torch.float32, device=x.device)
            elif torch.is_tensor(h) and len(h.shape) == 0:
                h = h[None].to(x.device)
            h = h.float()
            h = h * torch.ones(x.shape[0], dtype=h.dtype, device=h.device)
            t = t + self.h_mlp(h)

        # Interval-CFG conditioning, additive like h_mlp (held constant in the JVP).
        if self.interval_cfg:
            if omega is not None:
                t = t + self._embed_scalar(self.omega_mlp, omega, x)
            if t_min is not None:
                t = t + self._embed_scalar(self.tmin_mlp, t_min, x)
            if t_max is not None:
                t = t + self._embed_scalar(self.tmax_mlp, t_max, x)

        # v2: the visual latent leaves the time path entirely and becomes per-block
        # FiLM. `t` therefore stays [B, dim] all the way to the blocks, which is why
        # v1's ORDER MATTERS constraint does not apply here.
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

        h_stack = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            h_stack.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, cond=cond_emb)
        x = self.mid_block2(x, t, cond=cond_emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h_stack.pop()), dim=1)
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            x = upsample(x)

        # `trunk` is the shared post-up feature; u and (optional) v both read it.
        trunk = x
        u = self.final_conv(trunk)
        u = einops.rearrange(u, 'b t h -> b h t')

        if self.calc_energy:
            energy = ((u - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]

        if return_v and self.dual_head:
            v = self.v_final_conv(trunk)
            v = einops.rearrange(v, 'b t h -> b h t')
            return u, v
        return u

    def get_pred(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x horizon x transition ]
            returns : [batch x horizon]
        '''
        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

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

        h_stack = []

        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            h_stack.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t, cond=cond_emb)
        x = self.mid_block2(x, t, cond=cond_emb)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h_stack.pop()), dim=1)
            x = resnet(x, t, cond=cond_emb)
            x = resnet2(x, t, cond=cond_emb)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x
