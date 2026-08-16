import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange
from torch.distributions import Bernoulli

from .helpers import (
    SinusoidalPosEmb,
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
)

# ══════════════════════════════════════════════════════════════════════════════
# Gen14 — TWO-TIME + VISUAL backbone.  GRAFT, not a rewrite.
#
# Body: `flow_matcher_v3_meanflow/models/unet1d_temporal_cond.py` copied VERBATIM
#       (Gen3v6 == Gen3v7 modulo package name).  Gives us `h_mlp` (two-time
#       h-conditioning) + the dual_head / interval_cfg switches.
#
# Grafted in from `fm_visual_aligning/models/unet1d_temporal_cond.py` (Gen7),
# pasted verbatim, TWO blocks only — both marked `Gen14 GRAFT` below:
#   1. __init__ : `use_cond_projection` kwarg + the `cond_mlp` construction +
#                 `embed_dim = dim + cond_embed_dim`
#   2. forward  : the tensor-cond pooling/projection/concat block
#
# 🔴 ORDER MATTERS.  `h_mlp` and the interval-CFG embeddings are ADDED to `t`
# (both emit [B, dim]), so they must run while `t` is still [B, dim] — i.e.
# BEFORE the cond CONCAT widens it to [B, dim + cond_embed_dim].  Reordering
# these silently produces a shape error at the first ResidualTemporalBlock.
#
# Gen7's own `unet1d_temporal_cond.py` is NOT touched: the ddpm/fm arms keep
# importing it unchanged, which is what makes their fidelity structural.
# ══════════════════════════════════════════════════════════════════════════════

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.InstanceNorm2d(dim, affine = True)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 128):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])
        
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)

        return out + self.residual_conv(x)

# TODO: [Structural Modification] This class is the core for the new U-Net v2.
# Structural changes to the layer depth, attention mechanisms, and skip connections 
# should be implemented within this class.
class Flow_matcher_U_Net_v2(ModelMixin, ConfigMixin):

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
        # U5 Phase 1 — real-iMF additions (both default OFF ⇒ byte-for-byte unchanged):
        dual_head=False,     # add a v-head sharing the backbone trunk (official u_heads/v_heads split)
        interval_cfg=False,  # condition on (omega, t_min, t_max) for interval-CFG
        use_cond_projection=False,   # ← Gen14 GRAFT (Gen7): FiLM visual conditioning
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        # print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.h_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        # ── Gen14 GRAFT (from Gen7 unet1d_temporal_cond.py, verbatim) ─────
        # Conditioning Projection (FiLM-style).
        # Projects the external conditioning vector (e.g. visual embeddings)
        # into the same space as the time embedding, so it can modulate
        # the ResidualTemporalBlocks via concatenation with t.
        # Only enabled when use_cond_projection=True (visual pipelines).
        # The state-based pipeline passes cond as a dict for inpainting,
        # not as a tensor, so it must NOT enable this.
        if use_cond_projection and cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, dim),
                nn.Mish(),
                nn.Linear(dim, dim),
            )
            cond_embed_dim = dim  # will be concatenated with time_dim
        else:
            self.cond_mlp = None
            cond_embed_dim = 0
        # ── end Gen14 GRAFT ───────────────────────────────────────────────

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy

        # Gen14 GRAFT: embed_dim widened by cond_embed_dim (Gen7 line). With
        # use_cond_projection=False this is `dim` / `2*dim` exactly as Gen3v6.
        embed_dim = dim + cond_embed_dim
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

        # TODO: [Structural Modification] Initialize new architectural components such as 
        # Self-Attention, Cross-Attention, or Transformer blocks here for Gen3 U-Net.
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size)
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size),
                ResidualTemporalBlock(dim_in, dim_in, embed_dim=embed_dim, horizon=horizon, kernel_size=kernel_size),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=kernel_size),
            nn.Conv1d(dim, transition_dim, 1),
        )

        # U5 Phase 1b — shared-backbone v-head (official imfDiT u_heads/v_heads split).
        # Mirrors final_conv; reads the SAME post-up trunk feature, so v shares the backbone.
        self.dual_head = dual_head
        if dual_head:
            self.v_final_conv = nn.Sequential(
                Conv1dBlock(dim, dim, kernel_size=kernel_size),
                nn.Conv1d(dim, transition_dim, 1),
            )

        # U5 Phase 1c — interval-CFG conditioning (omega, t_min, t_max), summed into the
        # time/h embedding exactly like h_mlp. Off by default.
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

        # Maybe remove
        timesteps = time
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.float32, device=x.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.float()
        timesteps = timesteps * torch.ones(x.shape[0], dtype=timesteps.dtype, device=timesteps.device)

        # t = self.time_mlp(time)
        t = self.time_mlp(timesteps)

        if h is not None:
            if not torch.is_tensor(h):
                h = torch.tensor([h], dtype=torch.float32, device=x.device)
            elif torch.is_tensor(h) and len(h.shape) == 0:
                h = h[None].to(x.device)
            h = h.float()
            h = h * torch.ones(x.shape[0], dtype=h.dtype, device=h.device)
            t = t + self.h_mlp(h)

        # U5 Phase 1c — interval-CFG conditioning, additive like h_mlp (held constant in the JVP).
        if self.interval_cfg:
            if omega is not None:
                t = t + self._embed_scalar(self.omega_mlp, omega, x)
            if t_min is not None:
                t = t + self._embed_scalar(self.tmin_mlp, t_min, x)
            if t_max is not None:
                t = t + self._embed_scalar(self.tmax_mlp, t_max, x)

        # ── Gen14 GRAFT (from Gen7 unet1d_temporal_cond.py, verbatim) ─────
        # Integrate external conditioning: pool over the temporal axis, project
        # to dim, then CONCAT with t.
        # NOTE: In the state-based pipeline, `cond` is a dict {0: state}, not a
        # tensor. The cond_mlp path only fires for tensor conditioning (i.e. the
        # visual embedding handed down by VisualUNetTwoTime).
        # 🔴 Must stay AFTER the h_mlp / interval-CFG additions above — see the
        # ORDER MATTERS note at the top of this file.
        if self.cond_mlp is not None and cond is not None and isinstance(cond, torch.Tensor):
            if len(cond.shape) == 3:
                # cond: [B, T, cond_dim] → pool → [B, cond_dim]
                cond_pooled = cond.mean(dim=1)
            else:
                # cond: [B, cond_dim] — already pooled
                cond_pooled = cond
            cond_emb = self.cond_mlp(cond_pooled)  # [B, dim]
            t = torch.cat([t, cond_emb], dim=-1)
        # ── end Gen14 GRAFT ───────────────────────────────────────────────

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


        # TODO: [Structural Modification] Update forward pass for new blocks\n        # Ensure new attention/transformer blocks are correctly integrated into the\n        # down-sampling, bottleneck, and up-sampling paths.
        for resnet, resnet2, downsample in self.downs:
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        # import pdb; pdb.set_trace()

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        # `trunk` is the shared post-up feature; u and (optional) v both read it ⇒ shared backbone.
        trunk = x
        u = self.final_conv(trunk)
        u = einops.rearrange(u, 'b t h -> b h t')

        if self.calc_energy:
            # Energy function
            energy = ((u - x_inp)**2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x_inp, create_graph=True)
            return grad[0]

        # U5 Phase 1b — return the shared v-head alongside u when requested.
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
            x = resnet(x, t)
            x = resnet2(x, t)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_block2(x, t)

        for resnet, resnet2, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, t)
            x = resnet2(x, t)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')

        return x

class MLPnet(nn.Module):
    def __init__(
        self,
        transition_dim,
        cond_dim,
        dim=128,
        dim_mults=(1, 2, 4, 8),
        horizon=1,
        returns_condition=True,
        condition_dropout=0.1,
        calc_energy=False,
    ):
        super().__init__()

        if calc_energy:
            act_fn = nn.SiLU()
        else:
            act_fn = nn.Mish()

        self.time_dim = dim
        self.returns_dim = dim

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            act_fn,
            nn.Linear(dim * 4, dim),
        )

        self.returns_condition = returns_condition
        self.condition_dropout = condition_dropout
        self.calc_energy = calc_energy
        self.transition_dim = transition_dim
        self.action_dim = transition_dim - cond_dim

        if self.returns_condition:
            self.returns_mlp = nn.Sequential(
                        nn.Linear(1, dim),
                        act_fn,
                        nn.Linear(dim, dim * 4),
                        act_fn,
                        nn.Linear(dim * 4, dim),
                    )
            self.mask_dist = Bernoulli(probs=1-self.condition_dropout)
            embed_dim = 2*dim
        else:
            embed_dim = dim

        self.mlp = nn.Sequential(
                        nn.Linear(embed_dim + transition_dim, 1024),
                        act_fn,
                        nn.Linear(1024, 1024),
                        act_fn,
                        nn.Linear(1024, self.action_dim),
                    )

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False):
        '''
            x : [ batch x action ]
            cond: [batch x state]
            returns : [batch x 1]
        '''
        # Assumes horizon = 1
        t = self.time_mlp(time)

        if self.returns_condition:
            assert returns is not None
            returns_embed = self.returns_mlp(returns)
            if use_dropout:
                mask = self.mask_dist.sample(sample_shape=(returns_embed.size(0), 1)).to(returns_embed.device)
                returns_embed = mask*returns_embed
            if force_dropout:
                returns_embed = 0*returns_embed
            t = torch.cat([t, returns_embed], dim=-1)

        inp = torch.cat([t, cond, x], dim=-1)
        out  = self.mlp(inp)

        if self.calc_energy:
            energy = ((out - x) ** 2).mean()
            grad = torch.autograd.grad(outputs=energy, inputs=x, create_graph=True)
            return grad[0]
        else:
            return out

class TemporalValue(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        time_dim=None,
        out_dim=1,
        dim_mults=(1, 2, 4, 8),
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = time_dim or dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])

        # print(in_out)
        for dim_in, dim_out in in_out:

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            horizon = horizon // 2

        fc_dim = dims[-1] * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out