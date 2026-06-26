"""Trajectory iMF DiT backbone — a faithful port of the official imfDiT.

This is the U6 `imf_backbone='dit'` implementation. It ports the official
`/workspaces/imeanflow/models/imfDiT.py` transformer (RoPE attention + QK-RMSNorm,
SwiGLU MLP, zero-init residual vector-gates, shared backbone → u/v heads,
in-context conditioning tokens, zero-init final layers, scaled-variance init)
and adapts the I/O from 2D images to 1D trajectories `[B, H, D]`:

  • PatchEmbedder (2D conv)  → TrajPatchEmbedder (1D linear over `patch_size` steps)
  • unpatchify (image)       → trajectory un-patchify back to `[B, H, D]`
  • LabelEmbedder(class y)   → optional null-class token used ONLY as the CFG
                                content-dropout switch (trajectory conditioning is
                                the pinned observation, applied to x externally)
  • conditions on h, ω, τ_min, τ_max (and optionally t) via per-signal tokens

It satisfies the SAME `velocity_net` contract as `Flow_matcher_U_Net_v2`:
    forward(x, cond, time, *, h, force_dropout, omega, t_min, t_max, return_v)
      -> u                       (return_v=False)
      -> (u, v)                  (return_v=True)
so `iMFTrajectoryModel` swaps it in with no change to the objective/JVP/sampler.

JVP-safety note (the iMF objective differentiates this backbone with
`torch.func.jvp`): every component here is forward-AD friendly — RMSNorm is
per-token (no batch coupling), attention/softmax/SwiGLU are pointwise+matmul, and
RoPE is implemented with a REAL-valued interleaved rotation that is mathematically
identical to the official complex-bitcast version but avoids the non-differentiable
`view(torch.complex64)` cast. This is the single deliberate deviation from the
verbatim port; all learned components are unchanged.
"""

import math
from functools import partial
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── primitive layers (ported) ─────────────────────────────

class TorchLinear(nn.Module):
    """Linear with scaled-variance / zeros init (ported from imf torch_models)."""

    def __init__(self, in_features, out_features, bias=True,
                 weight_init="scaled_variance", init_constant=1.0, bias_init="zeros"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        if weight_init == "scaled_variance":
            nn.init.normal_(self.linear.weight, std=init_constant / math.sqrt(in_features))
        elif weight_init == "zeros":
            nn.init.zeros_(self.linear.weight)
        else:
            raise ValueError(f"Invalid weight_init: {weight_init}")
        if bias:
            nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x)


class RMSNorm(nn.Module):
    """Root-Mean-Square norm (per-token ⇒ JVP-safe). Ported verbatim."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        ms = torch.mean(x.float().square(), dim=-1, keepdim=True)
        return (x * torch.rsqrt(ms + self.eps)).to(x.dtype) * self.weight


class SwiGLUMlp(nn.Module):
    """Swish-Gated Linear Unit MLP. Ported verbatim."""

    def __init__(self, in_features, hidden_features, weight_init="scaled_variance",
                 weight_init_constant=1.0):
        super().__init__()
        kw = dict(bias=False, weight_init=weight_init, init_constant=weight_init_constant)
        self.w1 = TorchLinear(in_features, hidden_features, **kw)
        self.w3 = TorchLinear(in_features, hidden_features, **kw)
        self.w2 = TorchLinear(hidden_features, in_features, **kw)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TimestepEmbedder(nn.Module):
    """Scalar → sinusoidal → MLP embedding. Ported from imf embedder."""

    def __init__(self, hidden_size, frequency_embedding_size=256,
                 weight_init="scaled_variance", init_constant=1.0):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            TorchLinear(frequency_embedding_size, hidden_size, bias=True,
                        weight_init=weight_init, init_constant=init_constant),
            nn.SiLU(),
            TorchLinear(hidden_size, hidden_size, bias=True,
                        weight_init=weight_init, init_constant=init_constant),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


# ───────────────────────────── RoPE (real-valued, JVP-safe) ─────────────────────────────

def precompute_rope_cos_sin(dim: int, seq_len: int, theta: float = 10000.0):
    """cos/sin tables for an interleaved rotary embedding over `seq_len` positions.

    Mathematically identical to the official complex `precompute_rope_freqs`, but
    returns real tensors so forward-mode AD (the MeanFlow JVP) flows through cleanly.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos = torch.arange(seq_len, dtype=torch.float32)
    ang = torch.outer(pos, freqs)            # [seq_len, dim/2]
    return torch.cos(ang), torch.sin(ang)


def apply_rotary_pos_emb(x, cos, sin):
    """Interleaved RoPE: pairs (x_even, x_odd) rotated by (cos, sin).

    x   : [B, S, num_heads, head_dim]
    cos/sin : [S, head_dim/2]
    Matches the official complex `x.view(complex64) * (cos + i·sin)` exactly:
        out_even = x_even·cos − x_odd·sin
        out_odd  = x_even·sin + x_odd·cos
    """
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    return torch.stack([out_even, out_odd], dim=-1).flatten(-2)


# ───────────────────────────── transformer blocks (ported) ─────────────────────────────

class RoPEAttention(nn.Module):
    """Multi-head self-attention with RoPE + QK RMSNorm. Ported (real RoPE)."""

    def __init__(self, hidden_size, num_heads, weight_init="scaled_variance",
                 weight_init_constant=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        kw = dict(in_features=hidden_size, out_features=hidden_size, bias=False,
                  weight_init=weight_init, init_constant=weight_init_constant)
        self.q_proj = TorchLinear(**kw)
        self.k_proj = TorchLinear(**kw)
        self.v_proj = TorchLinear(**kw)
        self.out_proj = TorchLinear(**kw)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x, cos, sin):
        b, s, _ = x.shape
        q = self.q_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, s, self.num_heads, self.head_dim)
        q = apply_rotary_pos_emb(self.q_norm(q), cos, sin)
        k = apply_rotary_pos_emb(self.k_norm(k), cos, sin)
        query = q / math.sqrt(self.head_dim)
        attn = torch.einsum("bqhd,bkhd->bhqk", query, k)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(v.dtype)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, v).reshape(b, s, -1)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer block with zero-init residual vector-gates. Ported verbatim."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=8 / 3,
                 weight_init="scaled_variance", weight_init_constant=1.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = RoPEAttention(hidden_size, num_heads, weight_init, weight_init_constant)
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUMlp(hidden_size, int(hidden_size * mlp_ratio),
                             weight_init, weight_init_constant)
        # zero-init gates ⇒ block starts as identity (stable under the stiff JVP target)
        self.attn_scale = nn.Parameter(torch.zeros(hidden_size))
        self.mlp_scale = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin) * self.attn_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x


class FinalLayer(nn.Module):
    """RMSNorm + zero-init linear projection. Ported verbatim."""

    def __init__(self, hidden_size, out_features):
        super().__init__()
        self.norm = RMSNorm(hidden_size)
        self.linear = TorchLinear(hidden_size, out_features, bias=True,
                                  weight_init="zeros", bias_init="zeros")

    def forward(self, x):
        return self.linear(self.norm(x))


# ───────────────────────────── trajectory patch embed ─────────────────────────────

class TrajPatchEmbedder(nn.Module):
    """`[B, H, D]` → `[B, H//patch, hidden]` via a linear lift over `patch` steps."""

    def __init__(self, transition_dim, patch_size, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.transition_dim = transition_dim
        self.proj = TorchLinear(patch_size * transition_dim, hidden_size, bias=True)

    def forward(self, x):
        b, h, d = x.shape
        assert h % self.patch_size == 0, f"horizon {h} not divisible by patch {self.patch_size}"
        x = x.reshape(b, h // self.patch_size, self.patch_size * d)
        return self.proj(x)


# ───────────────────────────── the trajectory iMF DiT ─────────────────────────────

class IMFDiTTrajectory(nn.Module):
    """Trajectory port of imfDiT. Drop-in for `Flow_matcher_U_Net_v2` as `velocity_net`.

    Shared backbone of `depth - aux_head_depth` blocks → equal-depth u/v heads.
    Returns u, or (u, v) when return_v=True (v dropped at inference).
    """

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 8 / 3,
        aux_head_depth: int = 2,
        patch_size: int = 1,
        condition_dropout: float = 0.1,      # accepted for signature parity (CFG via null class)
        condition_on_t: bool = False,        # official conditions only on h; t optional
        num_classes: int = 1,
        num_time_tokens: int = 2,
        num_cfg_tokens: int = 2,
        num_interval_tokens: int = 1,
        weight_init_constant: float = 0.32,
        embedding_init_constant: float = 1.0,
        token_init_constant: float = 1.0,
        **unused,                            # tolerate UNet-only kwargs from the engine
    ):
        super().__init__()
        assert horizon % patch_size == 0
        self.hidden_size = hidden_size
        self.transition_dim = transition_dim
        self.patch_size = patch_size
        self.condition_on_t = condition_on_t
        self.num_classes = num_classes
        self.num_patches = horizon // patch_size

        self.x_embedder = TrajPatchEmbedder(transition_dim, patch_size, hidden_size)

        ek = dict(hidden_size=hidden_size, weight_init="scaled_variance",
                  init_constant=embedding_init_constant)
        self.h_embedder = TimestepEmbedder(**ek)
        self.t_embedder = TimestepEmbedder(**ek) if condition_on_t else None
        self.omega_embedder = TimestepEmbedder(**ek)
        self.cfg_t_start_embedder = TimestepEmbedder(**ek)
        self.cfg_t_end_embedder = TimestepEmbedder(**ek)
        # null-class label embedding: index `num_classes` is the CFG-dropout (uncond) token
        self.y_embedder = nn.Embedding(num_classes + 1, hidden_size)
        nn.init.normal_(self.y_embedder.weight, std=embedding_init_constant / math.sqrt(hidden_size))

        tok = partial(nn.init.normal_, std=token_init_constant / math.sqrt(hidden_size))
        self.time_tokens = nn.Parameter(tok(torch.empty(num_time_tokens, hidden_size)))
        self.class_tokens = nn.Parameter(tok(torch.empty(num_classes, hidden_size)))
        self.omega_tokens = nn.Parameter(tok(torch.empty(num_cfg_tokens, hidden_size)))
        self.t_min_tokens = nn.Parameter(tok(torch.empty(num_interval_tokens, hidden_size)))
        self.t_max_tokens = nn.Parameter(tok(torch.empty(num_interval_tokens, hidden_size)))

        self.prefix_tokens = (num_classes + num_cfg_tokens + 2 * num_interval_tokens
                              + num_time_tokens)
        total_tokens = self.prefix_tokens + self.num_patches
        head_dim = hidden_size // num_heads
        cos, sin = precompute_rope_cos_sin(head_dim, total_tokens)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        bk = dict(hidden_size=hidden_size, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  weight_init="scaled_variance", weight_init_constant=weight_init_constant)
        shared_depth = depth - aux_head_depth
        self.shared_blocks = nn.ModuleList([TransformerBlock(**bk) for _ in range(shared_depth)])
        self.u_heads = nn.ModuleList([TransformerBlock(**bk) for _ in range(aux_head_depth)])
        self.v_heads = nn.ModuleList([TransformerBlock(**bk) for _ in range(aux_head_depth)])

        out_features = patch_size * transition_dim
        self.u_final_layer = FinalLayer(hidden_size, out_features)
        self.v_final_layer = FinalLayer(hidden_size, out_features)

    # ── helpers ─────────────────────────────────────────────────────────────────

    def _as_batched(self, val, b, device, default=0.0):
        if val is None:
            return torch.full((b,), default, dtype=torch.float32, device=device)
        if not torch.is_tensor(val):
            val = torch.tensor([val], dtype=torch.float32, device=device)
        elif val.ndim == 0:
            val = val[None]
        return (val.float() * torch.ones(b, dtype=torch.float32, device=device))

    def _unpatchify(self, x):
        b = x.shape[0]
        return x.reshape(b, self.num_patches * self.patch_size, self.transition_dim)

    def _build_sequence(self, x, t, h, omega, t_min, t_max, force_dropout):
        b = x.shape[0]
        dev = x.device
        x_embed = self.x_embedder(x)

        h_embed = self.h_embedder(self._as_batched(h, b, dev))
        time_tok = self.time_tokens[None] + h_embed[:, None]
        if self.condition_on_t and t is not None:
            time_tok = time_tok + self.t_embedder(self._as_batched(t, b, dev))[:, None]

        # ω is conditioned as (1 − 1/ω) per the official recipe; ω≤0 ⇒ 0 (guidance off)
        w = self._as_batched(omega, b, dev)
        w_arg = torch.where(w > 0, 1.0 - 1.0 / w.clamp(min=1e-6), torch.zeros_like(w))
        omega_tok = self.omega_tokens[None] + self.omega_embedder(w_arg)[:, None]
        tmin_tok = self.t_min_tokens[None] + self.cfg_t_start_embedder(self._as_batched(t_min, b, dev))[:, None]
        tmax_tok = self.t_max_tokens[None] + self.cfg_t_end_embedder(self._as_batched(t_max, b, dev))[:, None]

        # class token: y=0 (sole conditioning class) normally; null index on CFG dropout
        y_idx = torch.full((b,), self.num_classes if force_dropout else 0, dtype=torch.long, device=dev)
        class_tok = self.class_tokens[None] + self.y_embedder(y_idx)[:, None]

        return torch.cat([class_tok, omega_tok, tmin_tok, tmax_tok, time_tok, x_embed], dim=1)

    # ── contract: matches Flow_matcher_U_Net_v2.forward ──────────────────────────

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False,
                h=None, omega=None, t_min=None, t_max=None, return_v=False):
        """x:[B,H,D] → u (or (u,v)). `cond`/`returns`/`use_dropout` accepted for parity."""
        seq = self._build_sequence(x, time, h, omega, t_min, t_max, force_dropout)
        cos, sin = self.rope_cos, self.rope_sin

        for block in self.shared_blocks:
            seq = block(seq, cos, sin)

        u_seq = seq
        for block in self.u_heads:
            u_seq = block(u_seq, cos, sin)
        u = self._unpatchify(self.u_final_layer(u_seq[:, self.prefix_tokens:]))

        if return_v:
            v_seq = seq
            for block in self.v_heads:
                v_seq = block(v_seq, cos, sin)
            v = self._unpatchify(self.v_final_layer(v_seq[:, self.prefix_tokens:]))
            return u, v
        return u
