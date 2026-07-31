"""Trajectory port of α-Flow's **own** backbone, the **SiT** — `imf_backbone='sit'` (Gen3v7 U2).

This mirrors the α-Flow reference repo's network
`/workspaces/aux_repo/alphaflow/src/training/dit.py` (class `SiT`), the SAME way Gen3v6 U2
mirrored the *MeanFlow* repo's `MFDiT` for its objective. It is NOT MeanFlow's MFDiT and NOT
the iMF DiT that `imf_backbone='dit'` (`af_dit_trajectory.py`) uses. All three are the DiT/SiT
family but differ on every learned detail:

                      α-Flow SiT (here)                 MeanFlow MFDiT            iMF DiT ('dit')
    conditioning      adaLN-zero, c = t_emb+r_emb        adaLN-zero, t+r+w        in-context prefix tokens
    norm              **LayerNorm** (affine off, fp32)   RMSNorm                  RMSNorm
    QK-norm           **OFF** (`qk_norm=False`)          ON (QK-RMSNorm)          ON (RoPE + QK-RMSNorm)
    positions         frozen sin-cos (requires_grad=F)   learned sin-cos (grad)   RoPE
    MLP               GELU(tanh), mlp_ratio=4.0          GELU(tanh), 4.0          SwiGLU, 8/3
    time embed        freq=256, **no scale**             freq=256, **scale=1000**  (rope)
    heads             **single** (u only) + analytic v   twin u/v FinalLayers     shared trunk → head blocks

Every LEARNED component is ported verbatim from α-Flow's `SiT`/`SiTBlock`/`FinalLayer`/
`TimestepEmbedder`: the adaLN-zero block (6·d modulation, gate applied outside the norm),
`torch.nn.LayerNorm(eps=1e-6, elementwise_affine=False)` run in fp32 inside `modulate`
(`x*(scale+1)+shift`), timm-style softmax `Attention` with **`qk_norm=False`**, `Mlp` with
`approx_gelu = GELU(approximate="tanh")` at `mlp_ratio=4.0`, the two time embedders
(`noise_labels` = t, `noise_labels_next` = r, freq=256, **no `scale` multiplier** — this is a
real difference from MeanFlow), and α-Flow's exact `initialize_weights` (xavier trunk, 0.02
time-MLPs, zeroed adaLN, zeroed output linear).

Two families of deviation, both explicitly flagged:

  (A) FORCED by the data being 1-D trajectories `[B,H,D]` instead of 2-D images `[N,C,H,W]`
      (no math change — the same class Gen3v6's MFDiT port carried):
      • PatchEmbed (2-D conv)      → TrajPatchEmbedder (1-D linear over `patch_size` steps)
      • 2-D sin-cos pos_embed      → 1-D sin-cos pos_embed (the sequence is 1-D), still FROZEN
      • unpatchify (image reshape) → `[B,H,D]` reshape
      • timm `Attention`/`Mlp`     → inline equivalents (no timm on the cluster), qk_norm=False
      • class-label `y_embedder`   → OFF: α-Flow's unconditional path (trajectory conditioning
                                     is the pinned observation applied to x externally, exactly
                                     as the other two backbones do) ⇒ c = t_emb + r_emb only.

  (B) FORCED by the FM-PCC Gen3v7 lineage contract, NOT by α-Flow:
      • α-Flow's SiT is SINGLE-head (predicts u; the instantaneous v is analytic in α-Flow).
        The FM-PCC lineage trains a v-head as an auxiliary loss against the analytic `v_inst`
        (`af_diffusion.py:735`) and its wrapper expects `(u, v)` — exactly as the 'unet'
        (dual_head) and 'dit' arms already do. So we add a SECOND `final_layer_v` reading the
        SAME trunk (a twin-head SiT). This changes nothing α-Flow deploys: v is dropped at
        inference and the deployed field is u, so the α-Flow-faithful u-path is untouched; the
        v-head only feeds the lineage's shared-trunk aux regulariser. If a future run wants the
        strictly-single-head α-Flow, drop `final_layer_v` and route v through the wrapper's
        legacy aux MLP.

JVP-safety: α-Flow's α=0 branch differentiates this net with `torch.func.jvp`
(`af_diffusion.py:575`, tangents `(v_inst, +1, −1)`). Safe by construction: LayerNorm,
softmax attention (no QK-norm here), and GELU are all forward-AD friendly, and there is no RoPE
complex-bitcast hazard (SiT uses a plain frozen sin-cos pos_embed). We use the native
(non-flash) attention path.

Contract (identical to `Flow_matcher_U_Net_v2` / `AFDiTTrajectory`):
    forward(x, cond, time, *, h, force_dropout, omega, t_min, t_max, return_v)
      -> u                       (return_v=False)
      -> (u, v)                  (return_v=True)
so `AFTrajectoryModel` swaps it in with NO change to the objective/JVP/sampler.

Time mapping (Gen3v7 DATA-AT-1 → SiT's two boundary times): the objective queries the backbone
at anchor `time = r` with interval `h = t − r`, and JVP-differentiates w.r.t. `(x, time, h)`
with tangents `(v_inst, +1, −1)`. SiT wants two times `noise_labels` (its t) and
`noise_labels_next` (its r); we feed:
    r_abs = time         (anchor)     → noise_labels_next (r), JVP tangent +1
    t_abs = time + h     (endpoint)   → noise_labels        (t), JVP tangent (+1)+(−1) = 0
so the perturbation lands on the anchor — the mirror image of α-Flow's own noise-at-1
`(t, r=t_next)` under the noise-at-1 ↔ data-at-1 convention flip (see the af_diffusion
convention note). α-Flow's `noise_labels ≥ noise_labels_next` invariant holds here since
t_abs = r_abs + h ≥ r_abs for h ≥ 0. These are the same tangents the iMF 'dit' arm uses, so the
mapping is valid for this arm unchanged.
"""

import math

import numpy as np
import torch
import torch.nn as nn


# ───────────────────────────── adaLN modulate (ported verbatim from α-Flow) ─────────────────────────────

def modulate(norm_func, x, shift, scale):
    """α-Flow's `modulate`: run the norm in fp32, then `x*(scale+1)+shift`.

    x:(B,N,D); shift/scale:(B,D) global → unsqueezed to (B,1,D), or already (B,N,D).
    """
    dtype = x.dtype
    x = norm_func(x.float())
    if scale is not None:
        assert shift is not None
        scale, shift = [s.unsqueeze(1) if s.ndim == 2 else s for s in (scale, shift)]
        x = x * (scale + 1) + shift
    return x.to(dtype)


# ───────────────────────────── primitive layers ─────────────────────────────

class Mlp(nn.Module):
    """timm `Mlp` equivalent (drop=0) with α-Flow's `approx_gelu = GELU(tanh)`."""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=True)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Attention(nn.Module):
    """timm `Attention` equivalent, `qk_norm=False` (as α-Flow's SiTBlock passes at dit.py:121).

    Native softmax path only — JVP-safe, and the trajectory sequences are tiny so flash is moot.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # no QK-norm (qk_norm=False)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TimestepEmbedder(nn.Module):
    """α-Flow `TimestepEmbedder`: sinusoidal (freq=256, cos-then-sin) → MLP. NO `scale` factor
    (unlike MeanFlow's `scale=1000`) — kept exactly as the reference (dit.py:351)."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

    def initialize_weights(self):
        # α-Flow: initialize_time_mlp_weights → normal std=0.02 on mlp[0] and mlp[2]
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)


class SiTBlock(nn.Module):
    """α-Flow `SiTBlock`: adaLN-zero, LayerNorm(affine off), qk_norm=False, GELU(tanh) Mlp.

    Note the gate is applied OUTSIDE the (modulated) sublayer — `x + gate·f(modulate(...))` —
    exactly as α-Flow (dit.py:44-53).
    """

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1, x, shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2, x, shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """α-Flow `FinalLayer`: adaLN-zero (2·d), LayerNorm(affine off), Linear → out_features."""

    def __init__(self, dim, out_features):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(dim, out_features, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final, x, shift, scale)
        return self.linear(x)


# ───────────────────────────── trajectory patch embed ─────────────────────────────

class TrajPatchEmbedder(nn.Module):
    """`[B,H,D]` → `[B, H//patch, hidden]` via a linear lift over `patch` steps (1-D analogue
    of α-Flow's Conv2d `PatchEmbed`)."""

    def __init__(self, transition_dim, patch_size, hidden_size):
        super().__init__()
        self.patch_size = patch_size
        self.transition_dim = transition_dim
        self.proj = nn.Linear(patch_size * transition_dim, hidden_size)

    def forward(self, x):
        b, h, d = x.shape
        assert h % self.patch_size == 0, f"horizon {h} not divisible by patch {self.patch_size}"
        x = x.reshape(b, h // self.patch_size, self.patch_size * d)
        return self.proj(x)


# ───────────────────────────── 1-D sin-cos position embedding (α-Flow's, dit.py:292) ─────────────────────────────

def get_1d_sincos_pos_embed(embed_dim, length):
    """[length, embed_dim] sin-cos table (α-Flow's `get_1d_sincos_pos_embed_from_grid`,
    sin-then-cos order). The 1-D analogue of SiT's frozen 2-D grid embed."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.arange(length, dtype=np.float64).reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)  # sin then cos (α-Flow order)
    return emb


# ───────────────────────────── the trajectory α-Flow SiT ─────────────────────────────

class AFSiTTrajectory(nn.Module):
    """Trajectory port of α-Flow's `SiT`. Drop-in `velocity_net`.

    Single trunk of `depth` adaLN-zero SiTBlocks conditioned on `c = t_emb + r_emb`, then two
    FinalLayers (u, v) reading the SAME trunk output. α-Flow itself is single-head (u); the v
    head is the FM-PCC-lineage aux head (dropped at inference). Returns u, or (u, v) when
    `return_v=True`.
    """

    def __init__(
        self,
        horizon: int,
        transition_dim: int,
        hidden_size: int = 256,
        depth: int = 8,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        patch_size: int = 1,
        **unused,  # tolerate UNet-/iMF-only kwargs threaded by the engine
    ):
        super().__init__()
        assert horizon % patch_size == 0
        self.hidden_size = hidden_size
        self.transition_dim = transition_dim
        self.patch_size = patch_size
        self.num_patches = horizon // patch_size

        self.x_embedder = TrajPatchEmbedder(transition_dim, patch_size, hidden_size)
        # α-Flow's two time embedders: noise_labels (t) + noise_labels_next (r).
        self.noise_labels_embedder = TimestepEmbedder(hidden_size)
        self.noise_labels_next_embedder = TimestepEmbedder(hidden_size)

        # FROZEN sin-cos pos-embed (requires_grad=False, as α-Flow's SiT).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            SiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        out_features = patch_size * transition_dim
        self.final_layer_u = FinalLayer(hidden_size, out_features)   # α-Flow's single head
        self.final_layer_v = FinalLayer(hidden_size, out_features)   # FM-PCC aux head (see docstring B)

        self.initialize_weights()

    def initialize_weights(self):
        # xavier trunk, zero bias (α-Flow `_basic_init`, linear_init_scale=1.0)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # frozen 1-D sin-cos pos-embed
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_patches)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # patch-embed proj xavier (α-Flow inits the conv proj xavier; kept explicit)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # time-embedder MLPs: normal std=0.02
        self.noise_labels_embedder.initialize_weights()
        self.noise_labels_next_embedder.initialize_weights()

        # zero-out adaLN in SiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out output layers (adaLN + linear), both heads
        for final_layer in (self.final_layer_u, self.final_layer_v):
            nn.init.constant_(final_layer.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(final_layer.adaLN_modulation[-1].bias, 0)
            nn.init.constant_(final_layer.linear.weight, 0)
            nn.init.constant_(final_layer.linear.bias, 0)

    # ── helpers ──────────────────────────────────────────────────────────────────

    def _as_batched(self, val, b, device, default=0.0):
        if val is None:
            return torch.full((b,), default, dtype=torch.float32, device=device)
        if not torch.is_tensor(val):
            val = torch.tensor([val], dtype=torch.float32, device=device)
        elif val.ndim == 0:
            val = val[None]
        return val.float() * torch.ones(b, dtype=torch.float32, device=device)

    def _unpatchify(self, x):
        b = x.shape[0]
        return x.reshape(b, self.num_patches * self.patch_size, self.transition_dim)

    # ── contract: matches Flow_matcher_U_Net_v2.forward ────────────────────────────

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False,
                h=None, omega=None, t_min=None, t_max=None, return_v=False):
        """x:[B,H,D] → u (or (u,v)). `cond`/`returns`/CFG kwargs accepted for parity.

        α-Flow's SiT is unconditional here (y-embedder off): c = t_emb(t) + r_emb(r). CFG knobs
        `omega/t_min/t_max` are ignored (Gen3v7 has no interval-CFG).
        """
        b = x.shape[0]
        dev = x.device

        r_abs = self._as_batched(time, b, dev)          # anchor time (tangent +1 under JVP) → r
        h_b = self._as_batched(h, b, dev, default=0.0)
        t_abs = r_abs + h_b                              # endpoint time (tangent 0 under JVP) → t

        x = self.x_embedder(x) + self.pos_embed         # (B, num_patches, D)
        # c = noise_labels(t) + noise_labels_next(r), matching α-Flow's dit.py:199 (y dropped).
        c = self.noise_labels_embedder(t_abs) + self.noise_labels_next_embedder(r_abs)

        for block in self.blocks:
            x = block(x, c)

        u = self._unpatchify(self.final_layer_u(x, c))
        if not return_v:
            return u
        v = self._unpatchify(self.final_layer_v(x, c))
        return u, v
