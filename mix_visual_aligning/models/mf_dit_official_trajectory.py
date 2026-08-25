"""Trajectory port of the **official MeanFlow** DiT — `imf_backbone='mf_dit'` (Gen3v6 U2).

This is a 100%-faithful port of the MeanFlow paper's reference network
`/workspaces/aux_repo/MeanFlow/models/dit.py` (`MFDiT`), NOT the iMF DiT that
`imf_backbone='dit'` (`mf_dit_trajectory.py`) uses. The two are architecturally distinct:

                      official MeanFlow MFDiT (here)        iMF MFDiTTrajectory ('dit')
    conditioning      adaLN-zero (t+r+w → 6·d modulation)   in-context prefix tokens
    positions         learned ABSOLUTE sin-cos pos-embed    RoPE
    attention         softmax + QK-RMSNorm                  softmax + QK-RMSNorm (RoPE)
    MLP               GELU(tanh), mlp_ratio=4.0             SwiGLU, mlp_ratio=8/3
    u/v heads         two FinalLayers on the SAME trunk     shared trunk → separate head blocks

Every LEARNED component is ported verbatim from `MFDiT`: the t/r/w `TimestepEmbedder`s
(freq=256, scale=1000), the adaLN-zero `DiTBlock`, QK-RMSNorm `Attention`, the twin
`FinalLayer`s, and the exact `initialize_weights` scheme (xavier trunk, sin-cos pos-embed,
0.02 timestep-MLP, zeroed adaLN + zeroed output layers).

The ONLY deviations are the ones FORCED by the data being 1-D trajectories `[B, H, D]`
rather than 2-D images `[N, C, H, W]` — and one dependency swap that changes no math:

  • PatchEmbed (2-D conv)         → TrajPatchEmbedder (1-D linear over `patch_size` steps)
  • get_2d_sincos_pos_embed       → get_1d_sincos_pos_embed (sequence is 1-D)
  • unpatchify (image reshape)    → `[B, H, D]` reshape
  • timm `Mlp` / `PatchEmbed`     → inline equivalents (no timm dependency on the cluster)
  • `nn.RMSNorm`                  → local `RMSNorm` (behaviourally identical; avoids a hard
                                    dep on a recent torch, mirroring the iMF port's choice)
  • class-label conditioning      → OFF (`num_classes=None`): trajectory conditioning is the
                                    pinned observation applied to x externally, exactly as the
                                    iMF port does; the unconditional MFDiT path (use_cond=False)

JVP-safety: the MeanFlow repo differentiates THIS network with
`torch.autograd.functional.jvp` (`meanflow.py:175`, `use_flash_attention=False`), so every
component is forward-AD friendly by construction. There is no RoPE complex-bitcast hazard
(this DiT has no RoPE). We always use the native (non-flash) attention path.

Contract (identical to `Flow_matcher_U_Net_v2` / `MFDiTTrajectory`):
    forward(x, cond, time, *, h, force_dropout, omega, t_min, t_max, return_v)
      -> u                       (return_v=False)
      -> (u, v)                  (return_v=True)
so `MFTrajectoryModel` swaps it in with NO change to the objective/JVP/sampler.

Time mapping (Gen3v6 DATA-AT-1 → MFDiT's two boundary times): the objective queries the
backbone at the anchor `time = r` with interval `h = t − r`, and JVP-differentiates w.r.t.
`(x, time, h)` with tangents `(v_inst, +1, −1)`. We feed MFDiT its two native times as
    r_abs = time         (anchor)     tangent → +1
    t_abs = time + h     (endpoint)   tangent → (+1) + (−1) = 0
so the induced perturbation lands on the anchor time — the mirror image of MeanFlow's own
`(t:+1, r:0)` under the noise-at-1 ↔ data-at-1 convention flip (see mf_diffusion.py:404).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────────── adaLN helpers (ported verbatim) ─────────────────────────────

def expand_ada_params(param, x):
    """Expand global (B, D) adaLN params to (B, 1, D); pass through token-level params."""
    if param.ndim == 2:
        return param.unsqueeze(1)
    return param


def modulate(x, scale, shift):
    """adaLN modulation. x:(B,T,D); scale/shift:(B,D) global or (B,T,D) token-level."""
    scale = expand_ada_params(scale, x)
    shift = expand_ada_params(shift, x)
    return x * (1 + scale) + shift


# ───────────────────────────── primitive layers ─────────────────────────────

class RMSNorm(nn.Module):
    """Behaviourally identical to `nn.RMSNorm` (the class MFDiT uses), written out so the
    port does not hard-depend on a recent torch. Per-token ⇒ JVP-safe.

    `elementwise_affine=False` reproduces MFDiT's `norm1/norm2/norm_final`; the default
    (affine=True) reproduces the QK RMSNorms.
    """

    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        ms = torch.mean(x.float().square(), dim=-1, keepdim=True)
        out = (x * torch.rsqrt(ms + self.eps)).to(x.dtype)
        if self.weight is not None:
            out = out * self.weight
        return out


class Mlp(nn.Module):
    """timm `Mlp` equivalent (drop=0): fc1 → GELU(tanh) → fc2. Ported."""

    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TimestepEmbedder(nn.Module):
    """Scalar → sinusoidal → MLP embedding. Ported verbatim from MFDiT (freq=256, scale=1000).

    Note the cos-then-sin concatenation order — kept exactly as the reference.
    """

    def __init__(self, dim, nfreq=256, scale=1000.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq
        self.scale = scale

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t = t * self.scale
        t_freq = self.timestep_embedding(t, self.nfreq)
        return self.mlp(t_freq)

    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)


class Attention(nn.Module):
    """Multi-head self-attention with QK-RMSNorm. Ported verbatim (native softmax path only).

    We keep ONLY the native attention path — `MFDiT`'s flash branch is JVP-unsafe and the
    reference disables it under jvp; the trajectory sequences are tiny so flash is moot.
    """

    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = self.q_norm(q)
        k = self.k_norm(k)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class DiTBlock(nn.Module):
    """adaLN-zero transformer block. Ported verbatim from MFDiT."""

    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = RMSNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True, qk_norm=True)
        self.norm2 = RMSNorm(dim, elementwise_affine=False)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + expand_ada_params(gate_msa, x) * self.attn(
            modulate(self.norm1(x), scale_msa, shift_msa)
        )
        x = x + expand_ada_params(gate_mlp, x) * self.mlp(
            modulate(self.norm2(x), scale_mlp, shift_mlp)
        )
        return x


class FinalLayer(nn.Module):
    """adaLN-zero final layer → per-patch output. Ported verbatim (1-D `out_features`)."""

    def __init__(self, dim, out_features):
        super().__init__()
        self.norm_final = RMSNorm(dim, elementwise_affine=False)
        self.linear = nn.Linear(dim, out_features)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), scale, shift)
        return self.linear(x)


# ───────────────────────────── trajectory patch embed ─────────────────────────────

class TrajPatchEmbedder(nn.Module):
    """`[B, H, D]` → `[B, H//patch, hidden]` via a linear lift over `patch` steps.

    The 1-D analogue of MFDiT's Conv2d `PatchEmbed`. Initialised xavier-uniform like the
    reference's patch-embed proj (see `initialize_weights`).
    """

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


# ───────────────────────────── 1-D sin-cos position embedding ─────────────────────────────

def get_1d_sincos_pos_embed(embed_dim, length):
    """[length, embed_dim] sin-cos table (the 1-D analogue of MFDiT's 2-D grid embed)."""
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000 ** omega
    pos = np.arange(length, dtype=np.float64).reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1)
    return emb


# ───────────────────────────── the trajectory official-MeanFlow DiT ─────────────────────────────

class MFDiTOfficialTrajectory(nn.Module):
    """Trajectory port of the official MeanFlow `MFDiT`. Drop-in `velocity_net`.

    Single trunk of `depth` adaLN-zero blocks conditioned on `c = t_emb + r_emb + w_emb`,
    then two FinalLayers (u, v) reading the SAME trunk output (as in the reference).
    Returns u, or (u, v) when `return_v=True` (v dropped at inference).
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
        # ── Gen14 U8 ── visual conditioning. 0 = state-only (byte-identical to pre-U8).
        # >0 = the 128-D dual-cam latent is PREPENDED as one token (not summed into adaLN's `c`
        # — see logs_in_develop/Gen14/U8/DECISION_Gen14_U8_injection_choice.md §2/§4.1).
        cond_dim: int = 0,
        # ── Gen14 U9 ── WHERE the visual latent enters. 'token' == U8, bit-identical.
        #   'token' : prepended as one sequence token (U8, the shipped design point)
        #   'adaln' : summed into adaLN's `c`, the way DiT conditions on a class label —
        #             the transformer analogue of VisualUNet v1's cond_mlp-into-`t`, which
        #             is the best-scoring mechanism this generation has (0.3425 pooled)
        #   'both'  : token AND modulation
        vis_cond_mode: str = 'token',
        max_cfg: float = 4.0,
        **unused,  # tolerate UNet-/iMF-only kwargs threaded by the engine
    ):
        super().__init__()
        assert horizon % patch_size == 0
        self.hidden_size = hidden_size
        self.transition_dim = transition_dim
        self.patch_size = patch_size
        self.num_patches = horizon // patch_size
        self.max_cfg = max_cfg

        self.x_embedder = TrajPatchEmbedder(transition_dim, patch_size, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.r_embedder = TimestepEmbedder(hidden_size)
        self.w_embedder = TimestepEmbedder(hidden_size)


        # ── Gen14 U8 ── visual token (Option 2). Prepended to the patch sequence rather than
        # summed into `c`: diffusion_policy — the upstream of THIS repo's vision encoder —
        # conditions its transformer on obs tokens and reserves modulation (FiLM) for its U-Net,
        # which is the design point our VisualUNet already occupies.
        self.use_visual = cond_dim > 0
        self.cond_dim = cond_dim
        self.num_visual_tokens = 1 if self.use_visual else 0
        # ── Gen14 U9 ── in 'adaln' the latent never joins the sequence, so it claims no
        # position. num_tokens is computed on the NEXT line and picks this up, which keeps
        # pos_embed, the sin-cos table and the strip in forward() consistent automatically
        # (G-B6 asserts the three agree; that assertion now covers all three modes).
        self.vis_cond_mode = str(vis_cond_mode)
        if self.vis_cond_mode not in ('token', 'adaln', 'both'):
            raise ValueError(
                f"[ MFDiTOfficialTrajectory ] vis_cond_mode='{self.vis_cond_mode}' is not one "
                "of 'token' | 'adaln' | 'both'.")
        if self.use_visual and self.vis_cond_mode == 'adaln':
            self.num_visual_tokens = 0
        self.num_tokens = self.num_visual_tokens + self.num_patches
        if self.use_visual:
            self.vis_projector = nn.Linear(cond_dim, hidden_size)
            self.vis_token = nn.Parameter(torch.zeros(1, 1, hidden_size))
            # ── Gen14 U9 ── 'adaln' has no sequence token, so vis_token would be an
            # untrainable dead parameter sitting in every checkpoint. Remove it: the
            # state_dict then states the conditioning mode honestly, and a checkpoint
            # trained in one mode cannot be silently loaded into another.
            if self.vis_cond_mode == 'adaln':
                del self.vis_token

        # learned ABSOLUTE pos-embed, sin-cos initialised (requires_grad=True, as in MFDiT).
        # Gen14 U8: sized over num_tokens = num_visual_tokens + num_patches, so the prepended
        # visual token gets its own position (mirrors diffusion_policy's separate `cond_pos_emb`,
        # in the simplest form that keeps ONE table).
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size), requires_grad=True)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)
        ])
        out_features = patch_size * transition_dim
        self.final_layer_v = FinalLayer(hidden_size, out_features)
        self.final_layer_u = FinalLayer(hidden_size, out_features)

        self.initialize_weights()

    def initialize_weights(self):
        # xavier trunk, zero bias (MFDiT `_basic_init`)
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # sin-cos pos-embed (1-D)
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.num_tokens)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # patch-embed proj like nn.Linear (already covered by _basic_init; kept explicit
        # to mirror the reference's separate xavier on the patch proj)
        nn.init.xavier_uniform_(self.x_embedder.proj.weight)
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # timestep embedding MLPs (t, r, w): normal std=0.02
        for embedder in (self.t_embedder, self.r_embedder, self.w_embedder):
            embedder.initialize_weights()

        # zero-out adaLN in DiT blocks
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # zero-out output layers (adaLN + linear)
        for final_layer in (self.final_layer_v, self.final_layer_u):
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

    @staticmethod
    def _pool_cond(cond):
        """Gen14 U9 — (B, cond_dim) out of whatever the wrapper handed down.

        Mirrors the window pooling already inside _prepend_visual so the two conditioning
        paths can never disagree about what 'the latent' is. At window_size=1 (the shipped
        aligning setting) the mean is a no-op.
        """
        if cond is None:
            raise ValueError(
                '[ MFDiTOfficialTrajectory ] vis_cond_mode needs a visual latent but got '
                'None — the wrapper must resolve cond -> (B, cond_dim).')
        return cond.mean(dim=1) if cond.ndim == 3 else cond

    def _prepend_visual(self, x, cond):
        """Gen14 U8 — prepend the visual token. No-op (and `cond` ignored) when state-only.

        `cond` is the (B, cond_dim) latent already encoded upstream by VisualDiTTwoTime, so
        inside the MeanFlow / alpha-Flow JVP it is a captured CONSTANT: its forward-mode tangent
        is zero by construction and the ResNets never enter the differentiated function.
        """
        if not self.use_visual:
            return x
        if cond is None:
            raise ValueError(
                f"[ {type(self).__name__} ] cond_dim>0 but no visual latent reached the backbone. "
                "The wrapper (VisualDiTTwoTime) must resolve cond -> (B, cond_dim) and pass it as "
                "`cond`; training image-blind is exactly what this guard prevents.")
        if cond.ndim == 3:                 # (B, T_win, C) -> pool the window
            cond = cond.mean(dim=1)
        # ── Gen14 U9 ── the guards above still run in 'adaln' (an image-blind run must
        # fail loudly in EVERY mode); only the concatenation is skipped.
        if self.vis_cond_mode == 'adaln':
            return x
        vis = self.vis_token + self.vis_projector(cond)[:, None]   # (B, 1, D)
        return torch.cat([vis, x], dim=1)

    # ── contract: matches Flow_matcher_U_Net_v2.forward ────────────────────────────

    def forward(self, x, cond, time, returns=None, use_dropout=True, force_dropout=False,
                h=None, omega=None, t_min=None, t_max=None, return_v=False):
        """x:[B,H,D] → u (or (u,v)). `cond`/`returns`/CFG kwargs accepted for parity.

        Gen3v6 has NO CFG: `omega/t_min/t_max` are ignored and w falls back to ones (⇒ the
        w-embed is a constant, guidance off), exactly as the reference does when `w is None`.
        """
        b = x.shape[0]
        dev = x.device

        r_abs = self._as_batched(time, b, dev)          # anchor time (tangent +1 under JVP)
        h_b = self._as_batched(h, b, dev, default=0.0)
        t_abs = r_abs + h_b                              # endpoint time (tangent 0 under JVP)

        # w=None → ones (as MFDiT); normalise by max_cfg before embedding.
        w = torch.ones(b, dtype=torch.float32, device=dev) / self.max_cfg

        x = self.x_embedder(x)                          # (B, num_patches, D)
        x = self._prepend_visual(x, cond)               # Gen14 U8 — no-op when state-only
        x = x + self.pos_embed                          # (B, num_tokens, D)
        c = self.t_embedder(t_abs) + self.r_embedder(r_abs) + self.w_embedder(w)
        # ── Gen14 U9 ── vision into the MODULATION path. Fully skipped in 'token', so
        # that mode stays BIT-IDENTICAL to U8 (G-B9 asserts it) — this is an added branch,
        # never an added no-op arithmetic term.
        if self.use_visual and self.vis_cond_mode in ('adaln', 'both'):
            c = c + self.vis_projector(self._pool_cond(cond))

        for block in self.blocks:
            x = block(x, c)

        x = x[:, self.num_visual_tokens:]               # Gen14 U8 — strip the visual position
        u = self._unpatchify(self.final_layer_u(x, c))
        if not return_v:
            return u
        v = self._unpatchify(self.final_layer_v(x, c))
        return u, v
