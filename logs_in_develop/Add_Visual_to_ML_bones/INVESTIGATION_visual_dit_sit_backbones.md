# Investigation: Adding Visual Conditioning to DiT / SiT / iMF-DiT Backbones

> **Date**: 2026-08-11
> **Context**: Conversation-driven investigation — is there any visual model in FM-PCC that uses a real Transformer/DiT backbone instead of the VisualUNet? And how hard would it be to add one?

---

## Part 1 — Aggregated Findings: Current Visual Model Landscape

### 1.1 Question: Do ANY visual models use DiT / SiT / Transformer?

**Answer: No.** Every active visual pipeline in FM-PCC uses `VisualUNet` (or `VisualUNetTwoTime`). This is a 1D temporal convolutional U-Net, NOT a Transformer / DiT.

### 1.2 What `VisualUNet` actually is

All visual models across Gen6 → Gen14 use this architecture:

```
┌────────────────────────────────────────────────┐
│  VisualUNet / VisualUNetTwoTime                │
│  ┌──────────────────────────────┐              │
│  │ MultiImageObsEncoder         │ ← D3IL       │
│  │ (2× ResNet-18/64)            │              │
│  │ agentview_image → 64D        │              │
│  │ in_hand_image   → 64D        │              │
│  │ concat → 128D visual latent  │              │
│  └──────────┬───────────────────┘              │
│             │  FiLM conditioning               │
│             ▼                                  │
│  ┌──────────────────────────────┐              │
│  │ UNet1DTemporalCondModel      │ ← 1D Conv   │
│  │ (Conv1d residual blocks,     │              │
│  │  stride-2 down/up,           │              │
│  │  time embedding via MLP)     │              │
│  └──────────────────────────────┘              │
└────────────────────────────────────────────────┘
```

This is used across:
| Gen | Folder | Visual Backbone |
|:----|:-------|:----------------|
| Gen6v4 (Diffusion/DDPM) | `diffuser_visual_aligning/` | `VisualUNet` → `UNet1DTemporalCondModel` |
| Gen7 (FM ODE) | `fm_visual_aligning/` | `VisualUNet` → `UNet1DTemporalCondModel` |
| Gen9 (FM Avoiding) | `fm_visual_avoiding/` | `VisualUNet` → `UNet1DTemporalCondModel` (single-cam) |
| Gen8 (iMF Visual) | `imf_visual_aligning/` | `VisualUNet` → `UNet1DTemporalCondModel` |
| Gen14 (Mix-ML, diffusion arm) | `mix_visual_aligning/` | `VisualUNet` → `UNet1DTemporalCondModel` / `FiLMModel` |
| Gen14 (Mix-ML, fm arm) | `mix_visual_aligning/` | `VisualUNet` → `UNet1DTemporalCondModel` / `FiLMModel` |
| Gen14 (Mix-ML, mf arm) | `mix_visual_aligning/` | `VisualUNetTwoTime` → `Flow_matcher_U_Net_v2` (two-time) |
| Gen14 (Mix-ML, af arm) | `mix_visual_aligning/` | `VisualUNetTwoTime` → `Flow_matcher_U_Net_v2` (two-time) |

### 1.3 Where DiT / SiT backbones exist (state-only)

The repo has **four** distinct DiT/SiT backbone implementations, all **state-only** (no visual conditioning):

| Backbone Class | File | Architecture | Used By |
|:---------------|:-----|:-------------|:--------|
| `MFDiTTrajectory` | `mf_dit_trajectory.py` | iMF-DiT: RoPE + QK-RMSNorm, SwiGLU, in-context prefix tokens, shared trunk → u/v head blocks | MF engine (`imf_backbone='dit'`) |
| `MFDiTOfficialTrajectory` | `mf_dit_official_trajectory.py` | Official MeanFlow MFDiT: adaLN-zero, learned sin-cos pos-embed, GELU, twin u/v FinalLayers on same trunk | MF engine (`imf_backbone='mf_dit'`) |
| `AFDiTTrajectory` | `af_dit_trajectory.py` | Same arch as `MFDiTTrajectory` (iMF port), shared by α-Flow | AF engine (`imf_backbone='dit'`) |
| `AFSiTTrajectory` | `af_sit_trajectory.py` | α-Flow SiT: adaLN-zero, LayerNorm (not RMSNorm), qk_norm=OFF, frozen sin-cos pos-embed, GELU, twin u/v FinalLayers | AF engine (`imf_backbone='sit'`) |

**All four are blocked from visual use** by explicit safety assertions in both `MFTrajectoryModel` and `AFTrajectoryModel`:

```python
# mix_visual_aligning/models/mf_trajectory_model.py:82-87
if if_vision:
    if imf_backbone not in ('unet',):
        raise ValueError(
            f"[ MFTrajectoryModel ] if_vision=True requires imf_backbone='unet' "
            f"(got '{imf_backbone}'). The DiT/SiT backbones have no visual "
            f"conditioning path and would train image-blind."
        )
```

### 1.4 What about D3IL's DDPM-ACT (Transformer)?

D3IL's `ddpm_encdec_vision_agent` does use a real Transformer (encoder + decoder from ACT, `agents.models.act.act_vae.TransformerEncoder/Decoder`), but:
- It is a **vendored external benchmark**, not your model.
- Gen5 tried to bridge it into FM-PCC but it was **archived/abandoned** in `Archived_Codes/ddpm_encdec_vision_Legacy/`.
- It was never integrated into the DPCC/FM/MF/AF engine lineage.

### 1.5 `aux_repo` — no visual models at all

All repositories in `/workspaces/aux_repo/` (`HardFlow`, `MeanFlow`, `alphaflow`, `imeanflow`, `SafeFlowMPC`, `UAV-Flow`, `dpcc`, etc.) are **purely state-only**. Zero visual conditioning, zero image inputs.

---

## Part 2 — Feasibility Analysis: Adding Visual Conditioning to DiT/SiT

### 2.1 What needs to change

The DiT/SiT backbones currently see trajectories as `[B, H, D]` vectors. To add visual conditioning, two things must happen:

1. **The ResNet vision encoder must run** (to produce a 128D latent from the dual cameras).
2. **The 128D latent must reach the DiT/SiT blocks** as a conditioning signal.

Step (1) is already solved — `VisualUNetTwoTime` owns the `MultiImageObsEncoder` and the `encode_visual()` + `resolve_visual_cond()` logic. The question is entirely about step (2): how to inject the visual latent into the Transformer blocks.

### 2.2 Three injection strategies (increasing difficulty)

#### Strategy A: adaLN Injection (Easiest — MFDiT / SiT only)

**Difficulty: Low–Medium. ~100 lines of code per backbone. No architectural redesign.**

The `MFDiTOfficialTrajectory` and `AFSiTTrajectory` already use **adaLN-zero conditioning**: each block modulates its norm layer via a conditioning vector `c` (e.g. `c = t_emb + r_emb + w_emb`). Adding vision is as simple as:

```python
# Before:
c = self.t_embedder(t_abs) + self.r_embedder(r_abs) + self.w_embedder(w)

# After:
c = self.t_embedder(t_abs) + self.r_embedder(r_abs) + self.w_embedder(w)
if visual_latent is not None:
    c = c + self.vis_embedder(visual_latent)   # new nn.Linear(128, hidden_size)
```

The visual latent is projected into the adaLN conditioning space and **summed in** alongside the time embeddings. Every block already processes `c`, so the visual signal modulates every layer automatically.

**Pros:**
- Minimal code change (~1 new MLP + 1 line per forward).
- Stays faithful to the official DiT conditioning paradigm (this is literally how class labels condition in the original DiT paper).
- JVP-safe by construction — the visual latent is a captured constant in the JVP closure (see PLAN §6.1 in `visual_unet_twotime.py`), so its tangent is zero.

**Cons:**
- The conditioning is GLOBAL — the same visual modulation hits every patch/step uniformly. No spatial/temporal attention between image features and trajectory steps.

**Applies to:** `MFDiTOfficialTrajectory`, `AFSiTTrajectory` (both use adaLN-zero natively).

#### Strategy B: In-Context Prefix Token Injection (Medium — iMF DiT)

**Difficulty: Medium. ~80 lines per backbone. Slight RoPE complication.**

The `MFDiTTrajectory` (and its α-Flow copy `AFDiTTrajectory`) use **in-context prefix tokens** for conditioning: time, class, ω, t_min, t_max are all projected into `hidden_size` vectors and prepended to the trajectory patch sequence. The self-attention over the full sequence naturally lets trajectory patches attend to conditioning tokens.

Adding vision means creating **visual prefix tokens**:

```python
# In _build_sequence():
# New: project visual latent to prefix token(s)
if visual_latent is not None:
    vis_tok = self.vis_tokens[None] + self.vis_projector(visual_latent)[:, None]
    # vis_tok: [B, num_vis_tokens, hidden_size]
```

Then prepend them to the sequence alongside the existing conditioning tokens:

```python
return torch.cat([class_tok, omega_tok, tmin_tok, tmax_tok, time_tok,
                  vis_tok,   # ← NEW
                  x_embed], dim=1)
```

**Complication:** Adding tokens changes `total_tokens`, which shifts the RoPE cos/sin buffer length and the `prefix_tokens` count. The RoPE table must be recomputed for the new total, and `prefix_tokens` must be bumped so the output layer still strips the prefix correctly.

**Pros:**
- Architecturally natural for the iMF DiT — this is exactly how it conditions on everything else.
- Trajectory patches can **attend to** the visual tokens via self-attention, giving richer interaction than adaLN.

**Cons:**
- Must update RoPE buffer dimensions.
- Slightly more complex than Strategy A.
- Still a global signal (one pooled 128D vector), not spatially rich.

**Applies to:** `MFDiTTrajectory`, `AFDiTTrajectory` (both use in-context prefix tokens).

#### Strategy C: Cross-Attention with Spatial Features (Hardest)

**Difficulty: High. ~300+ lines per backbone. Significant architectural change.**

Instead of pooling the ResNet output to a single 128D vector, keep the **spatial feature map** (e.g. `[B, 64, H_feat, W_feat]` from ResNet) and add **cross-attention layers** in each DiT/SiT block where trajectory patches attend to image feature tokens.

```python
class VisualDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, ...):
        self.self_attn = RoPEAttention(...)
        self.cross_attn = CrossAttention(hidden_size, num_heads, kv_dim=img_feat_dim)
        self.mlp = SwiGLUMlp(...)
        ...

    def forward(self, x, cos, sin, img_tokens):
        x = x + self.self_attn(self.norm1(x), cos, sin) * self.attn_scale
        x = x + self.cross_attn(self.norm_cross(x), img_tokens) * self.cross_scale
        x = x + self.mlp(self.norm2(x)) * self.mlp_scale
        return x
```

**Pros:**
- Richest interaction — trajectory patches can attend to specific spatial regions of the image.
- The standard approach in state-of-the-art vision-language and vision-control models (Stable Diffusion 3, RT-2, etc.).

**Cons:**
- Major architectural surgery: every block gains a new attention layer.
- Increases parameter count and memory substantially.
- The ResNet must produce **spatial** (not pooled) features, requiring changes to the `MultiImageObsEncoder` configuration.
- **JVP-safety is uncertain**: cross-attention with a large spatial feature map inside the JVP closure may have memory/compute issues. The pre-encode short-circuit still works (image features are a captured constant), but the cross-attention itself adds more ops inside the differentiated function.
- Way heavier than what the current task complexity warrants.

### 2.3 JVP Safety — The Critical Constraint for MF/AF

The MeanFlow (mf) and α-Flow (af) objectives differentiate the backbone with `torch.func.jvp`:

```python
u_pred, du_dr = jvp(_u_of, (x_r, r, h), (v_inst, ones, -ones))
```

**All components inside the JVP must be forward-AD friendly.** The existing visual pipeline solves this with the **pre-encode short-circuit** (`visual_unet_twotime.py:19-41`): the ResNet vision encoder runs ONCE outside the JVP, and the resulting 128D tensor is captured as a constant inside the closure.

For DiT/SiT, the same principle applies:
- **Strategy A (adaLN sum)**: Safe. The visual latent enters `c` as a constant additive term. adaLN modulation (`x * (1 + scale) + shift`) is forward-AD friendly.
- **Strategy B (prefix tokens)**: Safe. The visual token is prepended as a constant. Self-attention, softmax, RoPE (real-valued) are all forward-AD friendly.
- **Strategy C (cross-attention)**: Likely safe but untested. Cross-attention with a constant KV is mathematically equivalent to a learned linear projection of the query, which is AD-friendly. But the memory cost of running cross-attention inside JVP may be prohibitive.

### 2.4 Practical Recommendation

**Strategy A (adaLN injection) is recommended as the starting point.** It is the smallest diff, architecturally faithful, JVP-safe, and delivers the core capability (visual conditioning) with ~100 lines per backbone.

### 2.5 Concrete Implementation Sketch (Strategy A — `MFDiTOfficialTrajectory`)

The changes would be:

#### In `mf_dit_official_trajectory.py`:

```diff
 class MFDiTOfficialTrajectory(nn.Module):
     def __init__(self, ...,
+                 cond_dim: int = 0,        # visual latent dim (0 = no vision)
                  **unused):
         ...
         self.t_embedder = TimestepEmbedder(hidden_size)
         self.r_embedder = TimestepEmbedder(hidden_size)
         self.w_embedder = TimestepEmbedder(hidden_size)
+
+        # Visual conditioning (adaLN injection)
+        self.use_visual = cond_dim > 0
+        if self.use_visual:
+            self.vis_embedder = nn.Sequential(
+                nn.Linear(cond_dim, hidden_size),
+                nn.SiLU(),
+                nn.Linear(hidden_size, hidden_size),
+            )
         ...

     def forward(self, x, cond, time, ..., return_v=False):
         ...
         c = self.t_embedder(t_abs) + self.r_embedder(r_abs) + self.w_embedder(w)
+
+        # Inject visual conditioning into adaLN
+        if self.use_visual and cond is not None and isinstance(cond, torch.Tensor):
+            if cond.ndim == 3:
+                cond = cond.mean(dim=1)  # pool temporal
+            c = c + self.vis_embedder(cond)
+
         for block in self.blocks:
             x = block(x, c)
         ...
```

#### In `mf_trajectory_model.py`:

```diff
 if if_vision:
-    if imf_backbone not in ('unet',):
-        raise ValueError(...)
-    from .visual_unet_twotime import VisualUNetTwoTime
-    self.velocity_net = VisualUNetTwoTime(...)
+    if imf_backbone == 'unet':
+        from .visual_unet_twotime import VisualUNetTwoTime
+        self.velocity_net = VisualUNetTwoTime(...)
+    elif imf_backbone == 'mf_dit':
+        self.velocity_net = MFDiTOfficialTrajectory(
+            ..., cond_dim=128,   # visual latent dim
+        )
+    else:
+        raise ValueError(...)
```

#### A new wrapper: `VisualDiTTwoTime`

A thin wrapper (analogous to `VisualUNetTwoTime`) that:
1. Owns the `MultiImageObsEncoder` (ResNet vision encoder).
2. Implements `encode_visual()` and `resolve_visual_cond()`.
3. Delegates to the DiT backbone, passing the visual latent as `cond`.

This is ~120 lines, mostly copied from `VisualUNetTwoTime` with the backbone swap.

### 2.6 Effort Estimates

| Strategy | Lines of Code | Files Changed | Effort | Risk |
|:---------|:-------------|:-------------|:-------|:-----|
| **A: adaLN injection** (MFDiT, SiT) | ~100–150 per backbone | 3–4 files per arm | **1–2 days** | Low |
| **B: Prefix tokens** (iMF DiT) | ~80–120 per backbone | 3–4 files per arm | **1–2 days** | Low–Med (RoPE resize) |
| **A+B: All four backbones** | ~400–600 total | ~8–10 files | **3–5 days** | Medium |
| **C: Cross-attention** | ~300+ per backbone | Heavy refactor | **1–2 weeks** | High |

### 2.7 What would need validation

1. **Training convergence**: Does the DiT visual model learn as well as the VisualUNet on the aligning/avoiding tasks?
2. **JVP correctness**: Verify the gradient through the visual-conditioned DiT matches expected behavior (can be tested with a small integration test).
3. **Parameter count**: The DiT backbones are sized differently from the UNet. Need to ensure comparable parameter counts for fair comparison.
4. **Checkpoint compatibility**: New `cond_dim` parameter means new checkpoints; existing state-only DiT checkpoints would NOT be loadable.

---

## Part 3 — Summary Table

| Question | Answer |
|:---------|:-------|
| Any visual model using DiT/SiT/Transformer? | **No.** All visual = VisualUNet (1D Conv U-Net). |
| D3IL's DDPM-ACT (Transformer) is your model? | **No.** It's a vendored external benchmark, Gen5 tried + abandoned it. |
| `aux_repo` has visual models? | **No.** All repos there are state-only. |
| Is it hard to add visual to DiT/SiT? | **Not very** for Strategy A (adaLN injection) or B (prefix tokens). ~1–2 days per backbone, ~3–5 days for all four. |
| Main blocker? | JVP safety (solved by pre-encode short-circuit) and the wrapper plumbing (new `VisualDiTTwoTime` class). |
| Recommended approach? | Strategy A (adaLN sum) for MFDiT/SiT, Strategy B (prefix tokens) for iMF-DiT. |
| Gen10 (Planned)? | Listed in MASTER_TEST_HISTORY as a planned VAE+Transformer upgrade — this investigation is directly relevant to that goal. |
| Is D3IL's vision encoder the same across all scenes? | **Yes.** Identical `MultiImageObsEncoder` + `get_resnet` config across all 11 vision agents and all 5 vision scenes. |
| Can we reuse the existing vision encoder with DiT/SiT? | **Yes.** The encoder outputs a flat `[B, 128]` latent. The generative backbone (UNet, DiT, SiT) only ever sees this vector, never raw images. Swapping the backbone does not touch the encoder. |

---

## Part 4 — D3IL's Universal Vision Encoder & Direct Reuse with DiT/SiT

### 4.1 D3IL uses ONE vision encoder — everywhere, every scene, every agent

Quick-verified across the entire vendored D3IL benchmark (`d3il/configs/agents/`):

**All 11 vision agent configs** use byte-identical vision encoder settings:

| Setting | Value | Identical Across All? |
|:--------|:------|:---------------------:|
| `_target_` | `agents.models.vision.multi_image_obs_encoder.MultiImageObsEncoder` | ✅ |
| `rgb_model._target_` | `agents.models.vision.model_getter.get_resnet` | ✅ |
| `input_shape` | `[3, 96, 96]` | ✅ |
| `output_size` | `64` | ✅ |
| `use_group_norm` | `True` (replaces BatchNorm2d → GroupNorm) | ✅ |
| `share_rgb_model` | `False` (independent ResNet per camera) | ✅ |
| `imagenet_norm` | `True` (ImageNet mean/std normalisation) | ✅ |
| Camera keys | `agentview_image` + `in_hand_image` (both `[3, 96, 96]` RGB) | ✅ |

The agents verified:

| # | Agent Config | ML Backbone | Same Vision? |
|:--|:------------|:------------|:------------:|
| 1 | `act_vision_agent.yaml` | ACT (VAE + Transformer Enc/Dec) | ✅ |
| 2 | `bc_vision_agent.yaml` | Behavioral Cloning (MLP) | ✅ |
| 3 | `beso_vision_agent.yaml` | BESO (Score-based) | ✅ |
| 4 | `bet_mlp_vision_agent.yaml` | BeT-MLP (Behavior Transformer) | ✅ |
| 5 | `bet_vision_agent.yaml` | BeT (Behavior Transformer) | ✅ |
| 6 | `cvae_vision_agent.yaml` | CVAE | ✅ |
| 7 | `ddpm_encdec_vision_agent.yaml` | DDPM Enc-Dec (Transformer) | ✅ |
| 8 | `ddpm_transformer_vision_agent.yaml` | DDPM + Transformer | ✅ |
| 9 | `ddpm_vision_agent.yaml` | DDPM (U-Net) | ✅ |
| 10 | `gpt_vision_agent.yaml` | GPT-BC (Autoregressive Transformer) | ✅ |
| 11 | `ibc_vision_agent.yaml` | IBC (Implicit BC) | ✅ |

**Also verified across all 5 D3IL vision scenes** (aligning, avoiding, sorting_4, sorting_6, stacking) — the scene configs override the dataset and environment, never the vision encoder.

**Conclusion: D3IL's design philosophy is that the vision encoder is ONE universal module** — `MultiImageObsEncoder(2× ResNet-18 → 64D each → concat → 128D)` — shared across every possible downstream ML backbone (MLP, Transformer, DDPM U-Net, BESO, BeT, CVAE, GPT, IBC, ACT). The backbone never sees images. It only sees the flat 128D latent.

### 4.2 How FM-PCC reuses D3IL's vision encoder today

FM-PCC took this exact module and wired it into the VisualUNet via two FiLM modes:

```
                    ┌─────────────────────────────┐
                    │  MultiImageObsEncoder        │  ← REUSED FROM D3IL
                    │  2× ResNet-18 (GroupNorm)     │     (vendored, unchanged)
                    │  → 128D latent (aligning)     │
                    │  → 64D latent  (avoiding)     │
                    └──────────┬──────────────────┘
                               │
                    ┌──────────▼──────────────────┐
                    │  cond_mlp (128 → dim → dim)  │  ← FM-PCC's own projection
                    └──────────┬──────────────────┘
                               │
              ┌────────────────┼────────────────────┐
              │                │                    │
     ┌────────▼───────┐  ┌────▼──────────┐  ┌──────▼──────┐
     │ Fake FiLM (v1) │  │ True FiLM (v2)│  │ ??? (DiT?)  │
     │ concat with t   │  │ γ scale + β   │  │  NOT YET    │
     │ → embed_dim     │  │ per-block     │  │             │
     └────────┬───────┘  └────┬──────────┘  └──────┬──────┘
              │               │                    │
              ▼               ▼                    ▼
     ┌────────────────────────────────────────────────────┐
     │         Generative Backbone (denoises)              │
     │  UNet1DTemporalCondModel  (v1)                      │
     │  UNet1DTemporalFiLMModel  (v2)                      │
     │  ??? DiT / SiT / iMF-DiT  (future)                 │
     └────────────────────────────────────────────────────┘
```

The critical observation: **the 128D latent is the clean interface boundary**. Everything above it (ResNet encoder, image preprocessing, window pooling) is backbone-agnostic. Everything below it (how the backbone consumes the 128D) is encoder-agnostic.

### 4.3 Aligning vs. Avoiding: same encoder, different camera count

| Scene | Cameras | Latent Dim | Wrapper |
|:------|:--------|:-----------|:--------|
| **Aligning** (Gen6v4, Gen7, Gen14) | 2 cams: `agentview_image` + `in_hand_image` | 128D (64+64 concat) | `VisualUNet` / `VisualUNetTwoTime` |
| **Avoiding** (Gen9) | 1 cam: `agentview_image` only | 64D (single ResNet) | `VisualUNet` (avoiding variant) |

The difference is just the `shape_meta` dict passed to `MultiImageObsEncoder` — one camera entry vs two. The encoder class itself is identical.

### 4.4 Why this means DiT/SiT reuse is trivial

Since D3IL itself already pairs **the exact same vision encoder** with 9 completely different ML backbones (MLP, Transformer, DDPM U-Net, BESO, CVAE, GPT, IBC, ACT, BeT), the vision encoder is **architecturally agnostic by design**. The 128D output is just a flat vector.

For DiT/SiT, you do NOT need:
- ❌ A new vision encoder
- ❌ Changes to `MultiImageObsEncoder`
- ❌ Changes to `get_resnet`
- ❌ Changes to the image preprocessing pipeline

You ONLY need:
- ✅ A **projection layer** inside the DiT/SiT to consume the 128D latent (Strategy A: adaLN sum, or Strategy B: prefix token)
- ✅ A **wrapper class** (`VisualDiTTwoTime`) that owns the `MultiImageObsEncoder` and passes the latent to the backbone — structurally identical to `VisualUNetTwoTime`, just swapping which backbone class it delegates to

### 4.5 Architecture comparison: what changes vs what stays

```
                              VisualUNet (today)          VisualDiT (proposed)
                              ────────────────────       ─────────────────────
  Vision Encoder              MultiImageObsEncoder       MultiImageObsEncoder
                              (UNCHANGED)                (UNCHANGED — same class,
                                                         same ResNet, same config)

  Latent Interface            128D flat vector            128D flat vector
                              (UNCHANGED)                (UNCHANGED)

  Pre-encode short-circuit    encode_visual() outside    encode_visual() outside
  (JVP safety)                JVP, capture as constant   JVP, capture as constant
                              (UNCHANGED)                (UNCHANGED — same pattern)

  Conditioning Injection      cond_mlp → concat with t   vis_embedder → sum into c
                              (Fake FiLM v1)             (adaLN injection)
                              OR                         OR
                              film_proj → γ,β per-block  vis_projector → prefix tok
                              (True FiLM v2)             (in-context conditioning)
                              ──── ONLY THIS CHANGES ──── ──── ONLY THIS CHANGES ────

  Generative Backbone         UNet1DTemporalCondModel    MFDiTOfficialTrajectory /
                              / Flow_matcher_U_Net_v2    AFSiTTrajectory /
                                                         MFDiTTrajectory / etc.
                              ──── DIFFERENT ──────────── ──── DIFFERENT ────────────
```

**Bottom line: the vision encoder is a commodity module. Reusing it with DiT/SiT requires zero changes to the encoder and ~100 lines of new conditioning plumbing in the backbone. This is the same level of effort as the Fake FiLM v1 → True FiLM v2 upgrade that was already done.**
