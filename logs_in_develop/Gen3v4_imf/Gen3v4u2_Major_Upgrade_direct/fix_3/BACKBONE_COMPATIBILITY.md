# iMF Backbone Compatibility — Is our U-Net the problem?

**Date**: 2026-06-02
**Parent**: extends [`CHANGELOG.md`](CHANGELOG.md) (fix_3 architectural-deviations record) and [`wandb_analysis/TRAIN_LOG_ANALYSIS.md`](wandb_analysis/TRAIN_LOG_ANALYSIS.md) (training-side root cause of post-fix_3 jitter).
**Triggering questions** (user, verbatim):
> "is our archtec. especially the Unet is not compaiitble as originla imf repo?"
> "what is htere ML bone? not Unet?"
> "or is not Unet Porblem, ie the imf indeed principlly work with Unet (but weak/better/irrelevant)?"

**Sources (vendored, direct file reads):**
- Reference: `/workspaces/imeanflow/models/imfDiT.py`, `embedder.py`, `imf.py`, `README.md` — the official iMF PyTorch re-implementation.
- Ours: `/workspaces/FM-PCC/flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py`, `imf_trajectory_model.py`, `imf_engine.py`, `imf_diffusion.py`.

---

## 1. Headline answer (read this first)

**No, U-Net is not "incompatible" with iMF.** iMF is a *training objective + sampling procedure* (mean-velocity matching, dual u/v heads, h-conditioning, one-step inference), **not a specific neural-net architecture**. The math is backbone-agnostic.

**However**, the reference iMF and ours differ in backbone *family*, *scale*, *conditioning style*, and *aux-head depth*. Of those, only **aux-head depth and overall training stability** plausibly explain the post-fix_3 jitter — and aux-head depth is already neutralised at inference by Deviation A. So the backbone is **probably not the dominant problem**; training-side instability (TRAIN_LOG_ANALYSIS.md's E4 spike) is more likely.

**Short verdict per sub-question:**

| Sub-question | Answer |
|---|---|
| Is our U-Net "incompatible" as reference? | No. iMF math doesn't require any specific backbone. |
| What backbone does reference iMF use? | A custom DiT-style **Transformer** (`imfDiT`), 12–48 blocks, hidden 768–1024, RoPE + RMSNorm + SwiGLU. Patches a 32×32 VAE latent of ImageNet images. |
| Is iMF "principally a U-Net thing"? | No — historically U-Net was used in DDPM/EDM. iMF's flagship implementation is **Transformer (DiT)**, not U-Net. But both work in principle. |
| Is U-Net the cause of our jitter? | Probably not the *primary* cause. The training log shows a stability spike at E4 (numerical, not architectural). U-Net is a legitimate choice at our scale; sample-quality bottleneck is u-head training residual, not backbone capacity. |

---

## 2. Side-by-side architectural comparison

| Property | Reference `imfDiT` (`/workspaces/imeanflow/`) | Our `Flow_matcher_U_Net_v2` (`flow_matcher_v3_imeanflow/`) |
|---|---|---|
| **Domain** | ImageNet 256×256 class-conditional generation | 1-D trajectory prediction (D3IL aligning) |
| **Input shape** | `[B, 4, 32, 32]` (VAE latent) | `[B, transition_dim, horizon]` e.g. `[B, 23, 8]` |
| **Tokenisation** | PatchEmbed (32×32, patch_size=2 → 256 tokens) | None — Conv1d operates on horizon dim directly |
| **Backbone family** | **Transformer (DiT-style)** | **1-D U-Net** (Conv1d + downsample/upsample) |
| **Depth (B variant)** | 12 transformer blocks | 4 down + 1 mid + 4 up residual temporal blocks |
| **Depth (XL variant)** | 48 transformer blocks | n/a — we don't scale beyond one size |
| **Hidden width** | 768 (B/M), 1024 (L/XL) | `freq_dim=256` with `dim_mults=(1,2,4,8)` → channels `{256, 512, 1024, 2048}` peak |
| **Approx. params** | iMF-B/2 ≈ 130M, iMF-XL/2 ≈ 675M | ~few million (small) |
| **Attention** | Multi-head self-attention with **RoPE** + QK-norm (RMSNorm) | LinearAttention helper exists in file but isn't wired into the main forward path; effectively Conv1d-only |
| **Normalisation** | **RMSNorm** (modern Transformer norm) | **InstanceNorm2d** + Mish activation (older U-Net norm) |
| **MLP** | **SwiGLU** with mlp_ratio = 8/3 | Linear → Mish → Linear (standard MLP) |
| **Position encoding** | **RoPE** (rotary) on Q, K | None explicit (Conv1d has implicit translation equivariance) |
| **Conditioning style** | **Token-level**: class label, t (`h_embedder`), omega, cfg-start, cfg-end each become learnable conditioning tokens *prepended to the patch sequence* | **Additive FiLM-like**: `time_mlp(t) + h_mlp(h)` added into each ResidualTemporalBlock's feature map |
| **Conditioning capacity** | Each conditioning variable can attend to every patch token bidirectionally | Each conditioning variable is added as a per-channel offset |
| **u/v dual-head structure** | **Shared backbone of `(depth − aux_head_depth)` layers**, then **two parallel head branches each `aux_head_depth=8` transformer blocks deep**, then `FinalLayer` for each → u, v | **Shared backbone (the U-Net) produces u; aux is a tiny 2-layer MLP applied to the input `x` directly** (NOT to a backbone feature). Aux's last layer is zero-initialised. |
| **Eval-mode optimisation** | `v_heads = ModuleList([] if eval_mode else [TransformerBlock] * head_depth)` — v-branch literally **not constructed** at eval | Aux head always exists; fix_3 Deviation A simply doesn't read its output at sampling |
| **Output stage** | `FinalLayer` projects each token to `patch_size × patch_size × out_channels`, then `unpatchify` reconstructs the image | `Conv1dBlock(dim, dim) → Conv1d(dim, transition_dim, 1)` directly outputs per-step velocity |
| **Training data scale** | ImageNet (1.28M images) | D3IL aligning task (~hundreds of trajectories × 256 horizon) |
| **NFE goal** | 1 (with FID ~3.32 on B/2, ~1.72 on XL/2) | 10 by default (`flow_steps_v3=10`), 1 is the iMF promise but our u-head residual makes it currently rough |

---

## 3. What "compatible" actually means here

The user's word "compatible" can mean three different things; only one is critical:

1. **API/interface compatibility** — does our backbone receive the same arguments and return the same shape the iMF training/sampling code expects?
   - **Yes.** Both backbones implement `forward(x, t, h, cond) → velocity`. Our `iMFTrajectoryModel.forward()` returns `(velocity, aux)`. The reference's `imfDiT` `_forward` also returns `(u, v)`. Same contract.

2. **Mathematical compatibility** — does the iMF objective (mean-velocity target `(x_t − x_r) / h`, dual u/v heads, h-conditioning, forward-Euler 0→1 sampling) *require* a Transformer?
   - **No.** The iMF paper (Kaiming He et al., arXiv:2502.13129) derives the loss in terms of an abstract `f_θ(x, t, h)`. It does not specify the parameterisation. Any model that can take `(x, t, h)` and emit a velocity field works.
   - The **predecessor MeanFlow paper** (the "MF" before "iMF") used a CNN-based velocity network, not a Transformer. iMF's choice of DiT is for ImageNet generation; it's not theoretical.

3. **Empirical-quality compatibility** — given equal training compute, does U-Net match Transformer iMF quality?
   - **Probably not at image-generation scale**, where attention's global context is critical and Transformers benefit from scale. The DiT family has dominated image generation since 2022.
   - **Probably yes at trajectory-prediction scale**, where horizon is short (8 steps), the relevant dependencies are local (your action depends on neighbouring steps), and Conv1d's translation equivariance matches the temporal structure. This is also why the entire **D3IL benchmark** uses Conv1d/U-Net backbones in `Diffuser`, `Diffusion-Policy`, `Beso`, etc. The diffusion-trajectory-policy literature is U-Net-dominated.

So when the user asks "is our U-Net incompatible," the only "no" that matters is empirical — and at our task scale the empirical answer is "U-Net is the field-standard choice; it's not the bottleneck."

---

## 4. What the reference iMF actually does (the most useful single fact)

From `imfDiT.py:149-230` + `embedder.py`:

```python
class imfDiT(nn.Module):
    def __init__(self,
        input_size=32, patch_size=2, in_channels=4,    # ← VAE latent of 32×32×4
        hidden_size=1152, depth=28, num_heads=16,
        mlp_ratio=8/3, num_classes=1000,
        aux_head_depth=8,                              # ← 8 transformer blocks per head
        num_class_tokens=8, num_time_tokens=4,
        num_cfg_tokens=4, num_interval_tokens=2,
        ...
    ):
        self.x_embedder = PatchEmbedder(...)
        self.h_embedder = TimestepEmbedder(...)         # h, NOT t
        self.omega_embedder = TimestepEmbedder(...)
        self.cfg_t_start_embedder = TimestepEmbedder(...)
        self.cfg_t_end_embedder = TimestepEmbedder(...)
        self.y_embedder = LabelEmbedder(num_classes, ...)
```

This tells you four things at once:

1. **The reference iMF is built for class-conditional ImageNet generation**, not for trajectory prediction. The class-label embedding (`LabelEmbedder` with 1000 classes) is a load-bearing piece of the conditioning. We have nothing analogous in our trajectory pipeline.

2. **There is no `t` embedder.** Look at the embedders: `h_embedder` exists but `t_embedder` does not. The reference iMF conditions *only on h*, never on t. This is exactly what `fix_3/CHANGELOG.md` calls out as **Deviation B** — our training conditions on (t, h) but reference conditions on (h, ω, …). Our fix_3 freezes t=0.5 at inference to approximate the reference's h-only condition, but our *trained weights* still embed t.

3. **The auxiliary (v) head is structurally substantial.** `aux_head_depth=8` Transformer blocks. Ours is a 2-layer MLP. The reference paper's reported one-step FID gains are partly attributable to a richly-parameterised v-branch — at inference, v isn't used (per Deviation A), but at *training* the v-head's gradient flow back through the shared backbone shapes the learned u-head representation. Truncating the v-head changes what the u-head learns.

4. **Conditioning is token-prepended, not added.** The reference builds a sequence `[class_tokens, h_tokens, omega_tokens, cfg_start_tokens, cfg_end_tokens, ...patch_tokens]` and feeds this into the Transformer. Each patch token can attend to every conditioning token at every layer. Our U-Net adds `time_mlp(t) + h_mlp(h)` as a channel-wise bias inside each residual block — a much narrower information channel.

---

## 5. What our backbone actually does

From `unet1d_temporal_cond.py:87-180`:

```python
class Flow_matcher_U_Net_v2(ModelMixin, ConfigMixin):
    def __init__(self, horizon, transition_dim, cond_dim,
                 dim=128, dim_mults=(1, 2, 4, 8), ...):
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(dim), Linear→Mish→Linear)
        self.h_mlp    = nn.Sequential(SinusoidalPosEmb(dim), Linear→Mish→Linear)
        # downs: ResidualTemporalBlock × 2 + Downsample1d (×4 levels)
        # mid:   ResidualTemporalBlock × 2
        # ups:   ResidualTemporalBlock × 2 + Upsample1d (×4 levels)
        self.final_conv = Sequential(Conv1dBlock(dim,dim), Conv1d(dim, transition_dim, 1))
```

`ResidualTemporalBlock(inp, out, embed_dim, horizon, kernel_size=5)`:
- `Conv1dBlock(inp, out, k=5) → out + time_mlp(t) → Conv1dBlock(out, out, k=5) → + residual`

This is the **Janner Diffuser**-family temporal U-Net (used by D3IL `Diffuser`, `Diffusion-Policy`, `Beso`, etc., since 2022). It is the *standard* trajectory-prediction backbone in this field, not an unusual choice.

`iMFTrajectoryModel.forward()` (`imf_trajectory_model.py:55-70`):
```python
def forward(self, x, t, h=None, cond=None, force_dropout=False):
    velocity = self.velocity_net(x, cond, t, h=h, ...)   # U-Net produces u
    aux = self.aux_head(x)                                # 2-layer MLP on raw input — NOT on backbone features
    return velocity, aux
```

**Critical detail to underline**: our aux head consumes the *input* `x`, not the U-Net's intermediate features. The reference iMF's v-branch consumes shared-backbone features. So ours has **even less coupling** between u and v than the reference — the v-head can't usefully shape the u-head's representation through the shared backbone, because there is no shared-backbone path for v.

In practical effect:
- Reference u and v share `depth − aux_head_depth` layers, then split.
- Ours: u uses the full U-Net; v skips the U-Net entirely and reads `x` directly with a 2-layer MLP.

This is a structural deviation we did not previously catalogue. It's not a bug — both designs are valid — but it means the reference's "v-head supervision shaping the shared trunk" property is **completely absent in our implementation**. That likely contributes to why our u-head trained to a worse plateau (TRAIN_LOG_ANALYSIS.md §4): without v's gradient flowing back through the trunk, the trunk has only the (mean-velocity) signal to learn from.

---

## 6. Mapping each architectural difference to its plausible inference-quality cost

| Difference | Direction of effect | Magnitude estimate |
|---|---|---|
| U-Net vs Transformer | Reduced global-context for the velocity field | **Small** at horizon=8 (locality dominates). |
| Param count (few M vs 130M+) | Reduced expressive capacity | **Small** for D3IL-scale problem (proven across the literature). |
| Additive FiLM-style conditioning vs token-prepended | Reduced conditioning bandwidth | **Small-Medium** — t and h are scalar; FiLM is adequate for scalars. |
| Aux head: 2-layer MLP on x vs 8-block branch on trunk | No v-gradient flowing back through trunk | **Medium** — this is the *most plausible architectural contributor* to a weak u-head. Even though Deviation A disables aux at inference, the training-time signal it provides is missing. |
| t-conditioning vs h-only | Reference uses h alone; we use both | **Medium** — addressed by Deviation B (freeze t=0.5) at inference; training-side mismatch remains. |
| RMSNorm vs InstanceNorm | Training stability | **Small-Medium** — possibly contributes to the E4 spike in TRAIN_LOG_ANALYSIS.md. |
| Position encoding (RoPE vs implicit Conv1d locality) | Different temporal-structure prior | **Small** at horizon=8. |
| SwiGLU vs MLP+Mish | Marginal expressive gain in MLP layer | **Negligible**. |
| Patch-tokenization vs direct Conv1d | Different input representation | **N/A** — the input modalities differ (images vs trajectories), this gap can't be unified. |

**Aggregate**: the architectural differences are real but, at this task scale, **none of them individually explains the symptom** the user reports ("trajectories not exploding but not smooth"). The aux-head-depth + training-stability combination is the strongest candidate; backbone-family (U-Net vs Transformer) is not.

---

## 7. Direct answers to the user's three sub-questions

### Q1 — "Is our architecture, especially the U-Net, not compatible with original iMF repo?"

It is **interface- and math-compatible**. It is **family-different** (U-Net vs Transformer) and **scale-different** (few M vs 100M+ params). Neither difference makes it "incompatible" — both implement the same `(x, t, h, cond) → (u, v)` contract that iMF requires.

What is *not* compatible is the **conditioning convention** (we condition on both t and h; reference conditions on h only). That's the structural mismatch fix_3 addresses with Deviation B. The "U-Net vs Transformer" axis is a *family choice*, not a *compatibility issue*.

### Q2 — "What is their ML backbone? Not U-Net?"

It is a **custom DiT-style Transformer** named `imfDiT`. From `imfDiT.py:149`:
- Patch-embed the 32×32 VAE latent into 256 tokens
- Prepend conditioning tokens (class label, h, omega, cfg start/end)
- Run through 12–48 transformer blocks (size variants B/M/L/XL)
- Branch into two 8-block heads (u and v)
- `FinalLayer` projects each head's output back to patches
- `unpatchify` reconstructs the image
- Modern Transformer ingredients: RoPE attention, RMSNorm, SwiGLU MLP, scaled-variance init

It is **not a U-Net**. The reference repo's pre-trained checkpoints (iMF-B/2 through iMF-XL/2 on HuggingFace) are all DiT.

### Q3 — "Is U-Net the problem, i.e. does iMF principally work with U-Net (but weaker/better/irrelevant)?"

**U-Net is not principally what iMF "works with."** Reference iMF works with DiT. But the iMF *objective* is backbone-agnostic, so U-Net is also a valid implementation choice.

**Whether U-Net is the cause of our jitter:** *probably not the primary cause*. Evidence:
- TRAIN_LOG_ANALYSIS.md shows a training-time stability spike (E4: `diffusion_loss` 0.39 → 1.57) that the U-Net never fully recovers from. This is an instability event, not an architecture-capacity event.
- D3IL-scale trajectory tasks are routinely solved by Conv1d/U-Net backbones at similar parameter counts (Janner Diffuser, Diffusion-Policy, etc.). The capacity is there.
- The structural deviation that *could* matter (aux head on raw input vs aux head on trunk features) affects what the u-head learns at *training* time but is decoupled from any specific backbone choice — you'd have the same issue if you wrapped a Transformer instead of a U-Net while keeping the aux-head wiring as-is.

**If you wanted to escalate beyond TRAIN_LOG_ANALYSIS.md's Option B (h-clamp + grad-clip + lower LR), the architecture-side escalation order would be:**

1. **Fix aux-head wiring first** (cheap structural change, no scale increase): make aux read from a shared U-Net feature, not from raw `x`. Restores the "v-gradient shapes the trunk" property the reference relies on.
2. **Add explicit position embeddings** to the U-Net inputs (sinusoidal along horizon dim). Cheap; helps the trunk distinguish trajectory positions.
3. **Replace InstanceNorm with RMSNorm or LayerNorm.** Cheap; possibly addresses the E4 stability spike.
4. **Only if all three above are exhausted: replace backbone with a small DiT** (8–12 blocks, hidden 256–512). Larger lift; may help marginally; not first-priority.

---

## 8. Quick sanity table — "if we were to copy reference iMF's design choices into our trajectory U-Net, which are easy to copy?"

| Reference design choice | Trivially portable to our U-Net? | Effort |
|---|---|---|
| h-only conditioning (drop t) | Yes (training-side change; fix_3 already does inference-side via Deviation B) | One-line in `imf_diffusion.py` p_losses |
| Aux head reads trunk features, not raw `x` | Yes — wire `aux_head` to take the U-Net's mid-block output | ~10 lines in `imf_trajectory_model.py` |
| Aux head = several blocks deep | Yes — replace 2-layer MLP with deeper module | ~20 lines |
| Token-prepended conditioning | No — only meaningful for Transformer (token sequence is the model's data structure) | N/A |
| RoPE | No — only meaningful with attention | N/A |
| SwiGLU MLP | Yes (drop-in replacement for Linear→Mish→Linear) | ~5 lines per block |
| RMSNorm | Yes (replace InstanceNorm2d) | ~3 lines |
| Patch-tokenization | No — trajectory has no spatial dim to patch | N/A |

The portable ones (h-only, aux wiring, deeper aux, SwiGLU, RMSNorm) are all *aux-head and conditioning* changes — not the backbone family. Reinforces the section-7 conclusion: the U-Net family choice is fine; the *details* around it are where reference iMF derives its quality.

---

## 9. Connection to the rest of fix_3 and the wandb analysis

- `CHANGELOG.md` (fix_3) addressed **two inference-time deviations** (A: drop aux contribution; B: freeze t to 0.5). Both were structurally correct fixes per the reference iMF's own code. **Neither retrains.**
- `wandb_analysis/TRAIN_LOG_ANALYSIS.md` traced **post-fix_3 jitter** to a **training-side spike** at E4 leading to a permanent u-head MSE plateau ~3× the briefly-achieved floor. Recommendation: retrain with h-clamp + grad-clip + lower LR.
- **This file** answers: **is the U-Net itself the bottleneck?** No — but the *aux-head wiring* (input vs trunk features) is a structural deviation we hadn't catalogued, and it plausibly contributes to the u-head's training residual. If we ever do retrain, fixing aux-head wiring is a cheap structural improvement that's worth bundling.

---

## 10. One-line summaries

- **Reference iMF's backbone is a custom DiT-style Transformer** (`imfDiT`), 12–48 blocks, hidden 768–1024, ~130–675M params, with RoPE / RMSNorm / SwiGLU and an 8-block aux head sharing a depth-20 trunk.
- **Our backbone is a 1-D temporal U-Net** (`Flow_matcher_U_Net_v2`) in the Janner Diffuser family, a few M params, with FiLM-style additive (t, h) conditioning and a 2-layer MLP aux head **that reads raw `x`, not trunk features**.
- **The iMF objective is backbone-agnostic** — U-Net is a legitimate trajectory-domain choice and is the field standard for D3IL-scale problems. The post-fix_3 jitter is not best explained by "U-Net is wrong"; it is best explained by training-time instability (TRAIN_LOG_ANALYSIS.md) compounded by a structural deviation in how our aux head is wired (this file §5). Backbone-family swap (U-Net → DiT) is a *low-priority* escalation; aux-head re-wiring + retraining with stability guardrails is *higher* priority.
