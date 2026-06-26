# UNet vs DiT for iMF — Is the UNet Backbone Suboptimal? (Principle Analysis)

**Date:** 2026-06-16
**Question:** The FM→iMF objective change is settled (see [iMF_vs_FM_Math_Principle](../U3/iMF_vs_FM_Math_Principle.md)).
But Gen3v4 keeps the **conv UNet** (`Flow_matcher_U_Net_v2`) while the official iMF uses a
**Transformer (imfDiT)**. Does that architectural choice *itself* hurt iMF in a way it never hurt
FM? Is the UNet **principally** suboptimal for the MeanFlow objective — and if so, why, and how much?
**Code anchors:** `unet1d_temporal_cond.py` (ours), `/workspaces/imeanflow/models/imfDiT.py` (official).

---

## 0. TL;DR

- **FM never stressed the architecture.** Its target `x₁−ε` is a constant, local, low-frequency
  field; an `h`-input it can ignore. *Any* reasonable backbone — conv UNet included — fits it.
- **iMF stresses exactly the two things a conv UNet is weakest at:** (i) **conditioning capacity**
  (it must represent a *family* of fields indexed by `h, ω, [τ_min,τ_max]`), and (ii) a **smooth,
  globally-coupled field whose own Jacobian `∂u/∂z` is the training target** (the JVP).
- **The headline DiT advantage — global receptive field via attention — mostly evaporates here**,
  because our horizon is `H=8` and the UNet bottleneck is already fully global (§4). This is the key
  nuance: the image-scale reasons DiT dominates do **not** transfer to an 8-step trajectory.
- **What *does* survive at trajectory scale, and is principled, not cosmetic:**
  1. **Conditioning injection** — UNet collapses `t, h, ω, τ_min, τ_max` into **one additive scalar
     bias**; DiT gives each its **own learnable tokens that interact through attention**. iMF needs
     this; FM did not. **This is the real, scale-independent gap.**
  2. **Head-branch depth** — our "dual head" is 2 conv layers off a shared trunk; DiT's is **8 full
     transformer blocks per head**. u (averaged) and v (instantaneous) must genuinely diverge.
  3. **Stiff-target stability machinery** — DiT's zero-init residual gates + RMSNorm are designed
     for self-referential objectives; the UNet has none.
- **Verdict:** the UNet is **mildly suboptimal in principle for iMF**, but the dominant cause is the
  **conditioning bottleneck**, not the conv/attention receptive field. It is **not a barrier**
  (JVP-safe, trains, works) — but if real-iMF quality plateaus, the conditioning pathway is the first
  thing to fix, and a DiT swap is justified mostly for that reason, not for "attention is better."

---

## 1. The two architectures, precisely

| Aspect | Our UNet (`Flow_matcher_U_Net_v2`) | Official `imfDiT` |
|---|---|---|
| Core op | 1D temporal **conv**, kernel=5, `dim_mults=(1,2,4,8)` | **RoPE self-attention** + SwiGLU MLP |
| Receptive field | **Local**; global only at bottleneck | **Global** from layer 1 |
| Norm | InstanceNorm (in ResBlocks) | RMSNorm |
| Conditioning inputs | `t, h, ω, τ_min, τ_max` | `h, ω, τ_min, τ_max` (**not `t`**) |
| Conditioning **injection** | **All summed into one `dim`-vector**, added as a bias inside every ResBlock (`t = time_mlp(t)+h_mlp(h)+omega_mlp(ω)+…`, `unet1d_temporal_cond.py:236-254`) | **Separate learnable tokens** per signal, **prepended** to the patch sequence; interact via attention (`imfDiT.py:308-351`) |
| Dual head | 2 conv layers off the shared trunk (`final_conv` / `v_final_conv`, `:179-191`) | **`aux_head_depth=8` transformer blocks per head** off a shared backbone (`:374-388`) |
| Residual stability | none special | **zero-init vector gates** `attn_scale=mlp_scale=0` (`:113-118`) |
| Conditions on `t`? | **Yes** (`time_mlp(t)`) | **No** — only `h=t−r` (`:370`) |

The last row is a real divergence: the official iMF deliberately conditions **only on `h`** (the paper's
recipe), while ours feeds both `t` and `h`. Ours is *richer*, not wrong — but it means our network must
learn to use `t` consistently, and the sampler must never freeze `t` (the Deviation-B lesson).

---

## 2. Why FM was indifferent to the backbone

Recall the FM target on the linear interpolant (proven in the U3 math doc):

```
v_target = x₁ − ε        — constant along the path, independent of (r, t, h)
```

Three consequences for architecture choice:

1. **No conditioning capacity needed.** The Bayes-optimal output does not depend on `h, ω, τ_*`. So
   the UNet's weak additive-scalar conditioning is *sufficient* — there is nothing to condition on.
   The `h`-input is provably inert; the network just learns to ignore it.
2. **No global field-coupling needed for the target.** The regression label is computed **per sample**
   from `(ε, x₁)` alone. The network never has to represent how its own output varies across the
   trajectory; it just regresses a fixed vector. A local conv field suffices.
3. **No self-reference → no stability machinery needed.** Plain MSE to a fixed label is a benign
   optimization landscape. InstanceNorm + Mish is plenty.

**This is why diffusion→FM was a free swap and FM→{any backbone} was a free swap.** FM exercises none
of the axes on which UNet and DiT differ. Architecture was a non-question.

---

## 3. Why iMF changes the question — the three principled stresses

iMF's target (U4/U5) is structurally different:

```
u_target = v_inst + h · du/dr,     du/dr = ∂_z u·v + ∂_r u − ∂_h u     (a JVP of the network)
```

This stresses exactly the UNet's weak points.

### 3.1 Stress A — conditioning capacity (the real, scale-independent gap)

iMF asks the network to represent a **family of fields** `u(z, r, t)` parameterised by the interval,
and at inference a **further family** indexed by `(ω, τ_min, τ_max)`. These are not nuisance inputs —
they *change the correct output*: `u` over `[0,1]` (1-NFE, `h=1`) must differ from `u` over `[0.4,0.5]`.

- **UNet injection (`:236-254`):** `t = time_mlp(t) + h_mlp(h) + omega_mlp(ω) + tmin_mlp + tmax_mlp`.
  Five distinct conditioning signals are **summed into a single `dim`-dimensional vector**, then added
  as the *same bias to every time-position* inside each ResBlock. This is the **weakest** conditioning
  primitive: a scalar-driven global shift. It cannot express "attend differently depending on `h`"; it
  can only translate activations. Worse, summation means the network must **linearly disentangle** five
  signals that arrive pre-mixed — an information bottleneck precisely where iMF demands the most
  conditional expressivity.
- **DiT injection (`:333-344`):** each of `h, ω, τ_min, τ_max` gets its **own set of learnable tokens**
  (`time_tokens + h_embed`, `omega_tokens + omega_embed`, …), prepended to the content tokens. Through
  attention, **every patch can read each conditioning signal independently and content-dependently**.
  The conditioning is *routable*, not a uniform bias.

**This gap is independent of sequence length.** It is the single most defensible reason the UNet is
suboptimal for iMF, and it is invisible to FM (which has nothing to condition on).

### 3.2 Stress B — the JVP needs a globally-coupled, smooth field

`du/dr` includes `∂_z u · v`: the directional derivative of the output w.r.t. the **whole latent
trajectory** along `v`. For the MeanFlow target to carry true *marginal* curvature, the Jacobian
`∂u/∂z` must couple distant trajectory positions — the field at step `k` should respond to perturbations
at step `j≠k`.

- A **conv** UNet has a **band-limited Jacobian** at fine resolution: `∂u_k/∂z_j ≈ 0` for `|k−j|`
  beyond the receptive field, until the bottleneck mixes globally. So the curvature it can represent is
  spatially local except through the coarse bottleneck path.
- **Attention** gives a **dense Jacobian** at every layer → a richer, smoother `du/dr` target.

FM never differentiated the network, so this never mattered. iMF differentiates `u` *as its objective*,
so the geometry of `∂u/∂z` directly shapes the learning signal. **However**, see §4 — at `H=8` the
"band limit" is small relative to the sequence, so this stress is **partially** mitigated.

### 3.3 Stress C — head specialization depth

u (averaged over `[r,t]`) and v (instantaneous) are **different fields** that share low-level features
but must diverge at the output. The official split gives each **8 transformer blocks** of private depth
(`aux_head_depth=8`) after the shared backbone. Our U5 `dual_head` gives each **2 conv layers**
(`final_conv` / `v_final_conv`). With so little private capacity, the shared trunk is forced to encode a
compromise representation; the v-head's regularizing pull on the trunk (the whole point of U5 1b) is
real but shallow. FM didn't care — its u and v targets were the *same vector* (§2), so zero head
specialization was needed.

### 3.4 Stress D — stiff self-referential optimization

The JVP target is bootstrapped (it contains `u_θ`'s own derivative). DiT's **zero-init residual gates**
(`attn_scale=mlp_scale=0` at init, `:113-118`) make every block start as identity, so a deep network
trains stably under a stiff target — the network eases into the self-referential regime. The UNet has no
analogue; it relies on the adaptive loss weight (U4) and the `r=t` anchor alone for stability. FM, being
plain MSE, needed none of this.

---

## 4. The crucial counter-nuance: H=8 collapses the receptive-field argument

The reasons DiT **dominates on images** are mostly Stress B (global coupling) and the scale of the
sequence (256×256 → thousands of tokens, where conv locality is a severe handicap). **Our trajectories
are `H=8`.** With `dim_mults=(1,2,4,8)` the UNet downsamples three times:

```
H = 8 → 4 → 2 → 1     (bottleneck horizon = 1)
```

At the bottleneck the UNet's receptive field is the **entire trajectory** — it is already globally
coupled there. So:

- **Stress B is heavily mitigated.** Unlike images, an 8-step trajectory is fully mixed by the
  bottleneck; `∂u/∂z` is *not* meaningfully band-limited across `H=8`. The conv-vs-attention
  receptive-field gap — DiT's biggest image-scale win — is **largely irrelevant here**.
- The trajectory-diffusion literature (Diffuser, Decision-Diffuser) corroborates: conv U-Nets are
  competitive backbones for short-horizon planning; DiT's edge there is modest, not categorical.

**What does NOT get mitigated by short horizons:** Stress A (conditioning injection) and Stress C (head
depth) are about *capacity and routing of conditioning*, not sequence length. A bias-summed scalar is a
bottleneck whether `H=8` or `H=8000`. **So the principled residual disadvantage of our UNet for iMF is
concentrated in conditioning, not in the conv operator itself.**

---

## 5. Putting it together — is the UNet suboptimal for iMF?

| Stress axis | Hurts iMF? | Survives at H=8? | UNet exposure |
|---|---|---|---|
| A. Conditioning injection (scalar-bias vs tokens) | **Yes — core to iMF** | **Yes (scale-free)** | **High** — 5 signals summed to one bias |
| B. Global field coupling for the JVP | Yes (image-scale) | **Mostly no** — bottleneck is global at H=8 | Low here |
| C. Head specialization depth | Yes | Yes | Medium — 2 conv layers vs 8 blocks |
| D. Stiff-target stability gates | Yes | Yes | Medium — mitigated by adaptive weight + anchor |

**Conclusion:**
- **In principle, yes, the conv UNet is suboptimal for iMF** — but for a *specific* reason: it
  under-conditions (Stress A) and under-specializes the heads (Stress C). It is **not** principally
  suboptimal because "conv < attention" — that argument (Stress B) is the one that fails at `H=8`.
- **It is not a barrier.** The UNet is JVP-safe (InstanceNorm → no batch coupling), already conditions
  on both `t` and `h`, and U5 added the shared v-head and interval-CFG embeds. It will train and run the
  *real* iMF objective. The U3/U4 "UNet is not a barrier" claim stands for *feasibility*.
- **But "not a barrier" ≠ "optimal."** If real-iMF quality at low NFE plateaus below expectation, the
  most likely architectural culprit is the **additive-scalar conditioning bottleneck**, then **head
  depth** — *not* the conv receptive field.

---

## 6. Practical implications & the migration order

If/when the UNet's iMF quality disappoints, fix in this order (cheapest, most-principled first):

1. **Strengthen conditioning injection on the UNet (cheap, high-value).** Replace the additive scalar
   bias with **FiLM** (per-channel scale+shift from the conditioning embedding) inside each ResBlock,
   so `h, ω, τ_*` *modulate* features instead of merely shifting them. This closes most of Stress A
   without leaving the UNet. (DiT-style adaLN-Zero is the image analogue.)
2. **Deepen the head split.** Give `u`/`v` a few private ResBlocks (not just one conv) so they can
   specialize — partial Stress C fix, still on the UNet.
3. **Only then** consider the **`IMFBackbone` DiT drop-in** (the U5 placeholder, `imf_trajectory_model.py:11`).
   The contract `forward(x,t,h,cond,ω,t_min,t_max)->(u,v)` is already the swap point; the objective,
   JVP, and sampler need no change. A DiT buys you tokenized conditioning (Stress A), deep dual heads
   (Stress C), and zero-init stability (Stress D) **natively** — but its receptive-field win (Stress B)
   is *not* the reason to do it at `H=8`.

> The honest framing: **the FM→iMF objective change is what turned "any backbone is fine" into "the
> conditioning pathway matters."** The UNet's conv-ness is largely innocent at trajectory scale; its
> *conditioning bottleneck* is the principled weakness iMF exposes. Fix conditioning first; swap to DiT
> only if you also want the deep dual heads and stability gates for free.

---

## 7. Caveats

- Image-scale claims about DiT>UNet (e.g. the iMF paper's NFE-1 FID) are at `256²` resolution with
  thousands of tokens; they do **not** transfer wholesale to `H=8` trajectories. This doc deliberately
  separates the scale-free stresses (A, C, D) from the scale-dependent one (B).
- The `t`-conditioning divergence (ours feeds `t`, official only `h`) is orthogonal to UNet-vs-DiT; it
  is a recipe choice and is covered by the Deviation-B "never freeze t" guardrail in `p_sample_loop`.
- This is a principled architecture analysis, not a benchmark. Confirm against actual low-NFE eval once
  the U5 all-power run completes (Phase 2).
