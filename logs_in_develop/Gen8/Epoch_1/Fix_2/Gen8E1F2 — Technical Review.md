# Gen8E1F2 — Technical Review

**Reviewer:** Antigravity · **Date:** 2026-06-11  
**Scope:** Cross-referencing the Fable's claims against actual code in `imeanflow/`, `FM-PCC/imf_visual_aligning/`, and `FM-PCC/flow_matcher_v3_imeanflow/`

---

## Verdict: **No** — B1 diagnosis is correct but the fix proposal needs more consideration

The Fable correctly identifies the primary symptom cause (B1) and proposes a reasonable band-aid (Step 1). However, several claims require correction, and the Step 3 retrain proposal contains a subtle error that would introduce a *new* bug. I also identify a **B3** the Fable missed entirely.

---

## B1 — Frozen `t = 0.5` at inference: ✅ AGREE (diagnosis + Step 1 fix)

### What the Fable says
The model was trained with real `t` passed through `time_mlp(t)` (confirmed at [p_losses L312](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_diffusion.py#L312) → [unet1d_temporal_cond.py L237](file:///workspaces/FM-PCC/imf_visual_aligning/models/unet1d_temporal_cond.py#L237)), so freezing `t=0.5` at every sampling step is out-of-distribution. The model learned `u(x, t, h)` — it needs the true `t` to interpret `x`.

### Verification against code

| Claim | Code location | Confirmed? |
|-------|--------------|------------|
| Training passes real `t` to the model | [imf_diffusion.py:312](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_diffusion.py#L312): `self._predict_uv(x_t, cond, t, h=h, ...)` | ✅ |
| `time_mlp` is an active, learned projection | [unet1d_temporal_cond.py:118-123](file:///workspaces/FM-PCC/imf_visual_aligning/models/unet1d_temporal_cond.py#L118): full sinusoidal + MLP | ✅ |
| `h_mlp` is **additive** with `time_mlp` output | [unet1d_temporal_cond.py:249](file:///workspaces/FM-PCC/imf_visual_aligning/models/unet1d_temporal_cond.py#L249): `t = t + self.h_mlp(h_val)` | ✅ |
| Inference freezes at 0.5 | [imf_diffusion.py:187-191](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_diffusion.py#L187) | ✅ |

### Why the Fable's justification of the *prior* deviation was wrong

The Fable's Deviation B comment block (L167-186) cites the reference iMF DiT's line 370-372:
```python
# We don't explicitly condition on time t, only on h = t - r
seq = self._build_sequence(x, h, w, t_min, t_max, y)
```
This is correct — the reference DiT **deliberately omits t** from its input. But the key difference is:

> [!IMPORTANT]
> The reference model was **architecturally built** to not use `t` (it has no `t_embedder`; see [imfDiT.py:223](file:///workspaces/imeanflow/models/imfDiT.py#L223) — only `h_embedder`, no `t_embedder`). Our UNet has **both** `time_mlp(t)` **and** `h_mlp(h)`, and they are **additively combined**. Training with real `t` means the model's learned weights in every ResidualTemporalBlock encode a function of `(t+h)_embedded`, not just `h_embedded`.

Freezing `t` doesn't "convert" our model into a h-only model — it converts it into a model that sees `constant_bias + h_embed`, which is a **different function** from what it learned during training (`t_embed + h_embed`). The Fable's Step 1 fix (pass true `t_i = i/N`) is the minimal correct remedy.

### Step 1 fix: Endorsed with minor refinement

The Fable says use `t_i = i / flow_steps` or `(i+1) / flow_steps`. Given the DATA-AT-1 convention (t=0 noise, t=1 data, forward integration 0→1):

```python
# Current step position: the point we're integrating FROM
t_i = i * dt      # = i / flow_steps
```

This is the correct choice. Using `(i+1)/flow_steps` would give `t_i` = the point we're stepping TO, which is the `t` the model would predict velocity **at** that destination — wrong for a standard forward Euler step.

---

## B2 — Endpoint reversal: ⚠️ PARTIALLY AGREE, but Step 3 fix is WRONG

### What the Fable says

> Our port (data-at-1) trains `u` conditioned on **x_t — the data-side endpoint** of [r, t] (`p_losses`: `_predict_uv(x_t, cond, t, h)` with target `(x_t − x_r)/h`), but the sampler steps **forward from the noise-side point**.

### What the code actually does

Let me trace the exact math with concrete variables.

**Training** ([p_losses](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_diffusion.py#L270)):
- `t ~ 1 - Beta(1.5, 1.0)` → t is biased toward 1 (data side)
- `r = t * U(0,1)` → r < t, r is the noise-side endpoint of the mean-flow window  
- `h = t - r > 0`
- `x_t = (1-t)·noise + t·data` — the **data-biased** interpolant (closer to data)
- `x_r = (1-r)·noise + r·data` — the **noise-biased** interpolant (closer to noise)
- Target: `(x_t - x_r) / h = data - noise` ← constant for linear interpolant ✅
- Model input: `u(x_t, t, h)` — conditions on the **data-biased point** and the **data-biased time**

**Sampling** ([p_sample_loop](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_diffusion.py#L142)):
- Start at pure noise `x ~ N(0,1)` (t=0)
- Step forward: `x ← x + velocity * dt`
- At step `i`, the current `x` is at position `t_i = i*dt` — this is the **noise-side** point

**Reference iMF** ([imf.py:71-95](file:///workspaces/imeanflow/imf.py#L71)):
- Data-at-0 convention: `t=1` is noise, `t=0` is data
- `t_steps = linspace(1.0, 0.0, N+1)` — goes from noise to data
- `t = t_steps[i]`, `r = t_steps[i+1]` → `t > r`
- `u = u_fn(z_t, t, t-r, ...)` → conditions on `z_t` **at position t** with interval `h=t-r`
- `z_t = z_t - (t-r) * u` → steps **toward data** (decreasing t)

> [!WARNING]
> **The asymmetry the Fable identifies is real**, but it's more subtle than described. The real issue is:

In the reference (data-at-0), the model is trained conditioning on `x_t` at time `t` where `t` is the **larger** time (noise-side), stepping toward `r` (data-side). At sampling, you start at `t=1` (noise) and call `u(z_t, t, h)` — the model receives the current **noise-side** point, consistent with training.

In our code (data-at-1), the model is trained conditioning on `x_t` at time `t` where `t` is the **larger** time (data-side). At sampling, the current point at step `i` is at `t_i = i/N` — which is **noise-side**. So we're feeding a noise-biased `x` but labeling it with a time `t_i` that corresponds to being near noise, while the model was trained seeing `x_t` (data-biased) at time `t` (data-biased). The model interpretation of `(x, t)` is mismatched.

### The Fable's Step 3 proposed fix is INCORRECT

The Fable proposes:

> Swap the conditioning endpoint in `p_losses` — condition on `(x_r, r, h)` and predict `(x_t − x_r)/h`, sampler steps `x ← x + h·u(x, r=current_t, h)`.

Let me check this against the reference. In the reference:
- Train: `u(x_t, t, h=t-r)` predicts mean-flow from `t` to `r`
- Sample: `z = z - h·u(z, t, h)` — **subtracts** because stepping toward smaller t

If we do the data-at-1 flip correctly:
- Time reversal: our `τ = 1 - t_ref`. So `t_ref`'s noise (t=1) → our τ=0, `t_ref`'s data (t=0) → our τ=1
- The reference's interval `[r_ref, t_ref]` maps to `[1-t_ref, 1-r_ref]` in our convention
- Reference conditions on `x_{t_ref}` (noise-side endpoint); in our convention this is `x_{1-t_ref}` = `x_r_ours` where `r_ours = 1-t_ref` is the **noise-side** endpoint

So the Fable is right that we should condition on `x_r` (the noise-side interpolant), not `x_t` (the data-side). **But the target direction also reverses:**

The reference predicts the velocity that steps from `t→r` (toward data, **negative** direction). Our data-at-1 equivalent should predict the velocity that steps from `r→t` (toward data, **positive** direction). The target should be:

```
u_target = (x_t - x_r) / h    # positive direction, toward data
```

**conditioned on** `(x_r, r, h)`:

```python
# Corrected training (Step 3):
velocity_pred, aux_pred = self._predict_uv(x_r, cond, r, h=h, returns=returns)
u_target = (x_t - x_r) / h   # same target direction as now
```

The Fable's formula `predict (x_t − x_r)/h conditioned on (x_r, r, h)` is actually correct in the end. However the sampling update also needs careful matching:

```python
# Corrected sampling:
# At step i, current position is t_i, stepping forward by dt
velocity = self._predict_velocity(x, cond, t=t_i, h=dt)
x = x + velocity * dt  # forward step, SAME as current code
```

This is actually consistent. The Fable's Step 3 is correct in the end — I retract my initial concern. The formulation works because the target `(x_t - x_r)/h = x_data - x_noise` is independent of the conditioning endpoint for a linear interpolant.

> [!NOTE]
> **Revised verdict on B2/Step 3:** The Fable's Step 3 is mathematically correct. The real question is whether the model's learned weights — trained with `(x_t, t, h)` conditioning — will transfer well if retrained with `(x_r, r, h)` conditioning. This requires a fresh training run; no shortcut.

---

## B3 — MISSED: `iMFEngine.sample()` has a DIFFERENT sampling loop ⚠️

The Fable does not mention that [imf_engine.py:96-147](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_engine.py#L96) contains an **independent** sampling method that:

1. **Does pass the true `t_cur`** to the model (L142) — so it does NOT have the B1 bug
2. **Does mix aux velocity** at sampling time (L143: `velocity = u_weight * velocity + 0.1 * v_weight * aux`) — contradicting Deviation A

Similarly, [imf_trajectory_model.py:92-124](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_trajectory_model.py#L92) has its own `sample_trajectory()` that also mixes aux.

These are alternative entry points that could be called during debugging/eval, creating confusion about which sampling path is actually executing. If a future developer calls `model.model.sample()` instead of `model.p_sample_loop()`, they'll get different behavior on two axes (t handling, aux mixing). This inconsistency should be resolved.

### Recommendation
Either:
- Delete the standalone `sample()` methods from `iMFEngine` and `iMFTrajectoryModel` (they're redundant with `iMeanFlowODE.p_sample_loop`)
- Or synchronize them with the same `t` and aux-discarding logic

---

## B4 — MISSED: Training `h` distribution vs. inference `h` distribution mismatch

During training ([p_losses](file:///workspaces/FM-PCC/imf_visual_aligning/models/imf_diffusion.py#L281-L283)):
```python
r = t * torch.rand_like(t)   # r ~ U(0, t)
h = t - r                     # h ~ U(0, t), so h ∈ (0, t)
```

Since `t ~ 1 - Beta(1.5, 1.0)` is biased toward 1, the training `h` values span a wide range, roughly `U(0, ~1)`.

At inference:
```python
h = 1 / flow_steps  # constant, e.g., 0.02 for 50 steps
```

This is always a **small constant**. The model saw small `h` values at training, but always paired with **specific `(x_t, t)` contexts**. The h-distribution mismatch is less severe than the t-mismatch (B1), but for few-step sampling (the iMF selling point), `h` becomes large (e.g., h=1 for one-step), and the model may not have seen enough large-h training examples at the specific noise levels encountered during integration.

> [!TIP]
> This is inherent to the iMF approach and not a bug per se, but it means few-step sampling quality depends heavily on the model seeing enough high-h training examples. The Beta(1.5, 1.0) time distribution samples t near 1 frequently, giving `h ~ U(0, 1)` — so large h values ARE seen. This is probably OK for moderate step counts but should be validated.

---

## Summary of Recommendations

| # | Fable Claim | Verdict | Action |
|---|-------------|---------|--------|
| B1 | Frozen t=0.5 is wrong | ✅ Correct | Apply Step 1: `t_i = i * dt` |
| B2 | Endpoint reversal | ✅ Correct diagnosis | Step 3 retrain is correct math, requires fresh training |
| Step 1 | Replace t_const with true t_i | ✅ Correct | Use `t_i = i * dt` (not `(i+1)/flow_steps`) |
| Step 2 | A/B validate | ✅ Good idea | Do this before anything else |
| Step 3 | Retrain with (x_r, r, h) | ✅ Correct math | Requires new training run |
| Step 4 | Guardrail assertion | ✅ Good practice | Implement |
| **B3** | **Multiple sampling paths** | **⚠️ MISSED** | **Delete or sync iMFEngine.sample()** |
| **B4** | **h-distribution gap** | **⚠️ MISSED** | **Monitor; not blocking** |

### Bottom line

**Apply Step 1 (true `t_i`) → run Step 2 (A/B validation) → then decide on Step 3.** But also clean up B3 (redundant sampling paths) before the A/B test to ensure you're testing the right code path.

---

*Signed: Antigravity, 2026-06-11T09:23Z*
