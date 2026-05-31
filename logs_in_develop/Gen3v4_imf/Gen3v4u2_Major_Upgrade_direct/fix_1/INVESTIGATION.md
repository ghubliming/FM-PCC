# Gen3v4u2 — fix_1 Investigation: Why iMF trajectories are chaotic straight lines

**Date**: 2026-05-31
**Symptom**: Eval rollout PNG at
`logs/avoiding-d3il/plans/flow_matching_v3_imeanflow(incorrect)/H8_Dflow_matcher_v3_imeanflow.models.iMFDiffusion_a1.5_b1.0_aw10/H8_K10_Meuler_Dflow_matcher_v3_imeanflow.models.iMFDiffusion/6/results/halfspace_both-hard/diffuser.png`
shows trajectories shooting off as **chaotic straight lines** from the
initial conditions — not following the avoiding-task's curved expert
distribution at all.

**User hypothesis**: "is the time reversed? ie ODE 0→1 mixed with 1→0?"

**Verdict**: **NOT a time-reversal bug.** Time direction is consistent
DATA-AT-1 everywhere. The bug is in the **training target formula** — it
contains a wrong variable that scales the regression target by a factor
that grows like ≈ N at the start of sampling. The model dutifully learns
this wrong scaling, sampling then takes ~N×-too-large Euler steps, and
trajectories shoot off in straight lines (the direction of the learned
mean velocity) regardless of where they should curve.

---

## 1. Verifying the time-direction hypothesis (it's NOT the cause)

Every relevant code site uses the same DATA-AT-1 convention (t=0 noise,
t=1 data):

| Site | File:line | Code | Direction |
|---|---|---|---|
| `q_sample` interpolant | `imf_diffusion.py:128` | `return (1.0 - t) * noise + t * x_start` | t=0 → noise, t=1 → data ✓ |
| `p_losses` time sampling | `imf_diffusion.py:236` | `t = 1.0 - Beta.sample(...)` | t ∈ [0,1] ✓ |
| `p_losses` r ≤ t | `imf_diffusion.py:251` | `r = t * torch.rand_like(t)` ⇒ `r ∈ [0,t]` | r ≤ t ✓ |
| `p_losses` h ≥ 0 | `imf_diffusion.py:252` | `h = t - r` | h = forward interval ✓ |
| `p_sample_loop` time grid | `imf_diffusion.py:158-163` | `t_cont = loop_idx / flow_steps`, i ∈ [0, N) | t grows 0 → 1 ✓ |
| `p_sample_loop` Euler step | `imf_diffusion.py:165` | `x = x + velocity * dt` | forward integration ✓ |
| `iMFTrajectoryModel.sample_trajectory` | `imf_trajectory_model.py:108` | `z_t = z_t + h * combined` (with `t_steps = linspace(0, 1, N+1)`) | forward 0 → 1 ✓ |

All seven sites agree: time runs noise → data, forward Euler integrates
0 → 1. **No reversal anywhere.** Rule the hypothesis out cleanly.

---

## 2. The actual bug — `u_target` has `x_start` where it must have `x_t`

`flow_matcher_v3_imeanflow/models/imf_diffusion.py:265-266`:

```python
# Mean flow target: (x_data - x_r) / h  — comment says "average velocity from x_r to x_data over interval h"
u_target = (x_start - x_r) / (h_expand + 1e-8)
```

This is **mathematically wrong** for a mean-flow target over the interval
`[r, t]`. Derivation below.

### 2.1 What mean-flow `u(x_t, t, h)` is supposed to be

iMeanFlow (Lim 2025 "Mean Flow Matching") defines:

```
u(x_t, t, h) := (1/h) · ∫_{t−h}^{t} v(x_τ, τ) dτ
```

where `v(x_τ, τ)` is the **instantaneous** velocity of the
probability-flow ODE at `(x_τ, τ)`. The mean-flow loss enforces this
identity directly (or its self-consistency relaxation
`u = v − h · ∂_t u`), so a *single* Euler step with `u` reproduces the
exact integral over `[t−h, t]`.

For the **linear interpolant** used here
`x_τ = (1−τ)·noise + τ·x_data`, the instantaneous velocity is
constant per sample:

```
v(x_τ, τ) = dx_τ/dτ = x_data − noise =: v_const
```

So the mean-flow target collapses to that same constant:

```
u_target = (1/h) · ∫_{t−h}^{t} v_const dτ = v_const = x_data − noise
```

Equivalently, in **bootstrap form**:

```
u_target = (x_t − x_{t−h}) / h
        = ((1−t)·noise + t·x_data − (1−t+h)·noise − (t−h)·x_data) / h
        = (h·x_data − h·noise) / h
        = x_data − noise = v_const                          ✓
```

So the correct formula in the code's notation is:

```python
u_target = (x_t - x_r) / h            # bootstrap form
# OR equivalently for linear paths:
u_target = x_start - x_base           # constant per sample
```

### 2.2 What the code actually computes

```python
u_target = (x_start − x_r) / h
        = (x_data − (1−r)·noise − r·x_data) / (t − r)
        = ((1−r)·(x_data − noise)) / (t − r)
        = ((1−r)/h) · v_const
```

That extra factor **`(1−r)/h`** is the bug. It blows up at small `h`,
which is exactly the sampling regime.

### 2.3 The scaling factor by numbers (DATA-AT-1, N = `flow_steps`)

At sampling time the loop queries the model at `t = i/N` with
`h = dt = 1/N`. In training, the closest analogue is `r = t − h`. So the
target the model was fit to, at those (t, h) values, has:

| Sampling step `i` | `t = i/N` | `(1−r)/h` factor | `u_target / v_const` |
|---|---|---|---|
| 0 (start)   | 0       | (1+1/N)/(1/N) = N+1 | **≈ N+1** |
| N/4         | 0.25    | (0.75+1/N)/(1/N)   | **≈ 0.75·N + 1** |
| N/2         | 0.50    | (0.50+1/N)/(1/N)   | **≈ 0.50·N + 1** |
| 3N/4        | 0.75    | (0.25+1/N)/(1/N)   | **≈ 0.25·N + 1** |
| N−1 (end)   | (N−1)/N | (1/N)/(1/N) = 1    | **≈ 1** |

For the eval run that produced the chaotic PNG (`K10_Meuler` ⇒ N = 10):
the model was trained to output ≈ 11×v at t=0, ≈ 6×v at t=0.5, ≈ 1×v at
t=0.9. The first Euler step alone advances by `u·dt ≈ 11·v·0.1 = 1.1·v`
— larger in magnitude than the whole noise→data displacement should be
across the entire integration.

### 2.4 Why the resulting trajectories look "chaotic straight lines"

- The over-scaled target dominates the model's regression early in the
  trajectory (high-magnitude target at small t).
- At sampling, the first one or two Euler steps shoot the latent by
  several times the expected total displacement.
- Once outside the data manifold, the model is far OOD; its remaining
  predictions are essentially arbitrary, but they're being added to an
  already-displaced latent — visually this reads as a **straight line**
  in the (already-projected-to-data-space) plot, because the dominant
  displacement happened in the first few steps and the directions are
  small noise around the (data − noise) axis.
- Direction is `≈ x_data − noise`, which for the avoiding task is some
  random-looking constant per episode, hence "chaotic": each rollout
  shoots off in a different direction, but each is approximately a
  straight line.

This matches the visual symptom precisely.

---

## 3. Cross-checks (other potential issues; not the primary culprit)

I looked at these as well — flagging in case Fix 1 doesn't fully resolve
the visual:

| Concern | Code site | Status |
|---|---|---|
| `flow_steps_v3` for sampling vs. training-time `h` distribution | `imf_diffusion.py:144`, `:154` | Sampling fixes `h = 1/N` for every step; training samples `h = t − r` with `r ~ U(0, t)`, so h_train ∈ [0, t]. Distribution shift exists but the model is h-conditioned (via `h_mlp` at `unet1d_temporal_cond.py:211`), so it can in principle generalise. **Not load-bearing for this bug.** |
| First sampling step at `t = 0` is OOD | `imf_diffusion.py:160` | At t=0 in training, h must equal 0 (since r ∈ [0, t=0]). Sampling queries t=0 with h=dt>0 — a slice the model never saw. Minor OOD; subsumed by the §2 bug. |
| `apply_conditioning` calls on noise vs. data | `imf_diffusion.py:248, 256, 267, 271, 275` | All consistent with DATA-AT-1; no anomaly. |
| Aux head bias | `imf_trajectory_model.py:52-53` | Zero-initialised; `aux_loss_weight` typically small. Not load-bearing. |
| Sign of velocity step | `imf_diffusion.py:165` | `x = x + velocity * dt` is correct for forward Euler 0→1 (target u points noise→data, dt > 0). ✓ |

None of these alone produces "chaotic straight lines"; the §2 bug does
directly produce that exact symptom.

---

## 4. Proposed Fix

Single-line change in `flow_matcher_v3_imeanflow/models/imf_diffusion.py`:

```python
# BEFORE (line 266, BUG):
u_target = (x_start - x_r) / (h_expand + 1e-8)

# AFTER (correct mean-flow bootstrap target for linear interpolants):
u_target = (x_t - x_r) / (h_expand + 1e-8)
```

Alternative equivalent forms (any is fine; pick whichever is most
readable):

```python
# Form A — bootstrap (most explicit about the interval [r, t]):
u_target = (x_t - x_r) / (h_expand + 1e-8)

# Form B — instantaneous velocity directly (works because the linear
# interpolant has constant velocity x_data − noise):
u_target = x_start - x_base    # no division by h needed

# Form C — for non-linear paths in the future (most general):
u_target = (q_sample(x_start, t, x_base) - q_sample(x_start, r, x_base)) \
           / (h_expand + 1e-8)
```

Recommend **Form B** for clarity and to avoid the `+1e-8` numerical
guard. Form A is equally correct.

---

## 5. Sanity-Check Plan After the Fix

Run the same eval config (`K10_Meuler`, `halfspace_both-hard`) and
expect:

1. `u_target` magnitude at sampling time ≈ `|v_const| = |x_data − noise|`
   ≈ 1 (since noise is unit-variance and data is normalised). The
   chaotic-scale ≈10× overshoot disappears.
2. `diffuser.png` shows curved trajectories that **look like the data
   distribution** — no longer straight lines.
3. Mean-flow loss curve during a fresh training run should be **lower
   and more stable** than the pre-fix run; the previous training was
   fitting a noisy `(1−r)/h` target with huge variance at small h.

If the visual still looks wrong after this fix, secondary suspects in
priority order:
- `apply_conditioning` interactions with the new target (re-check the
  `noise=True` flag on lines 248, 256, 267, 271, 275 — should be
  unchanged but worth confirming).
- `h` distribution shift between training and sampling (consider
  uniform-h training schedule, decoupled from t).
- First-step OOD at t=0 (consider starting sampling at t=ε or skipping
  the first step entirely).

---

## 6. Recommended Sanity Test Without Retraining

To rapidly confirm §2 is the issue **before** committing the fix and
re-training:

1. Load the current (broken) checkpoint.
2. Patch `_predict_velocity` to divide its output by `(1−r)/h`-equivalent
   factor at inference: `velocity_corrected = velocity * h / (1 - (t - h))`.
3. Re-run a single eval rollout.
4. If trajectories curve correctly, §2 is **definitively** the cause.

This is a 5-line monkey-patch in the eval script — no checkpoint
re-training required. If §2 is the cause, the bias is consistent enough
that the inference-time correction should recover most behavior.

---

## 7. One-Line Summary

The iMF training target uses `(x_data − x_r)/h` instead of the correct
mean-flow target `(x_t − x_r)/h`. For DATA-AT-1 linear interpolants the
former equals `((1−r)/h) · v_const`, which inflates the regression
target by a factor of up to **N at small t** (where N = `flow_steps`).
The model learns the wrong-scale velocity, the first few Euler sampling
steps overshoot by ~N×, the latent leaves the data manifold, and the
resulting rollout is a chaotic straight line in the dominant
`(x_data − noise)` direction. **Time direction is not reversed; the
target formula has the wrong variable.**
