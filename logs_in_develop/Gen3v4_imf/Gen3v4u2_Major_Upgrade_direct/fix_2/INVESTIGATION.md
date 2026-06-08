# Gen3v4u2 — fix_2 Investigation: iMF trajectories no longer explode but are not smooth

**Date**: 2026-05-31
**Predecessor**: [`../fix_1/INVESTIGATION.md`](../fix_1/INVESTIGATION.md) (mean-flow target formula bug; resolved by `x_start → x_t`).
**Symptom**: After applying Fix-1 and retraining, the iMF rollouts no longer
shoot off as chaotic straight lines (over-scaling cured), but they are also
**not as smooth/clean as the old diffusion / FM rollouts**. Visually they
look jittery / step-quantized / not converging crisply onto the data
manifold.

---

## 1. What "correct" looks like for iMF on linear interpolants

The training target after Fix-1 is `(x_t − x_r)/h` = **the constant**
`v_const = x_data − noise` for the linear `q_sample` interpolant.

If the model perfectly learned `u(x_t, t, h) ≡ v_const`, every Euler step
during sampling would emit the same vector and the trajectory would be a
straight line from `noise` to `noise + v_const = x_data`. Curvature
visible in the rollout image means the model's output **varies between
Euler steps** — i.e., the model has not perfectly learned a constant
function and there's a step-to-step source of jitter.

That jitter is the remaining bug. Three candidates, ranked by likelihood
and ease-of-test.

---

## 2. Hypothesis A — aux head injects step-to-step jitter (most likely)

**Code**: `flow_matcher_v3_imeanflow/models/imf_diffusion.py:115-120`

```python
def _predict_velocity(self, x, cond, t, h=None, returns=None):
    velocity, aux = self._predict_uv(x, cond, t, h=h, returns=returns)
    ...
    return velocity + self.sample_aux_weight * aux
```

`aux` is produced by `iMFTrajectoryModel.aux_head` — a stateless 2-layer
MLP that depends **only on the current latent x** (no `t`, no `h`
conditioning). Trained to predict `v_target = x_start − x_base` for the
linear path (constant per sample), but at sampling time `x` drifts every
step, so the same target function produces **different outputs at every
step**.

`sample_aux_weight = 0.1 * v_mix ≈ 0.009` (`imf_diffusion.py:85` with the
default `v_loss_weight=0.1`). Each step's velocity output is therefore:

```
v_pred(step_i) = v_main(x_i, t_i, h) + 0.009 * aux_head(x_i)
```

Even though the magnitude factor is small, **the aux output varies
arbitrarily as `x_i` drifts off-manifold during integration**, and at
step 0 the input is pure noise (far OOD for an aux head trained on
interpolated `x_t` values), so its prediction is essentially noise. This
puts a per-step wobble on the trajectory that compounds over N steps.

### How to test (no retrain)

Disable aux at inference only. One-line monkey-patch in
`_predict_velocity`:

```python
# return velocity + self.sample_aux_weight * aux   # FIX-2 hypothesis A test
return velocity
```

Or use the sibling script [`disable_aux_at_inference.py`](disable_aux_at_inference.py)
which monkey-patches at runtime without editing source.

Re-eval the same checkpoint. **If trajectories become visibly smoother
and more straight-line-like → Hypothesis A is the cause.** Permanent fix
is then to either set `sample_aux_weight = 0` in `__init__` or remove
the aux branch from inference entirely.

---

## 3. Hypothesis B — first sampling step at t=0 is OOD

**Code**: `flow_matcher_v3_imeanflow/models/imf_diffusion.py:156-163`

```python
for i in range(total_steps):
    loop_idx = min(i, flow_steps - 1)
    t_cont = torch.full((batch_size,), loop_idx / max(flow_steps, 1), ...)
    velocity = self._predict_velocity(x, cond, t_cont, h=h_batch, returns=returns)
    x = x + velocity * dt
```

Sampling queries `t = 0, 1/N, 2/N, …, (N−1)/N` with `h = 1/N` at every
step. But training samples `r ~ Uniform(0, t)` ⇒ `h = t − r ∈ [0, t]`.
At `t = 0`, training has `h = 0` always; the model has **never seen
(t = 0, h > 0)** during training.

The first Euler step is therefore based on an extrapolated prediction.
If it lands off the data manifold, every subsequent step sees an
off-manifold `x` and predictions deteriorate.

This effect is also worse at the *end* of sampling: at `t = (N−1)/N` with
`h = 1/N`, training saw `h ∈ [0, (N−1)/N]` so `h = 1/N` is well-covered,
but the model's output here directly determines the final landing point.

### How to test (no retrain)

Shift the t-grid by half a step:

```python
# t_cont = loop_idx / max(flow_steps, 1)               # current: 0, 0.1, ..., 0.9
t_cont = (loop_idx + 0.5) / max(flow_steps, 1)         # shifted: 0.05, 0.15, ..., 0.95
```

Re-eval same checkpoint. If smoother → Hypothesis B. Midpoint queries
also better match the mean-flow semantics (mean velocity over an
interval is best queried at its midpoint).

---

## 4. Hypothesis C — (t, h) joint distribution mismatch (requires retraining)

Training samples `(t, h)` from a coupled distribution: `h ∈ [0, t]`.
Sampling queries with **constant `h = 1/N`** regardless of `t` — so
roughly half of sampling queries have a `(t, h)` pair the model never
saw in training (specifically, when `h > t`).

This is structural — the h-MLP was conditioned on coupled values; its
extrapolation to decoupled queries may be poor.

### How to fix (requires retraining)

In `p_losses`, sample `h` independently from `t` (e.g., `h ~ Uniform(0,
1)`), then set `r = max(0, t − h)`. Or sample `h ~ Uniform(0, 1)` and
allow `r < 0` (treated as pure-noise extrapolation; might need careful
handling).

**Only attempt if A and B don't explain the problem** — retraining cost.

---

## 5. Hypothesis D — N too small for an imperfect model

The model isn't perfect; each step's `v_pred` differs slightly from the
true `v_const`. With small N (10), the per-step error compounds visibly.
With larger N (50, 100), errors average out.

### How to test (no retrain)

Sweep `flow_steps_v3 ∈ {10, 25, 50, 100}` at eval. If quality
monotonically improves with N → just a step-budget issue, model is
working as intended, pick N according to compute budget.

This is also a useful diagnostic: if quality plateaus by N=50 with
acceptable smoothness, you can ship at N=50 and have an empirical floor
on integration error.

---

## 6. Recommended Test Order

| # | Hypothesis | Test | Effort | Retrain? |
|---|---|---|---|---|
| 1 | **A — aux jitter** | Run `disable_aux_at_inference.py` wrapper, re-eval | 5 min | No |
| 2 | **D — step budget** | Sweep `flow_steps_v3 ∈ {10, 50, 100}` | 15 min (3 evals) | No |
| 3 | **B — t=0 OOD** | Edit `t_cont` line in `p_sample_loop`, re-eval | 5 min | No |
| 4 | **C — (t,h) decouple** | Edit `p_losses` r-sampling, retrain + re-eval | hours | Yes |

Run 1–3 in series; each rules in/out one candidate cheaply. Only consider
4 if 1–3 don't recover smoothness.

---

## 7. What's NOT under suspicion this round

- The mean-flow target formula itself — Fix-1 resolved it; if it were
  still wrong, trajectories would still explode, not just be jittery.
- The sampling time direction — verified DATA-AT-1 throughout in
  [Fix-1 §1](../fix_1/INVESTIGATION.md#1-verifying-the-time-direction-hypothesis-its-not-the-cause).
- `apply_conditioning` semantics — consistent between training and
  sampling; verified during Fix-1 cross-check.
- Allocation / U-Net shape — would manifest as crashes, not jitter.

---

## 8. Deliverables in This Folder

- `INVESTIGATION.md` — this file.
- `disable_aux_at_inference.py` — monkey-patch script for Hypothesis A
  test. Standalone, no source edits, no retrain.
- (After tests) `CHANGELOG.md` — what was confirmed + applied.
