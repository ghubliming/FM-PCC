# Gen3v4u2 — fix_1 Changelog

**Date**: 2026-05-31
**Branch**: `update_into_FM`
**Investigation**: [`INVESTIGATION.md`](INVESTIGATION.md)
**Symptom**: iMF eval rollouts visualised as chaotic straight lines instead of curved data-distribution trajectories (`logs/avoiding-d3il/.../iMFDiffusion/.../halfspace_both-hard/diffuser.png`).
**Root cause**: training-target formula in `p_losses` used `(x_data − x_r)/h`, which equals `((1−r)/h)·v_const` for the linear interpolant — over-scaling the regression target by up to ~N at the start of N-step Euler sampling.
**Runtime status**: ⏭ **Code fix applied; not yet retrained or re-evaluated on cluster.** Validation pending.

---

## Files Modified

| File | Lines | Change |
|---|---|---|
| `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | 265-266 → 265-274 | Replaced wrong target `(x_start − x_r)/h` with correct mean-flow bootstrap `(x_t − x_r)/h`. Added explanatory comment + FIX-1 marker pointing at `INVESTIGATION.md`. |

### Diff (the one substantive change)

```diff
-        # Mean flow target: (x_data - x_r) / h  — average velocity from x_r to x_data over interval h
-        u_target = (x_start - x_r) / (h_expand + 1e-8)
+        # Mean flow target: (x_t - x_r) / h  — average instantaneous velocity over interval [r, t].
+        # For the linear interpolant q_sample(τ) = (1−τ)·noise + τ·x_data this equals
+        # the constant v = x_data − noise (since dx/dτ is constant for a linear path),
+        # which matches the iMeanFlow definition u(x_t, t, h) := (1/h) ∫_{t−h}^t v dτ.
+        # FIX-1: previous code had (x_start − x_r)/h = ((1−r)/h)·v, which over-scales
+        # the target by ~N at small t for N-step Euler sampling and causes the trained
+        # model to output velocities so large that the first sampling Euler step lands
+        # outside the data manifold (chaotic-straight-line rollouts). See
+        # logs_in_develop/Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_1/INVESTIGATION.md
+        u_target = (x_t - x_r) / (h_expand + 1e-8)
```

## Files Created

| File | Purpose |
|---|---|
| `logs_in_develop/Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_1/INVESTIGATION.md` | Full diagnosis: ruled out time-reversal hypothesis, derived the math, listed verification plan |
| `logs_in_develop/Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_1/CHANGELOG.md` | This file |

## Files Touched Elsewhere

**None.** The only logic change is one line in `imf_diffusion.py`.

The two other `imf_diffusion.py` paths in the repo are not separate
implementations:
- `diffuser/flow_matcher_v3_imeanflow/models/imf_diffusion.py` — a one-line
  re-export shim (`from flow_matcher_v3_imeanflow.models.imf_diffusion
  import iMeanFlowODE`). The fix propagates automatically.
- `Archived_Codes/diffuser_visual_aligning(Outdated)/flow_matcher_v3_imeanflow/...`
  — explicitly archived; not on any import path. Left as-is.

---

## What Changed Semantically

**Training target** for the iMeanFlow main loss:

| | Before (buggy) | After (FIX-1) | Per-sample value for linear interpolant |
|---|---|---|---|
| `u_target` | `(x_data − x_r) / h` | `(x_t − x_r) / h` | `((1−r)/h) · v_const` → **`v_const = x_data − noise`** |

Sampling code (`p_sample_loop`, `iMFTrajectoryModel.sample_trajectory`)
**unchanged** — direction (forward Euler 0→1, DATA-AT-1), step size
(`dt = 1/N`), and h-conditioning all stay the same.

No checkpoint compatibility concern at the *interface* level (model
class, sampling API, projection plumbing all unchanged). However:

⚠️ **Existing checkpoints are NOT directly usable.** Models trained
against the buggy target learned a function that outputs `~(1−r)/h · v`
instead of `v`. Their parameters bake in the wrong scale; loading them
and re-running with the corrected sampling code would still emit
over-scaled velocities. **Retrain from scratch** to benefit from the
fix.

---

## How to Validate (Recommended Order)

### Option A — fastest (no retraining)

Monkey-patch sanity check from `INVESTIGATION.md` §6: at inference,
divide model output by the `(1−r)/h` factor with `r = t − dt`:

```python
# in eval_flow_matching_v3_imeanflow.py, wrap _predict_velocity:
correction = h / (1.0 - (t - h) + 1e-8)
velocity_corrected = velocity * correction
```

Re-run a single rollout. **If trajectories curve correctly**, the
FIX-1 diagnosis is definitively confirmed. ~5-line edit in eval, no
training.

### Option B — full retrain + eval (real fix)

1. Re-train from scratch with the patched `imf_diffusion.py`.
   Expected: loss curve **lower and more stable** than the previous
   training run, since `u_target` magnitude is now ~1 instead of
   `~N · v` at small t.
2. Re-run eval with the same K10_Meuler config that produced the
   chaotic PNG.
3. Inspect `diffuser.png`. Expected: trajectories follow the avoiding
   data distribution (curved expert paths through the obstacle field),
   no straight-line shoot-offs.
4. Expected metrics improvement:
   - Final-step success rate up substantially.
   - Constraint-satisfaction up (over-scaled steps were the dominant
     source of bound violations).
   - Mean displacement-per-step at sampling no longer dominates the
     intended noise→data scale.

---

## Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Fix doesn't resolve the visual | Low — math is airtight for linear interpolants | INVESTIGATION.md §3 lists secondary suspects in priority order if this doesn't fully fix it |
| Other parts of the iMF codebase depended on the over-scaled target | Low — only `p_losses` uses `u_target`; no other reference | Grepped: `u_target` is only used at the `loss_fn(velocity_pred, u_target)` call on line 277, nowhere else |
| Stale checkpoints get accidentally re-evaluated post-fix | Medium (cosmetic confusion) | Note above ⚠️; recommend deleting / renaming the existing `H8_Dflow_matcher_v3_imeanflow.models.iMFDiffusion_*` checkpoint dir to force retrain |
| FIX-1 marker comment goes stale | Low | Marker references the INVESTIGATION.md path; if that doc is moved, update the comment |

---

## Reversibility

```bash
git diff -- flow_matcher_v3_imeanflow/models/imf_diffusion.py
# Revert if needed:
git checkout HEAD -- flow_matcher_v3_imeanflow/models/imf_diffusion.py
```

The investigation and changelog MDs are append-only documentation; leave
them in place even if the code is reverted, as they record the
diagnosis for future re-discovery.

---

## One-Line Summary

`flow_matcher_v3_imeanflow/models/imf_diffusion.py:266` had
`u_target = (x_start − x_r)/h` (over-scales by `(1−r)/h` ≈ N at small
t for N-step sampling). Fixed to `u_target = (x_t − x_r)/h` (= `v` for
linear interpolants, matches iMeanFlow definition). One-line change,
one file. Retrain required for existing checkpoints to benefit.
