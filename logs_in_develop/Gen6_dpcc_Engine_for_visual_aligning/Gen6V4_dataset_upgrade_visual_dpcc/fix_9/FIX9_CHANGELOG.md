# Fix 9 — Changelog

**Date:** 2026-05-19
**Source:** FIX9_ANALYSIS.md — Joint Conclusion (Antigravity × Claude Code)
**Applies:** Fix 9.1, 9.2, 9.3, 9.4 + B1 unit test

---

## Files Changed

| File | Fixes Applied |
|---|---|
| `diffuser_visual_aligning/sampling/projection.py` | 9.1, 9.2, 9.3 |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | 9.4 |
| `diffuser_visual_aligning_test/test_projector_b1.py` | B1 unit test (new file) |

---

## Fix 9.1 — No-op Guard in `project()` (`projection.py` line 100-103)

**Problem:** With `constraint_types: []`, the Projector's `A` and `C` matrices are empty and `obstacle_constraints.P_list` is empty. Despite no actual constraints, `project()` still ran SLSQP with bounds `[-5, 5]` and the QP minimum-distance cost. The solver "explored" the bound space, corrupting all 6 batch trajectories to boundary-saturated ±5 normalized actions — identical to K=16 noise amplification behaviour even with a healthy K=100 model. Confirmed directly by DIAG data in Job 20551: diffuser range `[-0.78, 0.99]` vs post_processing range `[-5.0, 5.0]`.

**Change:** Early return in `project()` before any SLSQP setup when no constraints are active:

```python
# Fix 9.1: skip SLSQP entirely when no constraints are active
if self.A.shape[0] == 0 and self.C.shape[0] == 0 and len(self.obstacle_constraints.P_list) == 0:
    batch_size = trajectory.shape[0]
    return trajectory, np.zeros(batch_size, dtype=np.float32)
```

Returns the input trajectory unchanged with zero projection costs. When constraints are enabled (C1), this guard is inactive and the full SLSQP path runs as before.

---

## Fix 9.2 — No-op Guard in `compute_gradient()` (`projection.py` line 192-194)

**Problem:** Same root cause as 9.1. `compute_gradient()` is called for gradient-based projector variants. With empty constraint matrices, the gradient is mathematically zero but the code still executed all gradient accumulation loops unnecessarily.

**Change:** Early return before any gradient computation:

```python
# Fix 9.2: skip gradient computation entirely when no constraints are active.
if self.A.shape[0] == 0 and self.C.shape[0] == 0 and len(self.obstacle_constraints.P_list) == 0:
    return torch.zeros_like(trajectory)
```

---

## Fix 9.4 — Trajectory Selection for `post_processing` / `model_free` (`eval_visual_aligning_dpcc.py` lines 765-766)

**Problem:** Both `post_processing` and `model_free` variants fell through to `trajectory_selection = 'random'`, picking one of 6 batch samples at random with no cost criterion. With `batch_size=6` and random selection, there is a 5/6 chance of picking a worse-than-best sample. The identical results between the two variants in Job 20551 also suggest the same random index was drawn each step (no re-seeding between variants), making any threshold comparison meaningless.

**Change:**

```python
# Before:
trajectory_selection = 'random'
if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
elif 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'

# After:
trajectory_selection = 'random'
if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
elif 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
elif 'post_processing' in variant or 'model_free' in variant:
    trajectory_selection = 'minimum_projection_cost'  # Fix 9.4: cost-based over random from batch=6
```

With Fix 9.1 applied, projection costs for empty-constraint runs will all be zero (uniform), so `minimum_projection_cost` degrades gracefully to index-0 selection — identical to `diffuser` batch=1 behaviour. When real constraints are enabled (C1), cost-based selection will actively prefer the trajectory most consistent with constraint satisfaction.

---

## Fix 9.3 — SLSQP Delta Logging in `project()` (`projection.py` lines 173-175)

**Purpose:** Diagnostic — logs when SLSQP meaningfully modifies a trajectory during projection. Inactive in no-constraint runs (Fix 9.1 exits before reaching this code). Active and useful once C1 is enabled to confirm the projector is enforcing constraints as expected.

**Change:** Added after `projection_costs[i]` computation inside the batch loop:

```python
# Fix 9.3: log when SLSQP meaningfully modifies the trajectory
delta = np.linalg.norm(sol_np[i] - trajectory_np[i])
if delta > 1e-4:
    print(f'[ projector ] sample {i}: SLSQP delta={delta:.6f} '
          f'success={res.success} nit={res.nit} status={res.status}')
```

The old commented-out `print('Equality constraints not satisfied!')` blocks were removed — they checked the same thing but less informatively.

---

## B1 Unit Test — `test_projector_b1.py` (new file)

**Location:** `diffuser_visual_aligning_test/test_projector_b1.py`

**What B1 is:** Fix 8 B1 changed the initial-state row coefficient in `DynamicConstraints.build_matrices()` from `1` to `x_diff`, and updated the `b` vector in `project()` / `compute_gradient()` correspondingly. Without B1, the initial-state equality constraint was proportionally weaker than the dynamics rows by ~2.5× (for typical Franka workspace values), causing SLSQP to relax the initial-state anchor more than intended.

**B1 is a code fix already applied in Fix 8.** This test validates that the fix is correct.

**Run on cluster:**
```bash
cd /path/to/FM-PCC
python diffuser_visual_aligning_test/test_projector_b1.py
```

**Three tests:**

| Test | What It Checks |
|------|---------------|
| `test_b1_initial_row_coefficient()` | Structural: `A[0, x_idx] == x_diff` (not 1). Directly confirms B1 coefficient. |
| `test_b1_projection_satisfies_constraints()` | Functional: `‖A @ sol − b‖ < 1e-3` for all batch elements after projection. |
| `test_b1_initial_state_preserved()` | Functional: `projected[0, pos] ≈ s_0[pos]` — initial state anchor is tight. |

**Expected output:**
```
Test 1 — Structural: A[0, x_idx] == x_diff
  [PASS] A[0, x_idx] = 0.4000 == x_diff = 0.4000
  [PASS] Dynamics rows also use x_diff consistently.

Test 2 — Functional: equality constraints satisfied after projection
  [PASS] Sample 0: max constraint residual = ...
  [PASS] Sample 1: max constraint residual = ...
  [PASS] Sample 2: max constraint residual = ...

Test 3 — Functional: initial state is preserved in projected solution
  [PASS] s_0[pos]=..., projected[0,pos]=..., err=...

All B1 tests passed.
```

---

## What Is NOT Fixed Here

| Item | Status |
|---|---|
| C1 — Re-enable constraints | Still disabled per user decision. One-line uncomment in `config/visual_aligning_eval.yaml` when ready |
| imageio/ffmpeg on cluster | Overridden by user — not in scope |

---

## Verification Criterion (D6)

After deploying Fix 9.1+9.2, re-run with `n_contexts=2` (quick check). The `post_processing` and `model_free` DIAG blocks should show:
- Normalized action range: `~[-0.78, 0.99]` (matching diffuser)
- Step-0 denormalized magnitude: `~0.0002 m` (matching diffuser)
- Context 0 distance: `~0.09 m` (matching diffuser)

If the range is still wide (`> ±2`) after Fix 9.1, a second corruption source is present — investigate `apply_conditioning` double-snap in `p_sample_loop` (snaps obs dims after SLSQP at line ~194 in `diffusion.py`).

---

*Author: Claude Code — claude-sonnet-4-6*
*2026-05-19*
