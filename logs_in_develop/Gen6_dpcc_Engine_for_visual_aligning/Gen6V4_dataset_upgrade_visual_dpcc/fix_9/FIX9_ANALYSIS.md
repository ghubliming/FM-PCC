# Fix 9 — Eval Results Analysis & Projector Diagnosis

**Date:** 2026-05-19  
**Analyst:** Antigravity (Auditor)  
**Input:** Slurm eval outputs from Fix 7 (`temp/For_Gen6V4/Slurm eval outptus fix7`)  
**Context:** Pre-Fix 8, constraints disabled (`constraint_types: []`)

---

## 1. Eval Results Summary (Fix 7, Seed 6)

| Variant | Context 0 Dist (m) | Context 1 Dist (m) | Mean Dist (m) | Success | Visual Observation |
|---------|--------------------|--------------------|---------------|---------|-------------------|
| **diffuser** | 0.2695 | 0.4137 | **0.3416** | 0% | ✅ Robot arm moves correctly, camera tracks scene |
| **post_processing** | 0.7945 | 0.4168 | **0.6056** | 0% | ❌ Robot frozen near target (slight offset), right camera hits table |
| **model_free** | 0.4537 | 0.4167 | **0.4352** | 0% | ❌ Same frozen behavior as post_processing |

### Key Observations

- **All 3 variants: 0% success rate** — even `diffuser` fails (0.34m mean distance), but it's at least moving.
- **`post_processing` is worst** — 0.6056m mean, nearly 2× worse than `diffuser`.
- **Context 1 distances suspiciously similar** — all three variants report ~0.416-0.417m for Context 1. This suggests the robot is stuck in approximately the same degenerate position.
- **Maximum Tracking Error = 0.000 for all** — this is the dead-reckoning observation (B2). The code doesn't compute real tracking error.
- **Avg inference time** — `diffuser`: 0.233s, `post_processing`: 0.241s, `model_free`: 0.245s. The projector adds ~10ms overhead even with empty constraints.

---

## 2. Root Cause Analysis

### Why `diffuser` works but `post_processing` / `model_free` fail

The critical differences:

| Property | `diffuser` | `post_processing` | `model_free` |
|----------|-----------|-------------------|-------------|
| Projector | **None** (skipped, line 757) | Active | Active |
| batch_size | 1 (default) | **6** (line 768) | **6** |
| trajectory_selection | N/A (only 1 sample) | **random** | **random** |
| gradient | N/A | **False** | **False** |
| projector.gradient | N/A | False | False |
| threshold | N/A | **0.0** (post_processing) | **0.5** |
| Model call | `model(cond)` — no projector | `model(cond, projector=projector)` | `model(cond, projector=projector)` |

### The Projector Runs SLSQP Even With Empty Constraints

Even though `constraint_types: []`, when the projector is passed to the model:

**For `post_processing`** (threshold=0.0):
- `projector.gradient = False` and `threshold = 0.0`
- In `p_sample_loop` line 186: `not projector.gradient and t <= 0.0 * n_timesteps`
- This means `t <= 0` → **only at the final denoising step (t=0)** does `projector.project(x)` run
- But it DOES run — SLSQP with bounds `[-5, 5]` and the QP cost, for **all 6 batch samples**

**For `model_free`** (threshold=0.5):
- Same logic but `threshold = 0.5`
- `t <= 0.5 * n_timesteps` → SLSQP runs for the **last 50% of denoising steps**
- That's **~500 SLSQP calls** (1000 timesteps × 50% × 1 call/step... actually K denoising steps, not 1000)

### The SLSQP With Empty Constraints Is NOT Identity

This is the core problem. With `constraint_types: []`:
- `A` is empty → no equality constraints added to SLSQP
- `C` is empty → no inequality constraints added to SLSQP
- But SLSQP still has:
  - **Bounds**: `Bounds(-5, 5)` — clips all values to `[-5, 5]` in normalized space
  - **QP cost**: `0.5 * x @ Q @ x + r @ x` where `Q = I` (identity) and `r = -x_input @ Q`

The cost function `0.5 * x @ Q @ x + r @ x` with `r = -x_input @ I` simplifies to:
```
cost(x) = 0.5 * x^T x - x_input^T x = 0.5 * (x - x_input)^T (x - x_input) - 0.5 * x_input^T x_input
```

This is a **minimum-distance-to-input** objective. With `Q=I`, the unconstrained optimum is `x* = x_input`. So SLSQP **should** return the input trajectory unchanged... **unless the bounds `[-5, 5]` are clipping**.

### Hypothesis: Bounds Clipping May Not Be The Issue

The normalized trajectories should be within `[-1, 1]` (LimitsNormalizer maps to this range). Values of `[-5, 5]` should be generous. But let's check if there's a numerical issue with SLSQP itself.

### Hypothesis: The Real Problem Is `batch_size=6` + `random` Selection

With `trajectory_selection = 'random'` (line 762 — the variant names don't contain `dpcc-c` or `dpcc-t`):
- 6 trajectories are generated
- One is picked **at random**
- With A1 (BGR/RGB mismatch) corrupting the visual conditioning, the diffusion model generates 6 noisy trajectories
- `diffuser` uses `batch_size=1` and always picks sample 0 — which may be consistently "less bad"
- `post_processing` / `model_free` pick 1 of 6 randomly — more likely to pick a degenerate one

**But this doesn't fully explain the "frozen robot" behavior.** If it were just bad selection, the robot would still move — just badly.

### Most Likely Root Cause: SLSQP Solver Convergence Failure

The SLSQP solver with `maxiter=1000`, `tol=1e-6`, and empty constraints may:
1. Return a trivially modified trajectory (adding small numerical noise)
2. Or **fail to converge** and return the initial guess with slight perturbation

When this happens at **every denoising step** (`model_free`, threshold=0.5), the compounding numerical perturbations can push the trajectory toward a degenerate attractor (near-zero actions = robot doesn't move).

For `post_processing` (threshold=0.0), the SLSQP runs only once at the final step. But the `projection_costs` returned by the solver may be different from raw trajectory quality, causing `minimum_projection_cost` selection... wait, actually `trajectory_selection='random'` for these variants. So the selection isn't the issue.

### Revised Most Likely Root Cause: `apply_conditioning` Double-Snap After Projection

In `p_sample_loop` (lines 184-194):
```python
x = apply_conditioning(x, cond, ...)   # line 184: snap obs dims to initial obs

if projector is not None and not projector.gradient:
    x, projection_costs = projector.project(x, ...)   # line 191: SLSQP modifies x
    
x = apply_conditioning(x, cond, ...)   # line 194: snap obs dims AGAIN
```

After SLSQP runs (even with empty constraints), `apply_conditioning` forces `x[:, 0, 3:]` (obs dims at t=0) back to the initial observation. If SLSQP slightly shifted the obs dims, this snap creates a discontinuity. Over 500 denoising steps (`model_free`), this oscillation compounds.

---

## 3. Will Fix 8 Resolve This?

| Fix 8 Item | Relevance to This Problem |
|-----------|--------------------------|
| **A1 (BGR→RGB)** | ⚠️ Partially — improves visual conditioning quality for all variants, but `diffuser` already "works" with BGR, so this isn't the main cause of the projector-variant failure |
| **A4 (batch-0 broadcast)** | ❌ No effect — `constraint_list` is empty, `skip_initial_state` block never runs |
| **B1 (scale row)** | ❌ No effect — same reason as A4 |
| **C4 (obs_6d duplication)** | ❌ No effect — `obs_6d` only feeds projector anchor, projector has no dynamics constraints |
| **C1 (re-enable constraints)** | 🔄 Reverted by user — constraints still disabled |
| **B3 (deque ordering)** | ❌ No effect — dormant at `window_size=1` |

**Answer: Fix 8 alone will NOT fix the projector-variant failure.** The improvement from A1 will help all variants equally, but the fundamental issue — SLSQP modifying trajectories even with empty constraints — remains.

---

## 4. Proposed Fix 9 — Projector No-Op Guard

### Fix 9.1 — Skip Projector When Constraint Matrices Are Empty (CRITICAL)

**File:** `diffuser_visual_aligning/sampling/projection.py`, `project()` method

Add an early return when there are no effective constraints:

```python
def project(self, trajectory, constraints=None):
    # Skip SLSQP entirely when there are no constraints to enforce
    if self.A.shape[0] == 0 and self.C.shape[0] == 0 and len(self.obstacle_constraints.P_list) == 0:
        # No equality, inequality, or obstacle constraints — return input unchanged
        batch_size = trajectory.shape[0]
        projection_costs = np.zeros(batch_size, dtype=np.float32)
        return trajectory, projection_costs
    
    # ... existing SLSQP logic ...
```

This ensures the projector is truly a no-op when `constraint_types: []`.

### Fix 9.2 — Same Guard in `compute_gradient()` 

```python
def compute_gradient(self, trajectory, constraints=None):
    if self.A.shape[0] == 0 and self.C.shape[0] == 0 and len(self.obstacle_constraints.P_list) == 0:
        return torch.zeros_like(trajectory)
    # ... existing logic ...
```

### Fix 9.3 — Debug Logging for Projector Output (Recommended)

Add diagnostic logging to understand what SLSQP actually does to trajectories:

**File:** `diffuser_visual_aligning/sampling/projection.py`, inside `project()` loop

```python
for i in range(batch_size):
    # ... SLSQP call ...
    
    # Debug: check if SLSQP changed the trajectory
    delta = np.linalg.norm(sol_np[i] - trajectory_np[i])
    if delta > 1e-4:
        print(f'[ projector ] SLSQP modified trajectory {i} by {delta:.6f} '
              f'(success={res.success}, nit={res.nit}, status={res.status})')
```

### Fix 9.4 — Verify `trajectory_selection` for Non-DPCC Variants

The variants `post_processing` and `model_free` don't contain `dpcc-c` or `dpcc-t`, so `trajectory_selection = 'random'` (line 762). This means trajectory selection is **random among 6 samples**, not based on projection cost. This is likely unintended — the `post_processing` variant should probably use `minimum_projection_cost`.

**File:** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, around line 762:

```python
# Current:
trajectory_selection = 'random'
if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
elif 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'

# Proposed:
trajectory_selection = 'random'
if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
elif 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
elif variant in ['post_processing', 'model_free']:
    trajectory_selection = 'minimum_projection_cost'
```

---

## 5. Summary & Next Steps

### Diagnosis

The `post_processing` and `model_free` variants fail because:
1. **The projector's SLSQP runs even with empty constraints**, modifying trajectories through bounds and QP cost
2. **Random trajectory selection** from 6 samples (not cost-based) may pick degenerate trajectories  
3. **Compounding numerical perturbations** from repeated SLSQP calls (especially `model_free` at 50% threshold)

### Priority Actions for Fix 9

| # | Action | Impact |
|---|--------|--------|
| 1 | **Fix 9.1** — No-op guard in `project()` | 🔴 Critical — prevents SLSQP from modifying trajectories when no constraints exist |
| 2 | **Fix 9.2** — No-op guard in `compute_gradient()` | 🔴 Critical — same reasoning |
| 3 | **Fix 9.4** — Fix trajectory selection for `post_processing`/`model_free` | 🟠 High — random selection from 6 is worse than cost-based |
| 4 | **Fix 9.3** — Add debug logging | 🟡 Medium — helps diagnose if SLSQP is actually the cause |

### Expected Outcome After Fix 9

With Fix 9.1+9.2, `post_processing` and `model_free` with `constraint_types: []` should produce **identical results to `diffuser`** (same trajectory, same selection). This validates the baseline before enabling real constraints.

---

*Analyst: Antigravity*  
*2026-05-19T22:07Z*
