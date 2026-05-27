# UF-16.3: Constraint Satisfaction / Violation Metrics

**Date**: 2026-05-27
**Branch**: `update_into_FM`
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py`,
           `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`
**Metric reference**: [CONSTRAINT_METRICS.md](CONSTRAINT_METRICS.md)

---

## Motivation

Before UF-16.3 the eval pipeline reported task-level metrics only (success rate,
mean distance, physical tracking error). There was no quantitative answer to the
question: *"did the robot actually stay inside the geometric constraints?"*

Constraint visualisation (UF-15) showed the geometry; UF-16.3 adds the numbers.
Two measurement levels are captured:

| Level | Question answered |
|---|---|
| **Execution** | Did the real executed `c_pos_history` stay inside bounds / halfspace / obstacles? |
| **Planning** | Did the post-projection planned trajectories satisfy constraints? (residual violation → SLSQP did not fully converge) |

---

## New functions added (both eval files)

### `check_trajectory_constraints(c_pos_traj, act_traj, geo_config, enlarge)`

Vectorised NumPy check of a `(T, 3)` actual EE trajectory against all active
geometric constraints.  Returns a dict of 15 `exec_*` metrics (all float/int/bool,
JSON-serialisable).  See [CONSTRAINT_METRICS.md](CONSTRAINT_METRICS.md) for full
definitions.

Key design decisions:
- Works entirely in **physical metres** (unnormalised `c_pos_history`).
- Handles `±inf` bounds by clamping to `±1e9` before comparison.
- Tightening applied via `enlarge` arg — same margin convention as the projector.
- Each constraint type guarded by presence in `constraint_types`.

### `_check_planned_violations(cands_xyz, geo_config, enlarge)`

Checks `(B, H, 3)` unnormalised planned c_pos candidates (post-projection, at each
replan step) against bounds + halfspace + obstacles.  Returns fraction of
`(sample, horizon_step)` pairs that still violate.  Non-zero → SLSQP did not
fully enforce constraints for at least one candidate.

---

## Integration points

### `VisualAgentWrapper.__init__` / `reset()`

Two new accumulators:
```python
self.history_constraint_metrics = []   # per-rollout exec metric dicts
self._plan_post_viol_rates      = []   # per-replan planned violation fraction
```
Both cleared in `reset()` at each rollout start.

### `VisualAgentWrapper.predict()` — after candidates are unnormalised

After `curr_rollout_all_candidates.append(...)`, calls `_check_planned_violations`
on the latest candidate set and appends the result to `_plan_post_viol_rates`.

### `VisualAgentWrapper.update_rollout_info()` — at rollout end

Calls `check_trajectory_constraints(curr_rollout_c_pos, history_desired_actions,
geo_config, enlarge)` and aggregates `_plan_post_viol_rates` into a single
`_cmetrics` dict.  Stored in:
- `master_rollout_history[f'rollout_{ridx}']['constraint_metrics']`
- `history_constraint_metrics` list (for cross-rollout aggregation)

### `_export_rollout_realtime()` — per-rollout JSON

`constraint_metrics` added to `rollout_{idx}_stats.json`.  A 3-line summary is
printed to console immediately after each rollout:
```
  [ constraints ] sat=0.923  violated=12steps  (bounds=8 hs=4 obs=0)
    first_viol_step=47  longest_safe=183  margin=0.0312m  dyn_err=0.0021m
    plan_post_viol_rate=0.0082  zero_viol=False
```

### Eval summary block — per-variant aggregate

After all rollouts for a variant, prints a full aggregate table and saves
`constraint_metrics.json` in the variant's `save_path`:

```
--- Constraint Metrics [dpcc-c] ---
  Execution satisfaction rate:    0.891 ± 0.074
  Violated steps/rollout:         17.3 ± 8.1  (bounds=12.1  hs=5.2  obs=0.0)
  Max bounds viol (m):            0.0231 ± 0.0108
  Max halfspace viol (m):         0.0094 ± 0.0043
  Max obstacle penetration (m):   0.0000 ± 0.0000
  Constraint margin mean (m):     0.0418 ± 0.0122
  First violation step:           39.7 (n_rollouts_with_viol=7)
  Longest safe streak (steps):    164 ± 52
  Dynamics consistency err (m):   0.0019 ± 0.0006
  Plan post-proj viol rate:       0.0061 ± 0.0031
  Zero-violation rollouts:        2 / 9  (22.2%)
  → Saved: .../dpcc-c/constraint_metrics.json
```

---

## Output files

| File | Written by | Contains |
|---|---|---|
| `diagnostics/rollout_{idx}_stats.json` | `_export_rollout_realtime` | Per-rollout constraint metrics merged into existing stats JSON |
| `{variant}/constraint_metrics.json` | Eval summary block | Cross-rollout mean ± std for all metrics + per-rollout list |

---

## Metrics catalogue

See [CONSTRAINT_METRICS.md](CONSTRAINT_METRICS.md) for full definitions, formulas,
interpretation guide, and comparison table between variants.

---

## Changed files summary

| File | Change |
|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | Added `check_trajectory_constraints()`, `_check_planned_violations()`, integration into `__init__`/`reset`/`predict`/`update_rollout_info`/`_export_rollout_realtime`/summary |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Identical changes |

---

## Post-release fixes (2026-05-27)

### Fix 1 — halfspace sign in `_check_planned_violations`

```python
# WRONG (original) — subtracts feasible-side normal → shifts boundary into infeasible direction
x1 -= enlarge * nx;  y1 -= enlarge * ny

# CORRECT — adds feasible-side normal → shifts boundary into feasible region (tighter)
x1 += enlarge * nx;  y1 += enlarge * ny
```

`(nx, ny)` is the unit normal pointing toward the feasible side.  The original sign
moved the halfspace boundary in the wrong direction, making `_check_planned_violations`
evaluate a *looser* halfspace for tightened runs than the projector actually enforced.

### Fix 2 — exec metrics always check against nominal boundary (matches original DPCC paper)

`check_trajectory_constraints` is now called with `enlarge=0.0` for **all** variants,
regardless of `is_tightened`.

**Why**: the original DPCC `eval.py` checks all variants against nominal constraint
boundaries (`constraint_list_polytopic_not_tightened`, un-enlarged obstacle radii).
Our previous code passed `enlarge=δ` for tightened runs, measuring those trajectories
against a harder standard than nominal runs — making cross-variant comparison unfair
and masking the safety benefit of tightening.

**Expected outcome after fix**: the tightened variant should show *better*
`exec_constraint_sat_rate` than the nominal variant, because its trajectories are
planned with a δ buffer and thus stay δ inside the nominal boundary even under
execution noise.  This is the intended DPCC tightening result.

**`_check_planned_violations` unchanged**: that metric still uses `enlarge=δ` for
tightened runs — it answers "did SLSQP enforce the constraints it was *given*?"
(the projector's own tighter boundary), not the paper-level safety comparison.

| Function | enlarge used | Question answered |
|---|---|---|
| `check_trajectory_constraints` | always 0 | Was the real trajectory safe vs nominal? |
| `_check_planned_violations` | δ for tightened, 0 for nominal | Did the projector succeed at the boundary it was given? |

### Changed files (fixes)

| File | Change |
|---|---|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | `_check_planned_violations` halfspace sign `+=`; `check_trajectory_constraints` call site `enlarge=0.0` |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Identical changes |
| `config/visual_aligning_eval.yaml` | `enlarge_constraints` comment expanded to explain planning-harder / metrics-nominal split |
| `u_f_15_constrainst_visual/TIGHTENING_CONVENTION.md` | TL;DR table added; sign bug section split into Fix 1 / Fix 2; code table updated |
