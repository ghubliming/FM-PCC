# UF-16.3 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-27
**Branch**: `update_into_FM`
**Source MD**: [u_f_16_constraint_metrics/CHANGELOG_UF16_3.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16_constraint_metrics/CHANGELOG_UF16_3.md)
**Metric reference**: [u_f_16_constraint_metrics/CONSTRAINT_METRICS.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16_constraint_metrics/CONSTRAINT_METRICS.md)
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

---

## Summary

See source MD for full detail.  Changes are shared infrastructure — no DPCC-specific
divergence from the FM version.  Both eval scripts receive identical additions.

---

## What was added

### Two new helper functions (module level)

**`check_trajectory_constraints(c_pos_traj, act_traj, geo_config, enlarge)`**
Vectorised NumPy check of a `(T, 3)` actual EE trajectory.  Returns 15 metrics
covering bounds/halfspace/obstacle violation counts and magnitudes, constraint
margin, first-violation step, longest safe streak, and Euler dynamics consistency.

**`_check_planned_violations(cands_xyz, geo_config, enlarge)`**
Lightweight per-replan check: fraction of `(B×H)` planned c_pos candidate positions
that still violate constraints after projection.  Non-zero → SLSQP did not fully
converge for those candidates.

### `VisualAgentWrapper` additions

| Location | Change |
|---|---|
| `__init__` | Added `history_constraint_metrics = []` and `_plan_post_viol_rates = []` |
| `reset()` | Both lists cleared |
| `predict()` (after candidates unnormalised) | Calls `_check_planned_violations` on latest candidates, appends to `_plan_post_viol_rates` |
| `update_rollout_info()` | Calls `check_trajectory_constraints` at rollout end; stores result in `master_rollout_history` and `history_constraint_metrics` |
| `_export_rollout_realtime()` | Adds `constraint_metrics` to per-rollout JSON; prints 3-line summary to console |
| Eval summary block | Prints aggregate table (mean ± std for all 15 metrics); saves `constraint_metrics.json` in variant save path |

### New output file per variant

`{variant}/constraint_metrics.json` — cross-rollout aggregate JSON with mean ± std
for every metric and per-rollout list.

---

## No changes to

- `diffuser_visual_aligning/sampling/projection.py` — projector internals untouched
- Existing `results.pkl` / `{variant}.npz` format — fully backwards compatible
- `constraint_overview.png` — visualisation unaffected

---

## Post-release fixes (2026-05-27)

**Source MD**: [u_f_16_constraint_metrics/CHANGELOG_UF16_3.md — Post-release fixes](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16_constraint_metrics/CHANGELOG_UF16_3.md)

Both fixes are shared infrastructure — no DPCC-specific divergence from the FM version.

**Fix 1 — halfspace sign in `_check_planned_violations`**: changed `x1 -= enlarge * nx` → `x1 += enlarge * nx` (and y). Previous sign moved the halfspace boundary in the infeasible direction for tightened runs, making the planned violation check looser than the projector's actual constraint.

**Fix 2 — exec metrics always check against nominal boundary**: `check_trajectory_constraints` now called with `enlarge=0.0` for all variants, matching original DPCC `eval.py` convention. Tightened variant expected to show better `exec_constraint_sat_rate` because its planned trajectories have a δ buffer over the nominal boundary.
