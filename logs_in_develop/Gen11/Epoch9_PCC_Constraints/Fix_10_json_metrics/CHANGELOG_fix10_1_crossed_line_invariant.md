# Epoch 9 Fix_10 — `success ⇒ success_relaxed` invariant was violated

**Date:** 2026-07-06. Triggered by a real cluster result (`s_curve`, pasted by the user):
```json
{ "success": true, "success_relaxed": false, "goal_reached": true, "safe": true,
  "crossed_line": false, "contact_frac": 0.00219, "n_violations": 107, ... }
```
`success=true` but `success_relaxed=false` — this contradicts the U7 design doc's own stated
invariant: *"`success ⇒ success_relaxed` always"* (`eval_fm_uav.py` docstring, line ~37).

## Root cause

`crossed_line` (the basis of `success_relaxed`) is a **fixed-orientation half-plane** through
`goal`, oriented along the **expert's** final-approach heading — computed once, upfront, before
the rollout even starts:
```python
_p_before_goal = np.asarray(traj_fn(max(dur - 0.1, 0.0))[0], dtype=float)
_line_dir_xy = (goal - _p_before_goal)[:2]
line_dir_xy = _line_dir_xy / np.linalg.norm(_line_dir_xy)   # fixed for the whole episode
...
_side = float(np.dot(data.qpos[:2] - goal[:2], line_dir_xy))
crossed_line = crossed_line or (_side >= 0.0)
```
This only fires correctly if the **actual** rollout approaches the goal along roughly the same
bearing the expert did. `success` (`goal_reached and safe`), by contrast, is orientation-
independent — it only checks radial distance to `goal` at the final step.

**Why it broke here specifically:** this episode had `track_err_mean=0.317` and
`n_violations=107` — a genuinely rough s_curve run (high tracking error, many soft-constraint
violations near the walls). Under that much deviation from nominal, the drone can end up
physically within `goal_radius` of `goal` (→ `goal_reached=True` → `success=True`, since
`safe=True` too) while approaching from a bearing that never satisfies `_side ≥ 0` for that
one fixed plane — so `crossed_line` stays `False` for the entire episode, and
`success_relaxed = crossed_line and safe = False`. The half-plane test is a reasonable
approximation for a clean, near-expert approach; it has no guarantee for a degraded one.

## Fix

Made `crossed_line` **also** latch on raw proximity to `goal` — using the exact same
`qpos`/`goal`/`goal_radius` the final-step `goal_reached` check uses:
```python
_side = float(np.dot(data.qpos[:2] - goal[:2], line_dir_xy))
_dist_now = float(np.linalg.norm(data.qpos[:3] - goal))
crossed_line = crossed_line or (_side >= 0.0) or (_dist_now < goal_radius)
```
This runs inside the same per-physics-step loop that eventually produces `p_final` (the state
`goal_reached` is computed from) — so whenever the **final** step satisfies
`goal_dist < goal_radius`, that **same** step already satisfied `_dist_now < goal_radius` in
this loop, latching `crossed_line = True` right then. `goal_reached ⇒ crossed_line` is now
guaranteed **by construction**, not just true for the typical/clean-approach case — which
guarantees `success ⇒ success_relaxed` (both also require `safe`, unchanged).

For the normal/clean case this changes nothing (the directional half-plane already fires well
before the drone gets that close in a typical approach) — it only adds coverage for the
degraded-approach edge case that broke the invariant.

## Second observation from the same JSON (flagged, not a code fix)

`total_over_budget: 610` equals `n_fm_steps: 610` — **every single FM control step** on this
episode exceeded the real-time budget (`budget_ms=30.3` vs. `total_ms_mean=130.8`, a **4.3x**
overrun). `fm_ms_mean` (84.7) + `proj_ms_mean` (46.2) ≈ `total_ms_mean` (130.8), consistent
internally — the overrun is real, not a logging artifact. This is the SLSQP-projector
real-time-cost risk the original E9 plan already flagged (`PLAN_E9_PCC_constraints.md` §6/§7:
"SLSQP projection × batch × ~hundreds of FM steps may be slow — profile"), now visible
concretely at 100% of steps over budget for `s_curve`'s constrained variants. Not something to
silently patch here (would need a real batch-size/horizon/threshold/solver decision, not a bug
fix) — flagging clearly since it calls into question whether this episode's *closed-loop*
dynamics (each step computed far slower than the control period assumes) is representative of
true real-time deployment, separate from the `crossed_line` correctness bug above.

## Verification
- `py_compile` clean on `eval_fm_uav.py`.
- Traced the fix's guarantee by hand: `goal_reached` and the new `_dist_now` check use
  identical inputs (`data.qpos[:3]`, `goal`, `goal_radius`) at the same final timestep, so the
  implication holds structurally, not empirically/probabilistically.
- Could not re-run the actual s_curve rollout here (cluster-only, no torch/MuJoCo runtime) —
  the fix should be spot-checked against a re-run of this same failing episode
  (`s_curve`, seed/homotopy that produced the pasted JSON) to confirm `success_relaxed` now
  reads `true`.

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `crossed_line` latch condition in `rollout_one`.
