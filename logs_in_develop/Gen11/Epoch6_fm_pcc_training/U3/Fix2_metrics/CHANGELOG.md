# U3 Fix2 — `success` must require reaching the goal (metric bug)

**Date:** 2026-06-25

## The bug
`success` was defined as **contact-free + airborne only** — it did NOT require reaching
the goal:
```python
airborne = bool(min_z > 0.2)
success  = bool(contact_frac <= limit and airborne)   # ← no goal check
```
So a drone that took off and flew around safely *without ever reaching the target*
scored `success=true`. This produced misleading headline numbers — e.g. `empty`
reported **100% success** while `goal_reached_rate = 0.00` (it ended 1.5 m from the
goal *every* trial). A metric that says "success" when the task is not completed is a bug.

Evidence (existing runs, all four scenes): `goal_reached_rate = 0.00` everywhere, yet
old `success_rate` was empty=1.00, s_curve=0.95.

## The fix (`FM_v3_uav_test/eval_fm_uav.py`)
`success` now **requires task completion AND safe flight**:
```python
goal_reached = bool(goal_dist < goal_radius)
safe         = bool(contact_frac <= limit and airborne)   # the OLD definition, renamed
success      = bool(goal_reached and safe)                 # NEW: must reach the goal too
```
- The old contact-free+airborne proxy is **kept and reported** as `safe` (per rollout) and
  `safe_rate` (per scene), so coherent-but-incomplete flights are still visible — we lose
  no information, we just stop mislabelling them as "success".
- `goal_dist < goal_radius` (the goal-reach test) is now a **CLI arg** `--goal-radius`
  (default `0.30 m`, was a hard-coded constant) so the tolerance is tunable per run.
- `rollout_one(...)` gains a `goal_radius` parameter; `eval_scene` passes `args.goal_radius`.
- Reporting updated: `results.json` summary adds `safe_rate`; the stdout line and
  `eval_<variant>.log` now show `success (goal+safe)`, `safe`, `goal_reached`, `goal_dist`.

## Impact on the 4-scene result (recomputed from existing npz)
`success = goal_reached AND safe`, and `goal_reached_rate = 0` everywhere, so:

| scene | OLD success (=safe) | goal_reached | **NEW success_rate** |
|---|---:|---:|---:|
| empty   | 1.00 | 0.00 | **0.00** |
| s_curve | 0.95 | 0.00 | **0.00** |
| corridor | 0.00 | 0.00 | **0.00** |
| pillars  | 0.00 | 0.00 | **0.00** |

**Honest task result: 0% across all scenes** — no scene reaches the goal. The earlier
100%/95% were the bug. This does NOT change the homotopy finding
(`../FINDING_homotopy_ambiguity_4scene_AB.md`), which rests on the explosion/tracking
split (`exec_maxabs`, `track_err`) — single-homotopy scenes still fly *coherently*, they
just (correctly) no longer count as task-success because they don't reach the goal. If
anything it sharpens the conclusion: **with no goal signal the FM cannot reach a goal at
all → goal conditioning (Epoch 7) is required.**

## Refinement (same day): scene-aware success — `empty` ≠ the goal-path scenes

Checking before the E7 upgrade surfaced that the scenes are **not** the same kind of task
(`generator._build_traj_and_init`):

- **`empty`** — `p_start` AND `p_goal` are both `rng.uniform(...)`: a **random** start→goal
  *every episode*. The state-only FM is never told that random goal, so requiring it to
  *reach* the goal is ill-defined/unfair. The right bar for `empty` is **stay stable**.
- **`corridor` / `s_curve` / `pillars`** — **fixed** start + a geometry-determined route to a
  fixed endpoint (`corridor_path` / `s_curve_scene_path` / `pillar_path`). Goal-reaching is a
  correct, fair metric here.

So `success` is now **scene-aware** (`eval_fm_uav.py`):
```python
GOAL_PATH_SCENES = {'corridor', 's_curve', 'pillars'}
...
if scene in GOAL_PATH_SCENES:
    success = bool(goal_reached and safe)     # must reach the fixed route endpoint
else:                                         # empty: random goal → just stay stable
    success = bool(safe)
```
`goal_reached`, `goal_dist`, and `safe` are still reported for **every** scene (no info lost).

### Effect on the 4-scene result (recomputed from existing npz)
| scene | criterion | **success** | (safe, goal_reached) |
|---|---|---:|---|
| empty   | stable (random goal) | **1.00** | (1.00, 0.00) |
| s_curve | goal_reached AND safe | **0.00** | (0.95, 0.00) |
| corridor | goal_reached AND safe | **0.00** | (0.00, 0.00) |
| pillars  | goal_reached AND safe | **0.00** | (0.00, 0.00) |

Now meaningful: `empty` correctly scores 1.00 (it flies stably — the appropriate task for a
random, unknowable goal), while the fixed-route scenes correctly score 0.00 (none complete
the route — `s_curve` flies coherently but ends ~5.7 m short; corridor/pillars explode).

## Notes
- `py_compile` clean. Working-tree only — re-run evals to regenerate `results.json` with the
  corrected `success_rate` (existing files carry the old number; tables here are recomputed).
- `safe_rate` ≈ the old global `success_rate`, so historical "safe flight" comparisons remain
  available under the new name.
- If a future scene also uses a random/unknown goal, add it OUTSIDE `GOAL_PATH_SCENES` so it
  is scored on stability, not goal-reaching.
