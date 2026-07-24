# U7 — `success_relaxed`: crossed the finish line, not "ended exactly on it"

**Date:** 2026-07-03
**Scope:** `FM_v3_uav_test/eval_fm_uav.py` (`rollout_one`, lines ~324–519)
**Status:** plan (not yet implemented)

## The problem

`success` requires the drone's position **at the very last physics step of the episode**
to be within `goal_radius` (0.30 m) of the fixed route endpoint:

```python
p_final = data.qpos[:3].copy()
goal_dist = float(np.linalg.norm(p_final - goal))
goal_reached = bool(goal_dist < goal_radius)
success = bool(goal_reached and safe)          # goal-path scenes: corridor, s_curve, pillars
```

Episodes run a **fixed** number of FM steps (no early-exit on reaching the goal). A rollout
that flies to the goal, arrives, and then drifts/overshoots for the remaining steps (because
the episode isn't over yet) is scored an outright `FAIL` — identical to one that never got
close at all. This is likely why success looks near-zero even on flights that visibly reach
the target.

## Fix: `success_relaxed` = crossed the finish line

Instead of a distance/radius check, treat the goal like a race finish line: a vertical plane
(infinite in z, defined by a line in the xy plane) positioned at the goal, oriented
perpendicular to the drone's final approach heading. `success_relaxed` = the drone's xy path
ever crosses to the far side of that line — a pass/fail crossing test, no tolerance radius.

```python
# once, after computing `goal` (eval_fm_uav.py:328):
p_before_goal = np.asarray(traj_fn(max(dur - 0.1, 0.0))[0], dtype=float)   # expert heading into goal
line_dir_xy = (goal - p_before_goal)[:2]
line_dir_xy /= (np.linalg.norm(line_dir_xy) + 1e-9)                       # unit xy heading at the goal

# per physics step, inside the existing inner loop (eval_fm_uav.py:425):
side = float(np.dot(data.qpos[:2] - goal[:2], line_dir_xy))
crossed_line = crossed_line or (side >= 0.0)

# at the end, alongside the existing success block:
success_relaxed = bool(crossed_line and safe) if scene in GOAL_PATH_SCENES else success
```

- `line_dir_xy`: the expert path's heading over its last 0.1s, i.e. which way it's "running"
  into the goal — the finish line sits perpendicular to that.
- `side >= 0` means "on the goal side of the line" (xy only — z/altitude is not part of the
  crossing test; `safe` already gates altitude/contact separately).
- `crossed_line` is a one-way latch: once true for any step, stays true for the rest of the
  rollout.
- `empty` scene unaffected (`success_relaxed == success` there — no fixed goal to cross).
- `success ⇒ success_relaxed` always (ending inside `goal_radius` of the goal implies you're
  on the goal side of a line through the goal).
- The "how many meters off" continuous signal already exists — `goal_dist` (final distance
  to goal, `eval_fm_uav.py:464`) is already computed and already in the per-rollout return
  dict (`:500`) and every summary. Nothing new needed there; `success_relaxed` only adds the
  binary crossing flag next to metrics that already exist.

## Implementation sketch

Init once per rollout, near `goal` (eval_fm_uav.py:328):
```python
p_before_goal = np.asarray(traj_fn(max(dur - 0.1, 0.0))[0], dtype=float)
line_dir_xy = (goal - p_before_goal)[:2]
_norm = np.linalg.norm(line_dir_xy)
line_dir_xy = line_dir_xy / _norm if _norm > 1e-9 else np.array([1.0, 0.0])
crossed_line = False
```

Inside the inner physics loop (next to where `min_z` is already tracked, eval_fm_uav.py:436):
```python
side = float(np.dot(data.qpos[:2] - goal[:2], line_dir_xy))
crossed_line = crossed_line or (side >= 0.0)
```

At the end, next to the existing `goal_dist`/`goal_reached`/`success` block:
```python
success_relaxed = bool(crossed_line and safe) if scene in GOAL_PATH_SCENES else success
success_and_constraints_relaxed = bool(success_relaxed and collision_free)
```

Add `success_relaxed` (and `crossed_line` for debugging) to the returned dict and to
`eval_scene`'s summary rollup (next to `success_rate`, ~line 628–632):
```python
'success_relaxed_rate': float(np.mean([r['success_relaxed'] for r in rollouts])),
```

## What this does and doesn't change

- Purely an evaluation-metric addition — no change to the FM policy, DPCC projection, or the
  `mjpc`/MJX controller.
- `success` is untouched, reported exactly as strict as today. `success_relaxed` is additive.
- No "no cheating" conflict: the controller still has to actually fly through the goal
  region — this only stops penalizing it for drifting *after* arrival, in an episode it has
  no way to end early.

## Open question for follow-up (not blocking this fix)

Episodes never terminate early on goal-reach today — always run the full expert-path
duration. Letting the episode stop the instant `crossed_line` first flips true would make
"arrival" and "episode end" the same event again, converging `success`/`success_relaxed`. That
touches the step loop itself, bigger scope than this — worth a U8 if the relaxed numbers show
overshoot-after-arrival is the dominant failure mode.
