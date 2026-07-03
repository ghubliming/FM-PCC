# U7 — `success_relaxed`: finish-line crossing metric

**Date:** 2026-07-03
**Scope:** `FM_v3_uav_test/eval_fm_uav.py`, `FM_v3_uav_test/eval_artifacts.py`
**Plan:** `U7_Succes_realaxed/PLAN_success_relaxed.md`

## Problem

`success` only checks the drone's position at the **last** physics step of a fixed-length
episode. Episodes never terminate early on goal-reach, so a rollout that arrives at the
goal and then drifts/overshoots for the remainder of the episode scored an outright
`FAIL` — identical to one that never got close. Likely why success looked near-zero even
on flights that visibly reached the target.

## Fix

Added `success_relaxed`: treats the goal like a race finish line instead of a point to
land on exactly.

- A vertical plane (a line in xy, any z) through the goal, oriented perpendicular to the
  expert path's final approach heading (`traj_fn(dur)` vs `traj_fn(dur - 0.1)`).
- `crossed_line` — one-way latch, true the first physics step the drone is ever on the
  goal side of that line; stays true regardless of what happens afterward.
- `success_relaxed = crossed_line AND safe` for goal-path scenes (corridor/s_curve/pillars);
  identical to `success` for `empty` (no fixed goal there).
- `success ⇒ success_relaxed` always holds — purely additive, `success` is untouched.
- `goal_dist` (final-position distance, already existed) remains the continuous "how many
  meters off" signal — no new distance metric was needed.

## Changes

**`eval_fm_uav.py`** (`rollout_one`):
- Compute `line_dir_xy` once per rollout (right after `goal`); init `crossed_line = False`.
- Per physics step (next to the existing `min_z` tracking): update `crossed_line` via the
  signed xy projection onto `line_dir_xy`.
- New `success_relaxed` / `success_and_constraints_relaxed`, computed the same way as
  their strict counterparts.
- Returned dict gains: `success_relaxed`, `success_and_constraints_relaxed`, `crossed_line`.
- `behaviour` dict (per-rollout `.log`) gains `result_relaxed`.
- Module docstring documents the new criterion alongside the existing one.

**`eval_fm_uav.py`** (`eval_scene`):
- `summary` gains `success_relaxed_rate`, `success_and_constraints_relaxed_rate`.
- stdout progress line now also prints `success_relaxed=`.

**`eval_artifacts.py`**:
- `save_npz`: `n_success_relaxed` array added to the legacy npz schema (so
  `npz_analysis/analyze_npz.py` picks it up automatically as a per-trial metric column,
  same as `n_success`).
- `write_eval_log`: per-rollout lines and the summary block now also report
  `success_relaxed`.

## What didn't change

No changes to the FM policy, DPCC projection, the `mjpc`/MJX controller, or any physics —
purely an evaluation-metric addition. `json_safe_rollouts`/`save_rollout_stats` needed no
changes (they filter dynamically by key, not an explicit list), so the new fields already
flow through to `results.json` and per-rollout diagnostics.

## Not done here (flagged as follow-up in the plan)

Episodes still never terminate early on goal-reach. A later change could stop the episode
the instant `crossed_line` first flips true, converging `success`/`success_relaxed` — bigger
scope (touches the step loop, not just scoring), worth revisiting if `success_relaxed`
results show overshoot-after-arrival is the dominant failure mode.
