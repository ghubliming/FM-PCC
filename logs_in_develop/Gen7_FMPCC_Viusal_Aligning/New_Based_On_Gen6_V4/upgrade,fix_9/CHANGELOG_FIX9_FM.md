# Gen7 Fix 9 — FM Visual Aligning Diagnostics Overhaul

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py`  
**Plan MD**: [PLAN_FIX9_DIAGNOSTICS_OVERHAUL.md](PLAN_FIX9_DIAGNOSTICS_OVERHAUL.md)

---

## I2 — MPC Candidate Visualization Fix

**Root cause**: `get_action()` stored normalized action dims (`traj_np[:, :, :3]`), then plotted `start + cumsum(normalized_deltas)` — mixing ~[-1,1] values with real-world meters → garbage trajectories.

**Fix**: Store `c_pos` dims (`6:9`) from the 9D trajectory instead. Unnormalize via `obs_normalizer` (pad 3 dummy zeros for the `des_c_pos` slot, unnorm full 6D, take `[:, 3:]`). No cumsum needed — `c_pos` is already absolute position.

**Changed**:
- `get_action()` candidate storage: `traj_np[:, :, :3].copy()` → unnormalized `c_pos` via `obs_normalizer`
- Per-rollout PNG (axes[0,0]): removed `cumsum` + `plan_starts`; direct `cands[b,:,0]` vs `cands[b,:,1]`; green/gray colors (green=selected)
- Aggregate PNG (axes[i,5]): same fix

---

## I3 — Physical Tracking Error (was always ~0)

**Root cause**: Old error = `|des_robot_pos_t - (des_robot_pos_{t-1} + action)|` = mental model self-consistency, always near zero.

**Fix**: Physical error = `|robot_pos_np[:2] - des_robot_pos_np[:2]|` = actual PD controller lag (meaningful).

**Changed**:
- `predict()` visual path: replaced `last_predicted_pos` error with `np.linalg.norm(robot_pos_np[:2] - des_robot_pos_np[:2])`
- `predict()` non-visual path: appends `0.0` (no separate `c_pos` available)
- `update_rollout_info()`: `max_err` → `max_phys_err`; rollout dict key `max_tracking_error` → `max_physical_tracking_error`
- Per-rollout PNG (axes[1,2]): title "Physical Tracking Error |c_pos - des| (m)"
- JSON stat key: `max_tracking_error` → `max_physical_tracking_error`

---

## I4 — Terminal Summary (removing avoiding boilerplate)

**Root cause**: Output was copied from D3IL avoiding eval — hardcoded "Constraints satisfied: 1.0000", "Avg number of constraint violations: 0.00 +- 0.00", "Avg total violation: 0.000 +- 0.000".

**Fix**: Replaced with aligning-specific metrics.

**New output**:
```
--- aligning-d3il-visual [default] <variant> seed=<s> ---
Success rate:              X.XXXX
Avg final mean distance:   X.XXXX m  +- X.XXXX m
Min final mean distance:   X.XXXX m
Avg steps (successful):    XXX.XX +- XXX.XX
Avg steps (all trials):    XXX.XX +- XXX.XX
Physical tracking error:   mean=X.XXXX m  max=X.XXXX m
Avg inference time/step:   X.XXX s
```

**Changed**: terminal summary block at end of variant loop.

---

## I5 — Remove `.txt` duplicate stats file

**Root cause**: `_save_diagnostics()` wrote `rollout_<r>_stats.txt` with the same content as `_export_rollout_realtime()`'s JSON.

**Fix**: `_save_diagnostics()` removed entirely. Video/GIF saving folded into `_export_rollout_realtime()` at top of function. JSON is the single stats format.

**Changed**:
- `_save_diagnostics()` replaced with single comment line
- `_export_rollout_realtime()`: added video/GIF block at top; dir `realtime_diagnostics` → `diagnostics`; removed `.txt` write
- `update_rollout_info()`: removed `_save_diagnostics` call; `_export_rollout_realtime` now handles all output

---

## I6 — Overlay c_pos on X/Y Position Panels

**Context**: Original DPCC/FMPCC avoiding had `x_des/y_des` (commanded) vs `x/y` (actual env obs) in the per-rollout plots. Aligning analog: `des_c_pos` (dims 3-5) = commanded, `c_pos` (dims 6-8) = actual.

**Fix**: Added `curr_rollout_c_pos` accumulator. In `predict()` visual path, appends `robot_pos_np`. In `predict()` non-visual, appends `des_robot_pos_np` (same — no lag). Stored in `master_rollout_history['c_pos_history']`.

**Changed**:
- `__init__`: added `curr_rollout_c_pos = []`, `history_rollout_mean_dist = []`
- `reset()`: added `curr_rollout_c_pos.clear()`
- `predict()`: appends to `curr_rollout_c_pos`
- `update_rollout_info()`: stores `c_pos_history` in rollout dict; appends to `history_rollout_mean_dist`
- Per-rollout PNG axes[0,1] (X) and axes[0,2] (Y): `des` black + `actual` red dashed
- Aggregate PNG axes[i,0] and axes[i,1]: same overlay

---

## I7 — NPZ Cleanup (remove fake avoiding keys)

**Removed from `np.savez()`**:
- `n_violations=np.zeros(...)` — not a concept in aligning
- `total_violations=np.zeros(...)` — same
- `collision_free_completed=successes...` — avoiding concept
- `pos_tracking_errors=...` (old mental-model error) — renamed

**Added**:
- `mean_dist_per_rollout=np.array(agent.history_rollout_mean_dist)` — per-rollout final distance
- `physical_tracking_errors=np.array(agent.history_pos_tracking_errors, dtype=object)` — real PD lag
