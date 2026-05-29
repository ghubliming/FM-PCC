# Gen7 Fix 9 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Source MD Reference**: [upgrade,fix_9/PLAN_FIX9_DIAGNOSTICS_OVERHAUL.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/upgrade,fix_9/PLAN_FIX9_DIAGNOSTICS_OVERHAUL.md) · [upgrade,fix_9/CHANGELOG_FIX9_FM.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/upgrade,fix_9/CHANGELOG_FIX9_FM.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

---

## Summary

Fix 9 applies the same diagnostics overhaul to the DPCC eval script that was applied to the FM eval. All six issue fixes (I2, I3, I4, I5, I6, I7) are mirrored symmetrically. The DPCC eval had the identical bugs: chaotic MPC candidate plots (normalized action dims + cumsum), tracking error always ~0, legacy avoiding terminal output, `.txt` + `.json` duplicate stats, no `c_pos` overlay on X/Y panels, and fake `n_violations`/`total_violations` in NPZ.

---

## I2 — MPC Candidate Visualization Fix

Same as FM Fix 9 I2. `get_action()` stored `traj_np[:, :, :3]` (normalized action dims) and plotted `start + cumsum(...)` — chaotic garbage.

**Fix**: Unnormalized `c_pos` dims (6:9) stored directly; no cumsum.

```python
# Before (Fix 8, broken):
self.curr_rollout_all_candidates.append(traj_np[:, :, :3].copy())

# After (Fix 9):
cpos_norm = traj_np[:, :, 6:9]
if self.obs_normalizer is not None:
    B_f, H_f = cpos_norm.shape[:2]
    dummy = np.zeros((B_f * H_f, 3), dtype=np.float32)
    obs6d = np.concatenate([dummy, cpos_norm.reshape(-1, 3).astype(np.float32)], axis=1)
    obs6d_un = self.obs_normalizer.unnormalize(obs6d)
    self.curr_rollout_all_candidates.append(obs6d_un[:, 3:].reshape(B_f, H_f, 3).copy())
else:
    self.curr_rollout_all_candidates.append(cpos_norm.copy())
```

Per-rollout PNG (axes[0,0]) and aggregate PNG (axes[i,5]): removed `plan_starts` + `cumsum`; green=selected, gray=others.

---

## I3 — Physical Tracking Error

Visual path: old error was `last_predicted_pos`-based (mental model, always ~0).

**Fix**: `np.linalg.norm(robot_pos_np[:2] - des_robot_pos_np[:2])` — DPCC already unpacks `robot_pos_np` from state via C4 fix. Non-visual path appends `0.0`.

Rollout dict key: `max_tracking_error` → `max_physical_tracking_error`. JSON key updated to match.

---

## I4 — Terminal Summary

Replaced avoiding boilerplate with aligning metrics (same format as FM Fix 9 I4):
```
Success rate, Avg final mean distance, Min final mean distance,
Avg steps (successful / all), Physical tracking error (mean + max),
Avg inference time/step
```

`history_rollout_mean_dist` feeds `dists` array. `history_pos_tracking_errors` feeds physical tracking error.

---

## I5 — Remove Duplicate `.txt` Stats / Consolidate into `diagnostics/`

`_save_diagnostics()` removed (comment left). Video/GIF saving moved into `_export_rollout_realtime()`.

Output directory: `realtime_diagnostics/` → `diagnostics/`. Single JSON stats file (no `.txt`).

`update_rollout_info()`: removed `_save_diagnostics` call; single `_export_rollout_realtime` handles all output.

---

## I6 — Overlay c_pos on X/Y Panels

Added `curr_rollout_c_pos` accumulator. Visual path appends `robot_pos_np`; non-visual appends `des_robot_pos_np`.

Stored as `c_pos_history` in `master_rollout_history`.

Per-rollout PNG axes[0,1]/[0,2] and aggregate PNG axes[i,0]/[i,1]: des=black, actual=red dashed.

---

## I7 — NPZ Cleanup

**Removed**: `n_violations`, `total_violations`, `collision_free_completed`, `pos_tracking_errors` (old mental-model).

**Added**: `mean_dist_per_rollout`, `physical_tracking_errors`.

---

## New `__init__` Fields

```python
self.history_rollout_mean_dist   = []   # Fix 9
self.curr_rollout_c_pos          = []   # Fix 9
# (existing) history_pos_tracking_errors now stores physical PD lag errors
```

`reset()`: added `self.curr_rollout_c_pos.clear()`.

---

## Fix 9.1 — Z Panel, Thin Lines, Standalone High-Res MPC Plot

Identical to FM Fix 9.1. Applied symmetrically to `eval_visual_aligning_dpcc.py`.

**Z des/actual overlay**:
- Per-rollout PNG `axes[1,1]`: was `real_pos[:,2]` red, titled "Z Height (Contact Stability)". Now black=des Z, red dashed=actual Z (`c_pos_h[:,2]`), title "Z — des (black) vs actual (red)".
- Aggregate PNG `axes[i,2]`: same fix with `c_pos_hist[:,2]`.
- Import: `from mpl_toolkits.mplot3d import Axes3D` added.

**Thinner MPC foresight lines** (per-rollout `axes[0,0]` and aggregate `axes[i,5]`):
- Selected: `linewidth=1.5` → `0.8`
- Non-selected: `linewidth=0.5, alpha=0.25` → `linewidth=0.2, alpha=0.2`

**Standalone high-res MPC foresight file** (`rollout_{idx}_mpc_foresight.png`):
- `figsize=(26, 11)`, `dpi=200` → ~5200×2200 px. Saved to `diagnostics/`.
- Left `ax_xy`: XY foresight, every-other-step sampling, `equal` aspect, grid.
- Right `ax_3d`: 3D projection using `cands[b,:,0/1/2]` — shows Z spread of DPCC predictions.
- Variable name in DPCC is `c_pos_h` (vs `c_pos_hist` in FM) — kept consistent with existing code.
