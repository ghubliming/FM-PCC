# U_2 Changelog — Dynamic Compare & Extended Metrics

**Date**: 2026-05-30  
**Branch**: `update_into_FM`  
**Scope**: DA Visual Aligning pipeline + HTML Visualizer  

---

## Summary

Extends the DA Visual Aligning pipeline to support the **expanded JSON metric schema** including `constraint_metrics` and extended `context_info` fields (`final_xy_dist`, `final_box_xy`, `final_box_angle_deg`). Adds a **DYNAMIC COMPARE** view mode to the HTML visualizer for interactive per-rollout metric comparison (scatter / bar / box charts).

---

## New JSON Fields Supported

### `context_info` (extended)

| JSON Field | Internal Key | Description |
|---|---|---|
| `context_info.final_xy_dist` | `context_final_xy_dist` | Final box→target XY distance (m) |
| `context_info.final_box_angle_deg` | `context_final_box_angle_deg` | Final box angle (deg) |
| `context_info.final_box_xy` | `context_final_box_xy` | Final box XY position `[x, y]` |

### `constraint_metrics` (full block)

| JSON Field | Internal Key |
|---|---|
| `exec_n_violated_steps` | `exec_n_violated_steps` |
| `exec_constraint_sat_rate` | `exec_constraint_sat_rate` |
| `exec_zero_violation_rollout` | `exec_zero_violation_rollout` |
| `exec_bounds_viol_count` | `exec_bounds_viol_count` |
| `exec_halfspace_viol_count` | `exec_halfspace_viol_count` |
| `exec_obstacle_viol_count` | `exec_obstacle_viol_count` |
| `exec_max_bounds_viol_m` | `exec_max_bounds_viol_m` |
| `exec_max_halfspace_viol_m` | `exec_max_halfspace_viol_m` |
| `exec_max_obstacle_penetration_m` | `exec_max_obstacle_penetration_m` |
| `exec_constraint_margin_mean_m` | `exec_constraint_margin_mean_m` |
| `exec_first_violation_step` | `exec_first_violation_step` |
| `exec_longest_safe_streak` | `exec_longest_safe_streak` |
| `exec_dynamics_consistency_error_mean` | `exec_dynamics_consistency_error_mean` |
| `exec_dynamics_consistency_error_max` | `exec_dynamics_consistency_error_max` |
| `plan_post_viol_rate_mean` | `plan_post_viol_rate_mean` |
| `plan_post_viol_rate_max` | `plan_post_viol_rate_max` |
| `plan_n_replan_steps` | `plan_n_replan_steps` |

---

## Modified Files

| File | Change |
|---|---|
| `DA_Visual_Aligning/config.py` | Added 19 new metrics to `METRICS` list and `METRIC_LABELS` dict |
| `DA_Visual_Aligning/data_loader.py` | Added `_cmet()` helper; extended `_load_json()` to extract all `constraint_metrics.*` and extended `context_info` fields |
| `DA_Visual_Aligning/reporter.py` | Extended `save_per_rollout_csv()` to include 6 new columns: `final_xy_dist_m`, `constraint_sat_rate`, `n_violated_steps`, `max_bounds_viol_m`, `max_halfspace_viol_m`, `longest_safe_streak` |
| `Visualizer_Visual_Aligning/index.html` | **v2.0**: Added COMPARE view mode with scatter/bar/box chart types, dynamic Y/X metric selection, per-variant filtering, auto-best summary card, PNG download. Updated PER-ROLLOUT table to show `FinalXY` and `CnstrSat` columns. Added `final_xy_dist_m` and `constraint_sat_rate` sort options. |

---

## HTML Visualizer — COMPARE Mode

### Controls

1. **Compare Metric (Y-axis)** — dropdown: `final_xy_dist_m`, `mean_dist_m`, `constraint_sat_rate`, `max_bounds_viol_m`, `max_halfspace_viol_m`, `phys_err_m`, `avg_time_s`
2. **X-axis** — dropdown: `rollout_idx`, `context_xy_dist_m`, `mean_dist_m`, `steps`
3. **Chart Type** — toggle: `SCATTER` / `BAR` / `BOX`
4. **Variants** — checkbox list with `[ALL]` / `[NONE]` bulk actions

### Output

- Dynamic matplotlib chart rendered in-browser via PyScript
- Auto-generated **summary card** showing best variant for the selected Y metric
- **DOWNLOAD_PNG** button for 300 DPI export

### Key Feature: `final_xy_dist` Comparison

Default Y-axis is `final_xy_dist_m` — the user's primary metric of interest. This enables direct visual comparison of how close each variant's final box position is to the target across all rollouts.

---

## Backward Compatibility

- All changes are **additive** — no existing fields removed
- Old CSVs without U_2 columns still load correctly (new columns show `—` in table)
- COMPARE mode gracefully shows error message if columns are missing
- AGGREGATE and PER-ROLLOUT modes unchanged in behavior

---

## Per-Rollout CSV Format (updated)

```
variant, rollout_idx, success, mean_dist_m, steps, phys_err_m, context_xy_dist_m, avg_time_s,
final_xy_dist_m, constraint_sat_rate, n_violated_steps, max_bounds_viol_m, max_halfspace_viol_m, longest_safe_streak
```
