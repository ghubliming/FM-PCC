# U_2 Hotfix-1 & Hotfix-2 — COMPARE Fallback & DataLoader Schema Updates

**Date**: 2026-05-31  
**File**: `Visualizer_Visual_Aligning/index.html`  
**Scope**: HTML-only hotfix — no backend pipeline changes  

---

## Problem

The COMPARE mode in the U_2 visualizer requires `per_rollout_detail.csv` (loaded into `df_roll`).  
The batch pipeline (`DA_VA_Batch`) does **not** generate this file — it only generates `va_candidates_dynamic.csv`.  
Result: `df_roll` was always `None` → COMPARE always showed *"No per-rollout CSV loaded"* and rendered nothing.

Secondary issue: `populate_compare_variants()` also returned early when `df_roll is None`, so the variant checkbox list in COMPARE mode was empty even after SYNC_SOURCE.

---

## Fix Summary

| Component | Before | After |
|-----------|--------|-------|
| `populate_compare_variants()` | Returns early if `df_roll is None` | Falls back to `df_agg` variants |
| `trigger_compare()` | Hard-errors if `df_roll is None` | Falls back to `df_agg` (aggregated mode) |
| BOX chart in fallback | N/A | Gracefully falls back to BAR with note |
| SCATTER chart in fallback | N/A | Plots mean per Candidate×Variant with error bars |
| BAR chart in fallback | N/A | Plots variant means with std error bars |
| Summary card in fallback | N/A | Orange background + "AGGREGATED MODE" label |
| 0.0 value warning | N/A | Shown when all values are zero (old eval format) |
| Original per-rollout path | Unchanged | Unchanged |

---

## New: `_CMP_METRIC_MAP`

Maps the COMPARE dropdown `y_col` values to metric names in `va_candidates_dynamic.csv`:

| Dropdown value | `df_agg` metric name |
|---|---|
| `final_xy_dist_m` | `context_final_xy_dist` |
| `mean_dist_m` | `mean_dist_per_rollout` |
| `constraint_sat_rate` | `exec_constraint_sat_rate` |
| `max_bounds_viol_m` | `exec_max_bounds_viol_m` |
| `max_halfspace_viol_m` | `exec_max_halfspace_viol_m` |
| `phys_err_m` | `max_phys_error_per_rollout` |
| `avg_time_s` | `avg_time` |

---

## Fallback Chart Behaviors

### SCATTER (aggregated)
- X-axis: candidate ID (A, B, H, K …)  
- Y-axis: `mean` from `va_candidates_dynamic.csv` for each Candidate×Variant  
- Error bars: `std` column from CSV  
- One series per selected variant

### BAR (aggregated)
- One bar per variant, height = mean of `mean` across all candidates  
- Error bar = mean of `std` across all candidates  
- Edged black bars, alpha=0.85

### BOX (aggregated fallback)
- BOX requires per-rollout data; with only aggregated means a box is meaningless  
- Prints a notice in the plot area, then renders BAR instead  
- Does **not** crash or show error

---

## Visual Indicators in Fallback Mode

- **Plot title**: appends `[AGGREGATED MODE — no per-rollout CSV]` in grey
- **Summary card**: orange background (`#fff3e0`) with italic note
- **Zero-value warning**: orange text shown if all values are `< 1e-9` (old eval format that did not write the metric)

---

## What Was NOT Changed

- AGGREGATE mode — untouched  
- PER-ROLLOUT mode — untouched  
- `load_data()` — untouched  
- `download_compare_plot()` — untouched  
- All backend DA pipeline files — untouched  
- Original `trigger_compare()` per-rollout path (bottom half) — untouched, only reachable when `df_roll is not None`

---

## Remaining Limitations (not fixed by this hotfix)

1. ~~`context_final_xy_dist` (and all `exec_*` metrics) are **0.0** for candidates B, H, I, K because the eval code that produced their JSONs predates the new schema. The hotfix correctly shows the zero-value warning for these.~~ *(Resolved by Hotfix 2: The pipeline now successfully reads the nested schema for all candidates.)*
2. ~~Candidates A, C, D, E, F, G, J failed loading entirely (old JSON format). They do not appear in COMPARE.~~ *(Resolved by Hotfix 2: Explicit `--geo-variant` support fixes discovery of these candidates.)*
3. BOX chart in aggregated mode degrades to BAR — true distribution plots require `per_rollout_detail.csv`.
4. COMPARE SCATTER in aggregated mode shows one point per Candidate, not one point per rollout — granularity is lower.

---

# U_2 Hotfix-2 — DataLoader Robust Auto-Retrieval for Nested Schemas

**Date**: 2026-05-31  
**Files**: `data_loader.py`  
**Scope**: Pipeline backend data loading fix  

## Problem
The data loader was hardcoded to look for JSON logs at `{seed}/results/{variant}/diagnostics/...`. However, new evaluations using geometric constraints (like `combined_5`) introduced an arbitrary new layer to the path schema: `{seed}/results/{geo_name}/{variant}/diagnostics/...`. 

Because the `DataLoader` did not account for this `geo_name` layer, it incorrectly parsed the directory structure, resulting in mass candidate loading failures (Candidates A, C, D, E, F, G, J) and returning `0.0` for new constraint metrics. 

## Fix Summary
A fully robust **Auto-Retrieval** system was implemented in `data_loader.py` to seamlessly flatten any nested directory structure.

| Component | Implementation Details |
|------|--------------|
| `data_loader.py` | Uses a deep recursive search (`glob.glob('**/diagnostics', recursive=True)` for JSON, or `**/*.npz` for NPZ) to find all actual data targets regardless of depth. |
| **Dynamic Variant Keys** | The relative path from the `results/` folder (e.g., `combined_5/dpcc-c`) is extracted and used as the unique variant key. This handles runs that contain multiple active geometric constraints simultaneously without collision. |
| **Zero Configuration** | No CLI flags or bash script changes are required. The script automatically traverses nested structures for newer runs while seamlessly supporting flat structures from older runs. |

## Usage
No new flags or parameters are needed! The original batch command now works flawlessly across all candidate schemas:
```bash
sbatch run_da_batch_visual_aligning.sh logs/aligning-d3il-visual/plans/fm_visual_aligning json
```
