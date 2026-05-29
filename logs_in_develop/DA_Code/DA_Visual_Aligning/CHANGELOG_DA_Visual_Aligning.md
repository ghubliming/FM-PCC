# Changelog: DA Visual Aligning

**Date**: 2026-05-22  
**Branch**: `update_into_FM`  
**New folder**: `Data_Analysis/DA_Visual_Aligning/`  
**Plan**: [PLAN_DA_Visual_Aligning.md](PLAN_DA_Visual_Aligning.md)

---

## Summary

New Data Analysis module for the Visual Aligning task (FM + DPCC).
Forked from `DA_Code_v3` (avoiding task). **DA_Code_v3 untouched.**

Key differences from DA_Code_v3:
- **Single seed focus** — multi-seed averaging loop deactivated (commented)
- **No geometric constraints** — constraint/halfspace dimension null
- **Per-rollout arrays** — load raw `(30,)` NPZ arrays; produce both rollout-level and summary views
- **Flat PNGs only** — hierarchical `plot_matrix_analysis()` deactivated
- **Dual source** — `--source npz` (default, U10.2+) or `--source json` (pre-U10.2 backward compat)

---

## New Files

### Python Package — `Data_Analysis/DA_Visual_Aligning/`

| File | Description |
|---|---|
| `config.py` | Single seed, no constraints, VA metrics list |
| `data_loader.py` | Single-seed loader; NPZ or JSON source |
| `batch_data_loader.py` | Multi-candidate loader (single seed each) |
| `aggregator.py` | `per_rollout` + `summary` two-level output |
| `batch_aggregator.py` | Cross-candidate stats |
| `visualizer.py` | Plots 00a–04b (per-candidate); hierarchical stub |
| `batch_visualizer.py` | Cross-candidate plots 00a–04b |
| `reporter.py` | TXT summary + per-rollout CSV |
| `batch_reporter.py` | Candidate summary + ranking CSVs |
| `utils.py` | Copied from DA_Code_v3 (logging, output dir) |
| `multi_candidate_discovery.py` | Copied from DA_Code_v3 (seed detection unchanged) |
| `main_da_batch.py` | Batch CLI — `--seed`, `--source`, `--variants` |
| `main_da.py` | Single-candidate CLI — `--seed`, `--source` |
| `__init__.py` | Package marker |

### Sbatch Scripts — `Slurm_Codes/sbatch/DA/`

| File | Description |
|---|---|
| `run_da_batch_visual_aligning.sh` | Batch analysis, default `--source npz --seed 6` |
| `run_da_single_visual_aligning.sh` | Single candidate; input path passed as `$1` |

### HTML Visualizer — `Data_Analysis/Visualizer_Visual_Aligning/index.html`

Two-mode dashboard:
- **AGGREGATE** — static PNG selector (00a–04b) with 4-chip scorecard
- **PER-ROLLOUT** — loads `per_rollout_detail.csv`, renders table with green/red row coding per success

---

## Deactivation Markers

| Marker | Where | What |
|---|---|---|
| `# MULTI-SEED-DEACTIVATED` | `data_loader.py`, `batch_data_loader.py`, `batch_aggregator.py` | Multi-seed loops |
| `# CONSTRAINT-DEACTIVATED` | `aggregator.py`, `config.py` | Constraint/halfspace dimension |
| `# HIERARCHICAL-DEACTIVATED` | `visualizer.py`, `batch_visualizer.py` | `plot_matrix_analysis()` stub |

To re-activate: remove comment markers.

---

## Bugfix — HTML Visualizer & Dynamic CSV (2026-05-22)

### Problem
Initial `index.html` rendered static PNG images (metric selector changed the `<img>` src). This was wrong — the visualizer should be fully dynamic like the existing `Visualizer/index.html` (PyScript + matplotlib, no static images).

Additionally, `batch_reporter.py` was not producing `va_candidates_dynamic.csv`, so the HTML had nothing to load.

### Fixes

| File | Change |
|---|---|
| `Visualizer_Visual_Aligning/index.html` | Rewritten as dynamic PyScript + matplotlib visualizer. Loads `va_candidates_dynamic.csv` (long-format). AGGREGATE mode: metric dropdown, variant/candidate checkboxes, dynamic bar chart. PER-ROLLOUT mode: loads `per_rollout_detail.csv`, table with green/red row coding. No env-select / halfspace / seed-mode controls (absent by design — VA has no constraint dimension). Fixed stale `id` reference (`controls-panel-agg` → `agg-controls`). |
| `batch_reporter.py` | Added `save_dynamic_csv()` method generating long-format `va_candidates_dynamic.csv` (columns: `Candidate, FolderName, FullPath, variant, metric, mean, std, n`). `save_all_reports()` now calls it. |

### CSV format (`va_candidates_dynamic.csv`)
Long-format, one row per Candidate × variant × metric. No halfspace/constraint columns.

---

## Bugfix — JSON Mode Loading (2026-05-22)

### Problem
`--source json` appeared to not work ("cannot load my path"). Three bugs combined:

1. **`batch_data_loader.py`** — when `--variants` not specified, the loader was substituting `DEFAULT_PROJECTION_VARIANTS` (13 variants) before passing to `DataLoader`. For a run that only has `diffuser`, 12 of 13 variants returned MISSING in the log, making it look like nothing loaded even though `diffuser` did load.

2. **`aggregator.py`** — per-rollout extraction used `arr.shape[0] > 1`, silently dropping arrays from runs with exactly 1 rollout. Changed to `>= 1`.

3. **Sbatch scripts** — both scripts hardcoded `--source npz` and wrong `--parent-path` / `INPUT_PATH` defaults, so the user's actual path was never searched.

### Fixes

| File | Change |
|---|---|
| `batch_data_loader.py` | Removed `DEFAULT_PROJECTION_VARIANTS` fallback; `variants=None` now passes through to `DataLoader` which auto-discovers via `os.listdir` |
| `aggregator.py` | `arr.shape[0] > 1` → `arr.shape[0] >= 1` in `_extract_per_rollout` |
| `run_da_batch_visual_aligning.sh` | `PARENT_PATH=$1` (default `logs/aligning-d3il-visual/plans/fm_visual_aligning`), `SOURCE=$2` (default `json`) |
| `run_da_single_visual_aligning.sh` | `INPUT_PATH=$1` (required, errors if missing), `SOURCE=$2` (default `json`); added comment explaining correct path level |

### Usage after fix
```bash
# Batch — auto-discovers H8_K100_..._mpc4/ recursively:
sbatch run_da_batch_visual_aligning.sh \
  logs/aligning-d3il-visual/plans/fm_visual_aligning json

# Single — must point directly at folder containing 6/:
sbatch run_da_single_visual_aligning.sh \
  logs/aligning-d3il-visual/plans/fm_visual_aligning/H8_Dfm_.../H8_K100_...mpc4 json
```

---

## `--source` Flag

Default `npz` loads `{seed}/results/{variant}/{variant}.npz` (requires U10.2).  
`--source json` reconstructs identical per-rollout arrays from `diagnostics/rollout_*_stats.json` — works on pre-U10.2 runs.

---

## Plot Map (00a–04b)

| File | Content |
|---|---|
| `00a_pareto.png` | Pareto: success_rate vs mean_distance |
| `00b_pareto.png` | Pareto: success_rate vs avg_time |
| `01a_success.png` | Bar: success rate per variant |
| `01b_success_rollouts.png` | Heatmap: variant × rollout coloured by success |
| `02a_mean_distance.png` | Bar: mean final distance per variant |
| `02b_distance_rollouts.png` | Boxplot: distance distribution (30 rollouts) |
| `03a_tracking_error.png` | Bar: avg max PD tracking error |
| `03b_steps.png` | Bar: avg steps per variant |
| `04a_time.png` | Bar: avg inference time/replan |
| `04b_context_scatter.png` | Scatter: init difficulty vs final distance per rollout |
