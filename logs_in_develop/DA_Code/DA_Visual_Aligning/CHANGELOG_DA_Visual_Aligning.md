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
