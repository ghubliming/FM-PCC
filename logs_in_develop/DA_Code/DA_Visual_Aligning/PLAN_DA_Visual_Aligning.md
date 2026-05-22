# Plan: DA Visual Aligning — Data Analysis (Rewritten v2)

**Date**: 2026-05-22  
**Branch**: `update_into_FM`  
**Source**: `Data_Analysis/DA_Code_v3/` (Avoiding task DA — do NOT touch)  
**Target (new)**: `Data_Analysis/DA_Visual_Aligning/`  
**Sbatch (new)**: `Slurm_Codes/sbatch/DA/run_da_batch_visual_aligning.sh` + single  
**Visualizer (new)**: `Data_Analysis/Visualizer_Visual_Aligning/index.html`

---

## Core design decisions (updated)

| Dimension | Avoiding DA_Code_v3 | Visual Aligning DA (new) |
|---|---|---|
| **Seeds** | Loop over 5 seeds, avg across them | **Single seed** focus. Multi-seed loop code deactivated (commented), not deleted |
| **Constraints / halfspace** | `halfspace_{variant}/` subfolder, constraint types loop | **Null** (no geo constraints now). Code structure kept, simply empty/skipped. When constraints added later, slot back in as a category |
| **Data granularity** | Load NPZ → avg on load (2 trials → scalar) | Load NPZ → **keep raw `(30,)` per-rollout arrays**. Show both per-rollout view AND rollout avg |
| **Hierarchical PNG** | `plot_matrix_analysis()` generates `hierarchical_analysis/` subfolder | **Deactivated** (commented in `plot_all()`). Only flat `00a`–`04b` PNGs produced |
| **Primary data** | `.npz` (avg metrics) | `.npz` with U10.2 per-rollout arrays: `n_success`, `n_steps`, `avg_time`, `mean_dist_per_rollout`, `max_phys_error_per_rollout`, `context_*` |

---

## NPZ fields available (after U10.2)

Per-rollout arrays `(N_rollouts=30,)` — **the DA primary source**:
```
n_success                    (30,)  bool/int  — 1=success per rollout
n_steps                      (30,)  int       — steps taken
avg_time                     (30,)  float     — seconds/replan
mean_dist_per_rollout        (30,)  float     — final box–target dist (m)
max_phys_error_per_rollout   (30,)  float     — max PD lag (m)  [U10.2]
context_box_init_xy          (30,2) float     — box start XY    [U10.2]
context_target_xy            (30,2) float     — target XY       [U10.2]
context_box_angle_deg        (30,)  float     — box angle (°)   [U10.2]
context_target_angle_deg     (30,)  float     — target angle (°)[U10.2]
context_init_xy_dist         (30,)  float     — init box→target dist (m) [U10.2]
```

Scalar summaries (keep, secondary):
```
success_rate, entropy, elapsed_seconds, seed
```

---

## Data path layout

```
logs/<task>/plans/<model_exp_name>/
  {seed}/
    results/
      {variant}/
        {variant}.npz          ← loader target
        diagnostics/
          rollout_N_stats.json  ← human reading, not loaded by DA
```

No halfspace subfolder. Loader path: `{root}/{seed}/results/{variant}/{variant}.npz`.

---

## New folder structure

```
Data_Analysis/
├── DA_Code_v3/                        ← UNTOUCHED
└── DA_Visual_Aligning/                ← NEW
    ├── __init__.py
    ├── config.py
    ├── data_loader.py
    ├── batch_data_loader.py
    ├── multi_candidate_discovery.py   ← copy as-is, title update only
    ├── aggregator.py
    ├── batch_aggregator.py
    ├── visualizer.py
    ├── batch_visualizer.py
    ├── reporter.py
    ├── batch_reporter.py
    ├── utils.py                       ← copy as-is
    ├── main_da_batch.py
    ├── main_da.py
    └── README.md

Slurm_Codes/sbatch/DA/
├── run_da_batch_v3.sh                 ← UNTOUCHED
├── run_da_single_v3.sh                ← UNTOUCHED
├── run_da_batch_visual_aligning.sh    ← NEW
└── run_da_single_visual_aligning.sh   ← NEW

Data_Analysis/
├── Visualizer/                        ← UNTOUCHED
└── Visualizer_Visual_Aligning/
    └── index.html                     ← NEW
```

---

## File-by-file change plan

### `config.py`

```python
DEFAULT_SEEDS = [6, 7, 8, 9, 10]   # kept but only index [0] used by default
ACTIVE_SEED   = 6                   # single-seed mode default

METRICS = [
    'n_success',                   # (30,) per-rollout success flags
    'success_rate',                # scalar — mean(n_success)
    'mean_dist_per_rollout',       # (30,) final box–target distance
    'n_steps',                     # (30,) steps per rollout
    'avg_time',                    # (30,) seconds/replan
    'max_phys_error_per_rollout',  # (30,) max PD tracking error
    'context_init_xy_dist',        # (30,) init scene difficulty proxy
]

# Constraint category — null for now, slot back in when geo constraints added
CONSTRAINT_TYPES  = []
HALFSPACE_VARIANTS = []

DEFAULT_PROJECTION_VARIANTS = [
    'diffuser', 'dpcc-r', 'dpcc-r-tightened',
    'dpcc-c', 'dpcc-c-tightened', 'dpcc-t', 'dpcc-t-tightened',
    'gradient', 'gradient-tightened',
    'post_processing', 'post_processing-tightened',
    'model_free', 'model_free-tightened',
]

OUTPUT_FOLDER_PREFIX = 'Visual_Aligning_Analysis'
```

---

### `data_loader.py`

**Key change**: load per-rollout arrays raw (no averaging on load).

```python
def load_results(self, root_path, seed, variants):
    """
    Single-seed loader. Returns {variant: metrics_dict} where
    metrics_dict values are (30,) arrays, not scalars.
    """
    for variant in variants:
        npz_path = f'{root_path}/{seed}/results/{variant}/{variant}.npz'
        data = np.load(npz_path, allow_pickle=True)
        metrics[variant] = {k: data[k] for k in data.files}
```

- Signature removes `seeds` list → single `seed` int (default: `ACTIVE_SEED`)
- Remove `halfspace_variants` and `constraint_types` args entirely
- No averaging on load — keep `(30,)` arrays as-is
- Multi-seed loop code: **commented out with `# MULTI-SEED-DEACTIVATED`** marker, not deleted

---

### `batch_data_loader.py`

- Update `load_all_candidates()` to pass single seed to `DataLoader`
- Remove `halfspace_variants` / `constraint_types` args
- Multi-seed loop: commented out

---

### `aggregator.py`

Two-level output per variant:

```python
class DataAggregator:
    def aggregate_all(self):
        return {
            'per_rollout': self._extract_per_rollout(),   # raw (30,) arrays
            'summary':     self._compute_summary(),        # mean±std scalars
        }

    def _extract_per_rollout(self):
        """Return {variant: {metric: (30,) array}}"""

    def _compute_summary(self):
        """Return {variant: {metric: {mean, std, min, max}}}"""
```

- `aggregate_by_constraint()`, `aggregate_by_halfspace()`: **commented out** (multi-seed/constraint code)
- `aggregate_by_variant()`: simplified — just mean±std across rollouts for each variant

---

### `batch_aggregator.py`

- Passes single seed; routes per-rollout and summary dicts through
- Multi-seed ranking: **commented out**

---

### `visualizer.py`

Keep flat `00a`–`04b` naming convention. **Deactivate** `plot_matrix_analysis()`.

New plot map for visual aligning:

| File | Plot | View |
|---|---|---|
| `00a_pareto.png` | Pareto: success_rate vs mean_distance (standard variants) | Aggregate avg |
| `00b_pareto.png` | Pareto: success_rate vs avg_time | Aggregate avg |
| `01a_success.png` | Bar: success_rate per variant | Aggregate avg |
| `01b_success_rollouts.png` | Heatmap: variant × rollout_idx, coloured by success | Per-rollout |
| `02a_mean_distance.png` | Bar: mean final distance per variant (lower=better) | Aggregate avg |
| `02b_distance_rollouts.png` | Box-plot: distance distribution per variant (30 rollouts) | Per-rollout |
| `03a_tracking_error.png` | Bar: avg max_phys_error per variant | Aggregate avg |
| `03b_steps.png` | Bar: avg steps per variant | Aggregate avg |
| `04a_time.png` | Bar: avg inference time/replan per variant | Aggregate avg |
| `04b_context_scatter.png` | Scatter: context_init_xy_dist vs mean_distance, coloured by variant | Per-rollout |

`plot_matrix_analysis()`: body **commented out**, method kept as empty stub.

---

### `batch_visualizer.py`

```python
def plot_all(self, output_dir, show=False):
    self.plot_candidate_pareto_frontier(output_dir)      # 00a, 00b
    self.plot_candidate_success_comparison(output_dir)   # 01a, 01b
    self.plot_candidate_distance_comparison(output_dir)  # 02a, 02b
    self.plot_candidate_tracking_error(output_dir)       # 03a, 03b
    self.plot_candidate_time_comparison(output_dir)      # 04a, 04b
    # self.plot_matrix_analysis(output_dir)  # DEACTIVATED — hierarchical
```

---

### `reporter.py` / `batch_reporter.py`

- Remove violation/halfspace sections
- Add `mean_distance`, `max_phys_error`, `context_init_xy_dist` summary sections
- Per-rollout breakdown table: rollout_idx | success | mean_dist | steps | context_xy_dist

---

### `main_da_batch.py`

```python
# Phase 1: Discover candidates (unchanged)
# Phase 2: Filter/rename (unchanged)
# Phase 3: Load data — single seed, no halfspace
# Phase 4: Aggregate — per_rollout + summary
# Phase 5A: Flat PNGs only (00a–04b), no hierarchical
# Phase 5B: Reports
```

Add `--seed` arg (default: `6`).  
Add `--source npz|json` arg (default: `npz`).  
Remove `--constraint-types` arg.  
Remove `--halfspace-variants` arg.

---

### Sbatch scripts

**`run_da_batch_visual_aligning.sh`**:
```bash
PYTHONPATH="$FMPCC:$FMPCC/Data_Analysis/DA_Visual_Aligning:$PYTHONPATH"
python Data_Analysis/DA_Visual_Aligning/main_da_batch.py \
    --parent-path logs/visual-aligning-dpcc/plans \
    --seed 6 \
    --output-path Data_Analysis/analysis_results/va_batch_$(date +%Y%m%d_%H%M%S)
```

**`run_da_single_visual_aligning.sh`**:
```bash
INPUT_PATH=${1:-"logs/visual-aligning-dpcc/plans/flow_matching_v3_imeanflow"}
PYTHONPATH="$FMPCC:$FMPCC/Data_Analysis/DA_Visual_Aligning:$PYTHONPATH"
python Data_Analysis/DA_Visual_Aligning/main_da.py \
    --input-path "$INPUT_PATH" \
    --seed 6 \
    --output-path "Data_Analysis/analysis_results/va_single_$(basename $INPUT_PATH)_$(date +%Y%m%d_%H%M%S)"
```

---

## Visualizer HTML (`Visualizer_Visual_Aligning/index.html`)

Copy `Visualizer/index.html` then:

### Mode toggle (new)
Two view modes:
- **AGGREGATE** — shows summary bar charts (00a–04b PNGs), same as current visualizer
- **PER-ROLLOUT** — shows rollout-level detail table/heatmap

### Aggregate view
- Scorecard chips: `success_rate`, `mean_distance`, `avg_time/replan`, `max_phys_error`
- Plot area: renders selected PNG (00a–04b)
- Metric selector: maps to plot files

### Per-rollout view (new)
- Variant selector dropdown
- Table: `rollout_idx | success | mean_dist (m) | steps | phys_err (m) | context_xy_dist (m)`
- Loaded from: reads the NPZ arrays via PyScript (same fetch mechanism as current)
- Colour coding: green row = success, red = fail

### Manifest
- Same `results_manifest.json` pattern as current visualizer
- Title: `FM-PCC VISUAL ALIGNING EXPLORER`

---

## Implementation order

1. `config.py`
2. `data_loader.py` (single-seed, raw arrays, multi-seed commented)
3. `batch_data_loader.py`
4. `aggregator.py` (per_rollout + summary)
5. `batch_aggregator.py`
6. `visualizer.py` (new plots, hierarchical deactivated)
7. `batch_visualizer.py`
8. `reporter.py` + `batch_reporter.py`
9. `main_da_batch.py` + `main_da.py`
10. `utils.py` + `multi_candidate_discovery.py` (copy as-is)
11. `__init__.py` + `README.md`
12. Sbatch scripts
13. HTML visualizer

---

## What is deactivated (commented, not deleted)

| Location | What | Marker |
|---|---|---|
| `data_loader.py` | Multi-seed loop | `# MULTI-SEED-DEACTIVATED` |
| `aggregator.py` | `aggregate_by_constraint()`, `aggregate_by_halfspace()` | `# CONSTRAINT-DEACTIVATED` |
| `batch_aggregator.py` | Multi-seed ranking | `# MULTI-SEED-DEACTIVATED` |
| `visualizer.py` | `plot_matrix_analysis()` body | `# HIERARCHICAL-DEACTIVATED` |
| `batch_visualizer.py` | `plot_matrix_analysis()` call | `# HIERARCHICAL-DEACTIVATED` |

Re-activating any of these later = remove the comment marker.

---

## `--source` flag (backward compatibility)

| Flag | Loader path | When to use |
|---|---|---|
| `--source npz` (default) | `{seed}/results/{variant}/{variant}.npz` | U10.2+ runs (has per-rollout context arrays) |
| `--source json` | `{seed}/results/{variant}/diagnostics/rollout_*_stats.json` | Pre-U10.2 runs (JSON-only) |

JSON loader reconstructs the same `(N_rollouts,)` arrays so downstream aggregator/visualizer is identical:

| NPZ key | JSON field |
|---|---|
| `n_success` | `success` |
| `n_steps` | `steps` |
| `avg_time` | `avg_inference_time_per_replan` |
| `mean_dist_per_rollout` | `mean_distance` |
| `max_phys_error_per_rollout` | `max_physical_tracking_error` |
| `context_init_xy_dist` | `context_info.init_xy_dist` |
| `context_box_init_xy` | `context_info.box_init_xy` |
| `context_target_xy` | `context_info.target_xy` |
| `context_box_angle_deg` | `context_info.box_init_angle_deg` |
| `context_target_angle_deg` | `context_info.target_angle_deg` |

Both modes produce identical output directory structure and PNG filenames.

---

## What does NOT change

- `multi_candidate_discovery.py` logic (seed-folder detection identical)
- `utils.py` (logging, output dir)
- Overall 5-phase pipeline
- Candidate letter assignment (A, B, C…)
- Manifest JSON pattern
- `00a`–`04b` flat file naming convention
- Slurm resource params
