# FM v3 ODE-Selectable Data Analysis Tool

Comprehensive evaluation analysis script for Flow Matching v3 ODE-Selectable model across multiple projection variants, constraint types, and random seeds.

## Quick Start

### Basic Usage

```bash
# Analyze results from a specific directory
python main_da.py --input-path /path/to/eval/results

# With custom output location
python main_da.py --input-path /path/to/eval/results --output-path ./my_analysis

# Only analyze specific variants
python main_da.py --input-path /path/to/eval/results --variants dpcc-c,dpcc-r,dpcc-t,diffuser

# Only analyze specific seeds
python main_da.py --input-path /path/to/eval/results --seeds 6,7,8,9

# Skip expensive plot generation
python main_da.py --input-path /path/to/eval/results --no-plots

# Verbose output
python main_da.py --input-path /path/to/eval/results --verbose
```

### Example: FM-PCC Workspace

```bash
cd /workspaces/FM-PCC

# Analyze FM_v3_ode_selectable_test results
python Data_Analysis/DA_Code/main_da.py \
    --input-path FM_v3_ode_selectable_test \
    --output-path ./analysis_output

# Quick analysis (no plots)
python Data_Analysis/DA_Code/main_da.py \
    --input-path FM_v3_ode_selectable_test \
    --no-plots
```

## Output Structure

The script creates a timestamped output directory with the following structure:

```
20260512_143022_FM_V3_ODE_Analysis/
├── plots/
│   ├── 00_pareto_frontier_accuracy_vs_time.png    ← THESIS MAIN PLOT
│   ├── 01_variants_n_success.png
│   ├── 01_variants_n_success_and_constraints.png
│   ├── 02_constraints_*.png
│   ├── 03_heatmap_variant_constraint_*.png
│   ├── 04_efficiency_*.png
│   ├── 05_boxplot_seeds_*.png
│   └── [MORE PLOTS...]
│
├── results_summary.txt                              ← THESIS SUMMARY
├── results_by_variant.csv                           ← DETAILED VARIANT METRICS
├── results_by_constraint.csv
├── results_by_halfspace.csv
├── detailed_results.csv
│
└── logs/
    ├── analysis.log                                 ← EXECUTION LOG
    ├── data_loading.log                             ← WHICH FILES LOADED
    └── warnings.log
```

### `Latest_Snapshot` — when was this run produced?

Batch mode (`main_da_batch.py`) records, per candidate, the config-snapshot
markers the eval pipeline leaves behind: every launch writes
`snapshot_<YYYYMMDD_HHMMSS>` into `<candidate>/<seed>/config_snapshot_<config>/`
and never deletes the earlier ones (`utils/setup.py::snapshot_configs`).

| file | grain |
|---|---|
| `candidates_multidimensional_raw.csv` | that **seed's** newest marker (blank when that seed has none) |
| `candidates_multidimensional_aggregated.csv`, `candidates_ranking.csv` | newest marker over all the candidate's seeds |
| `candidates_detailed.csv` | plus `First_Snapshot`, `Snapshot_Count`, `Snapshot_By_Seed` |
| `candidates_summary.txt` | human-readable, with a per-seed line when the seeds disagree |

`Visualizer/index.html` renders it as a **Last Run** column in the Path Audit Map
and the Plot Legend, and repeats it in the exported audit `.txt` / LaTeX. Older
batches have no such column and the page drops it instead of showing blanks.

## Plot Legend: highlight + seed coverage (viewer only)

The **Plot Legend — Selected Candidates** table under the chart carries two
columns that exist only in the page, not in any CSV:

- **HL** — a checkbox. Ticking it paints that candidate's *name* red wherever it
  is printed: the plot's x tick label, the legend row, every row head in the
  Result Matrices, and the Path Audit Map. Nothing else changes — no number, no
  cell background — so a highlight can never be mistaken for a data annotation
  the way the `(goal, constraint)` flags can. Purely a way to follow one row
  across ~18 variant columns. `[CLEAR HIGHLIGHTS]` in the legend header resets
  them all, including candidates you have since unticked in the sidebar. The
  exported `.txt` and `.tex` have no colour, so they say `[HIGHLIGHTED]` /
  `<-- highlighted in the viewer` in words instead.
- **Seeds** — which seeds that candidate actually has, from
  `candidates_multidimensional_raw.csv`, with a red **⚠ NOT FULL** naming the
  ones it is missing (against the batch's full seed set, or against the ticked
  seeds in Custom Seed Compare). A bar averaged over one seed and a bar averaged
  over five are identical on the chart; this is the only place next to the plot
  that says which you are looking at. With no raw CSV the page falls back to the
  `Missing_Seeds` column and says `n/a` for the present-seed list rather than
  claiming the candidate has none.

## Key Metrics

The tool analyzes:

- **n_success**: Goal reached success rate (%)
- **n_success_and_constraints**: Goal + constraint satisfaction (%)
- **collision_free_completed**: Collision-free rate (%)
- **n_steps**: Planning steps (mean ± std)
- **avg_time**: Computation time (ms)
- **n_violations**: Constraint violations per trial
- **total_violations**: Cumulative violation magnitude

## Key Plots

### For Your Thesis:

1. **00_pareto_frontier_accuracy_vs_time.png** (MAIN)
   - Shows accuracy vs. time tradeoff
   - Color-coded: dpcc-c/r/t (red/orange/yellow), Baseline (blue)
   - Clearly shows which method is best overall

2. **01_variants_n_success_and_constraints.png**
   - Bar chart ranking all variants by goal + constraint success
   - Shows top performers and baselines

3. **results_summary.txt**
   - Human-readable ranking table
   - Top 10 variants by different metrics
   - Performance breakdowns by constraint type

## Architecture

### Modules

- **main_da.py**: Entry point with CLI interface
- **config.py**: Default parameters and plot styling
- **data_loader.py**: Loads .npz result files from directory tree
- **aggregator.py**: Aggregates results across seeds, computes statistics
- **visualizer.py**: Creates publication-quality plots
- **reporter.py**: Generates summary reports (txt and csv)
- **utils.py**: Logging, file operations, helpers

### Data Flow

```
Input .npz files (eval results)
         ↓
    DataLoader (organize by seed/variant/constraint)
         ↓
    DataAggregator (compute mean/std across seeds)
         ↓
    ┌────┴─────┬─────────┬─────────┐
    ↓          ↓         ↓         ↓
 Reporter  Visualizer Summary   Histogram
(txt/csv)  (plots)    (txt)      (csv)
```

## Requirements

```
numpy
pandas
matplotlib
pyyaml (optional, for config parsing)
scipy (optional, for statistical tests in future)
```

Install with:
```bash
pip install numpy pandas matplotlib pyyaml
```

## Configuration

default parameters in `config.py`:
- Seeds: [6, 7, 8, 9, 10]
- Variants: 18 projection methods (dpcc-c/r/t and others)
- Constraint types: halfspace, obstacles, dynamics, bounds
- Halfspace variants: top-right-hard, top-left-hard, both-hard

Modify `config.py` to change defaults, or use CLI flags to override.

## Tips

1. **First run**: Use `--no-plots` to quickly check data is loading correctly
2. **Quick analysis**: Analyze only main variants with `--variants dpcc-c,dpcc-r,dpcc-t,diffuser`
3. **Thesis prep**: Focus on `00_pareto_frontier_accuracy_vs_time.png` and `results_summary.txt`
4. **Debugging**: Check `logs/data_loading.log` if files aren't found

## Performance

- **Data Loading**: ~10-30 seconds (depends on SSD speed)
- **Aggregation**: ~2-5 seconds
- **Visualization**: ~30-60 seconds (10+ plots)
- **Total**: ~1-2 minutes for full analysis

## Citation

If you use this analysis tool, please cite:

```bibtex
@software{fm_v3_ode_analysis,
  title={FM v3 ODE-Selectable Data Analysis Tool},
  author={FM-PCC Research},
  year={2025}
}
```

---

**Last updated**: May 12, 2025
