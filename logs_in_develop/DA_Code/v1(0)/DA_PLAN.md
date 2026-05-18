# Data Analysis (DA) Script - Plan

## Overview
Create a unified Data Analysis Python script to process, aggregate, and visualize evaluation data from FM v3 ODE-Selectable tests across 5 seeds and multiple test configurations.

---

## Current Problem
- **Data Volume**: 834+ `.npz` evaluation result files
- **5 Seeds**: [6, 7, 8, 9, 10]
- **18 Projection Variants**: dpcc-r, dpcc-c, dpcc-t, diffuser, gradient, post_processing, model_free, + tightened versions, + dt variants
- **4 Constraint Types**: halfspace, obstacles, dynamics, bounds
- **3 Halfspace Variants**: top-right-hard, top-left-hard, both-hard
- **Hard to Visualize**: No centralized analysis tool exists; current `load_results_*.py` scripts are one-off and limited in scope

---

## Objectives

### 1. **Unified Input Interface**
- CLI argument: folder path (or use default results directory)
- Auto-discover all `.npz` files in nested structure
- Support both single constraint type and cross-constraint analysis
- Handle missing/corrupted files gracefully

### 2. **Data Aggregation**
- Load and aggregate results across all seeds
- Compute statistics (mean, std, min, max) for each metric
- Preserve per-seed data for error bars
- Group results by:
  - **By Variant**: Compare projection methods
  - **By Constraint Type**: Compare constraint handling
  - **By Halfspace Variant**: Compare different constraint geometries
  - **Cross-tab**: Projection × Constraint × Halfspace

### 3. **Metrics to Extract & Compute**
From `.npz` files, each containing:
- `n_success` → Success rate (goal reached as %)
- `n_success_and_constraints` → Goal + constraint satisfaction (%)
- `n_steps` → Planning steps (mean ± std)
- `n_violations` → Constraint violations per trial
- `total_violations` → Cumulative violation magnitude
- `avg_time` → Computation time (mean ± std)
- `collision_free_completed` → Collision-free rate (%)

**Derived metrics**:
- Constraint Success Rate = n_success_and_constraints / total_trials
- Efficiency = success_rate / avg_time
- Robustness = constraint_success_rate across all variants

### 4. **Output Structure**

```
output_folder/
├── plots/
│   ├── 01_variants_comparison_success_rate.png       # Bar chart: all variants, goal SR
│   ├── 02_variants_comparison_constraint_success.png # Bar chart: all variants, constraint SR
│   ├── 03_variants_comparison_steps.png              # Bar chart: steps (mean ± std)
│   ├── 04_constraint_types_comparison.png            # Group bar: halfspace/obstacles/dynamics/bounds
│   ├── 05_variant_vs_constraint_heatmap.png          # Heatmap: variant × constraint success rate
│   ├── 06_efficiency_plot.png                        # Scatter: success_rate vs avg_time
│   ├── 07_success_vs_violations_scatter.png          # Scatter: success_rate vs n_violations
│   ├── 08_seed_variability_boxplot.png               # Boxplot: per-seed variation by variant
│   ├── 09_cumulative_violations_heatmap.png          # Heatmap: variant × constraint total_violations
│   └── 10_time_comparison_barplot.png                # Bar chart: computation time comparison
│
├── results_summary.txt                               # Human-readable summary (markdown format)
├── results_summary.csv                               # Tabular format (can import to Excel)
├── results_by_variant.csv                            # Detailed: variant, constraint, metric, seed, value
├── results_by_constraint.csv                         # Grouped by constraint type
└── logs/
    ├── data_loading.log                              # Which files found/missing
    ├── analysis.log                                  # Processing steps, errors
    └── warnings.log                                  # Data quality warnings
```

### 5. **Visualization Types**

| Plot | Purpose | X-Axis | Y-Axis | Color/Group |
|------|---------|--------|--------|-------------|
| Bar Chart | Variant ranking | Projection Variant | Success % | By Constraint Type |
| Heatmap | Variant vs Constraint | Variant | Constraint Type | Success % (color) |
| Boxplot | Seed variability | Variant | Metric (steps/violations) | By Seed |
| Scatter | Efficiency frontier | Time (ms) | Success Rate | Color by Variant |
| Line Plot | Multi-seed trends | Variant | Metric Value | Line per seed |
| Cumulative | Violations across seeds | Variant | Total Violations | Error bars |

---

## Implementation Details

### 6. **Python Structure** (`Data_Analysis/DA_Code/`)

**Files to create**:
1. **`main_da.py`** (Entry point)
   - Parse CLI arguments (input path, output path, config, plot options)
   - Coordinate data loading, aggregation, and analysis
   - Call viz and reporting modules

2. **`data_loader.py`**
   - `load_all_npz_files(root_path, constraint_types, variants, seeds)`
   - `parse_directory_structure()` → auto-discover seeds/variants
   - Error handling for missing files
   - Logging of loading process

3. **`aggregator.py`**
   - `aggregate_by_variant(data_dict)` → DataFrames
   - `aggregate_by_constraint_type(data_dict)` → DataFrames
   - `aggregate_by_halfspace_variant(data_dict)` → DataFrames
   - `compute_statistics(array)` → mean, std, min, max, median
   - `create_comparison_tables()`

4. **`visualizer.py`**
   - `plot_variant_comparison(df, metric, save_path)`
   - `plot_constraint_comparison(df, metric, save_path)`
   - `plot_heatmap_variant_vs_constraint(df, save_path)`
   - `plot_efficiency_scatter(df, save_path)`
   - `plot_boxplot_seed_variability(df, save_path)`
   - Consistent styling (colors, fonts, DPI=300)

5. **`reporter.py`**
   - `generate_summary_txt(data_dict, output_path)`
   - `generate_summary_csv(data_dict, output_path)`
   - `generate_detailed_csv(data_dict, output_path)`
   - Markdown formatting for txt output

6. **`config.py`**
   - Default paths (seeds, variants, constraint types)
   - Plot styling constants (colors, fonts, figsize)
   - Output folder naming convention

7. **`utils.py`**
   - Logger setup
   - File path utilities
   - Data validation

### 7. **CLI Interface**

```bash
python Data_Analysis/DA_Code/main_da.py \
    --input-path /path/to/eval/results \
    --output-path /path/to/output \
    [--seeds 6,7,8,9,10] \
    [--variants dpcc-c,dpcc-r,diffuser] \
    [--constraint-types halfspace,obstacles] \
    [--plot-all] \
    [--verbose]
```

**Default behavior**: Auto-discover all variants/seeds, generate all plots

### 8. **Error Handling**
- Missing `.npz` files → log warning, skip, continue
- Corrupted `.npz` → catch exception, log error, skip
- Empty results → warn but don't fail
- NaN values → replace with 0 or skip, log
- Output folder creation → auto-create with unique suffix if exists

### 9. **Output Naming Convention**
```
YYYYMMDD_HHMMSS_FM_V3_ODE_Analysis_[CONSTRAINT_TYPE]/
```
Example: `20260512_143022_FM_V3_ODE_Analysis_Halfspace/`

---

## Development Phases

### Phase 1: Core Infrastructure ✓ (Plan)
- [x] Understand data structure and format
- [x] Create plan document

### Phase 2: Minimal DA Tool
- [ ] Implement `data_loader.py` (load `.npz` files)
- [ ] Implement basic aggregation (`aggregator.py`)
- [ ] Create simple summary output (txt + csv)
- [ ] Test with small dataset

### Phase 3: Visualization
- [ ] Implement `visualizer.py` with 3-4 key plots
- [ ] Add styling and legends
- [ ] Test plot generation

### Phase 4: Polish
- [ ] Add error handling and logging
- [ ] CLI argument parsing (`main_da.py`)
- [ ] Documentation and comments
- [ ] Test with full dataset

### Phase 5: Optional Enhancements
- [ ] Interactive plots (plotly)
- [ ] Statistical significance testing (t-tests)
- [ ] Filtering/subsetting by metric threshold
- [ ] Comparison with baselines
- [ ] Per-seed trajectory visualization

---

## Key Design Decisions

1. **Format**: Pure NumPy + Pandas + Matplotlib (no heavy dependencies)
2. **Organization**: Modular structure (data, aggregation, visualization separate)
3. **Defaults**: Auto-discover → users don't need to enumerate everything
4. **Logging**: Verbose logging to help debug missing data
5. **Output**: Both human-readable (txt/plots) and machine-readable (csv)
6. **Robustness**: Skip missing files rather than fail entirely

---

## Expected Output Examples

### Summary Text (results_summary.txt)
```
=== FM v3 ODE-Selectable Evaluation Analysis ===
Date: 2025-05-12
Results Path: /path/to/results

Seeds Analyzed: [6, 7, 8, 9, 10]
Projection Variants: 18
Constraint Types: 4 (halfspace, obstacles, dynamics, bounds)
Halfspace Variants: 3 (top-right-hard, top-left-hard, both-hard)

--- OVERALL TOP 5 ---
1. dpcc-c-tightened           | SR: 0.92 | Time: 42.3ms | Violations: 0.12
2. dpcc-r-tightened           | SR: 0.89 | Time: 38.1ms | Violations: 0.18
3. dpcc-t-tightened           | SR: 0.87 | Time: 45.2ms | Violations: 0.21
...

--- BY CONSTRAINT TYPE ---
Halfspace:  Avg SR=0.85 | Best=dpcc-c (0.92)
Obstacles:  Avg SR=0.78 | Best=dpcc-c (0.89)
Dynamics:   Avg SR=0.72 | Best=dpcc-r (0.81)
Bounds:     Avg SR=0.88 | Best=diffuser (0.91)
```

### CSV Output (results_by_variant.csv)
```
variant,constraint_type,halfspace_variant,seed,n_success,n_success_and_constraints,n_steps_mean,n_violations,total_violations,avg_time
dpcc-c,halfspace,top-right-hard,6,92,88,12.3,0.08,0.045,41.2
dpcc-c,halfspace,top-right-hard,7,94,90,11.8,0.06,0.038,40.8
...
```

---

## Thesis-Focused Analysis

### Accuracy vs. Time Frontier
Create a **Pareto frontier plot** showing the accuracy-time tradeoff:
- **X-axis**: Computation time (avg_time in ms) — lower is better
- **Y-axis**: Goal + Constraint Success Rate (n_success_and_constraints, %) — higher is better
- **Color-coding**:
  - 🔴 **Red**: dpcc-c variants (main entry)
  - 🟠 **Orange**: dpcc-r variants (main entry)
  - 🟡 **Yellow**: dpcc-t variants (main entry)
  - 🔵 **Blue**: Baseline/ML methods (diffuser, gradient, etc.)
  - 🟢 **Green**: Other variants

**Key Output Table**:
```
Rank | Method           | Accuracy (%) | Time (ms) | Status
-----|------------------|--------------|-----------|--------
  1. | dpcc-c-tightened | 92.5 ± 2.1   | 42.3      | ✓ BEST
  2. | dpcc-r-tightened | 89.8 ± 2.5   | 38.1      | ✓ FAST
  3. | diffuser         | 78.2 ± 3.1   | 35.0      | ★ BASELINE
  4. | dpcc-t-tightened | 87.1 ± 2.8   | 45.2      | ✓ TRADE-OFF
```

**Analysis includes**:
- Which variant dominates (highest accuracy at acceptable time)
- Which variant is fastest (useful for real-time applications)
- How baseline diffuser compares (ML-only approach)
- Constraint success vs. goal-only success comparison
- Per-constraint analysis (halfspace/obstacles/dynamics/bounds)

---

## Success Criteria
✅ Script runs without errors on full dataset  
✅ All 834 `.npz` files processed (or logged if missing)  
✅ 10+ comparison plots generated (including Pareto frontier)  
✅ Summary CSV parseable in Excel  
✅ Results clearly show relative performance across variants  
✅ Seed-to-seed variance visible in output (error bars)  
✅ Execution time < 2 minutes for full analysis  
✅ **NEW**: Thesis-ready ranking table (accuracy vs. time)  
✅ **NEW**: Primary variants (dpcc-c/r/t) highlighted  
✅ **NEW**: Baseline comparison (diffuser as ML reference)  

---

## Dependencies
```
numpy
pandas
matplotlib
pyyaml  (if reading config)
scipy   (for statistical tests, optional phase 5)
```

