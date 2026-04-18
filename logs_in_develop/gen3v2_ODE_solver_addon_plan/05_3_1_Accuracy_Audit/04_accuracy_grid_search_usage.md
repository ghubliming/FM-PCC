# U4 Accuracy Grid Search: Automated Macro Analysis

File: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_accuracy_v3.py`  
Date: 2026-04-18

This document outlines the usage of the aggregated accuracy macro-benchmark runner. This script mirrors the automated grid search you used for latency testing, allowing you to sweep over multiple step counts, batch sizes, and horizons to definitively mathematically prove solver drift properties.

---

## 1. What This Script Does

Instead of running a single trial, `grid_search_accuracy_v3.py` acts as a master shell. It will:
1. Cross-multiply your `--grid-batch`, `--grid-steps`, and `--grid-horizon` arrays.
2. Spin up independent `benchmark_ode_accuracy_v3` processes for each combination.
3. Harvest all standalone `accuracy_summary.json` outputs.
4. Export a beautiful `MASTER_ACCURACY_MATRIX_V3_math.csv`.
5. Output a console ranking of the **Top 5 Most Accurate Configurations** (Lowest L2 Drift).
6. Automatically run `matplotlib` across all variables to print unified trendlines, utilizing a **Log Scale** on the y-axis to correctly highlight exponential differences ($O(h)$ vs $O(h^4)$).

---

## 2. Default Configuration

If run with no arguments, the solver executes the following multi-dimensional sweep targeting your `avoiding-d3il` dataset:
*   **Target Metrics**: `legacy:euler`, `legacy:rk4`
*   **Step Sweep**: `5, 10, 20`
*   **Batch Sweep**: `4, 32`
*   **Horizon Sweep**: `8, 16`
*(Total permutations: 12 independent test branches)*

---

## 3. Usage & Execution

Run this inside your Jupyter cell using the existing configured Conda environment:

### Standard Execution Flow
```bash
!/content/miniconda3/envs/FMPCC/bin/python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_accuracy_v3.py \
  --mode math \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --device cuda \
  --solver-spec legacy:euler,legacy:rk4 \
  --grid-batch 4,32 \
  --grid-steps 5,10,20 \
  --grid-horizon 8
```

---

## 4. Understanding the Macro Plots

When the sweep finishes aggregating data, it renders specific visual plots to `benchmark_accuracy_grid_v3/`:

### A. `macroplot_ACCURACY_vs_STEPS_v3_math.png`
This is your most important "Proof to the Advisor" chart. 
*   **X-Axis**: Integration Steps
*   **Y-Axis**: L2 Drift from Oracle (Log Scale)
*   **Expectation**: Euler's curve should drop shallowly (linearly) as steps increase. RK4's curve should crash sharply downward, achieving maximum accuracy at low step counts.

### B. `macroplot_ACCURACY_vs_HORIZON_v3_math.png`
*   **X-Axis**: Horizon length ($H$)
*   **Y-Axis**: L2 Drift
*   **Expectation**: Because diffusion trajectories are intrinsically tied to output vector length, higher horizons create physically longer integration paths, naturally amplifying the drift. RK4's advantage will look significantly larger here than at small horizons.
