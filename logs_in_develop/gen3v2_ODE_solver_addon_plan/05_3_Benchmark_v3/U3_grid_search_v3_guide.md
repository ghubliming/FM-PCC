# Grid Search Benchmark V3 — Usage Guide

Date: 2026-04-16
Script: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v3.py`

---

## 1) Overview
The V3 Grid Search is an automation wrapper that loops through combinations of **Batch Size**, **Steps**, and **Horizon** while strictly enforcing the V3 "Fair Pathing" logic.

### Key Upgrade in V3:
It produces a consolidated **MASTER_MATRIX_V3_[mode].csv** file. 
*   Because the execution mode (`math` vs `production`) now dictates the fundamental speed, the grid search keeps these results separate. 
*   You should run the grid search twice (once for each mode) to get a complete performance profile.

---

## 2) How to Run

### Scenario A: Proving Mathematical Scalability Sweep
This sweep runs on the **Math Mode** to show how different solvers scale theoretically across all batch sizes.

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v3.py \
  --mode math \
  --grid-batch 4,32,128 \
  --grid-steps 10,20 \
  --grid-horizon 8,32 \
  --solver-spec legacy:euler,legacy:midpoint,legacy:rk4 \
  --device cuda
```

### Scenario B: Production Latency Evaluation Sweep
This sweep runs on the **Production Mode** to identify the best solver for the real robot pipeline.

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v3.py \
  --mode production \
  --grid-batch 4,16,64 \
  --grid-steps 10,20 \
  --grid-horizon 8 \
  --solver-spec legacy:euler,legacy:rk4,torchdiffeq:rk4 \
  --device cuda
```

---

## 3) Output Structure

The results are saved to `FM_v3_ode_selectable_test/benchmark_grid_search_v3/` by default.

1.  **Individual Folders**: Each combination (e.g., `mode_math_h8_b4_s10`) gets its own folder with detailed JSON trials and plots.
2.  **The Master Matrix**: `MASTER_MATRIX_V3_math.csv` contains every data point from every run in a single sheet. 

---

## 4) Troubleshooting the Grid Search

### Mismatched Solver Specs
If you pass a solver spec like `torchdiffeq:rk4` in `math` mode, it will run correctly, but remember that **Torchdiffeq is always "Integrated"**. It always calls the model via the `diffusion.py` logic. Therefore, in `math` mode, `legacy` solvers will look significantly faster than `torchdiffeq` because they are cutting corners that the library cannot cut.

### Aggregation Errors
If the grid search finishes but the CSV is empty, check the console for mapping errors. Standardize your `--solver-spec` to use the `backend:method` format for best results.

---

## 5) Recommendation for Reports
When presenting to your advisor, use the **Math Mode CSV** to prove you understand the underlying physics, and use the **Production Mode CSV** to show your final engineering decision (e.g., why Batch 64 RK4 is the "sweet spot" for robot speed).
