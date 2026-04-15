# ODE Solver "Gigantic Matrix" Grid Search

This document explains how to use the rewritten `grid_search_benchmark_for_v2.py` wrapper script. This script automatically tests all supported Torch ODE methods across varying configuration architectures, saving all the results in a gigantic matrix array for macro-analysis. 

## 1. Goal of the Script
Running one benchmark tells you the speed of the script at that specific configuration. Running the Grid Search iterates over **every combination** of your parameters using an `itertools.product` engine.

The newest update introduces automatic aggregation:
1.  **Macro Matrix**: Combines all result folders into a single `MASTER_MATRIX_RESULTS.csv`.
2.  **Console Leaderboard**: Prints the top 5 fastest configurations directly to the terminal for immediate check.
3.  **Headless Plotting**: Uses a non-GUI backend (`Agg`) to save `.png` files silently without popping up window alerts.
4.  **Trend Analysis**: Generates 3 specific line plots (`macroplot_batch_influence.png`, `macroplot_horizon_influence.png`, and `macroplot_steps_influence.png`) to visualize exactly how your latency scales against architectural choice.

---

## 2. Explanation of the Python Code

The script (`FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v2.py`) does the following:

1. **Loads Base Constants**: Connects to the real model path using `/logs/avoiding-d3il/...`.
2. **Defines Sweep Variables**: 
   - `--grid-horizon`: A comma-separated list of horizon/sequence sizes (e.g. `8,16,32`).
   - `--grid-batch`: A comma-separated list of batch sizes (default: `4,32,128`).
   - `--grid-steps`: A comma-separated list of ODE chunks/steps (default: `10,20`).
   - `--grid-bridge`: A flag. If set, runs pure GPU vs GPU+Bridge comparison.
3. **Mega Matrix Engine**: Iterates over all configurations. Stores all metrics in a giant list. Writes to CSV. Generates matplotlib line-plots isolating standard scaling dependencies.

---

## 3. How to use in Colab / Notebook Cell

You can run this perfectly inside a Jupyter Notebook cell using the `!` bang command. Notice how the solver-spec automatically includes the whole TorchDiffeq continuum.

```bash
!python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v2.py \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --n-trials 20 \
  --device cuda \
  --solver-spec legacy_euler,torchdiffeq:euler,torchdiffeq:midpoint,torchdiffeq:rk4,torchdiffeq:dopri5 \
  --grid-horizon 8 \
  --grid-batch 4 \
  --grid-steps 1,3,5,7,9,11,13,15,17,19 \
  --base-out FM_v3_ode_selectable_test/benchmark_outputs_v2/GridSearch_MetaAnalysis
```

### What to expect:
1. The script will boot up `benchmark_ode_solvers_v2.py` consecutively for every parameter combination.
2. After finishing, navigate to `GridSearch_MetaAnalysis/MASTER_MATRIX_RESULTS.csv`. You can open this in Pandas or Excel to see a gigantic spreadsheet mapping every architectural choice to its execution cost.
3. You will also see `.png` graphs illustrating the scalability directly in the output folder.

---

## 4) The Horizon Verdict: Why H=8 is the Standard

If you attempt to run the grid search with `--grid-horizon 2` or other small numbers, the script will crash. 

**The Structural Reason**:
Your Temporal U-Net uses 3 downsampling layers (`dim_mults=(1, 2, 4, 8)`). This means the temporal sequence is halved 3 times ($2^3 = 8$). Structurally, the `horizon` **must be a multiple of 8** ($8, 16, 24, \dots$) or the skip connections will fail to align, causing a shape mismatch error.

**The Performance Reason**:
Since your FM-PCC model was specifically trained for $H=8$, testing larger horizons ($16, 32$) is only useful for theoretical curiosity. For the **real-world performance audit**, $H=8$ is the only value that accurately represents the robot's production logic. Setting a higher horizon will artificially inflate the "Integration Step" cost beyond what the robot actually experiences.

**Recommendation**: Keep `--grid-horizon 8` as your constant for all official benchmarking.
