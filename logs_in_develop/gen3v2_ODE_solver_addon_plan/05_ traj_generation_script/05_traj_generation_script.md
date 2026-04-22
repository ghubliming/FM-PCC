# Trajectory Generation Script for V4 Benchmarks Plan

## Objective
Introduce a standalone script to generate trajectory visualizations from the V4 ODE solver benchmarks. The solution must guarantee that saving trajectory data does **not** interfere with the core latency benchmarking, avoiding any "exhaustive data logging" speed penalty.

## 1. Hotfix: `benchmark_ode_solvers_v4.py`
To avoid I/O bottlenecks and memory overhead from affecting the speed test:
- **Add Optional Flag**: Introduce `--datalog-for-traj` (defaults to False).
- **Zero-Interference Logging**: 
  - The trajectory capture will only trigger for a single trial (e.g., `trial == 0`).
  - To guarantee zero impact on timing, any `.cpu().numpy()` conversions and file I/O operations (`np.save`) will execute **strictly after** `time.perf_counter()` is stopped for that specific trial.
  - The trajectory tensor `x` (shape `[batch_size, horizon, transition_dim]`) will be saved to `os.path.join(out_dir, f"traj_{backend}_{method}.npy")`.

## 2. Standalone Trajectory Generation Script
- **Path**: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v4/traj_gen_script_for_v4.py`
- **Responsibilities**:
  1. **Load Data**: Accept a `--benchmark-dir` argument to locate the `summary.json` and `traj_{backend}_{method}.npy` files.
  2. **Unnormalize**: Load the real U-Net model and dataset to extract the `normalizer`. The saved trajectory data is in the model's normalized space, so it must be unnormalized using `normalizer.unnormalize(traj, 'observations')`.
  3. **Console Output**: Print the raw, unnormalized trajectory position coordinates (e.g., the X/Y parameters over the horizon) directly to the console for exhaustive parameter inspection.
  4. **Diffuser-Style Plotting (Direct Calling Existing Code)**: 
     - We will NOT write new plotting logic from scratch. 
     - We will directly import `flow_matcher_v3_ode_selectable.utils` and `config/projection_eval.yaml`.
     - We will directly call the real code: `utils.plot_environment_constraints` and `utils.plot_halfspace_constraints` to overlay exactly the same constraints as the eval script.
     - Plot start points (green circles) and the morphing trajectories (blue lines) for a subset of the batch.
     - Save the final plot as `trajectory_comparison.png` in the benchmark output folder.

## User Review Required
Please review this plan. Once you approve, I will proceed to write the hotfix for `benchmark_ode_solvers_v4.py` and create the new `traj_gen_script_for_v4.py` script.
