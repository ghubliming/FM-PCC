# Trajectory Generation Update - V4 Benchmark Code Changes Report

This document details the code modifications made during the implementation of the trajectory logging and visualization tools for the V4 ODE solver benchmarks.

## 1. Modified File: `benchmark_ode_solvers_v4.py`
**Location**: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v4/benchmark_ode_solvers_v4.py`

### Changes Made (Update 27 April):
- **Universal Noise Fairness**: Refactored the `n_trials` loop to ensure that **both** `math` and `production` modes use a shared noise basis for every solver in a trial. 
  - This ensures that Euler B0 and RK4 B0 are always competing on the **exact same random challenge**.
- **Trial Determinism**: Updated the script so that all trials within a single run use the **identical noise basis** (`global_x_init`).
  - This removes mathematical variance between trials, making the `--n-trials` loop strictly about **timing statistics** (averaging system jitter).
- **CLI Argument Addition**: Added `--datalog-for-traj` to control exhaustive logging of the final synthesized trajectories.
- **Production Safety Shield (Double Anchor)**: Implemented a mandatory Step 0 anchor that snaps both **Observations** and **Actions** (Waypoints) to the physical robot position (`cond[0]`).
- **Persistent Snapping**: Updated the ODE loop to re-apply this anchor after every integration step to prevent numerical "leakage" at t=0.
- **Strict Benchmarking Assertion**: Added a hard check that forces the script to `raise AssertionError` and abort if it detects any drift (> 1e-4) in the initial state during Production Mode.
- **Zero-Parity Conditioning**: Ensured that the conditioning tensor is exactly `zeros` when no random initialization is active, maintaining perfect baseline parity.

---

## 2. New File: `traj_gen_script_for_v4.py`
**Location**: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v4/traj_gen_script_for_v4.py`

### Changes Made (Update 27 April):
- **Raw Environment Restoration**: Removed the "Projection" obstacles (Blue halfspaces and circles) from the plotting logic.
  - The script now strictly plots the **Original Dataset Environment** (Red circles) using `utils.plot_environment_constraints()`.
  - This ensures the audit shows the robot's performance against the actual training environment, not a modified "test" environment.
- **Dynamic Comparison System**: Implemented a master comparison plot (`solver_comparison_all.png`) that automatically assigns high-contrast colors from `tab10` to every solver found in the directory.
- **Oracle Priority (Red)**: Added logic to specifically detect and assign the **Red** color to the Oracle solver (`dopri5`) for immediate visual identification.
- **Per-Batch Audit Plots**: Implemented a new feature that generates individual plots for every batch index (e.g., `batch_comparison_B0.png`). These plots show all solvers overlaying the **exact same noise vector**, removing visual clutter.
- **High-Fidelity Rendering**:
  - Upgraded resolution to **300 DPI**.
  - Added **SVG** output for infinite-zoom vector analysis.
  - Reduced trajectory line thickness (`linewidth=1.0`) for clearer auditing of tight gaps.
- **Robust Model Loading**: Added `--loadbase`, `--diffusion-loadpath`, and `--diffusion-seed` to handle cross-environment model loading (e.g., Google Colab).
- **Dimension Slicing (Success Fix)**: Corrected the slice to `[action_dim:]` and the normalizer key to `observations`. This correctly isolates the physical robot state and removes zigzag artifacts.
- **Normalized-Space Assertion**: Added a redundant safety check in the plotter that verifies the solver's start point against the ground truth in normalized space before generating plots.
- **Clean Environment Overlay**: Removed redundant manual obstacle drawing ("blue shitts") and switched to high-resolution `utils` based constraint plotting with corrected `zorder`.

---

## 3. Mission Support: Solver Accuracy Comparison
The framework now explicitly supports a "Fair Competition" mission where solvers are judged on:
1.  **Numerical Precision** (Math Mode): Raw drift from the start point.
2.  **Safety Stability** (Production Mode): Ability to follow a safe path while start-point-clamped.
3.  **Timing Performance**: Stable averages across trials due to deterministic math workloads.
