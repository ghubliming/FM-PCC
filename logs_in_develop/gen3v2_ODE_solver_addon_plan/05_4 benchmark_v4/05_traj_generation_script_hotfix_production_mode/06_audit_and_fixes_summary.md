# Technical Audit & Rebuild Summary: Trajectory Generation (V4)

This document provides a final verdict on the updates made during the rebuild of the "Abandoned" V4 benchmark pipeline.

## 1. File 1: Benchmark Audit
**Comparison**: `old_main_ben_v4.py` $\rightarrow$ `benchmark_ode_solvers_v4.py`

| Change Component | Old Code Status | New Code Verdict | Reasonableness |
| :--- | :--- | :--- | :--- |
| **Manual Normalization** | **WRONG** (Passed raw 0.6) | **CORRECT** (Added `.normalize()`) | **CRITICAL**: Model cannot interpret raw meters; it requires normalized [-1, 1] space. |
| **Legacy Solver Snap** | **MISSING** in `math` loop | **ADDED** for `production` mode | **CORRECT**: Ensures Euler/RK4 solvers obey the same Step 0 anchoring as the main loop. |
| **Hybrid Sampling** | **MISSING** (Zeros only) | **ADDED** (Real-state pull) | **CORRECT**: Allows testing on realistic robot positions instead of just origin/noise. |
| **Dopri5 coefficients** | Contained typos (Line 328) | **FIXED** and standardized | **CORRECT**: Ensures numerical parity with `torchdiffeq`. |

## 2. File 2: Plotter Audit
**Comparison**: `old_traj_gen.py` $\rightarrow$ `traj_gen_script_for_v4.py`

| Change Component | Old Code Status | New Code Verdict | Reasonableness |
| :--- | :--- | :--- | :--- |
| **Dimension Slicing** | **WRONG** (`[:obs_dim]`) | **CORRECT** (`[action_dim:]`) | **MANDATORY**: Old code plotted Velocities as Positions. New code correctly isolates Physical State. |
| **Start Verification** | **ESTIMATED** (`traj[0]`) | **GROUND TRUTH** (`cond_true`) | **CORRECT**: Uses `cond_true_start.npy` to prove the model actually snapped to the Yellow Star. |
| **DGM Evolution** | Supported 3D only | Supports 4D (`datalog` mode) | **REASONABLE**: Allows auditing the frame-by-frame refinement of the entire plan. |
| **Batch Support** | Batch=1 Assumption | `n_init_points` Support | **CORRECT**: Required to handle the new multi-point "Hybrid" benchmark data. |

## 3. Core Verdict: The "Safety Shield" Audit
The audit of `diffusion.py` confirms the following ground-truth logic:
*   **Step 0 (Observation)**: Snapped 10/10 times (Every ODE iteration). **VERDICT: CORRECT.**
*   **Steps 1-7 (Actions/States)**: Floating and predicted by Vector Field. **VERDICT: CORRECT.**
*   **Obstacle Projection**: Applied only in the "Last Half" for stability. **VERDICT: CORRECT.**

**Conclusion**: The rebuild has transformed a "weird" and mathematically corrupted plotter into a precise audit tool. The "Jumps" seen in yesterday's plots were actually **corrupted unnormalization** caused by slicing the wrong dimensions.