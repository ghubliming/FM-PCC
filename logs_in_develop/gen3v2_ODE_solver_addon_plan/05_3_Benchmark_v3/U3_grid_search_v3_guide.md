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

## 2) Theoretical Context: What does sweeping "Steps" actually test?

When you run a Grid Search over ODE steps (e.g., `1, 2, 5, 10, 20`), you are testing the **"Survival Threshold" (The Pareto Frontier)** of the robot. 

Think of integrating the Flow Matching model like driving a car along a curvy road at night. You can only see the road when you flash your headlights. The ODE integration is always $t=0$ to $t=1$.
*   **"ODE Steps" is how many times you are allowed to flash your headlights.**
*   **Euler** assumes the road is a perfectly straight line between flashes. If you only have 5 steps ($dt=0.2$), Euler will drive off the cliff.
*   **RK4** uses those 5 flashes to calculate the *curvature* of the road, steering smoothly between the checkpoints. It might survive.

**The Ultimate Goal of the Steps Sweep:**
You are searching for the minimum number of steps a method needs to prevent the robot from crashing (drifting off the neural manifold). If the grid search shows that RK4 survives at **5 steps** (saving rendering latency) while Euler requires **40 steps** to survive, RK4 is practically superior for real-time control, even if its mathematical formula is heavier.

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

### Scenario C: The "Survival Threshold" Sweep (Steps 1 to 20)
This sweep fixes the batch limits to an aggressive realtime requirement and deeply sweeps the Step budget to find the exact point where solvers begin to fail or become too slow.

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v3.py \
  --mode production \
  --grid-batch 4 \
  --grid-steps 1,2,5,10,15,20 \
  --grid-horizon 8 \
  --solver-spec legacy:euler,legacy:rk4 \
  --device cuda
```

---

## 3) Output Structure

The results are saved to `FM_v3_ode_selectable_test/benchmark_grid_search_v3/` by default.

1.  **Individual Folders**: Each combination (e.g., `mode_math_h8_b4_s10`) gets its own folder with detailed JSON trials and plots.
2.  **The Master Matrix**: `MASTER_MATRIX_V3_math.csv` contains every data point from every run in a single sheet. 

---

## 4) Troubleshooting the Grid Search

### ⚠️ The "Identical Latency" Paradox (SOLVED)
**Observed Phenomenon**: Early versions of the benchmark showed `legacy:rk4` taking the same time as `legacy:euler`.
**Root Cause**: Confirmed as a **Logic Bias** where the `p_sample_loop` was hardcoded to only run Euler. 
**Verification**: Verified in V3. You should now see `legacy:rk4` taking ~4.3x longer than `legacy:euler` in both `math` and `production` modes.

### Mismatched Backend Performance
In `math` mode, you may notice that `torchdiffeq` is significantly faster than `legacy` for high-step solvers (like RK4). 
*   **Why?**: `legacy:rk4` performs 40 sequential Python trips. `torchdiffeq:rk4` performs **one single Python trip** and handles the 40 evaluations internally in C++.
*   **The Lesson**: The **Python Dispatch Tax** is the real execution bottleneck of the `legacy` backend, separating the theoretical FLOPs from real-world latency.

### Aggregation Errors
If the grid search finishes but the CSV is empty, check the console for mapping errors. Standardize your `--solver-spec` to use the `backend:method` format for best results.

---

## 5) Recommendation for Reports
When presenting to your advisor, use the **Math Mode CSV** to prove you understand the underlying physics, and use the **Production Mode CSV** to show your final engineering decision (e.g., why Batch 64 RK4 is the "sweet spot" for robot speed).
