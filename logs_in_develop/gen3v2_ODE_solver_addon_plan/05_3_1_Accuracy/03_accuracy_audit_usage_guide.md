# U4 Accuracy Audit: Individual Solver Documentation

File: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v3/benchmark_ode_accuracy_v3.py`  
Date: 2026-04-18

This document serves as the technical guide for using the standalone accuracy auditor to quantify the mathematical fidelity of ODE solvers within the FM-PCC framework.

---

## 1. Technical Architecture: Statistical Batch Variance

Unlike the latency benchmarks (which measure timing), the accuracy auditor measures the **Euclidean Drift (L2 Distance)** between a candidate solver and a high-precision Ground Truth Oracle.

To ensure the results are scientifically robust without wasting compute time, we utilize **Batch-Wise Spatial Variance**:
*   **The Oracle**: Runs a single, high-precision `dopri5` integration (`atol=1e-10`) on a fixed batch of 128 noise samples.
*   **The Candidate**: Runs the same noise samples through the target solver (e.g., Euler 10-steps).
*   **The Logic**: We calculate the L2 distance for **every individual trajectory** in the batch. This allows us to report not just an average drift, but the **Standard Deviation ($\sigma$)**—showing how the solver's error varies across the Vector Field's spatial distribution.

---

## 2. Code Logic & Execution Flow

To ensure the benchmark is both "Fast" (one Oracle run) and "Reliable" (many candidate trials), the script follows this internal pipeline:

### A. The Golden Seed (Phase 1)
A single `global_noise` block is generated at the start of the script. This ensures that every solver starts at the exact same physical point in the latent space.

### B. The Oracle Baseline (Phase 2)
The script executes a **Single High-Precision "Perfect Run"**. 
It uses `torchdiffeq:dopri5` at 100 steps with ultra-tight numerical tolerances (`1e-10`). This generates the absolute "Ground Truth" target that all candidates must try to hit.

### C. The Candidate Sweep (Phase 3)
The script then loops through your `n-trials`.
*   **Math Mode**: Uses raw hand-rolled integration loops (Euler, RK4) to bypass library overhead and test the naked algorithm.
*   **Production Mode**: Uses the same `p_sample_loop_v3_fair` used in RL/Robot control.
*   **Monkey-Patching**: Since production models usually generate their own random noise, the script temporarily "monkey-patches" `torch.randn` during the run to FORCE the model to use our `global_noise`. This ensures the comparison remains 100% deterministic.

### D. Per-Item Statistical Analysis (Phase 4)
Instead of looking at the batch as a single block, the code flattens the result into 128 individual trajectories and runs `torch.linalg.vector_norm(..., dim=1)`. 
*   This yields 128 unique drift data points from a single run.
*   From this distribution, we calculate the **Mean** (the Bar height) and **Standard Deviation** (the Error Bars).

---

## 3. Command Line Usage

Execute this script to compare specific solvers on a fixed VF environment.

### Example: RK4 vs Euler Battle
```bash
!/content/miniconda3/envs/FMPCC/bin/python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v3/benchmark_ode_accuracy_v3.py \
  --mode math \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --device cuda \
  --batch-size 128 \
  --steps 10 \
  --solver-spec legacy:euler,legacy:rk4 \
  --plot
```

---

## 3. Interpreting the Results

The Y-Axis on your generated `accuracy_drift_plot.png` represents the **L2 Euclidean Distance** in normalized space.

### The Numerical Scale:
*   **0.00**: **Identity.** The solver matches the Oracle perfectly. This is the theoretical upper limit.
*   **0.01 (1%)**: **Scientific Grade.** The deviation is negligible. RK4 typically lands here.
*   **0.05 (5%)**: **Acceptable Drift.** Safe for slow-moving tasks, but trajectory "ghosting" starts to appear.
*   **0.10 (10%)**: **Critical Violation.** The solver's predicted path is significantly detached from the Neural Network's intention. Euler at low step counts often hits this boundary.
*   **> 0.10**: **Catastrophic Failure.** The integration error is so high that the robot is effectively "guessing" its path.

### The Error Bars (I-Beams):
*   The black bars on top of the orange columns represent the **Standard Deviation**.
*   **Short Bars**: The solver is consistently accurate across the entire workspace.
*   **Long Bars**: The solver struggles in specific "high-curvature" regions of the Vector Field, making it unpredictable.

---

## 4. Why this justifies RK4
By looking at the output, you will see that while Euler is faster, its bar is significantly taller. In robot control, a lower "Mathematical Drift" translates directly to **fewer constraint violations** and **smoother motion**, as the Safety Projector doesn't have to keep "snapping" the robot back to the manifold to correct for solver error.
