# 05.4: Plan for Standardizing Batch Noise Generation

Date: 2026-04-18  
Status: **Proposed Refactor**

## 1. The Current Problem: Spatial Noise Leakage

In our current V3 benchmarking suite, we have a fundamental inconsistency in how we initialize robot paths (the batch):

*   **Time Benchmarks (`benchmark_ode_solvers_v3.py`)**: These currently generate **new random noise** inside the trial loop. 
    *   *Effect*: Trial 1 and Trial 2 are solving different math problems.
    *   *Result*: The timing stats are "polluted" by spatial variance; if one path is naturally harder for the neural network, the latency result is not a pure measure of the solver.
*   **Accuracy Audits (`benchmark_ode_accuracy_v3.py`)**: These use a **global noise basis** but only for candidate comparison. 
    *   *Effect*: While more stable, it lacks a unified protocol for cross-script noise injection.

---

## 2. The Refactor Goal: "Pure Algorithmic Benchmarking"

To fix this, we must ensure that every single benchmark (Time or Accuracy) satisfies these three conditions:
1.  **Spatial Consistency**: Every item in the 128-batch follows a unique path (to test robustness).
2.  **Temporal Consistency**: Every *Trial* (1 to 50) uses the **exact same** 128 paths.
3.  **Solver Fairness**: Every solver (Euler, RK4, etc.) must solve the **identical** 128 paths.

---

## 3. Implementation Plan for 4 Files

### A. `benchmark_ode_solvers_v3.py` (Time)
*   **Change**: Move `torch.randn` outside the `for trial in range(n_trials)` loop.
*   **Action**: Define `x_init = 0.5 * torch.randn(...)` once at the start of the backend execution.
*   **Benefit**: Eliminates "Path Jitter" from the speed statistics.

### B. `benchmark_ode_accuracy_v3.py` (Accuracy)
*   **Change**: Formalize the `global_noise` injection as a standard wrapper.
*   **Action**: Ensure that even `production` modes (which usually generate their own noise) are forced to use the pre-generated batch basis.

### C. `grid_search_benchmark_for_v3.py` (Time Grid)
*   **Change**: Pass a fixed `--seed` to all subprocesses.
*   **Action**: Ensure that every batch/horizon configuration in the sweep is reproducible and isolated from noise randomness.

### D. `grid_search_accuracy_v3.py` (Accuracy Grid)
*   **Change**: Standardize the `--n-trials 1` and noise baseline protocol.
*   **Action**: Align logic with the new unified noise generation in the individual audit.

---

## 4. Verification of Success
*   **Zero-Variance Math**: If we run Trial 1 and Trial 50 of an accuracy test, the L2 drift should be **exactly the same** to 8 decimal places.
*   **Pure Speed Stats**: In the time benchmark, the `std_ms` should drop to representing only hardware interrupt/jitter, with zero contribution from "easy/hard" paths.

---

> [!IMPORTANT]
> **Conclusion for Today**: 
> The current V3 code is "Mathematically Unreliable" because of loop-internal noise generation. The above plan will be executed in the next session to ensure V3 provides "Clean Room" benchmarking data for management.
