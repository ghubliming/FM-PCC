# Batch Consistency & Noise Refactor Roadmap

Date: 2026-04-18  
Goal: Standardize noise generation across the V3 benchmarking suite to achieve "Pure Algorithmic" metrics.

## 1. The Current Problem: Spatial Variance Contamination

Currently, our V3 benchmarks handle batch initialization (noise) inconsistently:

*   **Temporal Contamination**: In the Time Benchmark, `torch.randn` is called **inside** the trial loop. This means every trial is solving a different math problem, mixing "Hardware Jitter" with "Path Difficulty."
*   **Metric Inconsistency**: Because different scripts generate noise differently, we cannot perfectly compare a Time-Benchmark result for RK4 with an Accuracy-Audit result for the same RK4 (the paths don't match bit-for-bit).
*   **Scientific Target**: To be valid, a benchmark must isolate the variables. Time tests should measure **Cycles**, not "Path Luck."

---

## 2. Refactor Plan for the 4 Python Files

The goal is to move to a **Global Noise Basis** protocol.

### A. `benchmark_ode_solvers_v3.py` (Time/Latency)
*   **Fix**: Move noise generation **outside** the `for trial in range(args.n_trials)` loop.
*   **Result**: All trials solve the exact same 128 robot paths. The variance in the report will represent only the CPU/GPU jitter.

### B. `benchmark_ode_accuracy_v3.py` (Accuracy Audit)
*   **Fix**: Strengthen the noise injection wrapper to ensure the `production` model path exactly matches the `math` model path.
*   **Result**: 100% deterministic comparison between raw algorithms and production controllers.

### C. `grid_search_benchmark_for_v3.py` (Time Sweep)
*   **Fix**: Synchronize the seed-passing logic to ensure that every point in the grid uses a reproducible noise basis.

### D. `grid_search_accuracy_v3.py` (Accuracy Sweep)
*   **Fix**: Align the aggregation logic to the new `n_trials=1` standard and the unified noise baseline.

---

## 3. Final Verification
After this refactor, running any benchmark multiple times with the same seed will produce **bit-identical mathematical results**, allowing us to measure performance differences caused **only** by the solver algorithms themselves.
