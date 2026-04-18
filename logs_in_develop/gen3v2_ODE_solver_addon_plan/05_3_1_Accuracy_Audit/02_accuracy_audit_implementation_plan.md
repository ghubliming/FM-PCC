# 02. Implementation Plan: U4 Accuracy Tracker

This plan details the creation of the `benchmark_ode_accuracy_v3.py` script. To guarantee the absolute integrity of the test against our previous speed metrics, this architecture is governed by two strict principles.

## Core Principles

1.  **Test Accuracy, Not Speed**: The script's output matrix will exclusively measure the final trajectory's mathematical deviation from an "Oracle" ground-truth line, stripping away all latency profiling.
2.  **Import, Do Not Clone**: To guarantee we are testing the exact same Vector Field, safety bounds, and computational physics, the script will strictly `import` from the existing speed benchmark rather than duplicating code.

---

## Technical Design

### `benchmark_ode_accuracy_v3.py`
A lightweight wrapper script placed inside `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/`.

#### 1. Drop-In CLI Parity
We will duplicate the exact `argparse` configuration from `benchmark_ode_solvers_v3.py`.
*   Supported arguments: `--vf-mode`, `--mode`, `--dataset`, `--steps`, `--solver-spec`, `--device`
*   **Benefit**: You can invoke the accuracy benchmark using the exact same bash variables/commands from your Grid Search toolkit.

#### 2. The Oracle Phase Tracker
Before testing the candidate solvers, the script will establish the "Oracle Math Line."
*   It generates a static global noise tensor: `x_noise`.
*   It passes `x_noise` into the imported `p_sample_loop_v3_fair` function using `torchdiffeq:dopri5` evaluated at dense steps (e.g., $S=100$) with tight tolerances (`atol=1e-10`).
*   It captures the resulting terminal trajectory state as `Oracle_Tensor`.

#### 3. The Drift Output
The script will loop through the solvers specified in `--solver-spec` (e.g., `Euler`, `RK4`). 
*   It feeds the candidates the exact same `x_noise` tensor through the exact same imported solver loop.
*   It calculates two strict mathematical metrics of error against the oracle projection:
    1.  **MSE** (Mean Squared Error)
    2.  **L2 Norm** (Euclidean Distance / Drift Magnitude)
*   **Output**: It prints an ASCII summary to the terminal comparing the drift limits of Euler vs. RK4 and saves a `.json` matrix for later grid plotting.

---

## Verification Plan

1.  **Code Execution**: I will build the script and run a dry-test in the terminal.
2.  **Math Check**: The script should definitively prove that `RK4` drift is mathematically fractioned compared to the base `Euler` drift when run identically over $10$ steps. 

Please review and approve this formal plan to begin the coding phase!
