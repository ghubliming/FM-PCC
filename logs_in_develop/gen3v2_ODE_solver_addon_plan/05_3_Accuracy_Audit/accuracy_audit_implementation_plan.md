# Implementation Plan: U2 Accuracy & Fidelity Audit

This plan establishes a rigorous methodology to quantify the "Drift" problem identified by your advisor. We will build a benchmarking suite that measures how much an ODE solver deviates from the mathematically "Exact" solution.

## User Review Required

> [!IMPORTANT]
> **The "Oracle" Definition**:
> We will use `torchdiffeq:dopri5` with an absolute and relative tolerance of **1e-12** as our Ground Truth (Oracle). 
> 
> **Key Metrics**:
> 1.  **Terminal L2 Drift**: The Euclidean distance between the Oracle and Candidate at $t=1.0$.
> 2.  **Step-wise Convergence**: How fast the error drops as we increase the number of steps ($S$).
> 3.  **Accuracy Efficiency**: A plot showing **Error vs. Latency**. This identifies the "Sweet Spot" (e.g., RK4 at $S=3$ may be more accurate than Euler at $S=20$ for the same time cost).

## Proposed Changes

### 1. Accuracy Audit Script [NEW]

#### [NEW] [benchmark_accuracy_audit_v1.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_accuracy_audit_v1.py)
A specialized script that focuses on trajectory comparison rather than just throughput.
*   **Initialization**: Fixes the environment and noise seed for 100% bit-identity comparison.
*   **Oracle Generation**: Runs a high-fidelity Dopri5 pass once.
*   **Sweep Engine**: Automatically iterates through solvers (Euler, Midpoint, RK4) and step counts (e.g., $S \in [2, 4, 8, 16, 32]$).
*   **Diffing Logic**: Calculates the L2 norm of the difference between the Oracle and the Candidate at the final time step.

### 2. Visualization Suite

#### [NEW] [plot_accuracy_results.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/plot_accuracy_results.py)
*   **Convergence Plot**: Number of Steps vs. Log-Error. (This proves the "Order" of the solver).
*   **Efficiency Plot**: Average Latency (ms) vs. Log-Error. (This is the most important plot for your advisor—it shows the best "Value for Money").

## Verification Plan

### Automated Tests
*   **Verify Zero Error**: Run the Oracle against itself. Result must be 0 (or machine epsilon).
*   **Verify Expected Slope**: Euler should show a drift that decreases linearly with $S$ ($O(h)$), while RK4 should show a much steeper drop ($O(h^4)$).

### Manual Verification
*   **Visual Proof**: Compare the generated "Accuracy Plot" with the image sent by your advisor. They should look identical in trend (Pink line far away, Olive line close).
