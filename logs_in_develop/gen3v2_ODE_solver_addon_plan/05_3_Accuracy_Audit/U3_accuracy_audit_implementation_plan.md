# U3: Accuracy & Fidelity Audit Implementation Plan

This plan establishes a rigorous methodology to quantify the "Drift" problem identified by your advisor. We will build a benchmarking suite that measures how much an ODE solver deviates from the mathematically "Exact" solution.

## 1. Methodology: The Standalone Math Audit

To prove accuracy to your advisor, we will use a **Standalone Math Benchmark** that isolates the ODE solver logic from any simulation noise.

### The 3-Step Process:
1.  **Oracle Generation**: We take a specific start point ($x_0$) and run the **Neural Vector Field** through `torchdiffeq:dopri5` with ultra-low tolerances ($10^{-12}$). This is our "Exact Solution."
2.  **Candidate Simulation**: We run the same $x_0$ through **Legacy Euler** ($S=10$) and **Legacy RK4** ($S=10$).
3.  **Drift Quantification**: We measure the Euclidean distance (L2 norm) between the Oracle trajectory and the Candidate trajectories at every time step $t$.

> [!IMPORTANT]
> **Why Standalone?**: By loading the Model Weights but bypassing the Simulator, we prove that any error seen is **100% caused by the numerical integrator**. This provides the "mathematical proof" required for advisor verification.

### Key Metrics:
1.  **Terminal L2 Drift**: The Euclidean distance between the Oracle and Candidate at $t=1.0$.
2.  **Step-wise Convergence**: How fast the error drops as we increase the number of steps ($S$).
3.  **Accuracy Efficiency**: A plot showing **Error vs. Latency**. This identifies the "Sweet Spot" (e.g., RK4 at $S=3$ may be more accurate than Euler at $S=20$ for the same time cost).

## 2. Implementation [NEW]

#### [NEW] [benchmark_accuracy_v3.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_accuracy_v3.py)
A specialized script that focuses on trajectory fidelity.
*   **Loading**: Loads real model checkpoints (e.g., H8_K20).
*   **Storage**: Unlike the throughput script, this stores the **full state tensor** at each step to allow for line-plotting.
*   **Sweep Engine**: Iterates through solvers ($Euler, Midpoint, RK4$) and step counts ($S \in [4, 8, 16, 32]$).
*   **Visualizer**: Automatically generates the "Advisory Plot" (Multiple colored lines showing the drift over time).

## 3. Verification Plan

### Automated Tests
*   **Verify Zero Error**: Run the Oracle against itself. Result must be 0 (or machine epsilon).
*   **Verify Expected Slope**: Euler should show a drift that decreases linearly with $S$ ($O(h)$), while RK4 should show a much steeper drop ($O(h^4)$).

### Manual Verification
*   **Visual Proof**: Compare the generated "Accuracy Plot" with the image sent by your advisor. They should look identical in trend (Pink line far away, Olive line close).
