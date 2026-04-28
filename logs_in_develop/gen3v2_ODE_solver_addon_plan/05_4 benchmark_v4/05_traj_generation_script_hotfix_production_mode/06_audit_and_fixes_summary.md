# Technical Audit & Rebuild Summary: Trajectory Generation (V4)

This document summarizes the audit of the "Abandoned" V4 benchmark code and the fixes implemented during the rebuild.

## 1. Confirmed Mathematical Bugs (Fixed)

### A. Plotter Dimension Corruption (`traj_gen_script_for_v4.py`)
*   **Yesterday (WRONG):** The script used `[:obs_dim]` to slice the trajectory. Because the tensor is structured as `[Action, Observation]`, this accidentally mixed Velocities and Positions together. Unnormalizing this "mixed" slice created corrupted coordinates.
*   **Today (FIXED):** We implemented `action_dim` offset detection. The script now correctly slices `[action_dim : action_dim + 2]` to isolate the physical $(Rx, Ry)$ coordinates for plotting.

### B. Normalization Mismatch (`benchmark_ode_solvers_v4.py`)
*   **Yesterday (WRONG):** Manually entered physical points (e.g., `0.6`) were passed directly to the model. Since the model was trained on normalized data (range ~[-1, 1]), passing a raw `0.6` caused the model to behave as if the robot was 100 meters away.
*   **Today (FIXED):** All manual inputs are now passed through `normalizer.normalize()` before the ODE integration begins.

## 2. Confirmed Correct Logic (Verified via Git)

### A. Projective Conditioning (The Snap)
*   **Audit Result (CORRECT):** Yesterday's code in both `diffusion.py` and the `p_sample_loop_v4_fair` benchmark function was **already correctly** snapping Step 0 at every ODE iteration.
*   **Status:** No change to core model logic was required. We only updated legacy manual loops (Euler/RK4) in the benchmark script to match this already-correct behavior.

## 3. New Features & Audit Capabilities

### A. Real Coordinate Sampling
*   Added the "Hybrid Method" to pull real start points from the training dataset. This allows auditing the model on "Hard" real-world cases instead of just random Gaussian points.

### B. Multi-Initialization Support
*   The plotter can now process a batch of trajectories at once, saving individual plots (`_init0.png`, `_init1.png`, etc.) for large-scale auditing.

## 4. Workload Definitions
*   **1 MPC Step** = **10 ODE "Brain" Iterations**.
*   **Result** = **1 Full 8-step physical plan**.
*   **Anchor**: Step 0 is snapped to reality 10 times during the Brain phase.

(What are wrong, what are right?)