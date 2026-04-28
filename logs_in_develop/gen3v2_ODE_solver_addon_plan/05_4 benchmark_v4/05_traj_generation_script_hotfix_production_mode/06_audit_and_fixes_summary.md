# Final Audit: What We Changed Today (Timeline of Today's Chaos)

This document is a precise record of all changes made to the V4 benchmark codebase today, comparing the current state to the **original commit from yesterday**.

## 1. `benchmark_ode_solvers_v4.py` (The Engine)

**A. Added the Multi-Init Point System (Brand New Today)**
*   **Yesterday:** The script was hardcoded to run exactly 1 initialization point at `(0,0)`.
*   **Today:** We added `--n-init-points`, `--randomize-cond`, `--init-x-range`, and `--init-y-range`.
*   **The Change:** We added a new logic block (lines 292-313) that handles the creation of multiple distinct starting points.

**B. Real Coordinate Sampling (The Hybrid Method)**
*   **Yesterday:** Did not exist.
*   **Today:** We added code to dive into `fm_exp.dataset.dataset['observations']` to pull real starting states from the training history. We then clip these to your requested range using `np.clip` on indices 2 and 3 (the robot's physical position).

**C. The Normalization Critical Fix**
*   **Yesterday:** The script used `torch.zeros()` (which is already normalized).
*   **Today:** When we added random physical points (like `0.6`), we initially forgot to normalize them. We fixed this by adding `fm_exp.dataset.normalizer.normalize()` before passing them to the model.

**D. Loop-Level Projective Conditioning**
*   **Yesterday:** Production mode didn't forcefully anchor the start point at every step of the ODE.
*   **Today:** We added `apply_conditioning` at the start of the sample loop AND inside the legacy integration steps (Euler, RK4, etc.) to ensure the "Safety Shield" is always active in Production mode.

## 2. `traj_gen_script_for_v4.py` (The Plotter)

**A. The Dimension Slicing Fix (The "Ghost" Bug)**
*   **Yesterday:** The script was hardcoded to take `[:obs_dim]`. Since your state is `[Action, Observation]`, this was accidentally plotting the robot's **Actions (Vx, Vy)** as its position.
*   **Today:** We added `action_dim` detection and changed the slice to `[action_dim : action_dim + obs_dim]`. The Green Dot now correctly tracks physical X/Y.

**B. Support for Multi-Init Plots**
*   **Yesterday:** The plotter only expected one set of trajectories.
*   **Today:** We added a master loop that iterates through all `n_init_points` and saves individual audit plots for each one (`_init0.png`, `_init1.png`, etc.).

**C. Coordination Debugging**
*   We added console prints for the raw unnormalized coordinates of both the Green Dot and the Yellow Star so you can verify they are identical in the data before plotting.

## 3. The "Conditioning Snap" Explained
If you see a jump from Step 0 to Step 1, it is because we are only anchoring **Step 0**. The rest of the plan is generated from random noise. The "snap" is the model trying to bridge the gap between our forced anchor and its imagined path.
