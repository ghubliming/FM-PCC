# Deep Dive: Math vs. Production Logic in FM-PCC

This document summarizes the design rationale for the two operational modes in the V4 ODE Solver Benchmark and how they relate to the real FM-PCC evaluation pipeline.

---

## 1. The Trajectory as a Single "State"
In FM-PCC, the ODE solver does not move a single point; it moves an **entire 8-step trajectory** at once. 
*   **The State ($X$)**: $[p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7]$
*   **The Transformation**: The ODE solver flows this whole "string of points" from a random noisy shape into a clean, smooth path.

---

## 2. Math Mode vs. Production Mode (The Start Point)

### Math Mode (The "Naked Truth" Audit)
*   **Starting State**: 100% Random Noise. Every point ($p_0$ to $p_7$) is a random number.
*   **No Anchoring**: We do **not** force the start point to be the ground truth.
*   **The Goal**: We want to see if the Neural Network is smart enough to "pull" that random noise down to the correct starting location by itself.
*   **What you see**: The start point "drifts" or starts in a different place for every batch. This is an **Audit Feature**—it shows you the model's internal error.

### Production Mode (The "Safety Shield" Controller)
*   **Starting State**: Random Noise, but with a **Forced Anchor**.
*   **Anchoring**: Before we start, and after **every** ODE step, we manually overwrite $p_0$ with the Ground Truth sensor reading.
*   **The Goal**: Ensure the robot's plan is always physically attached to the robot's current location.
*   **What you see**: All batches and all solvers start at the **exact same coordinate**. The error is "hidden" for the sake of safety.

---

## 3. The Role of Noise (Diversity)
Even in Production Mode where the start point is fixed, **noise is still used for the rest of the trajectory ($p_1 \dots p_7$)**.

*   **B0, B1, B2, B3**: All start from the **same anchor point** ($p_0$), but they have **different noise** for their future steps.
*   **Result**: This allows the robot to "brainstorm" different paths (e.g., go Left vs. go Right) from the exact same starting position. 

---

## 4. Environment & Plotting (Raw Truth)
In the V4 Trajectory Visualization, we prioritize the **Raw Dataset Environment**.

*   **Original Obstacles**: We plot the original Red Circles from the `avoiding-d3il` dataset.
*   **No Projection Clutter**: We specifically exclude the "new" blue obstacles and halfspaces from `projection_eval.yaml`. This ensures the audit shows the robot's performance against the environment it was actually trained on.

---

## 5. Hardware Parallelism
Batch processing (B0-B3) is not done one-by-one. It is processed **simultaneously** on the **CPU (Vectorization)** or **GPU (CUDA)**. Because the Neural Network handles the entire batch tensor in one forward pass, generating 4 or 16 paths costs almost the same amount of time as generating one.

---

## Summary Table
| Feature | Math Mode | Production Mode | Real Eval Script |
| :--- | :--- | :--- | :--- |
| **Start Point** | Drifts (Audit) | Fixed (Clamped) | Fixed (Clamped) |
| **Noise Basis** | Static (Deterministic) | Static (Deterministic) | Live (Random) |
| **Batch Purpose** | Reproducibility audit | Distribution audit | Sample-based planning |
| **Parallelism** | CPU/CUDA Parallel | CPU/CUDA Parallel | CPU/CUDA Parallel |

---

> [!TIP]
> Use **Math Mode** to judge the **precision** of the ODE solvers.
> Use **Production Mode** to judge the **safety and diversity** of the generated plans.
