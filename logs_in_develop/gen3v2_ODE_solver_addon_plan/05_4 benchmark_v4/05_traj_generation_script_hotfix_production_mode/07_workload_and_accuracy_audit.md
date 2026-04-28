# Workload Audit: What are we actually testing?

To help with your manual debugging, here is the exact breakdown of the mathematical workload being measured in the V4 Benchmark.

## 1. The "State" is a Path, not a Point
When you set `--steps 10` and `--horizon 8`, the ODE solver is not integrating a single (x, y) coordinate. It is integrating a **Trajectory Tensor** of shape:
`[Batch Size, Horizon (8), State Dimension (8)]`

## 2. Mathematical Workload per Trial
For each solver (Euler, RK4, etc.), the workload for a single trial is:
*   **Forward Passes**: Exactly `steps` (e.g., 10) calls to the Neural Network.
*   **Inference Size**: Each call processes `Batch * 8 * 8` elements.
*   **Total Work**: 10 inferences $\times$ 8 horizon steps = **80 "virtual" step predictions** per trajectory.

## 3. "Imagination" vs. "Physical" Steps
This is the most common point of confusion. Here is the breakdown:
*   **ODE Steps (e.g., 10)**: These are **Imagination Steps**. The robot sits at its current position and runs the ODE solver 10 times to "clean" a noisy 8-step plan into a smooth one. 
*   **Horizon (e.g., 8)**: These are **Physical Steps**. This is the length of the path the robot "imagines" on the map.

**The Workload Multiplier:**
To get **one** physical plan of length 8, the robot must think for **10** ODE steps. 
*   Total NN Inferences = 10 per MPC cycle.
*   Total States Predicted = 10 steps $\times$ 8 horizon = 80 states per MPC cycle.

### 3.1 The "Nested" Logic: MPC vs. ODE
It is critical to distinguish between Physical Time and Imagination Time:

*   **MPC Cycle (Physical Step 0)**:
    *   The Brain runs **10 ODE Iterations** (DGM sampling).
    *   Each iteration updates the **full 8-step horizon** simultaneously.
    *   **RESULT**: A full 8-step future plan is generated.
    *   **ACTION**: The robot executes only the 1st action.
*   **MPC Cycle (Physical Step 1)**:
    *   The robot has moved. The Brain runs **another 10 ODE Iterations**.
    *   **RESULT**: A brand new 8-step future plan is generated.

## 4. Why the "Conditioning" Matters for Accuracy
In `production` mode, we apply the "Safety Shield" (`apply_conditioning`) after **every single one** of those 10 Imagination Steps.
*   **The Anchor:** Step 0 ($t=0$) of your 8-step horizon is strictly overwritten with the robot's current real-world observation.
*   **The Floating Path:** Steps 1 through 7 remain unconditioned ("floating"). The Neural Network updates their values 10 times, attempting to "pull" them into a physically consistent trajectory that originates from the anchored Step 0.

## 5. Why the Time/Accuracy tradeoff is unique
Because the neural network uses a Transformer or UNet architecture, it can predict all 8 horizon steps **in parallel** during a single GPU/CPU forward pass. 
*   This means 10 steps of ODE integration gives you a **full 8-step future plan**, which is much more computationally efficient than a classic MPC that would have to solve 8 separate optimization problems.

## 6. Why Benchmark the Full Horizon?
You might ask: "If the robot only executes the 1st step in real life, why do we benchmark all 8 steps?"

The answer is **Mental Integrity**:
1.  **Mathematical Coupling**: The neural network calculates the 1st step based on its "imagination" of the **entire 8-step future**. If the 8-step plan is "broken" or drifts into a wall, the 1st step will also be corrupted.
2.  **Audit of the "Brain"**: By plotting the full 8 steps, we are auditing the model's reasoning. If the full plan is smooth and avoids obstacles, we can trust the 1st step. If the plan looks like "spaghetti," the ODE solver is failing.
3.  **Safety**: A "short-sighted" model that doesn't understand its full 8-step future is dangerous. We benchmark the whole path to ensure the "Safety Shield" and "Numerical Precision" hold up across the entire predicted window.

**Summary for Debugging:** 
## 7. The Anchoring Ratio (The "Hard Reset")
As confirmed in the `apply_conditioning` helper, our Production Mode uses a **Hard Reset** (also known as Inpainting), exactly like the **Janner et al. Diffuser** architecture.

### 7.1 Detailed Re-anchoring Audit (What happens at every sub-ODE step?)
At every single ODE iteration (e.g., 10 times per plan), the "Safety Shield" performs a surgical overwrite on the trajectory tensor:

| Horizon Step | Data Type | Status | Source of Value |
| :--- | :--- | :--- | :--- |
| **Step 0 ($t=0$)** | **Actions** | **Floating** | **Predicted** by Neural Network |
| **Step 0 ($t=0$)** | **Observation** | **FIXED** | **Snapped** to Real Robot Position |
| **Steps 1-7** | **Actions** | **Floating** | **Predicted** by Neural Network |
| **Steps 1-7** | **Observation** | **Floating** | **Predicted** by Neural Network |

*   **1/8 State Anchoring**: Only the physical state at the very first moment is locked. 
*   **7/8 Future Imagination**: The entire rest of the 8-step path is free to move and be optimized by the solver.

**Why the "Teleportation" Jump happens:**
Because the initial noise is centered at `(0,0)` but your Yellow Star might be at `(0.6, 0.1)`, there is a **conflict** between the 1/8 anchored state and the 7/8 noisy state. 
*   **State 0** is snapped to the Star (0.6, 0.1).
*   **State 1** is still sitting over where the Noise wanted it to be (near 0,0).
*   The "Jump" you see is the model trying to bridge that 0.6m gap in the very first timestep of the plan.

**Summary for Debugging:**
In a real robot run, "Warm-Starting" (starting the noise near the previous plan) makes this jump disappear. In the benchmark, starting from 100% random noise makes this jump massive.
