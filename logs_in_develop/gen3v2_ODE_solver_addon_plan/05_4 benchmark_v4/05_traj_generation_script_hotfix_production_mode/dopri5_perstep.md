# Understanding `dopri5` vs. Per-Step Evaluation

This document clarifies the mathematical differences between fixed-step ODE solvers and the adaptive `dopri5` solver, specifically regarding the concept of breaking the $t=0 \rightarrow 1$ flow process into 10 "sub-paths" or steps.

## 1. Fixed-Step Solvers (Euler, Midpoint, RK4)
For solvers like Euler, Midpoint (RK2), and RK4, the `--steps 10` argument dictates the exact behavior of the math. It divides the integration timeline (from noise at $t=0$ to the physical trajectory at $t=1$) into exactly **10 discrete intervals** ($dt = 0.1$). 

At each of these 10 steps, the solver evaluates the velocity field to move forward. The only difference is precision *within* the step:
*   **Euler:** 1 evaluation per step.
*   **Midpoint:** 2 evaluations per step.
*   **RK4:** 4 evaluations per step.

## 2. The Adaptive Oracle (`dopri5`)
`dopri5` (Dormand-Prince 5) is an **adaptive step size** solver. Its core mathematical principle is to automatically calculate and adjust its own internal step sizes to guarantee that the local error stays strictly below the thresholds you set (`--rtol 1e-10` and `--atol 1e-10`). 

Because of this, `dopri5` **cannot be forced** to take a specific number of math steps. It will completely ignore a `--steps 10` instruction. It might take 150 tiny steps where the curve is complex, and 2 huge steps where the curve is flat.

## 3. Can we get "10 Steps" out of `dopri5`?
If you want to visualize the flow process in 10 frames, it depends entirely on *how* you ask `dopri5` for those 10 frames.

### Approach A: The Illogical Way (Stopping and Restarting)
Imagine running `dopri5` from $0 \rightarrow 0.1$, extracting the output, and then launching a *brand new* `dopri5` run from $0.1 \rightarrow 0.2$, repeating this 10 times.

> [!WARNING]
> **This approach is mathematically illogical, extremely slow, and worsens the final error.**

Here is why:
1.  **Breaking Adaptive Momentum:** When `dopri5` runs continuously, it "learns" the complexity of the curve and optimizes its step size (taking larger, faster steps when safe). If you stop it at $t=0.1$ and restart it, you wipe its memory. It becomes overly cautious and starts with a tiny, slow step all over again.
2.  **Accumulating Boundary Errors:** Every time you start an ODE solver, you incur a small "local truncation error" at that boundary. By forcing 10 hard stops, you are artificially injecting 10 boundary errors into the integration. By the time you reach $t=1$, these accumulated boundary errors will make the final trajectory *less accurate* than if you had just let `dopri5` run continuously.

### Approach B: The Logical Way (Dense Output / Interpolation)
Instead of restarting the solver, you pass an array of the specific times you want to observe: `[0.0, 0.1, 0.2, ... 1.0]` into a *single* `dopri5` command. `torchdiffeq` handles this natively using a technique called **Dense Output**.

> [!TIP]
> **This approach is mathematically sound and does not alter the final $t=1$ error.**

Here is why:
`dopri5` still completely ignores your 10 checkpoints while it is doing the actual math. It integrates from 0 to 1 taking however many messy, adaptive steps it needs. However, as it integrates, Runge-Kutta methods like `dopri5` build a **continuous polynomial curve** behind the scenes. 

Because it has this continuous mathematical curve, the solver simply "looks up" (interpolates) the exact values at $t=0.1, 0.2, ...$ without ever stopping its integration momentum. The math remains pure, and the final state at $t=1$ is exactly as precise as it would be if you only asked for the start and end points.

---

**Conclusion:**
In our `benchmark_ode_solvers_v4.py` script, we intentionally only pass `[0.0, 1.0]` to the `dopri5` oracle. This skips the per-step evaluation entirely, allowing the adaptive solver to find the fastest and most mathematically pure path to the final $t=1$ robotic trajectory, which is all we care about for the final benchmark comparison.
