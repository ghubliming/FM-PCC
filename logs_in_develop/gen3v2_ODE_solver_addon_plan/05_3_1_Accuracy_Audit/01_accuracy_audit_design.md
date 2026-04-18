# 01. Accuracy Audit Design: "The Guidance Principle"

This document outlines the testing logic for the final phase of the ODE solver benchmark. It merges the "Old Idea" (Mathematical Drift) with our newly discovered "New Principle" (Interleaved Projector Guidance).

## 1. The Core Principle: Accuracy is Guidance
In standard generative AI, the intermediate trajectory steps don't matter—only the final image does. 
However, as established in our deep-dive analysis, **FM-PCC applies a Safety Projector at every sub-step** to guide the trajectory away from new obstacles. 

If we use a "lazy" math method (Euler) or heavily reduce the number of steps, the numerical math drifts significantly. The state handed to the Safety Projector becomes physically invalid. High ODE accuracy (RK4) ensures the state remains feasible, dramatically reducing the burden on the SLSQP safety optimizer.

We will design a dual-pronged benchmark to mathematically prove this.

---

## 2. Test A: The "Old Idea" (Mathematical Drift)
**Goal:** Prove that Euler drifts exponentially more than RK4 on our specific robot flow manifold, independently of any robotics physics.

### Methodology
1.  **The Oracle Setup**: We load a real `avoiding-d3il` model. We generate a "Ground Truth" trajectory from $t=0 \to 1$ using the `torchdiffeq:dopri5` solver with extremely tight error tolerances (`atol=1e-12`, `rtol=1e-12`) and high-density waypoints.
2.  **The Candidates**: We run the exact same noise input through `legacy:euler` and `legacy:rk4` at different step counts ($S \in [5, 10, 20]$).
3.  **The Metric**: We measure the **L2 Euclidean Distance** (the drift) between the Oracle trajectory and the candidate trajectory at corresponding intervals.
4.  **Expected Result**: Euler will drift significantly far away from the Oracle, while RK4 will stay magnetically attached to it, proving $O(h^4)$ accuracy.

---

## 3. Test B: The "New Principle" (Projector Thrashing)
**Goal:** Prove that numerical ODE drift directly inflates the computational cost of the Safety Optimization layer.

### Methodology
1.  **The Projector Setup**: We load the `Projector` class from `flow_matcher_v3` with a tight obstacle boundary constraint.
2.  **The Interleaved Execution**: We run the `p_sample_loop_v3_fair` (with the projector `apply_gradients` active).
3.  **The Metric**: Instead of just measuring overall latency, we explicitly track the **SLSQP Iteration Count / Internal Projector Cost** at each step.
4.  **Expected Result**: When running Euler at 5 steps, the drift will be so severe that the state violates the bounds egregiously. The SLSQP solver will suffer a massive latency spike trying to "recover" the physics. When running RK4 at 5 steps, the state will be highly accurate, and the boundary correction will be nearly instantaneous. 

---

## 4. Why this matters for the Paper / Advisor
This dual-test design completely bridges the gap between pure math and applied robotics.
*   **Test A** proves we know how ODEs work mathematically.
*   **Test B** proves why those math differences actually dictate the life-or-death latency limitations of a real-time Model Predictive Controller.
