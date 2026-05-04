# U3: The Accuracy vs. Time Payoff in FM-ODE

This document addresses the theoretical question: **"Is better ODE accuracy worth the extra computation time in robot planning?"**

## 1. The Metric of Success: NFE (Number of Function Evaluations)

In high-performance sampling, we don't just measure "Milliseconds." We measure **Accuracy per NFE**.

*   **Explicit Euler**: Requires many steps ($NFE \approx 50-100$) to suppress discretization error in curved manifolds.
*   **RK4 / Advanced Solvers**: While they take 4 model passes per step, they can achieve the same accuracy in **significantly fewer steps** ($NFE \approx 10-20$).

> [!IMPORTANT]
> **The Efficiency Paradox**: 
> A 5-step RK4 solver (20 total model passes) is often **more accurate AND faster** than a 40-step Euler solver (40 total model passes). 
> This is why advisors push for high-order math: it is a tool for **reducing the total work** required to reach a valid answer.

---

## 2. Manifold Drift & Success Rates

In robot control (Flow Matching), the ODE trajectory represents the robot's intended path through a "safe" manifold of the state space.

### The "Drift" Failure Mode
*   **Euler ($O(h)$ Error)**: The linearization error ($O(h^2)$ locally) causes the trajectory to drift outward. In a crowded environment, a 5% drift off the "safe path" means a **collision**.
*   **RK4 ($O(h^4)$ Error)**: The error is suppressed exponentially faster. The trajectory stays "locked" to the trained manifold.

### The "Success Rate Transition"
Empirical research in Diffusion Models shows that as you increase solver order, the **Success Rate** (or Sample Quality) reaches its plateau at much lower NFEs. 
*   Euler might reach its maximum success rate at $NFE=50$.
*   RK4 might reach that same plateau at $NFE=12$.
*   **Result**: RK4 is theoretically the "faster" path to a successful robot mission.

---

## 3. The "Straightness" Factor (Rectified Flow)

Does accuracy *always* matter? 
*   If the Flow Matching model was trained to be "Straight" (e.g., via **Rectified Flow** or **Optimal Transport**), then the curvature of the trajectory is zero.
*   In a **linear field**, Euler is technically "exact" (the error term $f''$ is zero).
*   **The Robot Reality**: Because our robot environments are non-convex (obstacles), the learned vector field is rarely perfectly straight. Higher-order methods remain the "safety net" that allows for reliable control in tight spaces.

## 4. Summary: The Trade-off

| Aspect | Euler Baseline | Advanced (RK4/DPM) |
| :--- | :--- | :--- |
| **Logic** | Simple, fast-per-step. | Complex, slow-per-step. |
| **Drift** | High (Drifts off manifold). | Low (Locked to manifold). |
| **Optimal NFE** | Needs many steps ($>50$). | Needs few steps ($<10$). |
| **Trustworthy?** | Risky in tight gaps. | Precision-grade. |

**Final Verdict**: Better accuracy doesn't just improve results; it **enables** lower total sampling time by allowing you to take larger, safer steps through the manifold.
