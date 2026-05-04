# Audit: Math vs. Production Mode Error Propagation

This document explains why the numerical error (Solver vs. Oracle) is significantly different between **Math Mode** and **Production Mode**, even though the computational workload (number of steps) remains identical.

## 1. The Computation Paradox
In both modes, a 10-step Euler solver performs exactly 10 evaluations of the neural network. From a hardware perspective, they cost the same. However, the **Numerical Error** in the final trajectory is vastly different because of how the data is handled between steps.

| Metric | Math Mode | Production Mode |
| :--- | :--- | :--- |
| **Network Calls** | 10 (for 10 steps) | 10 (for 10 steps) |
| **Error Propagation** | **Accumulated** | **Reset / Interrupted** |
| **Start Point ($t=0$)** | Floating (from noise) | **Anchored** (to Physical Robot) |
| **Final Accuracy** | Reflects True Solver Drift | Reflects Residual Drift |

---

## 2. Math Mode: The "Pure" ODE Audit
In Math mode, the solver is "pure." It starts at $t=0$ and integrates to $t=1$ without any external correction.
*   **Drift Accumulation:** If the solver makes a small error at Step 1, that error becomes the starting point for Step 2. 
*   **Divergence:** As the steps progress, the trajectory "wanders" further and further away from the ground truth (Oracle) path.
*   **Use Case:** This is the **only valid mode** for comparing solver precision (e.g., Euler vs. RK4), as it shows the true cost of numerical discretization.

---

## 3. Production Mode: The "Safety Shield" Effect
In Production mode, the [benchmark_ode_solvers_v4.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v4/benchmark_ode_solvers_v4.py) applies a physical anchor at every step.

```python
# [V4 PROD FIX] Persistently re-anchor the first waypoint at t=0 to the physical robot position
current_robot_pos = cond[0][:, fm_model.action_dim:fm_model.action_dim+fm_model.action_dim]
x[:, 0, :fm_model.action_dim] = current_robot_pos.clone()
```

### Why the error is lower:
1.  **Numerical Resets:** After each ODE step, the code "pulls back" the start of the trajectory to the robot. 
2.  **Killing the Compounded Error:** By resetting the $t=0$ position, you effectively "kill" any error that was starting to build up at the root of the trajectory.
3.  **Tethering:** Since the rest of the waypoints ($t=1 \dots 7$) are predicted relative to the anchor, the entire trajectory stays "tethered" to the robot.

**Effect:** A bad solver like **Euler** looks much more accurate in Production mode than it actually is, because the anchoring is constantly "cleaning up" its mistakes.

---

## 4. Visual Comparison Analogy

Imagine walking 100 meters in the dark with a 1-degree error in your compass:

*   **Math Mode:** You walk the whole 100 meters. By the end, you are **1.7 meters off-course**.
*   **Production Mode:** You walk 10 meters, then someone turns on the lights and puts you back on the center line. You do this 10 times. By the end, you are only **0.17 meters off-course**.

The "Computation" (walking 100 meters) is the same, but the "Error" (distance from target) is 10x smaller because of the corrections.

---

## 5. Conclusion for Benchmarking

> [!IMPORTANT]
> **Math Mode** measures the **Solver Accuracy.**
> Use this to decide if you need RK4 or if Euler is "good enough."

> [!NOTE]
> **Production Mode** measures the **Robotic Performance.**
> Use this to see how the physical robot will actually behave when the Safety Shield is active.
