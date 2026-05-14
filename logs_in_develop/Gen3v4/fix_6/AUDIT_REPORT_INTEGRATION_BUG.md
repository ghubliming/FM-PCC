# Audit Report: Single-Step Euler Integration Chaos (iMF)

**Date**: 2026-05-14  
**Component**: iMeanFlow (iMF) Sampling Engine  
**Status**: RESOLVED (Fix applied to `config/avoiding-d3il.py`)

---

## 1. Problem Identification
During the standardization of the iMeanFlow (iMF) pipeline (Fix #6), initial evaluation plots showed **chaotic, unstable trajectories**. The agent appeared to make sharp, erratic jumps that disregarded environment boundaries and obstacle constraints, leading to a near-zero success rate despite low training loss.

## 2. Technical Audit: The Root Cause
A comparison between the standard `FMv3ODE` pipeline and the `iMeanFlow` pipeline revealed a critical discrepancy in the inference configuration.

### 2.1 Configuration Mismatch
In `config/avoiding-d3il.py`:
*   **Standard FMv3ODE**: `'flow_steps_v3': 10`
*   **iMeanFlow (iMF)**: `'flow_steps_v3': 1`  <-- **CRITICAL BUG**

### 2.2 The Physics of Single-Step Euler
The iMF engine utilizes an **Explicit Euler** solver as its default sampling mechanism. The update rule is:
$$x_{t+dt} = x_t + v(x_t, t) \cdot dt$$

When `flow_steps_v3` is set to `1`, the solver performs the entire integration in a single jump:
1.  **Initial State ($t=0$)**: $x$ is pure Gaussian noise.
2.  **Velocity Prediction**: The model predicts the velocity $v$ based on the $t=0$ noise.
3.  **Extrapolation**: The solver takes a massive step of size $dt=1.0$.

Because the obstacle avoidance task (D3IL-Avoiding) has a **highly non-linear, curved vector field**, a straight-line extrapolation from $t=0$ to $t=1$ is mathematically guaranteed to shoot past the target manifold and ignore the complex curvature required to avoid obstacles.

---

## 3. The Fix: Restoring Numerical Stability
The fix involved aligning the iMF sampling resolution with the standard FMv3ODE baseline.

**File**: `config/avoiding-d3il.py`
```diff
- 'flow_steps_v3': 1,
+ 'flow_steps_v3': 10,
```

By increasing the resolution to 10 steps, we reduce the integration error by an order of magnitude. Each step now covers $dt=0.1$, allowing the model to "correct" its course as it approaches the high-curvature regions near obstacles.

---

## 4. Conclusion & Best Practices
The "chaos" was not caused by a failure in the model's weights or the iMF mathematical framework, but by **numerical starvation** during inference. 

### Key Lessons:
*   **Euler vs. RK4**: Unlike adaptive solvers (Dopri5), Explicit Euler requires a sufficient step count to handle non-linear flows.
*   **Sampling Parity**: When benchmarking new engines like iMF against FMv3ODE, ensure that the `NFE` (Number of Function Evaluations) is identical to ensure a fair comparison of the learned field, not the solver's error.

> [!IMPORTANT]
> A retrain is **NOT** required. The model had already learned a valid flow field; it simply needed a higher-resolution solver to navigate it correctly.
