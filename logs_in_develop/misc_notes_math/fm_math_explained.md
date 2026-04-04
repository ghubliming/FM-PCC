# Flow Matching (FM) Implementation: Mathematical Foundations

This document explains the mathematical concepts implemented in the Flow Matching (FM) code, focusing on the core equations and their correspondence to the code in `flow_matcher/models/diffusion.py`.

---

## 1. FM Objective: Velocity Field Learning

**Goal:**
Learn a velocity field $v(x_t, t)$ such that the ODE
$$
\frac{dx_t}{dt} = v(x_t, t)
$$
transports samples from a base distribution $p_0$ to a data distribution $p_1$ over $t \in [0, 1]$.

**Linear Interpolation Path:**
$$
x_t = (1 - t) x_0 + t x_1
$$
where $x_0 \sim p_0$, $x_1 \sim p_1$.

**Target Velocity:**
$$
v^*(x_t, t) = x_1 - x_0
$$

**Code:**
- In `p_losses`, the target is computed as `v_target = x_start - x_base`.
- The model predicts $v(x_t, t)$, and the loss is $\|v_{\text{pred}} - v^*\|$.

---

## 2. FM Training Loss

**Loss Function:**
$$
\mathcal{L} = \mathbb{E}_{x_0, x_1, t} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]
$$

**Code:**
- `loss, info = self.loss_fn(v_pred, v_target)` in `p_losses`.
- $x_t$ is sampled by linear interpolation between $x_0$ and $x_1$.

---

## 3. FM Sampling: ODE Integration

**Reverse ODE:**
$$
\frac{dx_t}{dt} = v_\theta(x_t, t)
$$

**Numerical Integration:**
- The code uses a reverse-time Euler method:
$$
x_{t-\Delta t} = x_t - v_\theta(x_t, t) \cdot \Delta t
$$
- Implemented in `p_sample_loop` as:
  - `model_mean = x - velocity * dt`
  - Looping from $t=1$ to $t=0$.

---

## 4. Conditioning and Constraints

- Conditioning is applied by fixing certain variables in $x_t$ (see `apply_conditioning`).
- Constraints (e.g., projections) are handled by modifying $x_t$ after each ODE step if needed.

---

## 5. Summary Table

| Code Function         | Math Equation / Concept                  |
|----------------------|------------------------------------------|
| `q_sample`           | $x_t = (1-t)x_0 + t x_1$                 |
| `p_losses`           | $\|v_{\text{pred}} - (x_1-x_0)\|$        |
| `p_sample_loop`      | $x_{t-\Delta t} = x_t - v(x_t, t)\Delta t$|
| `predict_velocity`   | $v(x_t, t)$ (model output)               |

---

*This document was generated to clarify the mathematical meaning of the FM implementation in code.*
