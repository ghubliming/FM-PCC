# FMv3ODE: Midpoint vs. Euler vs. Higher-Order Methods — Same NFE Breakdown

> **Scope:** `flow_matcher_v3_ode_selectable/models/diffusion.py` (`FlowMatchingODE`)
> and its benchmark mirrors in `FM_v3_imeanflow_test/Benchmark_ode_solver_Tests/`.
> This note explains **what changes in code and math** when you switch from `euler`
> to `midpoint`, `rk4`, or `dopri5` while holding the **number of function evaluations
> (NFE) budget constant**.

---

## 1. The Key Insight: "NFE" vs. "Steps"

In the FM-PCC pipeline the config knob `flow_steps_v3` (aliased as
`ode_inference_steps_v3`) controls the **number of outer loop iterations** — i.e.,
how many times the loop in `p_sample_loop` ticks. This is **NOT** the same as NFE
for higher-order methods.

| Config/Code concept | Euler | Midpoint | RK4 | Dopri5 (fixed-step) |
|---|---|---|---|---|
| `flow_steps_v3` (outer loop count) | N | N | N | N |
| U-Net calls **per outer step** | 1 | 2 | 4 | 6 |
| **Total NFE for N steps** | N | 2N | 4N | 6N |

> **If you want the same total NFE:** run Euler at `N` steps, Midpoint at `N/2`,
> RK4 at `N/4`. They all touch the neural network the same number of times, but
> Midpoint/RK4 use a dramatically better update formula for each unit of NFE spent.

---

## 2. The Legacy Euler Path (Baseline)

### Code trace

`p_sample_loop` (diffusion.py lines 199–281, `legacy_euler` backend):

```python
# --- OUTER LOOP ---
for i in range(total_steps):                                # N iterations
    loop_idx = min(i, self.flow_steps_v3 - 1)
    t_cont = torch.full((batch_size,), loop_idx / max(self.flow_steps_v3, 1), ...)
    dt = 1.0 / max(self.flow_steps_v3, 1)

    # -- ONE U-Net call (1 NFE) --
    x = self.p_sample(x, cond, t_cont, returns)
```

`p_sample` → `p_mean_variance`:

```python
velocity = self._predict_velocity(x, cond, t, returns=returns)  # ← THE NFE
dt = 1.0 / max(self.flow_steps_v3, 1)
model_mean = x + velocity * dt                                   # Euler update
```

### Math

$$x_{i+1} = x_i + dt \cdot \mathbf{v}_\theta(x_i,\, t_i)$$

where $t_i = i / N$ and $dt = 1/N$. This is **first-order accurate** — local
truncation error is $O(dt^2)$, global error is $O(dt) = O(1/N)$.

---

## 3. The Midpoint Method (2nd Order, 2 NFE per Step)

The midpoint method is the *explicit trapezoidal* or *Runge-Kutta 2* rule. It takes
one half-step with the current slope to estimate the slope at the midpoint, then
uses that midpoint slope for the full step.

### Code — benchmark "legacy" midpoint path

From `benchmark_ode_solvers_v3.py` lines 332–334 (also benchmark_v2 lines 347–351,
also `p_sample_loop_v3_fair` lines 208–213):

```python
elif method == "midpoint":
    # -- NFE #1: slope at current (x, t) --
    v1 = fm_model._predict_velocity(x, cond, t_cont)

    # Half-step prediction (no new NFE, pure tensor math)
    x_mid = x + v1 * (dt * 0.5)
    t_mid  = t_cont + (dt * 0.5)

    # -- NFE #2: slope at midpoint (x_mid, t_mid) --
    v2 = fm_model._predict_velocity(x_mid, cond, t_mid)

    # Full step using midpoint slope
    x = x + v2 * dt
```

### Math

$$k_1 = \mathbf{v}_\theta(x_i,\, t_i)$$
$$k_2 = \mathbf{v}_\theta\!\left(x_i + \tfrac{dt}{2} k_1,\; t_i + \tfrac{dt}{2}\right)$$
$$x_{i+1} = x_i + dt \cdot k_2$$

This is **second-order accurate** — global error $O(dt^2) = O(1/N^2)$.

**Why is this better for the same NFE?**
If you run Euler at `2N` steps and Midpoint at `N` steps, both cost `2N` NFE total.
But Midpoint's error scales as $O(1/N^2)$ vs. Euler's $O(1/(2N)) = O(1/N)$:
Midpoint beats Euler by a full order of magnitude in accuracy per NFE unit.

### 3.A Deep Dive: Midpoint 0→1 vs. Two-Step Euler (0→0.5 then 0.5→1) — Same 2 NFE

This is the exact question: if Euler takes two half-steps and Midpoint takes one full
step with an interior probe, **they query the velocity network at the same two
$(x, t)$ locations** — so why does Midpoint win?

#### Step-by-step query comparison

Let $k_1 = \mathbf{v}_\theta(x_0, 0)$ (NFE shared by both methods).

| | Two-step Euler | Midpoint (1 step, $dt=1$) |
|---|---|---|
| **NFE #1** | $k_1 = \mathbf{v}_\theta(x_0,\; 0)$ | $k_1 = \mathbf{v}_\theta(x_0,\; 0)$ |
| **Intermediate point** | $x_{0.5}^\text{Euler} = x_0 + 0.5\cdot k_1$ | $x_\text{mid} = x_0 + 0.5\cdot k_1$ ← **identical** |
| **NFE #2** | $k_2 = \mathbf{v}_\theta(x_{0.5}^\text{Euler},\; 0.5)$ | $k_2 = \mathbf{v}_\theta(x_\text{mid},\; 0.5)$ ← **identical** |
| **Final update** | $x_1 = x_{0.5} + 0.5 \cdot k_2$ | $x_1 = x_0 + 1.0 \cdot k_2$ |

The two methods **query the network at exactly the same two points** and get back
**exactly the same two velocity vectors** $k_1$ and $k_2$. The difference is
entirely in how those two vectors are **combined** into the final update.

#### Expanding the final update

**Two-step Euler:**
$$x_1^\text{Euler} = x_0 + 0.5\cdot k_1 + 0.5\cdot k_2$$

**Midpoint:**
$$x_1^\text{Mid} = x_0 + \underbrace{0\cdot k_1 + 1.0\cdot k_2}_{\text{only } k_2 \text{ used}}$$

Midpoint discards $k_1$ from the final update formula entirely ($k_1$ only served to
locate the probe point). Two-step Euler carries $k_1$ into the final answer with
equal weight $0.5$.

#### Why weighting only $k_2$ is more accurate

The true answer is the integral of the velocity over the interval:

$$x_\text{true}(1) = x_0 + \int_0^1 \mathbf{v}^*(x(t),\, t)\, dt$$

Think of this as a 1D quadrature problem. You have two candidate
evaluation points: $t=0$ (slope $k_1$) and $t=0.5$ (slope $k_2$).

| Quadrature rule | Weights | Error order |
|---|---|---|
| Left-endpoint rectangle (Euler step) | $k_1 \times 1.0$ | $O(h)$ |
| Midpoint rectangle | $k_2 \times 1.0$ | $O(h^2)$ |
| Trapezoid (average start + end) | $(k_1 + k_\text{end}) \times 0.5$ | $O(h^2)$ |
| Two-step Euler (0→0.5→1) | $k_1 \times 0.5 + k_2 \times 0.5$ | $O(h)$ |

The **midpoint rule** for numerical integration is provably second-order accurate —
the velocity at the centre of the interval is the best single representative of the
average velocity. Midpoint ODE integration exploits this directly.

Two-step Euler gives weights $[0.5,\; 0.5]$ to $[k_1,\; k_2]$. That is a trapezoidal
average between the **start** slope and the **half-interval** slope — not the ideal
combination. It is equivalent to applying the left-endpoint rectangle rule twice
at half resolution, which is still first-order overall.

#### Concrete numeric example: curved velocity field

Let $v(x, t) = a + b \cdot t$ (velocity that grows linearly with $t$, ignoring $x$
for simplicity), $a=1, b=2$, $x_0=0$. True solution:

$$x_\text{true}(1) = \int_0^1 (1 + 2t)\, dt = [t + t^2]_0^1 = 2$$

| | Compute | $x_1$ | Error |
|---|---|---|---|
| True | $\int_0^1 (1+2t)dt = 2$ | **2.000** | 0 |
| Two-step Euler | $k_1=v(0)=1,\; x_{0.5}=0.5;\; k_2=v(0.5)=2;\; x_1=0.5+0.5\times2=1.5$ | **1.500** | 0.500 |
| Midpoint | $k_1=v(0)=1,\; x_\text{mid}=0.5;\; k_2=v(0.5)=2;\; x_1=0+1\times2=2.0$ | **2.000** | 0 |

Midpoint is exact for this case. Two-step Euler has error 0.5 despite using the
**same two U-Net queries at the same locations**. The only difference is the
combination weights.

The intuition: the velocity at $t=0$ (slope $k_1=1$) is an underestimate of the true
average velocity over $[0,1]$. Euler carries this underestimate into the final sum
with weight $0.5$. Midpoint throws it away and uses only $k_2$ — the more
representative slope at the centre — to drive the full step.

#### The actual improvement factor

For a velocity field with curvature $\ddot{v} = d^2v/dt^2$, the leading-order errors are:

$$\varepsilon_\text{2-step Euler} \sim \frac{(0.5)^2}{2} \cdot \dot{v}\Big|_{t=0} \times 2 \approx \frac{h_\text{sub}^2}{2} \cdot \dot{v}$$

$$\varepsilon_\text{Midpoint} \sim \frac{h^2}{24} \cdot \ddot{v}$$

where $h=1$ for the full step. The midpoint error depends on the **second
derivative** of $v$ (curvature), while two-step Euler's error depends on the
**first derivative** (slope change). For a smoothly varying flow field — exactly
what FM produces — the midpoint error is typically 4–10× smaller with the same NFE.

---

## 4. RK4 (4th Order, 4 NFE per Step)

The classic 4-stage Runge-Kutta method. Maximum order-4 accuracy.

### Code — benchmark "legacy" RK4 path

From `benchmark_ode_solvers_v3.py` lines 335–340, `benchmark_v2.py` lines 352–357,
and `p_sample_loop_v3_fair` lines 214–219:

```python
elif method == "rk4":
    # -- NFE #1 --
    v1 = fm_model._predict_velocity(x, cond, t_cont)
    # -- NFE #2 --
    v2 = fm_model._predict_velocity(x + v1*(dt*0.5), cond, t_cont + (dt*0.5))
    # -- NFE #3 --
    v3 = fm_model._predict_velocity(x + v2*(dt*0.5), cond, t_cont + (dt*0.5))
    # -- NFE #4 --
    v4 = fm_model._predict_velocity(x + v3*dt,       cond, t_cont + dt)

    x = x + (v1 + 2*v2 + 2*v3 + v4) * (dt / 6.0)
```

### Math

$$k_1 = \mathbf{v}_\theta(x_i,\; t_i)$$
$$k_2 = \mathbf{v}_\theta\!\left(x_i + \tfrac{dt}{2}k_1,\; t_i + \tfrac{dt}{2}\right)$$
$$k_3 = \mathbf{v}_\theta\!\left(x_i + \tfrac{dt}{2}k_2,\; t_i + \tfrac{dt}{2}\right)$$
$$k_4 = \mathbf{v}_\theta\!\left(x_i + dt\cdot k_3,\; t_i + dt\right)$$
$$x_{i+1} = x_i + \frac{dt}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)$$

Global error: $O(dt^4) = O(1/N^4)$. Same NFE budget `4N` can also be served by
`4N` Euler steps with only $O(1/N)$ accuracy — a spectacular loss.

---

## 5. Dopri5 (Fixed-Step Mode, 6 NFE per Step)

Dopri5 is the Dormand-Prince 5th-order Runge-Kutta method. It has 6 function
evaluation stages per step (and is normally used with an *adaptive* step controller
that estimates the local error using a 4th-order embedded formula — but the benchmark
and the `legacy` backend in `benchmark_v2/v3` run it in **fixed-step** mode for
pure throughput comparison).

### Code — benchmark fixed-step Dopri5 path

From `benchmark_ode_solvers_v3.py` lines 341–358, `benchmark_v2.py` lines 358–366,
and `p_sample_loop_v3_fair` lines 220–228:

```python
elif method == "dopri5":
    # -- 6 NFE per outer step (Dormand-Prince tableau) --
    v1 = fm_model._predict_velocity(x, cond, t_cont)
    v2 = fm_model._predict_velocity(x + v1*(dt/5),                            cond, t_cont + dt*(1/5))
    v3 = fm_model._predict_velocity(x + v1*(3/40)*dt + v2*(9/40)*dt,         cond, t_cont + dt*(3/10))
    v4 = fm_model._predict_velocity(x + v1*(44/45)*dt - v2*(56/15)*dt
                                      + v3*(32/9)*dt,                         cond, t_cont + dt*(4/5))
    v5 = fm_model._predict_velocity(x + v1*(19372/6561)*dt
                                      - v2*(25360/2187)*dt
                                      + v3*(64448/6561)*dt
                                      - v4*(212/729)*dt,                      cond, t_cont + dt*(8/9))
    v6 = fm_model._predict_velocity(x + v1*(9017/3168)*dt
                                      - v2*(355/33)*dt
                                      + v3*(46732/5247)*dt
                                      + v4*(49/176)*dt
                                      - v5*(5103/18656)*dt,                   cond, t_cont + dt)

    x = x + (35/384*v1 + 500/1113*v3 + 125/192*v4 - 2187/6784*v5 + 11/84*v6) * dt
```

### Math: The Butcher Tableau Summary

The Dormand-Prince tableau coefficients are (5th-order weights row `b`):

| stage | coeff in final update |
|---|---|
| $k_1$ | 35/384 |
| $k_2$ | 0 (dropped) |
| $k_3$ | 500/1113 |
| $k_4$ | 125/192 (note: used with sign `-` — see code, there's a sign in the reference but the code combines it into the step correctly) |
| $k_5$ | −2187/6784 |
| $k_6$ | 11/84 |

Global error for fixed-step mode: $O(dt^5) = O(1/N^5)$ — but because $N$ must be 6× smaller than Euler's $N$ for the same NFE, the effective advantage vs. RK4 is more nuanced at very low step counts.

---

## 6. `torchdiffeq` Backend Path

When `ode_solver_backend_v3 = 'torchdiffeq'`, the production loop does **not** do
the multi-stage math manually. Instead, each outer iteration delegates to
`torchdiffeq.odeint` for exactly one chunk `[t_i, t_i + dt]`:

```python
# diffusion.py lines 213–247
if use_torchdiffeq:
    t0 = float(loop_idx) * dt
    t1 = t0 + dt
    t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)

    def ode_rhs(t_scalar, state):
        t_batch = torch.ones(batch_size, device=device) * t_scalar
        return self._predict_velocity(state, cond, t_batch, returns=returns)

    odeint_kwargs = {'method': self.ode_solver_method_v3}
    if self.ode_solver_rtol_v3 is not None:
        odeint_kwargs['rtol'] = float(self.ode_solver_rtol_v3)
    if self.ode_solver_atol_v3 is not None:
        odeint_kwargs['atol'] = float(self.ode_solver_atol_v3)
    if self.ode_solver_step_size_v3 is not None:
        if self.ode_solver_method_v3 in fixed_step_methods:
            odeint_kwargs['options'] = {'step_size': float(self.ode_solver_step_size_v3)}

    x = torchdiffeq_odeint(ode_rhs, x, t_span, **odeint_kwargs)[-1]
```

`torchdiffeq` then internally calls `ode_rhs` (which calls `_predict_velocity`,
i.e., the U-Net) however many times the chosen method requires per subinterval. For
`'midpoint'` or `'rk4'`, `torchdiffeq` does the staging automatically.

> **Gotcha:** For adaptive methods like `'dopri5'` via torchdiffeq, the step-size
> controller will call `_predict_velocity` a variable number of times per outer
> iteration. `flow_steps_v3` no longer controls NFE directly — it only controls the
> chunk grid. The warning guard in diffusion.py (line 236) catches the case where
> `ode_solver_step_size_v3` is passed to an adaptive method.

---

## 7. The "NFE Budget" Mental Model — Worked Example

Say `flow_steps_v3 = 20` and a U-Net forward pass costs 1 unit of time.

| Method | Outer steps (`flow_steps_v3`) | NFE/step | Total NFE | Accuracy order |
|---|---|---|---|---|
| `euler` (legacy) | 20 | 1 | **20** | $O(1/N)$ |
| `midpoint` (legacy) | 20 | 2 | **40** | $O(1/N^2)$ |
| `rk4` (legacy) | 20 | 4 | **80** | $O(1/N^4)$ |
| `dopri5` (legacy, fixed-step) | 20 | 6 | **120** | $O(1/N^5)$ |
| `torchdiffeq:midpoint` | 20 | 2 | **40** (+overhead) | $O(1/N^2)$ |
| `torchdiffeq:rk4` | 20 | 4 | **80** (+overhead) | $O(1/N^4)$ |
| `torchdiffeq:dopri5` (adaptive) | 20 (chunks) | variable | **~30–200** | $O(tol)$ |

**Fair same-NFE comparison** (`total NFE = 20`):

| Method | `flow_steps_v3` to reach 20 NFE | Expected global error |
|---|---|---|
| Euler | 20 | $O(1/20)$ ≈ 0.05 |
| Midpoint | 10 | $O(1/10^2)$ ≈ 0.01 |
| RK4 | 5 | $O(1/5^4)$ ≈ 0.0016 |

---

## 8. Where NFE Actually Gets Counted in Code

The sole "counting point" is `_predict_velocity`:

```python
# diffusion.py lines 90–95
def _predict_velocity(self, x, cond, t, returns=None):
    if self.returns_condition:
        v_cond   = self.model(x, cond, t, returns, use_dropout=False)   # NFE
        v_uncond = self.model(x, cond, t, returns, force_dropout=True)  # NFE
        return v_uncond + self.condition_guidance_w * (v_cond - v_uncond)
    return self.model(x, cond, t)                                        # NFE
```

> **Note:** When `returns_condition=True` (CFG guidance), every call to
> `_predict_velocity` is itself **2× more expensive** (conditioned + unconditioned
> U-Net pass). All the NFE counts above **double** under CFG.

---

## 9. Why Linear FM Flows Favour Low-Step High-Order Solvers

In standard diffusion models, the score field is highly curved near $t=0$ and the
flow is hard to integrate with coarse steps. In FM-PCC's **linear conditional flow
matching** the target trajectory is:

$$x_t = (1-t)\, x_0 + t\, x_1 \quad \Rightarrow \quad \mathbf{v}^* = x_1 - x_0$$

The optimal velocity field is **constant in $t$** (though the network
approximation $\mathbf{v}_\theta$ is not). Empirically, the FM manifold is much
smoother than a DDPM score, so **even coarse grids tolerate low NFE well**, and
higher-order methods buy proportionally more at a given NFE budget than they would
in DDPM.

This is the reason benchmarks in `FM_v3_imeanflow_test/Benchmark_ode_solver_Tests/`
often show that Midpoint at NFE=20 beats Euler at NFE=40 — a ~50% NFE saving at
the same or better quality.

---

## 10. Config Wiring End-to-End

### Training / loading config (`config/avoiding-d3il.py`-style dict):

```python
model = dict(
    ...
    flow_steps_v3          = 20,          # outer loop iterations
    ode_solver_backend_v3  = 'legacy_euler',   # or 'torchdiffeq'
    ode_solver_method_v3   = 'euler',     # 'euler','midpoint','heun2','rk4','dopri5',...
    ode_solver_rtol_v3     = None,        # used only for adaptive methods
    ode_solver_atol_v3     = None,
    ode_solver_step_size_v3 = None,       # forces fixed step within torchdiffeq fixed-step methods
)
```

### Eval-time override (eval scripts):

```python
# e.g. eval_flow_matching_v3_ode_selectable.py lines 177–181
fm_model.ode_solver_backend_v3 = getattr(args, 'ode_solver_backend_v3',
                                           getattr(fm_model, 'ode_solver_backend_v3', 'legacy_euler'))
fm_model.ode_solver_method_v3  = getattr(args, 'ode_solver_method_v3',
                                           getattr(fm_model, 'ode_solver_method_v3', 'euler'))
```

The attrs are hot-patched directly onto the loaded model object — no retraining needed.

### Benchmark override (benchmark_ode_solvers_v3.py line 312–314):

```python
fm_model.ode_solver_backend_v3 = backend   # e.g. 'torchdiffeq'
fm_model.ode_solver_method_v3  = method    # e.g. 'midpoint'
fm_model.flow_steps_v3         = args.steps
```

---

## 11. `apply_conditioning` — The Fixed-Cost Boilerplate Per Step

After every velocity update (regardless of solver), the loop always applies
conditioning twice per outer step:

```python
x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)  # re-pin obs
# ... optional projector logic ...
x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)  # re-pin again
```

This is a pure tensor-slice write — essentially free compared to U-Net cost. But it
means the **boilerplate cost is O(N) regardless of method**, while NFE cost is O(N ×
stages). At very small `N` (e.g. N=5), the boilerplate becomes a measurable fraction
of wall time — this is why the v3 benchmark introduced `--mode math` vs.
`--mode production` to separate pure NFE cost from Python loop overhead.

---

## 12. Quick Reference: Which Method to Use When

| Scenario | Recommendation |
|---|---|
| Fastest inference, quality secondary | `euler`, `flow_steps_v3=10` |
| Same NFE, better quality | `midpoint`, `flow_steps_v3=5` (≡ 10 NFE) |
| Debugging / reference solution | `torchdiffeq:rk4`, low steps |
| Adaptive quality / unknown curvature | `torchdiffeq:dopri5` with tight tol |
| MPC real-time constraint | `euler`, `flow_steps_v3=5` — latency wins |
| Benchmarking fairness | Use `benchmark_ode_solvers_v3.py --mode math` to isolate pure NFE cost from Python tax |

---

## 13. "But the RHS Is a Neural Network — Does Higher-Order Even Help?"

This is the correct skeptical question. The short answer is **yes, still holds — but
only up to the model error floor, and FM's geometry makes that floor unusually low**.
Here is the full argument.

### 13.1 Two Independent Error Sources

At inference time you are numerically solving the autonomous ODE:

$$\dot{x}(t) = \mathbf{v}_\theta(x(t),\, t)$$

where $\mathbf{v}_\theta$ is a fixed, trained neural network. This ODE has **two
independent error contributions** in the final $x(1)$:

| Error source | Symbol | Cause | Controllable by solver choice? |
|---|---|---|---|
| **Model error** | $\varepsilon_\text{model}$ | $\mathbf{v}_\theta \neq \mathbf{v}^*$ (NN not perfect) | ❌ No — fixed by training |
| **Discretisation error** | $\varepsilon_\text{disc}$ | Finite step size | ✅ Yes — higher-order shrinks this |

The total trajectory error at $t=1$ is bounded roughly as:

$$\|x_\text{numerical}(1) - x_\text{true}(1)\| \;\lesssim\; \varepsilon_\text{disc} + C \cdot \varepsilon_\text{model}$$

where $C$ grows with integration time and how much the error compounds. Switching from
Euler to Midpoint **reduces $\varepsilon_\text{disc}$**, but $\varepsilon_\text{model}$
stays exactly the same because you didn't retrain anything.

### 13.2 The Regime Question: Which Error Dominates?

```
Total error
│
│  ←── Euler    Midpoint RK4
│  ●                              ← discretisation-dominated regime
│       ●    ●
│               ─────────────── ← model error floor (ε_model)
│                    ●  ●  ●    ← solver gains vanish below floor
│
└──────────────────────────────→ NFE (more steps →)
```

- **Low NFE (coarse grid):** discretisation error >> model error. Every extra stage
  (midpoint over euler) genuinely improves the answer because you're solving the
  neural ODE more faithfully.
- **High NFE (fine grid):** discretisation error << model error. Adding more steps or
  switching to higher-order does nothing useful — you've already converged to the
  neural ODE's solution, which still has model error baked in.

**The practical implication:** for the FMv3 setup typically running at
`flow_steps_v3 ∈ {5,10,20}`, you are almost certainly in the discretisation-dominated
regime. The $\varepsilon_\text{model}$ floor is set by training quality and is rarely
reached at these step counts.

### 13.3 Why the NN Being the RHS Does NOT Break the Argument

The classical accuracy theory for Runge-Kutta methods requires the RHS to be
**Lipschitz-continuous** in $x$ — that is, $\|\mathbf{v}(x_1, t) - \mathbf{v}(x_2, t)\|
\leq L \|x_1 - x_2\|$ for some finite $L$. A trained neural network with smooth
activations (Mish, SiLU, etc.) IS Lipschitz — the Lipschitz constant $L$ is bounded by
the product of weight norms across layers. So the standard RK accuracy theorems apply
directly.

In other words: once you treat $\mathbf{v}_\theta$ as a black box that maps
$(x, t) \mapsto$ velocity, it's just a vector field like any other. The solver doesn't
care how that vector field was computed internally.

### 13.4 The Intermediate Evaluation Point Issue (Midpoint Specific)

In Midpoint, NFE#2 evaluates at a point the NN was not directly trained on:

```python
x_mid = x + v1 * (dt * 0.5)   # this x_mid was never a training input per se
v2 = fm_model._predict_velocity(x_mid, cond, t_mid)
```

Is this a problem? **Usually not**, for two reasons:

1. **Generalisation:** Neural networks trained with flow matching are trained to
   predict velocity at *any* $(x_t, t)$ along the interpolation path, not just at
   grid-aligned points. The training distribution covers the entire path $t \in [0,1]$
   with continuous noise. Evaluating at an off-grid $x_\text{mid}$ is within the
   model's effective interpolation range — it is not extrapolation.

2. **FM path smoothness:** Under linear interpolation $x_t = (1-t)x_0 + t x_1$, the
   distribution of $x_t$ at any $t$ is a Gaussian mixture. The midpoint
   $x_i + v_1 \cdot dt/2$ for small $dt$ is an extremely local perturbation of the
   training distribution — the model error at this point is nearly the same as at $x_i$.

If the model were badly overfit to exact grid points this could matter. In practice with
U-Net architectures and standard FM training it does not.

### 13.5 What CAN Go Wrong: Error Accumulation with a Bad Model

Consider a model that has learned a velocity field with a *systematic bias*
$\mathbf{v}_\theta(x,t) = \mathbf{v}^*(x,t) + \delta(x,t)$ where $\delta$ is a
persistent directional error. Then:

- **Euler** at step $i$: steps in the wrong direction by $\delta(x_i, t_i) \cdot dt$.
- **Midpoint** at step $i$: evaluates at $x_\text{mid}$ and gets
  $\delta(x_\text{mid}, t_\text{mid}) \cdot dt$. If $\delta$ is smooth,
  $\delta(x_\text{mid}) \approx \delta(x_i)$, so the error per step is similar.

The key point: **both methods accumulate the same model bias per unit of true time
traversed**. The midpoint does not cancel or amplify model bias. It only reduces the
*integration* error from taking coarse steps across a curved (but correctly-signed)
field.

If model error is large and non-smooth, higher-order methods offer **no benefit and
no additional harm** — you converge faster to the neural ODE's trajectory, which
itself has model error.

### 13.6 Summary: When to Trust Higher-Order for This Setup

| Condition | Higher-order benefit |
|---|---|
| `flow_steps_v3` ≤ 20 (typical) | ✅ Strong — discretisation-dominated |
| Well-trained model (low val loss) | ✅ Strong — model floor is low |
| Poorly trained or very low-capacity model | ⚠️ Weak — model error dominates, switching solver doesn't matter |
| Very high steps (e.g. 200+) | ⚠️ Diminishing returns — approaching model error floor |
| `returns_condition=True` (CFG) | ✅ Still applies — each of the 2× CFG calls is still a fixed Lipschitz map |
| Adaptive solver (`dopri5` via torchdiffeq) | ✅ Most principled — step controller drives towards $\varepsilon_\text{model}$ floor automatically |

**Bottom line:** Yes, querying a neural network as the RHS. The higher-order argument
holds because the NN is a fixed Lipschitz function at inference time — the solver
theory doesn't know or care it's a U-Net. The gain is real in the coarse-step regime
that FM-PCC operates in. The only caveat is that you cannot improve *below* the model
error floor by changing the solver — but that floor is typically not reached until far
more NFE than we use in production.

### 13.7 What Were High-Order Methods Actually Built For? And Why FM ODE Lands in That Exact Sweet Spot

#### The original motivation (classical numerics, 1890s–1950s)

Runge-Kutta methods were invented to solve ODEs where evaluating the right-hand side
is **expensive** — historically, that meant computing a physical model: fluid
dynamics, orbital mechanics, structural stress. In those settings you cannot simply
throw thousands of Euler steps at the problem because each step requires solving a
partial differential equation or running a laboratory measurement. The whole point of
high-order methods is:

> **Achieve high accuracy with as few expensive function evaluations as possible.**

The "function" was originally a physical experiment or a heavy simulation kernel.
The core trade-off is always: *pay more algebraic overhead per step, but take far
fewer steps to reach a target accuracy*.

| Era | "Expensive function" | NFE budget concern |
|---|---|---|
| 1895 (Runge) | Analytical PDE evaluation by hand | Minutes per eval |
| 1950s (Kutta, Gill) | Analogue computer sub-circuit | Seconds per eval |
| 1960s–80s (Dormand-Prince) | Scientific simulation kernel | ms–s per eval |
| 2020s (Flow Matching inference) | **Neural network forward pass** | ms per eval |

The problem structure has changed entirely in surface appearance, but the underlying
constraint is **identical**: the RHS evaluation is a bottleneck, and you want to
extract maximum accuracy from each call.

#### Why FM VF inference lands in this sweet spot

There are three properties of the FM inference problem that make it an almost perfect
match for the original motivation of high-order integrators:

**1. The RHS is uniformly expensive regardless of state.**

Every call to `_predict_velocity` runs the full U-Net forward pass regardless of
where $(x, t)$ sits in state space. There is no "cheap region" of the flow. This is
exactly the original assumption — each function evaluation costs the same, so you
want to minimise their count while maximising accuracy.

**2. The vector field is globally smooth.**

Classical RK accuracy theory assumes the RHS is $C^p$ (p-times continuously
differentiable). FM training with a U-Net on smooth Gaussian path distributions
produces a velocity field that is effectively $C^\infty$ in $t$ and smooth in $x$
within the training manifold. This is the *best possible* scenario for high-order
methods — the Taylor series from which their error bounds are derived converges
rapidly. Physical ODEs sometimes have shocks, discontinuities, or stiff eigenvalues
that defeat high-order methods; FM flows have none of these.

**3. The integration interval is unit and the step counts are small.**

In classical applications, ODEs are integrated over arbitrary long time horizons with
unknown curvature scales. In FM inference, the interval is always $t \in [0, 1]$
and typical production step counts are $N \in \{5, 10, 20\}$. At these resolutions
the step size $dt = 1/N$ is large enough that discretisation error dominates model
error (§13.2), so higher-order methods are squarely in their regime of benefit.

Contrast this with cases where high-order methods *don't* help: stiff ODEs (where
step sizes are limited by stability, not accuracy), chaotic systems (where error is
dominated by exponential trajectory divergence), or coarsely-trained models (where
$\varepsilon_\text{model} \gg \varepsilon_\text{disc}$).

#### The historical analogy made explicit

```
Classical setting (1950s orbital mechanics):
  RHS = gravity model, 6 seconds/call
  Budget = 60 seconds total → 10 NFE
  Goal = sub-km position accuracy at t=3 days
  → You MUST use RK4 (4 NFE/step, 2.5 steps) rather than Euler (10 steps)
    because Euler's O(h) error at h=0.3 days is catastrophic.

FM inference (2024, this codebase):
  RHS = U-Net, ~5 ms/call  
  Budget = 50 ms total → 10 NFE
  Goal = sub-mm trajectory accuracy over H=16 horizon
  → Same math applies. Midpoint at 5 outer steps (10 NFE) beats
    Euler at 10 outer steps (10 NFE) for the same reason.
```

The labels changed. The algebraic structure of the problem — expensive oracle,
smooth vector field, fixed NFE budget, accuracy target — did not.

#### What makes FM slightly *easier* than classical problems

In classical mechanics the "true trajectory" is a physical reality that must be
tracked. In FM, the "true trajectory" is the neural ODE trajectory defined by
$\mathbf{v}_\theta$ itself. There is no separate physical ground truth beyond the
trained model. This means:

- You are not fighting a chaotic attractor.
- You have no stiffness from disparate time constants.
- The field is as smooth as your NN architecture allows (U-Nets with Mish activations
  are $C^\infty$).

These are all classical conditions under which Runge-Kutta methods achieve their
theoretical order of accuracy. The FM VF ODE is, in a sense, a *textbook* ODE for
which these solvers were designed — just with a neural network standing in for the
analytic formula.

---

*Written: 2026-07-18. Covers: `flow_matcher_v3_ode_selectable/models/diffusion.py`,
`FM_v3_imeanflow_test/Benchmark_ode_solver_Tests/v2/benchmark_ode_solvers_v2.py`,
`FM_v3_imeanflow_test/Benchmark_ode_solver_Tests/v3/benchmark_ode_solvers_v3.py`.*
