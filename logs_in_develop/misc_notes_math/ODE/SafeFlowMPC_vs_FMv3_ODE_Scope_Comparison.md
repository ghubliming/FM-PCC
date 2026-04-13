# SafeFlowMPC vs FM-v3 in ODE Scope (Math + Code)

## 1. What Problem Scope You Asked

You asked specifically: how SafeFlowMPC handles the same ODE-solving scope, with math and code, and how that compares to your FM-v3 code.

Scope boundary used in this note:

1. Core flow integration stepper during inference.
2. How training is formulated for the vector field.
3. Whether Newton/SQP-type optimization is part of the core flow ODE solver or a separate safety/projection module.

## 1.1 Important Terminology: Two Different "Solvers"

In this codebase context, "solver" can refer to two different things:

1. ODE solver (flow integrator): advances the learned flow dynamics over time.
2. Optimization solver (projection/safety): solves constrained optimization problems.

These are not the same module and not the same math task.

For flow matching inference, the ODE is

$$
\dot{x}(t) = v_\theta(x(t), t, c),
$$

and the numerical integrator in both FM-v3 and SafeFlowMPC core flow loop is Euler-style forward stepping.

Projection/safety modules solve constrained problems like

$$
\min_{z} J(z) \quad \text{s.t.} \quad g(z)\le 0,\; h(z)=0,
$$

which is a different problem class from integrating $\dot{x}=v_\theta(\cdot)$.

Quick code anchors for this distinction:

- FM-v3 ODE integrator update: [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L135)
- FM-v3 projection optimizer (SLSQP): [flow_matcher_v3/sampling/projection.py](../../../flow_matcher_v3/sampling/projection.py#L138)
- SafeFlowMPC ODE-style update: [../../../../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py](../../../../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py#L390)
- SafeFlowMPC safety optimizer (SQP_RTI): [../../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py](../../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py#L157)

## 2. FM-v3: Core ODE Handling in Your Code

### 2.1 Math model

FM-v3 inference integrates a learned velocity field:

$$
\dot{x}(t) = v_\theta(x(t), t, c), \quad t \in [0,1].
$$

With $N$ fixed steps and $h=1/N$:

$$
x_{k+1} = x_k + h\, v_\theta(x_k, t_k, c), \quad t_k = k/N.
$$

This is explicit Euler.

### 2.2 Code evidence (FM-v3)

- Euler step size and update:
  - [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L133)
  - [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L135)

```python
dt = 1.0 / max(self.flow_steps_v3, 1)
model_mean = x + velocity * dt
```

- Fixed-step loop:
  - [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L173)

```python
for i in range(total_steps):
```

- Eval path calls this sampler (not another solver):
  - [FM_v3_test/eval_FM_v3.py](../../../FM_v3_test/eval_FM_v3.py#L180)
  - [flow_matcher_v3/sampling/policies.py](../../../flow_matcher_v3/sampling/policies.py#L52)
  - [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L295)

## 3. SafeFlowMPC: Core ODE Handling

### 3.1 Math model

SafeFlowMPC uses the same forward-additive structure in flow steps:

$$
x_{k+1} = x_k + \Delta x_k, \quad \Delta x_k = h\,\hat v_\theta(x_k,t_k,c), \quad h = 1/N.
$$

So algorithmically this is also fixed-step explicit Euler-style integration of a learned vector field.

### 3.2 Code evidence (SafeFlowMPC)

- Step size and iteration in planner step:
  - [../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py](../../../../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py#L339)
  - [../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py](../../../../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py#L340)

```python
dt = 1.0 / self.config.flow_steps
for flow_step in range(self.config.flow_steps):
```

- Velocity increment from model is already multiplied by dt:
  - [../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/FlowMatchingField.py](../../../../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/FlowMatchingField.py#L86)

```python
dx_flow = self.ema.ema_model(x_current_unet, t, condition[None, :]) * dt
```

- Additive state update:
  - [../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py](../../../../SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py#L390)

```python
self.x_current += dx_flow
```

## 4. Training Formulation: FM-v3 vs SafeFlowMPC

Both projects train the field with sampled-time regression, not rollout integration.

### 4.1 FM-v3 training evidence

- Interpolation path sample:
  - [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L259)
- Regression target and prediction:
  - [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L276)
  - [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L279)

```python
return (1.0 - t_cont) * noise + t_cont * x_start
v_target = x_start - x_base
v_pred = self._predict_velocity(x_t, cond, t, returns=returns)
```

### 4.2 SafeFlowMPC training evidence

- Imitation training uses sampled path and MSE to target derivative:
  - [../SafeFlowMPC/train_imitation_learning.py](../../../../SafeFlowMPC/train_imitation_learning.py#L103)
  - [../SafeFlowMPC/train_imitation_learning.py](../../../../SafeFlowMPC/train_imitation_learning.py#L109)

```python
path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
loss = torch.pow(dxc - path_sample.dx_t, 2).mean()
```

- Safe variant also uses interpolation and velocity regression:
  - [../SafeFlowMPC/train_imitation_learning_safe.py](../../../../SafeFlowMPC/train_imitation_learning_safe.py#L179)
  - [../SafeFlowMPC/train_imitation_learning_safe.py](../../../../SafeFlowMPC/train_imitation_learning_safe.py#L189)

```python
x_t = x_0 + t[:, None] * (x_1 - x_0)
dx_t = x_1 - x_0
```

Interpretation: training in both codebases is learning $v_\theta$; integration method choice matters at inference.

## 5. Where Newton/SQP-Type Solvers Actually Appear

### 5.1 In your FM-v3 project

Projection module uses SLSQP in a constrained optimization step:

- [flow_matcher_v3/sampling/projection.py](../../../flow_matcher_v3/sampling/projection.py#L135)
- [flow_matcher_v3/sampling/projection.py](../../../flow_matcher_v3/sampling/projection.py#L138)

```python
res = minimize(..., method='SLSQP', ...)
```

This is separate from the core Euler stepper.

### 5.2 In SafeFlowMPC

Safety filter uses acados NLP/QP settings:

- [../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py](../../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py#L155)
- [../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py](../../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py#L157)
- [../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py](../../../../SafeFlowMPC/safe_flow_mpc/SafetyFilter/SafetyFilterAcados.py#L164)

```python
ocp.solver_options.integrator_type = "DISCRETE"
ocp.solver_options.nlp_solver_type = "SQP_RTI"
ocp.solver_options.hessian_approx = "EXACT"
```

Again: this is a safety optimization layer, not the core neural flow integrator.

## 6. Direct Comparison for Your Advisor

| Question | FM-v3 (your code) | SafeFlowMPC |
|---|---|---|
| Core flow ODE stepping in inference | Explicit Euler | Explicit Euler-style additive step |
| Fixed step size | Yes ($h=1/N$) | Yes ($h=1/N$) |
| Training does rollout integration? | No, sampled-time VF regression | No, sampled-time VF regression |
| Newton/RK used in core flow stepper? | No | No |
| Optimization solver present elsewhere? | Yes, projection SLSQP | Yes, safety SQP/acados |

## 7. Plain-English Bottom Line

SafeFlowMPC and your FM-v3 handle the core ODE scope in the same numerical style: forward fixed-step Euler-like updates over a learned velocity field.

The major difference is not the core integrator order; it is how each system wraps that integrator with constraint/safety optimization modules.

## 8. If You Need One-Line Meeting Answer

Both FM-v3 and SafeFlowMPC perform inference by fixed-step forward Euler-style integration of a learned vector field, while Newton/SQP methods are used only in separate projection/safety optimization modules, not as the core flow ODE solver.

## 9. Short FAQ (for "is solver = projection?")

Q: Is "the solver" in your advisor discussion the projection/safety solver?

A: Usually no. In ODE-method discussion, "solver" means the time integrator for $\dot{x}=v_\theta$ (Euler/RK/implicit). Projection/safety solvers are separate constrained optimizers that can modify or filter trajectories.

## 10. Should You Use Advisor-Recommended ODE Methods to Optimize Steps?

Short answer: yes, this is a good next experiment, especially if you want fewer steps with similar or better trajectory quality.

### 10.1 Why this can help

Current FM-v3 and SafeFlowMPC use first-order Euler-style integration.

For a fixed horizon $[0,1]$, if you reduce step count $N$, Euler error increases quickly. Higher-order methods can keep error lower at the same $N$:

- Euler global error: $O(h)$
- RK2 global error: $O(h^2)$
- RK4 global error: $O(h^4)$

where $h=1/N$.

This means you may be able to use fewer integration steps while preserving rollout quality.

### 10.2 Cost-quality tradeoff you should expect

Per integration step, model evaluations are approximately:

- Euler: 1
- Euler half-step: 2
- RK2: 2
- RK4: 4

Total runtime is roughly proportional to (evaluations per step) x (number of steps), plus overhead from projection/safety modules.

So the optimization target is not "higher-order always"; it is best quality under your wall-clock budget.

### 10.3 Practical recommendation for your codebase

Use this staged strategy:

1. Keep Euler baseline with current `flow_steps_v3`.
2. Add RK2 first (best effort-to-benefit ratio).
3. Add RK4 only if RK2 cannot hit your quality target at acceptable runtime.
4. Consider implicit/Newton-style ODE stepping only if you see clear stiffness-like instability in the flow field dynamics.

For your current projects, projection/safety optimization already adds computational load, so RK2 is usually a better first upgrade than jumping directly to RK4 or implicit methods.

### 10.4 Minimal evaluation protocol (advisor-ready)

Run the same seeds, same constraints, same tasks, varying only integration method and effective compute budget.

Suggested candidate set:

1. Euler @ N
2. Euler @ N/2
3. RK2 @ N/2
4. RK4 @ N/4

Compare metrics:

1. Success rate
2. Constraint/collision violations
3. Tracking error / trajectory smoothness
4. Average inference latency per step and per episode
5. If relevant: projection/safety intervention frequency

Decision rule:

Adopt the new method only if it gives better quality at equal runtime, or equal quality at lower runtime.

### 10.5 Integration location in FM-v3 code

The method swap belongs in the sampling update path where Euler is currently implemented:

- [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L133)
- [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L135)
- [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L173)

Conceptually, replace single-evaluation update with multi-stage evaluations (RK2/RK4), while keeping conditioning and projection logic unchanged.

### 10.6 Bottom-line recommendation

Advisor suggestion is technically well-motivated for your ODE scope. Start with RK2 benchmarking against Euler under matched runtime budgets, then decide if RK4 is worth the extra cost.
