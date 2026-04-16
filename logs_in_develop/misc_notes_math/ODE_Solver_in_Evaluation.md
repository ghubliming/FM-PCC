# ODE Solver in Evaluation (FM-v3): Math + Code Reference

## 1. Purpose

This note answers two questions clearly:

1. What ODE solver is currently implemented in FM-v3 evaluation?
2. How does that relate to variants often discussed by advisors (Exact, Euler, Euler half-step, RK2, RK4, Newton-based implicit solve)?

It also clarifies a third common confusion:

3. Is VF training using the same numerical ODE integration loop as evaluation?

Short answer:

- Current ODE stepper in FM-v3 sampling is fixed-step Explicit Euler.
- Optional SLSQP optimization exists in projection, but that is not the ODE integrator.

## 2. Mathematical Formulation

The sampling dynamics are modeled as:

$$
\frac{d\mathbf{x}}{dt} = f(t, \mathbf{x}), \qquad f(t,\mathbf{x}) = \mathbf{v}_\theta(\mathbf{x}, t, \mathbf{c}), \quad t \in [0,1].
$$

- $\mathbf{x}(t)$: trajectory state (actions/states depending on representation)
- $\mathbf{c}$: conditioning information
- $\mathbf{v}_\theta$: learned velocity field

With $N$ integration steps:

$$
h = \Delta t = \frac{1}{N}, \qquad t_k = \frac{k}{N}, \quad k=0,\dots,N-1.
$$

## 3. Method Implemented in Current Code

### 3.1 Explicit Euler update

The implemented update is:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + h\,f(t_k,\mathbf{x}_k).
$$

For FM-v3 notation:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + h\,\mathbf{v}_\theta(\mathbf{x}_k,t_k,\mathbf{c}).
$$

This is a forward integration from initial noise toward data-like samples.

### 3.2 Initialization and conditioning

- Initial sample: $\mathbf{x}_0 \sim 0.5\,\mathcal{N}(0,I)$
- Conditioning is re-applied after updates so fixed entries remain fixed.

### 3.3 Optional projection stage

Near final integration time ($t \approx 1$), projection may be applied:

- Gradient correction mode
- Optimization projection mode (SLSQP)

This projection step is downstream of the ODE stepper and should not be confused with the ODE integration method.

### 3.4 Evaluation vs VF training (critical distinction)

Simple mental model:

- Training = teach the velocity function.
- Evaluation = use the taught velocity function to march through time.
- Same VF network, different job.

Another analogy:

- Training is like learning the slope field from examples.
- Evaluation is like drawing a trajectory by following that slope field step-by-step.

Evaluation (sampling/inference):

- Uses an explicit numerical ODE stepper (Explicit Euler), repeatedly over time steps.
- This is where integration method choice (Euler/RK/Newton-implicit) matters directly.

VF training:

- Does not run a multi-step ODE integration loop.
- Instead, it samples a time $t$, builds $x_t$ on the interpolation path, predicts velocity $v_{pred}$, and regresses to target velocity $v_{target}$.
- So training is vector-field regression, not numerical rollout with Euler/RK/Newton per batch item.

Side-by-side summary:

| Aspect | VF Training | Evaluation / Sampling |
|---|---|---|
| Goal | Learn $v_\theta(x,t)$ | Generate trajectory using learned $v_\theta$ |
| Time use | Sample one random $t$ per item | Step through many times $t_0, t_1, ...$ |
| Main operation | Regression loss on velocity | Numerical ODE integration |
| Solver type | No Euler/RK/Newton rollout loop | Explicit Euler currently |
| Where in code | `q_sample`, `p_losses`, `loss` | `p_mean_variance`, `p_sample_loop` |

## 4. Code Evidence (Direct Quotes + Links)

### 4.0 Call-chain confirmation (yes, eval uses diffusion sampling)

The evaluation script does not implement a separate ODE solver. It calls policy, and policy calls the diffusion model sampler. Therefore, eval sampling behavior is exactly the sampling logic in `flow_matcher_v3/models/diffusion.py`.

Call chain:

- [../../FM_v3_test/eval_FM_v3.py](../../FM_v3_test/eval_FM_v3.py#L180)
- [../../flow_matcher_v3/sampling/policies.py](../../flow_matcher_v3/sampling/policies.py#L52)
- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L295)
- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L172)

In short: eval script is the caller; diffusion.py is the ODE stepper implementation.

### 4.1 Evaluation calls policy

Reference:

- [../../FM_v3_test/eval_FM_v3.py](../../FM_v3_test/eval_FM_v3.py#L180)

Quote:

```python
action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon, disable_projection=disable_projection)
```

### 4.2 Policy calls diffusion sampler

Reference:

- [../../flow_matcher_v3/sampling/policies.py](../../flow_matcher_v3/sampling/policies.py#L52)

Quote:

```python
samples, infos = self.model(conditions, returns=returns, projector=projector, constraints=constraints, horizon=horizon, **self.sample_kwargs)
```

### 4.3 Euler step size and update

References:

- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L133)
- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L135)

Quote:

```python
dt = 1.0 / max(self.flow_steps_v3, 1)
model_mean = x + velocity * dt
```

### 4.4 Time-stepping loop

References:

- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L172)
- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L173)

Quote:

```python
total_steps = self.flow_steps_v3 + repeat_last
for i in range(total_steps):
```

### 4.5 Projection trigger near end

Reference:

- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L178)

Quote:

```python
near_end = loop_idx >= (1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3 \
           if projector is not None else False
```

### 4.6 Projection optimizer (not ODE stepper)

References:

- [../../flow_matcher_v3/sampling/projection.py](../../flow_matcher_v3/sampling/projection.py#L135)
- [../../flow_matcher_v3/sampling/projection.py](../../flow_matcher_v3/sampling/projection.py#L138)

Quote:

```python
res = minimize(fun=cost_fun,
               x0=trajectory_np_double[i],
               constraints=constraints,
               method='SLSQP',
               jac=jac_cost_fun,
               ...)
```

### 4.7 Integration-step parameter

Reference:

- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L16)

Quote:

```python
flow_steps_v3=None, ode_inference_steps_v3=None,
```

### 4.8 VF training evidence (regression, not ODE stepping loop)

References:

- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L259)
- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L269)
- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L276)
- [../../flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py#L279)

Quotes:

```python
def q_sample(self, x_start, t, noise=None):
    ...
    return (1.0 - t_cont) * noise + t_cont * x_start
```

```python
def p_losses(self, x_start, cond, t, returns=None):
    ...
    v_target = x_start - x_base
    ...
    v_pred = self._predict_velocity(x_t, cond, t, returns=returns)
    loss, info = self.loss_fn(v_pred, v_target)
```

## 5. Methods Your Advisor Mentioned: Precise Comparison

Given $\dot{\mathbf{x}}=f(t,\mathbf{x})$ and step size $h$:

### 5.1 Exact solution

$$
\mathbf{x}(t_{k+1}) = \mathbf{x}(t_k) + \int_{t_k}^{t_{k+1}} f(t,\mathbf{x}(t))\,dt.
$$

- Usually unavailable in closed form for neural $f$.

### 5.2 Explicit Euler (current method)

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + h f(t_k,\mathbf{x}_k).
$$

- Global error: $O(h)$
- 1 function evaluation per step

### 5.3 Explicit Euler with half-step

Two substeps of size $h/2$:

$$
\mathbf{x}_{k+\frac12} = \mathbf{x}_k + \frac{h}{2} f(t_k,\mathbf{x}_k),
$$

$$
\mathbf{x}_{k+1} = \mathbf{x}_{k+\frac12} + \frac{h}{2} f\left(t_k+\frac{h}{2},\mathbf{x}_{k+\frac12}\right).
$$

- Better accuracy than coarse Euler
- About 2x evaluations per original step

### 5.4 RK2 (midpoint)

$$
\mathbf{k}_1=f(t_k,\mathbf{x}_k), \qquad
\mathbf{k}_2=f\left(t_k+\frac{h}{2},\mathbf{x}_k+\frac{h}{2}\mathbf{k}_1\right),
$$

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+h\mathbf{k}_2.
$$

- Global error: $O(h^2)$
- 2 evaluations per step

### 5.5 RK4

$$
\mathbf{k}_1=f(t_k,\mathbf{x}_k),
$$

$$
\mathbf{k}_2=f\left(t_k+\frac{h}{2},\mathbf{x}_k+\frac{h}{2}\mathbf{k}_1\right),
$$

$$
\mathbf{k}_3=f\left(t_k+\frac{h}{2},\mathbf{x}_k+\frac{h}{2}\mathbf{k}_2\right),
$$

$$
\mathbf{k}_4=f(t_k+h,\mathbf{x}_k+h\mathbf{k}_3),
$$

$$
\mathbf{x}_{k+1}=\mathbf{x}_k+\frac{h}{6}(\mathbf{k}_1+2\mathbf{k}_2+2\mathbf{k}_3+\mathbf{k}_4).
$$

- Global error: $O(h^4)$
- 4 evaluations per step

### 5.6 Newton-based implicit solve (not currently implemented in ODE stepper)

Implicit Euler would require solving:

$$
\mathbf{x}_{k+1} = \mathbf{x}_k + h f(t_{k+1},\mathbf{x}_{k+1}),
$$

which is nonlinear in $\mathbf{x}_{k+1}$ and typically solved via Newton iterations.

## 6. Interpretation: What Your Advisor Could Mean

Most likely intent is one of these:

1. Name the exact numerical method you currently use.
2. Compare baseline Euler against higher-order alternatives (Euler half-step, RK2, RK4).
3. Possibly test an implicit/Newton variant for stability analysis.

So your response can be:

- Current code: Explicit Euler for ODE stepping.
- Optional optimization: SLSQP in projection only.
- VF training: velocity-matching regression at sampled times, not ODE rollout integration.
- Next technical task (if requested): add inference solver variants and compare performance/quality.

## 7. What to Say in Meeting

### 7.0 If asked "Are training and eval the same ODE thing?"

No. Training learns the vector field by regression at sampled times; evaluation numerically integrates that learned field over many time steps.

### 7.1 One-line version

Our FM-v3 sampling ODE is currently solved with fixed-step Explicit Euler; Newton-type solving is not used in the ODE stepper and appears only in the separate projection optimization stage.

### 7.2 20-second version

We model sampling as $d\mathbf{x}/dt=\mathbf{v}_\theta(\mathbf{x},t,\mathbf{c})$ and discretize with fixed-step Explicit Euler, so each step is $\mathbf{x}_{k+1}=\mathbf{x}_k+h\mathbf{v}_\theta(\mathbf{x}_k,t_k,\mathbf{c})$. The SLSQP solver exists only in the projection module, not in ODE integration. If needed, we can benchmark Euler-half, RK2, RK4, and implicit-Newton variants.

## 8. Actionable Extension Plan (If Advisor Requests Variants)

Implement a stepper switch in FM-v3 sampler:

- euler
- euler_half
- rk2
- rk4
- optional implicit_euler_newton

Compare each variant on:

- success rate and constraint satisfaction
- trajectory quality
- runtime per environment step
- sensitivity to number of integration steps

Important scope note:

- Keep the VF training objective unchanged initially.
- Modify only the inference sampler stepper first.
- If advisor specifically asks, then consider training-time changes separately (for example, consistency with a chosen higher-order integrator).

## 9. Clarification: Is "ODE is just integration" wrong?

Short answer: it is partly correct, but incomplete.

### 9.1 What is correct in your statement

- An ODE defines a continuous-time dynamics law.
- To generate trajectories in practice, we integrate that ODE over time.

So saying "we integrate" is not wrong.

### 9.2 What is incomplete (and why advisor pushes back)

In numerical work, saying only "integration" is too vague because the chosen numerical method matters:

- Explicit Euler
- Euler half-step
- RK2
- RK4
- Implicit + Newton solve

These methods can produce different:

- accuracy
- stability
- runtime

So the key missing detail is the exact numerical integrator, not the general idea of integration.

### 9.3 Precise corrected statement

Instead of "it is just integration", say:

"We solve the sampling ODE by fixed-step Explicit Euler integration in eval; training learns the vector field by regression at sampled times."

### 9.4 What your advisor recommendation likely means

Your advisor is likely recommending a solver-comparison study, not saying your current implementation is invalid.

Meaning in practice:

1. Keep current Explicit Euler as baseline.
2. Add alternative inference steppers (Euler half-step, RK2, RK4, maybe implicit+Newton).
3. Compare quality/stability/runtime.
4. Justify final solver choice with empirical results.

### 9.5 Practical interpretation for your project

- Current code is a valid FM-v3 implementation using Explicit Euler.
- Advisor is asking for stronger numerical-method justification.
- Best next action: implement inference-time stepper variants first, then benchmark.

## 10. Performance Insights: The Paradox of Advanced Solvers in Python

Recent benchmarking of the "Legacy" (Direct-Math) solvers versus the "Torchdiffeq" (Package) solvers revealed a counter-intuitive paradox: **Higher-order solvers (RK4) in pure Python can be significantly slower than library-wrapped versions, even without the library tax.**

### 10.1 The Python Interpreter Ceiling

When we implement RK4 directly in a Python loop inside the benchmark script:
*   **Euler (10 steps)**: 10 Python-to-CUDA calls to the U-Net.
*   **RK4 (10 steps)**: **40 Python-to-CUDA calls** (4 passes per step).

At small batch sizes (**Batch=4**), the actual math calculation on the GPU is nearly instantaneous (sub-millisecond). The majority of the `avg_time` is spent on **CPU/Interpreter overhead**:
1.  Managing the Python `for` loop.
2.  Allocating small intermediate tensors for $k_1, k_2, k_3, k_4$.
3.  The overhead of the Python-to-C++ dispatch for every single U-Net call.

### 10.2 Why Torchdiffeq "Wins" at High Orders
Even though `torchdiffeq` pays a ~10ms "Entry Tax" to initialize the solver, it is much more efficient at managing the multi-stage transitions (the 4 k-values of RK4) once it is running. 

In contrast, our "Direct Math" implementation in the benchmark script hits the **Python Interpreter Ceiling** 40 times. This is why `legacy:rk4` (~450ms) appeared 3x slower than `torchdiffeq:rk4` (~150ms).

### 10.3 Identifying the Regimes

*   **Interpreter-Limited (Small Batch)**: The number of Python calls is the bottleneck. Euler (10 calls) is much faster than RK4 (40 calls).
*   **Compute-Limited (Large Batch)**: The GPU math becomes the bottleneck. At Batch 2048+, the difference between a Python loop and a compiled library starts to shrink, but the interpreter still imposes a large "latency floor."

### 10.4 The Final Conclusion on "Failure"
The advanced methods are **mathematically superior** (more accurate), but they are **computationally expensive in a raw Python loop**. 
To achieve the **20Hz (50ms) target** with RK4, we cannot use a Python loop at all. We must either:
1.  **JIT Compile** the entire integration loop using `torch.compile` to fuse the 40 calls into 1 GPU kernel.
2.  **Vectorize** the solver stages across the GPU directly.
- Best next action: implement inference-time stepper variants first, then benchmark.

## 11. The "Fixed-Step" Fallacy: Why RK4 Cannot Beat Euler in Raw Time

A common point of confusion is why "Advanced" methods like RK4 often show **higher** latency than "Simple" methods like Euler in these benchmarks.

### 11.1 The Mathematical Tax
Mathematically, an ODE solver's complexity is defined by its **Stages**.
*   **Euler**: 1-stage (1 U-Net pass per step).
*   **Midpoint/RK2**: 2-stage (2 U-Net passes per step).
*   **RK4**: 4-stage (4 U-Net passes per step).

At a **Fixed Step Count** (e.g., $S=10$), RK4 is performing **40 model passes**, while Euler is only performing **10**. 
> [!IMPORTANT]
> There is no scenario where a 4-pass algorithm can be faster than a 1-pass algorithm at the same step count. The "Advanced" label refers to **Accuracy**, not raw execution speed.

### 11.2 Why they often look "Equal" (Sub-optimal regimes)
In many of our B=4 benchmarks, RK4 (~120ms) and Euler (~110ms) look nearly identical. This is the **Package Tax Paradox**:
*   The 10ms "Fixed Tax" of the Python/Library overhead is so large compared to the 0.1ms GPU math that the 4x difference in stages is completely hidden by the noise of the interpreter.
*   The solvers look "equal" only because the system is so inefficient that the actual math doesn't matter yet.

### 11.3 The Real Value of Advanced Methods
The goal of using RK4 or Midpoint is **Step Reduction**. 

Because RK4 is more accurate ($O(h^4)$), you can potentially achieve the same trajectory quality with **3 steps** of RK4 (12 total passes) that would requires **20 steps** of Euler (20 total passes). 

**The Comparison that Matters**:
*   **Euler (S=20)**: 20 passes.
*   **RK4 (S=5)**: 20 passes.
*   **The Winner**: In this scenario, RK4 will likely produce a **much more accurate** result for the same time budget. 

**Summary**: In a benchmark with **fixed steps**, the higher-order methods will always be slower. You only see the performance benefit of "Advanced" methods when you allow the step count to reflect the higher accuracy of the algorithm.

## 12. Strategic Conclusion: When to Use Advanced Solvers?

The raw latency (ms) is only half of the story. The real decision to use an advanced solver depends on **Accuracy per Second**.

### 12.1 The "Step-Reduction" Strategy
The most powerful way to use RK4 is to **reduce the total step count ($S$)**. Because RK4 is a 4th-order method, it can often achieve higher precision in very few steps than Euler can in many steps.

| Metric | Euler (S=10) | RK4 (S=3) |
| :--- | :--- | :--- |
| **Total Model Passes** | 10 passes | **12 passes** |
| **Approx. Latency** | ~110 ms | **~125 ms** |
| **Accuracy (Fidelity)** | High Drift (Pink Line) | **Perfect (Olive Line)** |

**Decision**: For an extra 15ms, you gain massive trajectory fidelity. This is almost always the correct engineering trade-off in robotics.

### 12.2 Decision Matrix

| Choose **Explicit Euler** if... | Choose **RK4 / Midpoint** if... |
| :--- | :--- |
| Motion is simple/linear (low curvature). | Motion involves tight gaps or complex obstacles. |
| Latency is the absolute primary constraint. | **Accuracy/Success Rate** is the primary constraint. |
| You have a high frequency "Self-Corrector." | You have a **Long Horizon** ($H \ge 8$) where error accumulates. |
| You are in a Compute-Limited regime (Large B). | You are in an Interpreter-Limited regime (Small B). |

### 12.3 Final Technical Recommendation
Don't compare **Euler (S=10) vs. RK4 (S=10)**. Instead, find the **Accuracy Break-even Point**. 

For FM-PCC v3, we recommend testing **RK4 with $S=4$**. This provides a significantly more stable and accurate rollout than the current Euler $S=10$ baseline, with a total execution time that is likely comparable or better due to the reduction in total integration steps.
