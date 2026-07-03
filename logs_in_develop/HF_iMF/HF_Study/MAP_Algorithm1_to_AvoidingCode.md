# Map: HardFlow paper Algorithm 1 ↔ real `avoiding` task code

**Date:** 2026-07-03
**Paper source:** `/workspaces/FM-PCC/temp/HF_paper/HF/main.tex` (line numbers below refer to this file)
**Code source:** `/workspaces/HardFlow/hardflow/models_flow/flow_policy.py` and `flow_matcher.py`
(the `d3il` branch — the "avoiding" robotic-manipulation task, the one that actually runs)

**Purpose of this doc:** every symbol in Algorithm 1 (`main.tex:526-556`), traced to the exact
file+line in the code that computes it, plus the algebra showing *why* they're the same thing
even when the code doesn't look like the paper at first glance (it doesn't — the paper is
written for a **general affine scheduler**; the code hard-codes the **specific CFM scheduler**,
and one line in particular — the final state update — looks completely different until you
plug in the numbers).

---

## 0. The one-paragraph version

Algorithm 1 samples a flow-matching model **N** steps, but at each step, instead of just
taking an Euler step, it (1) predicts where the *whole trajectory* would end up if you
stopped steering right now, (2) nudges that predicted endpoint to satisfy your constraint
(obstacle avoidance) via a small optimization problem, then (3) partially applies that
nudge to the *current* step — not all of it, only a fraction proportional to how close you
already are to the end. Repeat until you reach the end, at which point the "predicted
endpoint" and "current step" are the same thing, so the constraint is satisfied *exactly*.

---

## 1. The scheduler specialization — the thing that makes the code look different from the paper

Algorithm 1 is written for a **general differentiable affine scheduler** `(α_t, β_t)` with
`X_t = α_t·X_1 + β_t·X_0`. The code only ever uses **one specific scheduler**: the standard
conditional flow-matching (CFM) interpolant.

| | Paper (general) | Code (CFM, hard-coded) |
|---|---|---|
| Interpolant | `X_t = α_t·X_1 + β_t·X_0` | `flow_matcher.py:71` — `ConditionalFlowMatcher.compute_mu_t`: `t*x1 + (1-t)*x0` |
| ⇒ | `α_t = t`, `β_t = 1-t` | (read off directly) |
| `α̇_t`, `β̇_t` | scheduler derivatives | `1`, `-1` |
| `Λ_t := α_t β̇_t - α̇_t β_t` | (Lemma, `main.tex:373`) | `t·(-1) - 1·(1-t) = -1` (**constant**, never appears explicitly in code — it's already been divided out) |
| Training target `v_{t\mid Z} = α̇_t X_1 + β̇_t X_0` | (`main.tex:369`) | `flow_matcher.py:85` — `compute_conditional_flow`: `x1 - x0` (matches: `1·X_1 + (-1)·X_0`) |

Every `Λ_t` in the paper's formulas below has already been replaced by `-1` in the code —
that's the first reason the code's algebra looks simpler than the paper's.

---

## 2. Symbol glossary (paper → code)

| Paper symbol | Meaning | Code name | File : line |
|---|---|---|---|
| `v_t^θ(x)` | learned marginal velocity field | `self.flow_model(x, t)` (called via `constrained_flow_fn_torch`) | `flow_policy.py:1709`, `1310-1314` |
| `x` (generic point in `ℝ^d`) | — | for this task, `x` is a **flattened (action,state) trajectory**, not a point — see §4 | — |
| `x_i`, `\bar{x}_{i+1}`, `x_{i+1}` | current / nominal-next / corrected-next ODE state | `x_k`, `x_next_ref`, `x_next` | `flow_policy.py:1323,1325,1360` |
| `t_i`, `Δt_i` | time grid, step size | `t_k`, `dt` | `flow_policy.py:1322,1320` |
| `N` (discretization steps) | | `self.oc_N_steps` = `cfg.ode_t_steps` (default **20**) | `flow_policy.py:700`; `config/flow_matching.py:65` |
| `𝓜_t^θ(x) := E[X_1∣X_t=x]` (posterior mean, Lemma `main.tex:361-380`) | | `x_terminal_predicted_ref` (the *unconstrained* version, called `\bar{x}_N` below) | `flow_policy.py:1340` |
| `\bar{x}_N := 𝓜_{t_{i+1}}^θ(\bar{x}_{i+1})` | nominal (unconstrained) predicted terminal state | `x_terminal_predicted_ref` | `flow_policy.py:1340` |
| `\hat{x}_N^*` | **constrained** predicted terminal state (solution of the NLP) | `x_terminal_predicted` | `flow_policy.py:1353` (solver) / `1356` (fallback) |
| `h(x) ≤ 0` | hard constraint | pillar/boundary/quadrilateral inequalities | `flow_policy.py:280-348` (`_apply_obstacle_constraints`) — see §5 |
| `C(x)` | optional cost/reward | `-self.oc_distance_objective` | `flow_policy.py:734-743` (`objective="distance"` branch) |
| `λ_oc` (control-regularization weight) | | `self.cfg.hardflow_reg_scale` (default **1.0**) | `flow_policy.py:712`; `config/flow_matching.py:85` |
| `α_{t_{i+1}}` (in the cost weight `λ_oc/(2Δt_i)·α_{t_{i+1}}²`) | | `self.oc_t_param` = `t_k+dt` = `t_{i+1}` (since `α_t=t` under CFM) | `flow_policy.py:714,1342` |
| the whole NLP `\eqref{final_optimization_formulation}` | | `hardflow_formulate()` builds it once (CasADi `Opti`); `hardflow_new_forward()` re-solves it every step by updating parameters | `flow_policy.py:683-751` (build) / `1351-1358` (solve) |
| `u_i^*` (recovered control, from Problem 3) | | `u_k = (x_next - x_next_ref) / dt` | `flow_policy.py:1368` |

---

## 3. Algorithm 1, line by line, matched to code

Quoting `main.tex:526-556` (`\begin{algorithm}...\end{algorithm}`, `\label{main_algorithm}`):

### Line: `\KwIn{...}`
Inputs: `p_0`, `v_t^θ`, `C(·)`, `h(·)≤0`, `λ_oc`, `N`, time grid, scheduler `(α_t,β_t)`.

Code: all baked into `self.cfg` (`FlowMatchingEvaluationConfig`, `config/flow_matching.py`)
+ `self.flow_model` (the trained checkpoint) + `hardflow_formulate(constraint=..., objective=...)`
(`flow_policy.py:683`) which builds `h`/`C` into the CasADi problem once, ahead of the loop.

### Line: `Draw initial state \bar{x}_0 ~ p_0 and set x_0=\bar{x}_0`

Code (`flow_policy.py:1302,1318`):
```python
best_s0_np, best_dof_chain_np = self.warmstart(conditions)   # picks x_0 from a batch, see §6
X_optimized = [best_dof_chain_np[0, :]]
```
**Not literally `p_0`-sampling** — HardFlow's `warmstart()` (`flow_policy.py:753-795`) rolls out
`warmstart_batch` (default 1) candidates through the *unconstrained* ODE and keeps the one the
value model likes best. This is engineering (a better starting point than raw noise), not part
of Algorithm 1's math — flagged again in §6.

### Line: `for i = 0 to N-1:`

Code (`flow_policy.py:1321`): `for k in range(self.oc_N_steps):`

### Line: `Compute Δt_i = t_{i+1} - t_i`

Code (`flow_policy.py:1320,1322`): `dt = 1.0 / self.oc_N_steps`; `t_k = k * dt` — uniform grid,
so `Δt_i` is the same constant every step (not recomputed per-step, just baked into `dt`).

### Line: `Compute \bar{x}_{i+1} = x_i + v_{t_i}^θ(x_i)·Δt_i`

Code (`flow_policy.py:1323-1325`):
```python
x_k = X_optimized[k]
v_k = flow_eval_np(x_k, t_k)
x_next_ref = x_k + v_k * dt
```
Direct, literal match — a plain Euler step.

### Line: `Compute \bar{x}_N = (β̇_{t_{i+1}}·\bar{x}_{i+1} - β_{t_{i+1}}·v_{t_{i+1}}^θ(\bar{x}_{i+1})) / Λ_{t_{i+1}}`

This is `𝓜_{t_{i+1}}^θ(\bar{x}_{i+1})` — the posterior-mean/x-prediction formula from Lemma 1
(`main.tex:361-380`), evaluated at the nominal next state.

Substituting the CFM values from §1 (`β̇=-1, β=1-t, Λ=-1`):
```
\bar{x}_N = ((-1)·\bar{x}_{i+1} - (1-t_{i+1})·v_{t_{i+1}}(\bar{x}_{i+1})) / (-1)
          = \bar{x}_{i+1} + (1-t_{i+1})·v_{t_{i+1}}(\bar{x}_{i+1})
```

Code (`flow_policy.py:1339-1340`):
```python
v_next = flow_eval_np(x_next_ref, t_k + dt)
x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * v_next
```
Exact match: `x_next_ref` = `\bar{x}_{i+1}`, `t_k+dt` = `t_{i+1}`, `(1.0-t_k-dt)` = `(1-t_{i+1})`.

**This is the line to remember if you only remember one thing**: `\bar{x}_N` is just "shoot the
current velocity estimate straight to the end (`t=1`)". It's a Taylor-linearization of where the
trajectory will land, computed with **one extra network call**, no numerical ODE integration to
`t=1`.

### Line: `Solve the optimization problem \eqref{final_optimization_formulation}`

```
\hat{x}_N^* = argmin_{\hat{x}_N}  C(\hat{x}_N) + λ_oc/(2Δt_i)·α_{t_{i+1}}²·‖\hat{x}_N - \bar{x}_N‖²
              s.t. h(\hat{x}_N) ≤ 0
```

Code: the *problem structure* is built once in `hardflow_formulate()` (`flow_policy.py:683-751`);
each loop iteration only updates the **parameters** (`\bar{x}_N`, `t_{i+1}`) and re-solves:

- cost (`flow_policy.py:710-715`):
  ```python
  self.oc_control_cost = 0.5 * self.cfg.hardflow_reg_scale \
      * cs.sumsqr(self.oc_X_terminal_predicted - self.oc_X_terminal_predicted_ref) \
      * self.oc_t_param**2
  ```
  `hardflow_reg_scale` plays the role of `λ_oc/Δt_i` (the `1/Δt_i` is pre-folded into the single
  config constant since the grid is uniform — `Δt_i` never changes, so there's no reason to
  divide by it every step); `oc_t_param**2` is `α_{t_{i+1}}²` (`=t_{i+1}²` under CFM).
  The `- self.oc_distance_objective * self.cfg.hardflow_cost_scale` term at `flow_policy.py:742`
  is the optional `C(\hat{x}_N)` (paper's optional reward/cost term, `main.tex:576`) — a
  goal-distance bonus (`_generate_distance_objective`, `flow_policy.py:477`), off by default
  (`objective=""`).
- constraint `h(\hat{x}_N)≤0` (`flow_policy.py:717-728`, dispatches to `_apply_obstacle_constraints`
  + optionally `_apply_dynamics_constraints`) — see §5 for what `h` actually *is* on this task.
- parameter update + solve (`flow_policy.py:1342-1358`):
  ```python
  self.oc_cs_opti.set_value(self.oc_t_param, t_k + dt)
  self.oc_cs_opti.set_value(self.oc_X_terminal_predicted_ref, x_terminal_predicted_ref)
  self.oc_cs_opti.set_initial(self.oc_X_terminal_predicted, x_terminal_predicted_ref)  # warm-start the NLP itself
  sol = self.oc_cs_opti.solve_limited()             # IPOPT
  x_terminal_predicted = sol.value(self.oc_X_terminal_predicted)   # = \hat{x}_N^*
  ```

### Line: `Compute x_{i+1} = α_{t_{i+1}}·\hat{x}_N^* + β_{t_{i+1}}·(-α̇_{t_{i+1}}·\bar{x}_{i+1}+α_{t_{i+1}}·v_{t_{i+1}}^θ(\bar{x}_{i+1})) / Λ_{t_{i+1}}`

This is the line that looks **nothing like the code** until you expand it. It's
`α_{t_{i+1}}·\hat{x}_N^* + β_{t_{i+1}}·𝒲_{t_{i+1}}^θ(\bar{x}_{i+1})` where `𝒲` is the
*noise*-side posterior estimate (the other half of Lemma 1, `main.tex:377-379`).

Substitute CFM values (`α=t, β=1-t, α̇=1, Λ=-1`):
```
x_{i+1} = t_{i+1}·\hat{x}_N^* + (1-t_{i+1})·(-\bar{x}_{i+1} + t_{i+1}·v_{t_{i+1}}(\bar{x}_{i+1})) / (-1)
        = t_{i+1}·\hat{x}_N^* + (1-t_{i+1})·(\bar{x}_{i+1} - t_{i+1}·v_{t_{i+1}}(\bar{x}_{i+1}))
        = (1-t_{i+1})·\bar{x}_{i+1} + t_{i+1}·\hat{x}_N^* - t_{i+1}(1-t_{i+1})·v_{t_{i+1}}(\bar{x}_{i+1})      (*)
```

Code (`flow_policy.py:1360-1362`):
```python
x_next = x_next_ref + (t_k + dt) * (x_terminal_predicted - x_terminal_predicted_ref)
```
i.e. `x_{i+1} = \bar{x}_{i+1} + t_{i+1}·(\hat{x}_N^* - \bar{x}_N)`. Expand using
`\bar{x}_N = \bar{x}_{i+1} + (1-t_{i+1})·v_{t_{i+1}}(\bar{x}_{i+1})` from the previous line:
```
x_{i+1} = \bar{x}_{i+1} + t_{i+1}·\hat{x}_N^* - t_{i+1}·\bar{x}_{i+1} - t_{i+1}(1-t_{i+1})·v_{t_{i+1}}(\bar{x}_{i+1})
        = (1-t_{i+1})·\bar{x}_{i+1} + t_{i+1}·\hat{x}_N^* - t_{i+1}(1-t_{i+1})·v_{t_{i+1}}(\bar{x}_{i+1})
```
**Identical to (\*).** ✓ The code's one-liner `x_next_ref + t_{i+1}·(x̂ - \bar{x}_N)` (a "gain-`t_{i+1}`
pull-back toward the constrained endpoint") is algebraically the same update as the paper's more
opaque general-scheduler formula — it's just simplified using the CFM identity
`x = α_t·𝓜_t(x) + β_t·𝒲_t(x)` (`main.tex:389`, "posterior_identity") before being coded up.

### Line: `\KwOut{Sample x_N}`

Code (`flow_policy.py:1373`): `optimized_final_dof = X_optimized[-1]` → later reshaped into
`(action, observations)` trajectories and returned (`flow_policy.py:1401-1419`).

---

## 4. What `x` actually *is* on the avoiding task (domain-specific instantiation)

Algorithm 1's `x ∈ ℝ^d` is written for a generic vector. On this task, `x` is a **flattened
trajectory**: `H` timesteps of `(action, state)` pairs, i.e.
`x = [a_0, s_1, a_1, s_2, …, a_{H-1}, s_H]` — `H·(action_dim+state_dim)` numbers, **minus**
`state_dim` because `s_0` (the robot's actual current position) is **not a free variable** —
it's conditioned/inpainted, fixed to the real observation. Hence:

```python
self.oc_dof = self.horizon * (self.state_dim + self.action_dim) - self.state_dim   # flow_policy.py:701
```

`s_0` is carried separately as `self.oc_s0_param` (`flow_policy.py:703`) and spliced back in
whenever the full trajectory is needed for a real flow-model call
(`constrained_flow_fn_torch`, `flow_policy.py:1695-1719`):
```python
full_flat = torch.cat([dof[:, :self.action_dim], s0, dof[:, self.action_dim:]], dim=1)   # :1699-1701
```
This is why `flow_eval_np` (`flow_policy.py:1310-1314`) takes `best_s0_torch` as an extra
argument the paper's abstract `v_t^θ(x)` doesn't need — the network always sees the full
trajectory including `s_0`, but the *optimizer* never touches `s_0`, only the free `oc_dof`
coordinates.

---

## 5. What `h(x) ≤ 0` actually *is* (the avoiding task's obstacles)

`_apply_obstacle_constraints` (`flow_policy.py:280-348`) loops over every horizon step (except
the first, conditioned one) and, for each predicted `(x,y)` robot position in `\hat{x}_N`:

- **circular keep-out** around each pillar (`flow_policy.py:315-322`):
  `‖pos - pillar_center‖ ≥ radius + margin` ⟺ `h = radius+margin-‖pos-center‖ ≤ 0`
- for the harder "novel" constraint variants (`flow_policy.py:324-346`): an extra circular
  obstacle, halfspace boundary lines, and/or a convex-quadrilateral keep-out
  (`_quadrilateral_constraint_expr`, `flow_policy.py:261-278`, built as a max of signed
  edge-distances — standard "outside every edge's half-plane" formulation).

Optionally, `_apply_dynamics_constraints` (`flow_policy.py:350-441`, gated by
`self.cfg.dynamics_constraint`) adds a **second, different** kind of constraint: a fitted
linear dynamics model `A·s + B·a + c = s'` enforced between every consecutive predicted
(state, action) pair — this has nothing to do with Algorithm 1's `h(x_N)≤0` (that's a
*terminal* constraint on the final predicted trajectory); it's an extra equality constraint
inside the same NLP, standing in for physical plausibility, not part of the paper's Algorithm 1.

---

## 6. What's in the code but NOT in Algorithm 1 (engineering, not math)

| Code feature | File:line | Why it's not in Algorithm 1 |
|---|---|---|
| `warmstart()` — rolls out `warmstart_batch` unconstrained candidates, keeps the value-model-best one as `x_0` | `flow_policy.py:753-795` | Algorithm 1 just says "draw `x_0~p_0`"; this is HardFlow's specific (better) choice of *which* sample to steer, not part of the transformation being proven. Paper does note (`main.tex:1226`, quoted below) that `\bar{x}_N` is "a natural warm start" — for the *optimizer's* initial guess, not this. |
| `hardflow_activation` (`'all'` vs `'late'`) — skip constraining for the first half of steps | `flow_policy.py:1327-1336`; config default `'all'` (`config/flow_matching.py:81`) | Directly implements a design choice the paper discusses in prose but does **not** put in the boxed algorithm: *"it is not necessary to solve the constrained optimization problem at every sampling step. In the early stages of sampling, both the posterior estimator and the fixed-point approximation are less accurate... we can skip the early steps and activate constrained optimization only in the later stages"* (`main.tex:1230`, appendix §"Feasibility, Stability, and Efficiency"). |
| `oc_dof` / `s0` splitting | `flow_policy.py:701,703,1695-1719` | Domain-specific — see §4. Algorithm 1's `x` is a bare vector; nothing in it addresses "part of the state is fixed by conditioning." |
| `_apply_dynamics_constraints` | `flow_policy.py:350-441` | Extra physical-plausibility equality constraints, orthogonal to the paper's single terminal inequality `h(x_N)≤0`. |
| `objective="distance"` / `_generate_distance_objective` | `flow_policy.py:477,734-743` | Instantiates the paper's optional `C(\hat{x}_N)` (`main.tex:576`, "may be solved... depending on the specific downstream application") as a concrete goal-distance reward for this task. |
| One-step fixed-point truncation choice | Baked directly into the derivation, not a runtime knob | The paper is explicit this is a **deliberate design choice** (`main.tex`, "This one-step cut-off is a deliberate design choice") — code has no `k`-iteration knob because the paper always uses `k=1` (see `hardflow` vs `hardflow_new`: both solve the *same* one-step surrogate; `hardflow_new` just evaluates the reference flow in PyTorch instead of through the `l4casadi` bridge — a numerics/dependency difference, not an algorithmic one). |

---

## 7. Quick config-knob reference (paper symbol → CLI/config default)

| Paper | Config field | Default | File:line |
|---|---|---|---|
| `N` | `ode_t_steps` | `20` | `config/flow_matching.py:65` |
| `λ_oc` (× the folded `1/Δt_i`) | `hardflow_reg_scale` | `1.0` | `config/flow_matching.py:85` |
| cost-term scale (if `C≠0`) | `hardflow_cost_scale` | `1.0` | `config/flow_matching.py:86` |
| "activate late" design choice | `hardflow_activation` | `'all'` | `config/flow_matching.py:81` |
| warm-start candidate count | `warmstart_batch` | `1` | `config/flow_matching.py:82` |
| dynamics constraint on/off | `dynamics_constraint` | `False` | `config/flow_matching.py:77` |
| obstacle keep-out margin | `obstacle_margin` | `0.0` | `config/flow_matching.py:75` |

---

## 8. Proposition 1 (safety guarantee) in one line

`main.tex:564`: *"The output sample `x_N` of Algorithm 1 satisfies the hard constraints
`h(x_N)≤0`, provided the feasible set is nonempty."* — Why this is true in the code: at the
**last** iteration (`k = N-1`, so `t_k+dt = 1`), `x_terminal_predicted_ref` (`flow_policy.py:1340`)
reduces to `x_next_ref + 0·v_next = x_next_ref` (since `1-t_{i+1}=1-1=0`), and the final update
(`flow_policy.py:1360-1362`) becomes `x_next = x_next_ref + 1·(x_terminal_predicted - x_terminal_predicted_ref)
= x_terminal_predicted = \hat{x}_N^*` — i.e. **the last output *is* the constrained NLP solution**,
which by construction satisfies `h(·)≤0`. This is the algebraic content of the paper's
`𝓜_{t_N}^θ(x_N)=x_N` "posterior estimator becomes exact unconditionally at the final step"
remark (`main.tex:1226`, and again the derivation at `main.tex:944`, *"for i=N-1, we can
simplify the update rule in Line~compute_next_state"*).
