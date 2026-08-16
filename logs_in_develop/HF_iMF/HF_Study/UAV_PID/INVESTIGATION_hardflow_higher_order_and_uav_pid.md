# Investigation — Can HardFlow be pushed to 2nd/higher-order systems? The UAV+PID case, and what control theory is still on the table

**Date:** 2026-08-13
**Type:** research investigation / direction-finding. **NO CODE WRITTEN. NO CODE CHANGED.**
**Question asked:** *"expand HF into 2nd/higher-order or the UAV PID system — read what the paper says,
read HF on the 1st-order avoiding task, read our UAV Gen11 code and notes, and see whether there is a
more complicated control-system research direction to build on HF."*

**Sources read for this doc**
- Paper: `/workspaces/aux_repo/HardFlow_Paper_Files/arXiv-2511.08425v3/main.tex` (full: Problem 1–6,
  Alg. 1, Prop. 1, Experiments, Appendix "Methodological Discussion", "Experiment Details",
  "High-Order Solvers and Non-Uniform Time Grids").
- Code: `/workspaces/aux_repo/HardFlow/hardflow/models_flow/flow_policy.py`,
  `hardflow/utils/avoiding_geometry.py`, `run/fit_dynamics.py`.
- Ours: `flow_matcher_v3_uav/sampling/projection.py`, `flow_matcher_v3_uav/utils/constraints_helpers.py`,
  `FM_v3_uav_test/eval_fm_uav.py`, `FM_v3_uav_test/mjpc_tracker.py`, `uav_env_test/flight_controller.py`,
  `config/uav_projection.yaml`, `config/uav.py`.
- Prior HF_Study / Gen11 docs, cited inline in §11.

---

## 0. The one-paragraph answer

**"Higher order" is three different questions wearing one name, and only one of them is open research.**
HardFlow is *already* order-agnostic in sampling time (Heun and non-uniform grids are in the paper's own
appendix) and it *already* runs on a 2nd-order plant (maze2d: $s=(p,v)$, $a=$ force). What HardFlow has
never done — and what our UAV forces — is a plant that is **nonlinear, underactuated, of relative degree
4 in the planned variable, and driven through an inner-loop tracker whose closed-loop time constant is
~22× longer than one control step**. That is not a "bigger $A,B,c$"; it breaks three assumptions at once:
the fitted **LTI** dynamics model, the identification of the **sample** with the **executed** trajectory,
and the silent assumption that per-sample feasibility (HF Prop. 1) implies closed-loop safety across
replans. Each broken assumption is a concrete, publishable direction, and two of them (§5.2 flat-output
actuation constraints, §5.4 tracker tube) are the *only* places where HardFlow's NLP can express
something the DPCC linear projector structurally cannot — which is exactly the axis on which HardFlow
has to beat the DPCC projector for us. **Recommended entry point: §5.3 → §5.1 → §5.4.** The gating
constraint on all of it is the 30.3 ms real-time budget (§8), which HF's published 190 ms/replan misses
by 6.3×.

---

## 1. Disambiguating "2nd/higher order" — three axes, only one is open

| # | Axis | What "order" means | HardFlow's status | Open for us? |
|---|---|---|---|---|
| **A** | **Sampling-time solver order** | Euler vs Heun vs RK on $\dot{x}_\tau = v^\theta_\tau(x_\tau)$, $\tau\in[0,1]$ | **Solved & published.** Appendix "High-Order Solvers and Non-Uniform Time Grids": swap $\Psi_i^\theta$, all theory carries. Heun measured *faster* than default on image editing (38.4 s vs 51.3 s). | **No.** Nothing to invent. It *is* a compute lever (§8). |
| **B** | **Plant order / relative degree** in physical time, i.e. what $h(x)\le 0$ asserts about physics | D3IL avoiding: **1st order** (single integrator, $a$ = desired velocity). Maze2d: **2nd order** but **LTI** (double integrator, $a$ = force). PDE: linear FD stencil. | **Partly open.** HF handles 2nd-order *linear*. It has never handled *nonlinear, underactuated, high relative degree*. **This is the UAV.** |
| **C** | **MPC-layer order** — how many receding-horizon loops are stacked and which are formalized | HF formalizes **one** MPC layer, in *sampling* time (Problem 3, horizon = 1 step, terminal proxy = posterior mean). The *physical-time* receding-horizon loop is left entirely informal. | **Wide open, and the deepest hole.** See §5.5. |

> The user's phrase "HF on the 1st-order avoiding task" is correct and is the right observation: the
> D3IL avoiding state is $s=(p_{\text{cur}}, p_{\text{des}}) \in \mathbb{R}^4$ with $a\in\mathbb{R}^2$ a
> *desired velocity* — a kinematic single integrator with **no momentum anywhere in the model**. That is
> the easiest possible plant, and every headline number in the paper's Table (safety 1.00, 52.5 steps)
> is measured on it.

---

## 2. What HardFlow actually assumes about the plant (evidence, not impression)

### 2.1 The physics enters only through $h(x)\le 0$, as a *fitted LTI equality*

Paper, Experiment Details §Robotic Manipulation:

$$ s_{i+1} = A s_i + B a_i + c,\qquad i=0,\dots,H-2,\quad A\in\mathbb{R}^{4\times4},\ B\in\mathbb{R}^{4\times2},\ c\in\mathbb{R}^4 $$

*"fitted via least squares on the training data."* Code: `run/fit_dynamics.py` (26 numbers, scikit-learn
`LinearRegression`, one per output dim), consumed at
`hardflow/models_flow/flow_policy.py:350` `_apply_dynamics_constraints` and stamped into CasADi at
`:389–405` as `A·s + B·a + c == s_next`, plus an $s_0$ anchor to the measured state at `:423–439`.
Already dissected in [`../FIT_DYNAMICS_what_the_linear_dynamics_model_is.md`](../FIT_DYNAMICS_what_the_linear_dynamics_model_is.md).

Three properties follow, and they are the real boundary of the method as published:

1. **Linear** — so the terminal constraint set $\{x: h(x)\le0\}$ stays benign for IPOPT even though
   the *obstacle* rows are nonconvex. The paper's own caveat (Appendix "Feasibility, Stability, and
   Efficiency"): performance degrades "when the feasible set is extremely irregular."
2. **Time-invariant and state-independent** — one $(A,B,c)$ for the whole workspace.
3. **Open-loop** — it models *plant*, not *plant + controller*. Legitimate for D3IL, where the
   end-effector is position-controlled and tracks essentially perfectly. **Not** legitimate for a
   drone (§3.3).

### 2.2 What HardFlow's structure *does* buy, that a linear projector cannot

This is the part worth keeping. HF's per-step subproblem (Alg. 1) is

$$ \widehat{x}_N^{*}=\arg\min_{\widehat{x}_N}\ C(\widehat{x}_N)+\frac{\lambda_{\text{oc}}\,\alpha_{t_{i+1}}^2}{2\Delta t_i}\lVert \widehat{x}_N-\bar{x}_N\rVert_2^2 \quad \text{s.t.}\quad h(\widehat{x}_N)\le 0 $$

— a **general NLP over the terminal sample**, solved by IPOPT, in the *natural* trajectory space (the
whole point of the reverse reparameterization: the neural network never enters the feasible set). By
contrast our DPCC projector (`flow_matcher_v3_uav/sampling/projection.py:38`) is a **linear**
equality/inequality projection plus hand-coded sphere/halfspace rows.

**Consequence, and it is the strategic one:** any constraint that is *nonlinear but tractable* — second-order
cones, norm bounds, HOCBF conditions — is free for HardFlow and structurally unavailable to the DPCC
projector. Per [`benchmark-hierarchy-who-beats-whom`], HardFlow has to beat the DPCC projector on
*something*. **§5.2 is that something**, and no one has claimed it yet.

### 2.3 What HardFlow guarantees, and what it does not

**Prop. 1 (`safety_guarantee`):** the output sample satisfies $h(x_N)\le 0$ — unconditionally, because
$\mathcal{M}^\theta_{t_N}(x_N)=x_N$ exactly at the last step, regardless of how bad the posterior
estimate was earlier. Clean, and genuinely stronger than projection-based baselines.

**What it does not say — three gaps, each a direction below:**

| Gap | Statement | Direction |
|---|---|---|
| G1 | The *sample* is feasible. The *executed* trajectory is a different object whenever the tracker is imperfect. | §5.4 |
| G2 | Feasibility holds *for this sample*. Nothing links successive replans — no recursive feasibility, no terminal set, no cost-to-go. | §5.5 |
| G3 | Feasibility is w.r.t. the *fitted* model. Model error is unmodelled; there is no $w_t$ slack, no chance constraint. | §5.4 / §5.8 |

The paper is honest about scope (Limitations §: "vision-based and contact-rich manipulation" named as
future work). Underactuated flight is not even in the limitations list.

---

## 3. What our UAV actually is (and why it is a genuinely harder object)

### 3.1 The tensor and the constraint, as they exist today

Transition layout (`FM_v3_uav_test/eval_fm_uav.py:729–733`), $H=8$, transition dim 12:

```
[ act(0,1,2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11) ]      act = Δp_des  (position delta, dt = 1.0)
```

The **entire** dynamics constraint is six `deriv` rows:

```python
constraint_list += [('deriv',[3,0]),('deriv',[4,1]),('deriv',[5,2])]   # p_des ← act
constraint_list += [('deriv',[6,0]),('deriv',[7,1]),('deriv',[8,2])]   # p     ← act
```

with `deriv` defined (`projection.py:435–487`) as the scalar Euler relation
$x_{t+1}=x_t+\Delta t\,\dot{x}_t$ (or its trapezoidal variant), and `dt: 1.0`
(`config/uav_projection.yaml`) because the action *is* a position delta.

**Three verified facts that define the gap:**

1. **`v(9,10,11)` is never constrained.** It is in the tensor, the FM predicts it, the projector ignores
   it. There is no row asserting $p_{t+1}=p_t+\Delta t_{\text{phys}} v_t$, none asserting
   $v_{t+1}=v_t+\Delta t_{\text{phys}} a_t$. The velocity channel is decorative. Everything load-bearing
   is first order.
2. **`DynamicConstraints` carries a single global `self.dt`** (`projection.py:429`, used at `:474–480`
   for every row). So you *cannot today* mix an "action-is-a-delta" row ($\Delta t=1$) with a true
   kinematic row ($\Delta t=1/33$ s) in the same projector. **Per-row `dt` is a hard prerequisite** for
   anything in §5.1.
3. **The only cross-time constraint family is `deriv`, and it is an equality.** There is no
   finite-difference *inequality* builder — so no acceleration bound, no jerk bound, no snap bound can
   be expressed at all, in either the projector or (by inheritance) a future HF port.

### 3.2 The plant is relative degree 4, not 1

A quadrotor has 6 DoF and 4 inputs. Position responds to thrust only *through* attitude:
thrust magnitude sets $\lVert \ddot p + g e_z\rVert$, attitude sets its direction, body rates come from
$\dddot p$ (jerk), and motor torques from $p^{(4)}$ (snap) — the standard differential-flatness chain
(Mellinger & Kumar 2011: $(p,\psi)$ are flat outputs). Our controller is exactly that structure:
`uav_env_test/flight_controller.py:1` — *"Cascaded PID flight controller for the Skydio X2
(Lee/Mellinger structure)"*, layers pos-PD → $R_{des}$ → SO(3) attitude PD → allocation to 4 thrusts.

So a plan expressed in positions must be at least $C^3$-plausible to be trackable, and the actuator
limits of the real vehicle are **statements about the 2nd–4th derivatives of the planned path**. Today
we constrain the 0th and (implicitly) the 1st. That is the precise sense in which our system is
"1st-order" while the vehicle is not.

### 3.3 The number that makes this urgent: the tracker is ~22× slower than one step

Outer loop (`flight_controller.py:89–91`): $a_{cmd}=-K_p e_p - K_d e_v + a_{des}$ with
$K_p=[4,4,8]$, $K_d=[3,3,4]$. Under timescale separation ($\ddot p \approx a_{cmd}$) the closed loop is
a linear 2nd-order system per axis:

| axis | $\omega_n=\sqrt{K_p}$ | $\zeta = K_d/2\omega_n$ | error time constant $1/(\zeta\omega_n)$ |
|---|---|---|---|
| x, y | 2.00 rad/s | 0.75 | **0.667 s** |
| z | 2.83 rad/s | 0.707 | **0.500 s** |

Against a control step of $1/33 = 30.3$ ms and a plan horizon of $H{=}8 \Rightarrow 0.242$ s:

> The constraint row `p ← act` asserts that the drone's **real** position reaches the commanded delta
> **within one 30.3 ms step**. The actual closed loop needs ≈ **0.667 s ≈ 22 steps ≈ 2.8 plan horizons**.
> The dynamics constraint is not merely "first order" — it is **~22× optimistic**, and it is a *hard
> equality* with no slack.

The DPCC paper's own dynamics carries a model-mismatch term $w_t$ *"accounting for the low-level
controller and numerical error"*; the projector's `deriv` rows have **no counterpart to $w_t$**. On the
arm, tracking error ≈ 0, so the omission is invisible. On the drone it is the dominant error term — which
is exactly what
[`TRACKING_ERROR_Gen11E7.md`](../../../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/data_example_anlysis/TRACKING_ERROR_Gen11E7.md)
set out to measure (`max_track_err = 2.072 m` in the logged rollout) and what
[`CRITIQUE_three_layer_absurdity.md` §0.3](../../../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/data_example_anlysis/CRITIQUE_three_layer_absurdity.md)
identified as "the UAV lags — underactuated, second-order."

### 3.4 Corroborating evidence that dynamics is load-bearing

[`INVESTIGATION_geo_free_model_free_worse_than_diffuser.md`](../../../Gen11/Epoch9_PCC_Constraints/U_13/INVESTIGATION_geo_free_model_free_worse_than_diffuser.md)
found the variant ordering `model_free < geo_free-model_free < geo_free-bounds_free < diffuser`, i.e.
**geometry projected *without* the dynamics row actively destroys the raw FM output**, because dynamics
is the only thing coupling the executed action to the constrained states. Read forward: if the
*coupling* row is the load-bearing piece, then **making that row physically correct is the highest-leverage
change available**, above any new geometry.

---

## 4. The gap, in one table

| Physical requirement | Expressible today (DPCC projector)? | Expressible in HF's NLP? | Direction |
|---|:--:|:--:|---|
| $p$ integrates commanded $\Delta p_{des}$ | ✅ `deriv` | ✅ linear eq | shipped |
| $v$ is the true derivative of $p$ | ❌ (single global `dt`) | ✅ | §5.1 |
| acceleration / jerk / snap bounds | ❌ (no FD inequality family) | ✅ linear ineq | §5.1, §5.2 |
| thrust magnitude limit $\lVert\ddot p+g e_z\rVert\le T_{max}/m$ | ❌ (nonlinear) | ✅ **SOC** | §5.2 |
| tilt limit | ❌ | ✅ **SOC** | §5.2 |
| closed-loop (PID-in-the-loop) dynamics | ⚠️ only if fitted as LTI | ✅ | §5.3 |
| executed path safe (not just the plan) | ⚠️ hand-tuned `enlarge_constraints: 0.025` | ✅ computed tube | §5.4 |
| stoppable at the end of the horizon | ❌ | ✅ | §5.5 |
| momentum-aware obstacle margin (relative degree ≥ 2) | ❌ | ✅ HOCBF | §5.6 |

---

## 5. Research directions

Each entry: **idea → math → what it costs → what would count as a result → risk.**
Ranked recommendation at the end of the section.

### 5.1 D1 — Make the plan second-order: kinematic chain + finite-difference bounds *(cheap, linear, enabling)*

**Idea.** Promote `v` from decoration to constraint, and add the missing derivative bounds.

**Math.** With per-row $\Delta t$: kinematic row $p_{t+1}=p_t+\Delta t_{\text{phys}} v_t$ ($\Delta t_{\text{phys}}=1/33$),
alongside the existing $p_{des,t+1}=p_{des,t}+\text{act}_t$ ($\Delta t=1$). Then $k$-th order
finite-difference bounds, all **linear in $x$**:

$$ \lVert \Delta^{(2)} p_t \rVert_\infty \le a_{max}\Delta t^2,\qquad
   \lVert \Delta^{(3)} p_t \rVert_\infty \le j_{max}\Delta t^3,\qquad
   \lVert \Delta^{(4)} p_t \rVert_\infty \le s_{max}\Delta t^4 $$

**Cost.** Two additions to `projection.py`: per-row `dt` in `DynamicConstraints`, and a new
`fd_bound(order, dims, limit)` family. Both are linear → they work **unchanged in the DPCC projector
and in a future HF NLP**. This is the single piece of infrastructure every other direction needs.

**Result that would count.** Executed tracking error and contact rate under `dpcc-c` with vs without the
$v$ chain and $a/j$ bounds, on `corridor` / `pillars` / `s_curve`, 5 seeds. Hypothesis: bounded jerk
reduces `max_track_err` materially without hurting steps-to-goal.

**Risk.** $H=8$ at 33 Hz is 0.242 s — a 4th-order finite difference over 8 samples is numerically thin.
Expect $a$ and $j$ bounds to be meaningful and **snap to be noise**. Do not over-claim snap.

---

### 5.2 D2 — Flat-output actuation constraints: the convex-but-nonlinear regime *(highest novelty per unit of work)*

**Idea.** Stop bounding a proxy (`action_bounds: 'auto'`, derived from the dataset's own $\Delta p_{des}$
range) and bound the **actual actuator**, expressed purely in the planned positions via differential
flatness.

**Math.** Let $u_t = \Delta^{(2)}p_t/\Delta t^2 + g e_z$ (the required specific thrust, linear in $x$).

- **Thrust upper bound** — a second-order cone, convex: $\lVert u_t\rVert_2 \le T_{max}/m$
- **Tilt bound** — also SOC: $u_{t,z} \ge \cos\theta_{max}\lVert u_t\rVert_2$
- **Thrust lower bound** — $\lVert u_t\rVert_2 \ge T_{min}/m$ is a *reverse* cone, **nonconvex**; the
  standard fix is the linear sufficient condition $u_{t,z}\ge T_{min}/m$, valid under the tilt bound.
- **Body-rate bound** via jerk (flatness): $\lVert\omega\rVert \lesssim \lVert \dddot p_\perp\rVert/\lVert u\rVert$ — a §5.1 jerk bound is the tractable surrogate.

**Why this is the strategic direction.** These are exactly the constraints the DPCC linear projector
**cannot** represent without linearizing them away, and exactly the ones IPOPT eats without complaint.
It converts "HardFlow vs DPCC projector" from a tie on shared constraint classes into a comparison where
HF enforces a strictly richer physical specification. It also replaces a *learned-data-range* action
bound with a *vehicle-datasheet* bound — much stronger to write up.

**Result that would count.** (i) Fraction of executed steps at motor saturation
(`CascadedPID.last_raw_saturated` telemetry already exists, `flight_controller.py:76`) — plans satisfying
the thrust cone should saturate far less; (ii) tilt-limit violations; (iii) safety/steps held or improved.

**Risk.** Timescale separation ($\ddot p \approx a_{cmd}$) is an approximation — the attitude loop
($K_{p,att}=[70,70,4]$) is fast but not instantaneous. The cone bound is therefore *necessary-ish, not
sufficient*; state it as such. Also: mass/inertia must come from the MuJoCo model
(`body_subtreemass`, already read at `flight_controller.py:39`), not from a guess.

---

### 5.3 D3 — The tracker-in-the-loop model: what HardFlow's $A,B,c$ *should* be for us *(most HF-faithful; recommended first)*

**Idea.** HF fits its dynamics on the *plant*. For us the object between "what the planner commands" and
"where the drone goes" is **plant + cascaded PID**. Fit/derive *that*.

**Two routes:**

- **D3a — fitted (literally HF's `fit_dynamics.py`, retargeted).** Roll out with the PID in the loop, dump
  $\big((p_{des,t}, p_t, v_t),\ \Delta p_{des,t}\big)\rightarrow (p_{t+1},v_{t+1})$ pairs, least-squares an
  $(A,B,c)$. This is the *faithful* HF import and it is nearly free. Note the protocol asymmetry already
  flagged in [`FIT_DYNAMICS…` §6b](../FIT_DYNAMICS_what_the_linear_dynamics_model_is.md): HF *fits*
  where we *assert*. Today our UAV rows are asserted identity — the least defensible position of the three.
- **D3b — analytic.** We *know* the gains. With $s=(p,v)$, $u=p_{des}$:
  $$ \dot s = \begin{bmatrix}0&I\\-K_p&-K_d\end{bmatrix} s + \begin{bmatrix}0\\K_p\end{bmatrix}u
     \;\xrightarrow{\text{ZOH},\,\Delta t=1/33}\; s_{t+1}=A_d s_t + B_d u_t $$
  Exact, no data, no normalizer-units hazard, and it drops straight into HF's `A·s+B·a+c` slot.
  (Include the $v_{des}=\Delta p_{des}/\Delta t$ feed-forward term the eval already computes.)

**Why it matters.** This is the direct fix for the 22× optimism of §3.3. It makes the plan *feasible for
the controller we actually fly*, not for a hypothetical infinitely-fast one.

**Result that would count.** Predicted-vs-actual one-step position error of the constraint model,
measured on logged rollouts: identity rows vs D3a vs D3b. This is a **pure offline analysis on existing
logs** — no new training. Cheapest decisive experiment in this document.

**Risk.** ⚠️ **Normalizer units.** Same trap flagged for Gen12: an $(A,B,c)$ fitted under one normalizer
silently enforces wrong physics under another. D3b sidesteps it (build in SI, convert once, explicitly).
Second risk: with MJPC as tracker (`mjpc_tracker.py:25`) instead of PID, the closed loop is a *sampling
MPC*, not an LTI system — D3b does not apply, D3a does (and it becomes an approximation of an
optimizer). Choose the tracker deliberately; do not silently mix.

---

### 5.4 D4 — From a hand-picked margin to a computed tube: safety of the *flown* path *(closes gap G1)*

**Idea.** HF Prop. 1 guarantees the **sample** is safe. We fly the **execution**. Bridge them with the
classical tube/ISS argument, and let the tube *replace* the magic number.

**Math.** With the linear closed loop of §5.3 and error $e=p-p_{des}$, standard $\mathcal{H}_\infty$/ISS
bounds give
$$ \lVert e\rVert_\infty \;\le\; \gamma_1 \lVert \dot p_{des}\rVert_\infty + \gamma_2 \lVert \ddot p_{des}\rVert_\infty $$
with $\gamma_i$ computable from $(K_p,K_d)$. Since §5.1/§5.2 already bound $\dot p_{des}$ and
$\ddot p_{des}$ **in the same optimization**, the tube radius $\varepsilon$ is *self-consistently
determined by the plan's own constraints*. Then tighten every obstacle/halfspace by $\varepsilon$:
$$ h_{\text{tightened}}(x) = h(x) + \varepsilon \;\le\; 0 \quad\Longrightarrow\quad \text{the executed path is safe.} $$

**Why this is well-aimed at our codebase.** The tightening *mechanism already exists*:
`enlarge_constraints: 0.025` plus `inflation.{r_drone, margin_base}` in `config/uav_projection.yaml`, and
the `-tightened` variant family is already wired through every projection variant. Today 0.025 m is
hand-picked. D4 turns it into a **derived** quantity — same code path, real theory behind the number.
It also directly addresses the plan-vs-execution discrepancy `_exec_constraint_violations`
(`eval_fm_uav.py:278`) was written to detect: it checks the *flown* path against raw geometry, precisely
because plan feasibility was known not to imply execution safety.

**Result that would count.** Executed violation count vs $\varepsilon$: hand-picked 0.025 vs computed
tube, across scenes. Prediction: the computed tube is *larger* on aggressive scenes (`s_curve`) and
*smaller* on `corridor` — i.e. it reallocates conservatism where it is needed instead of applying a flat
margin. That reallocation, if observed, is the paper figure.

**Risk.** A tube that is too large makes tight scenes infeasible. Mitigation is standard and worth
stating: if the tightened problem is infeasible, fall back to the untightened one and log it (soft
degradation, never a crash) — and the fallback rate is itself a metric.

---

### 5.5 D5 — Terminal ingredients and recursive feasibility for the *physical-time* loop *(the deepest hole; highest ceiling)*

**Idea.** HF calls itself MPC, but its MPC lives in **sampling time** (Problem 3: horizon 1, terminal
proxy $\mathcal{M}^\theta_t$). The *physical-time* receding-horizon loop — replan, execute, replan — has
**no MPC theory attached at all**, in HF, in DPCC, or in ours.

Classical MPC (Mayne et al. 2000) says a receding-horizon scheme is recursively feasible and stabilizing
when it carries three **terminal ingredients**: a terminal cost, a terminal *control-invariant* set, and
a local controller that keeps you in it. A generative planner has **none** of the three.

**Math, made concrete for the UAV.** The natural terminal set is *stoppability*: require the plan's
terminal velocity to be brakeable within the free space ahead,
$$ \lVert v_H\rVert_2^2 \;\le\; 2\,a_{brake}\, d_{\text{free}}(p_H), \qquad a_{brake} = T_{max}/m - g $$
which is an SOC constraint for a fixed conservative $d_{\text{free}}$, and a per-obstacle SOC otherwise.
Physically: *"never plan into a state you cannot stop from."* This is precisely the failure mode of a
momentum-carrying vehicle guided by a momentum-blind planner, and it is one extra constraint row.

**Why it is the biggest idea here.** It reframes the contribution from "a better sampler" to **"the first
receding-horizon generative planner with terminal ingredients"** — the missing half of the MPC analogy
that HF's own title invokes. It composes with everything above rather than competing with it.

**Result that would count.** Crash-at-speed rate and the `steps` distribution with vs without the
terminal stoppability row; plus a *recursive-feasibility* metric — the fraction of replans where the
previous plan's tail remains feasible (the standard "shift-and-append" test). That metric does not exist
in any of the three codebases; building it is itself a contribution.

**Risk.** Overly conservative $d_{\text{free}}$ turns the drone into a crawler; steps-to-goal will
regress and that is a real Pareto trade-off, not a win — report it as such per
[`pareto-definition-of-good`].

---

### 5.6 D6 — HOCBF inside HF's terminal constraint *(the invariance-flavored alternative to D4)*

**Idea.** For a double integrator, a distance CBF $h(p)\ge0$ has **relative degree 2**: the input does
not appear in $\dot h$. The control-theoretic answer is a High-Order CBF (Xiao & Belta):
$\psi_0=h$, $\psi_1=\dot\psi_0+\alpha_1(\psi_0)$, $\psi_2=\dot\psi_1+\alpha_2(\psi_1)\ge0$. Enforcing
$\psi_2\ge0$ along the plan yields margins that **scale with approach speed** — the momentum-aware
version of `enlarge_constraints`.

**Positioning.** SafeDiffuser and SafeFlowMatcher apply CBFs **pathwise, in sampling time** — the exact
target of HF's central criticism (constraining transient iterates is "overly restrictive"). Applying a
**HOCBF to the terminal sample's physical-time trajectory** is not the same object and is not covered by
that criticism: it keeps HF's terminal-only philosophy while getting relative degree right. That is a
clean, defensible novelty claim, and `aux_repo/SafeFlowMPC` is on hand as the reference implementation
to diff against.

**Risk.** D4 and D6 solve overlapping problems (safe margins under momentum) by different means —
robustness vs invariance. **Do not build both first.** D4 is cheaper and reuses existing plumbing; D6 is
the theory upgrade. Recommend D4 → measure → D6 only if the tube proves too conservative.

---

### 5.7 D7 — Sampling-time order and step count as the *compute* lever, not a research question

Axis A of §1 is closed as research but decisive as engineering: every direction above adds NLP work per
sampling step, and §8 shows we have no headroom. The levers, in order of leverage:

1. **Fewer function evaluations** — MeanFlow / iMF (Gen3v6, Gen3v7, Gen13) collapse the ODE to $K\approx5$
   steps. [`ANALYSIS_hardflow_vs_dpcc_planning_structure.md`](../../Research/ANALYSIS_hardflow_vs_dpcc_planning_structure.md)
   measured 42 intermediate plans/episode for iMF vs 70 for FM — 1.95× fewer NFE, hence proportionally
   fewer NLP solves. **This is the direct enabler of everything in §5.2–§5.6 on the UAV**, and it is
   exactly the "Gen15 UAV Mix-ML" idea already on the roadmap. The synergy is real and worth stating
   explicitly: *iMF is what makes HardFlow affordable at 33 Hz.*
2. **Solve only in the late steps** — HF already does this (Experiment Details: second half only;
   Remark on posterior accuracy). Our projector's `diffusion_timestep_threshold: 0.5` is the same idea,
   already tuned.
3. **Solver class** — a pure QP (§5.1) or SOCP (§5.2) with warm start from $\bar x_N$ is far faster than
   general IPOPT on a nonconvex NLP. Keeping the constraint set **convex** is a compute decision as much
   as a modelling one.

### 5.8 D8 — Robust / chance constraints (speculative, park it)

Wind, model error, and estimator noise are unmodelled in HF (gap G3). Chance-constrained terminal
feasibility $\Pr[h(x_N)\le0]\ge 1-\delta$ or a disturbance-augmented tube is the standard extension, and
it is where D4 naturally grows. **Not** a starting point: we have no disturbance model in the MuJoCo
scenes today, so there is nothing to be robust *to* without first building one.

---

### 5.9 Recommended order

| Rank | Direction | Why here | Effort |
|---|---|---|---|
| 1 | **D3 (§5.3)** — closed-loop model, D3b analytic + D3a check | Decisive **offline** experiment on existing logs; fixes the 22× optimism; the most defensible gap to close | S |
| 2 | **D1 (§5.1)** — per-row `dt`, $v$ chain, FD-bound family | Infrastructure every later direction needs; linear, so it serves the DPCC arm too | S–M |
| 3 | **D4 (§5.4)** — computed tube replacing `enlarge_constraints` | Reuses existing plumbing; converts a magic number into theory; targets a measured failure | M |
| 4 | **D2 (§5.2)** — thrust/tilt SOC in flat outputs | The HF-beats-DPCC-projector axis; needs an NLP/SOCP arm in the UAV loop first | M–L |
| 5 | **D5 (§5.5)** — terminal stoppability + recursive feasibility | Highest ceiling, but only meaningful once 1–4 make the plan physically honest | L |
| 6 | **D6 (§5.6)** — HOCBF | Only if D4's tube proves too conservative | L |

---

## 6. Case study — could HardFlow solve the velocity/PID problem Gen11 dropped?

*(Added 2026-08-13, in response to: "we intended to control the velocity at first, then dropped into
`pid_stopgo`.")*

**Short answer: partially, and in a specific and interesting way.** HardFlow cannot fix the half of the
problem that is a *conditioning* problem — it never changes what the network sees. But the half that
killed velocity control in Gen11 was a **consistency and boundary-condition problem**, and that half is
exactly what a hard terminal constraint is for. Concretely: **HF can restore momentum awareness to the
current velocity-blind 9D checkpoint at inference time, with no retrain, via three linear rows.** That is
a cheap, decisive, and — as far as I can tell — unclaimed experiment.

### 6.0 First, disambiguate "velocity" — it means three different things in this project

Same trap as §1. These get conflated constantly and the arguments below are unreadable if they blur:

| symbol | name | lives in | who produces it |
|---|---|---|---|
| $v_\tau^\theta(x)$ | **flow velocity** — the learned field, in sampling time $\tau\in[0,1]$ | the U-Net | training |
| $v$ | **physical velocity** of the drone, m/s | `data.qvel[:3]`; tensor cols 9–11 *in 12D only* | the simulator |
| $v_{des}$ | **commanded velocity** — PID feed-forward | computed at runtime in `rollout_one` | `eval_fm_uav.py:986–990` |

Worth noting in passing: for the standard scheduler $(\alpha_t{=}t,\ \beta_t{=}1{-}t)$ we get
$\Lambda_t=-1$, so HF's posterior mean collapses to $\mathcal{M}_t(x)=x+(1-t)\,v^\theta_t(x)$ — "where
the *flow* velocity says this sample is heading." The name collision with physical velocity is
unfortunate and is not a relationship.

### 6.1 What actually happened in Gen11 (the record)

| | FM obs / transition | $v_{des}$ to PID | Flight | Status |
|---|---|---|---|---|
| **E7 Option 1** | `[p_des\|p\|v]` / **12D** | $\Delta p_{des}/dt_{fm}$ | continuous | ✅ called "working baseline" |
| **U2 Option 2** | `[p_des\|p]` / **9D** | $\mathbf{0}$ | **strict stop-and-go** | ⬅ **current default** (`config/uav.py:133–134`, `:190–191`) |
| **E8 Option 3** | 9D | ignored (MJPC internal) | continuous-ish | ⚠ deprioritised (~50× PID cost) |
| **Option 4** | 9D | $\Delta p_{des}/dt_{fm}$ | continuous | ❌ **rejected — "unreliable by design"** |
| **U3 `pid_const_v`** | 9D | $\text{unit}(a)\cdot\bar v$, $\bar v = \overline{\lVert\Delta p_{des}\rVert}\times 33$ | continuous | implemented, timing-free but magnitude-blind |

Sources: [`MEMO_controller_options.md`](../../../Gen11/Epoch8_UAV_Mjpc_thrust_control/U2_PID_Stop_Go/MEMO_controller_options.md),
[`PLAN_PID_Stop_Go.md`](../../../Gen11/Epoch8_UAV_Mjpc_thrust_control/U2_PID_Stop_Go/PLAN_PID_Stop_Go.md),
[`PLAN_pid_const_v.md`](../../../Gen11/Epoch8_UAV_Mjpc_thrust_control/U3_v_des_Patch/PLAN_pid_const_v.md),
`eval_fm_uav.py:986–990`, `:1205–1209`.

**The memo's verdict, which is the thing to engage with:**

> *"`v` in the FM tensor does two jobs — FM planning (momentum awareness) and implicit timing
> self-correction. Dropping it to 9D breaks both; **no controller can recover this at the tracker
> level**."*

That is correct, and HF does not contradict it. **HF's move is to stop trying to recover it at the
tracker level and recover it at the *planner constraint* level instead.** The memo enumerated four
options and all four are choices of *how to post-process the FM's blind action into a $v_{des}$*. None of
them touches the plan. That is the unexplored axis.

Note also what `pid_stopgo` costs physically: because the UAV replans **every control step** (§8, no
`replan_steps`), $v_{des}=0$ is issued **33 times per second**. The velocity loop's error is
$v_{real}-0=v_{real}$, so the PID is commanded to brake to a standstill continuously while the position
loop simultaneously pulls it forward. That is a controller fighting itself at 33 Hz, and it is a
plausible contributor to the 2.07 m `max_track_err` of §3.3.

### 6.2 The three things HF actually offers here

**(a) Velocity as a *constrained* quantity rather than a *derived* one.**
Option 4 was rejected because $v_{des}=\Delta p_{des}/dt_{fm}$ is computed after the fact from an
unconstrained output, with no guarantee of self-consistency and no correction path. Under HF, the
velocity relation becomes a row of $h(x)\le0$ and is enforced *inside* the sampler, jointly with obstacle
avoidance, before anything is executed. The quantity being differenced stops being a hallucination.

**(b) The initial-condition anchor — momentum awareness without the tensor channel.**
This is the sharp one. HF anchors $s_0$ of the plan to the *measured* state
(`flow_policy.py:423–439`). For our 9D tensor $x=[\,a_t \mid p_{des,t}\mid p_t\,]_{t=0}^{H-1}$ there is
no $v$ channel — but physical velocity is a **linear functional of channels that do exist**:

$$ \underbrace{\frac{p_1-p_0}{\Delta t_{phys}} = v_{\text{meas}}}_{\textbf{3 linear equality rows}},
\qquad
\underbrace{\lVert p_{t+1}-p_t\rVert_\infty \le v_{max}\Delta t_{phys}}_{\text{linear speed bound}},
\qquad
\underbrace{\lVert \Delta^{(2)}p_t\rVert \le a_{max}\Delta t_{phys}^2}_{\S5.1}$$

with $\Delta t_{phys}=1/33$ s. **Three equality rows inject the drone's measured momentum into the plan
even though the network never sees velocity and was never trained on it.** The plan is forced to *depart
from where the drone is actually going*, which is precisely job #1 of the `v` channel the memo said was
lost. Training-free inference-time constraint restoring a capability that was dropped from training is
about as on-thesis for HardFlow as it gets.

Note the channel choice matters and mirrors DC_FIX: differencing `p_des` just returns `act` (they are
tied by construction), so the velocity rows must act on the **`p` channel**.

**(c) The speed target becomes a cost, not a runtime hack.**
`pid_const_v`'s $\bar v$ is the dataset's mean speed, applied blindly to the action direction. Under HF
the same intent is a terminal cost, traded off against obstacle constraints by the same optimizer:

$$ C(x) = \sum_t \Big( \tfrac{\lVert p_{t+1}-p_t\rVert}{\Delta t_{phys}} - \bar v \Big)^2 $$

Slow down near obstacles, resume speed in the open — automatically, instead of one constant for the whole
flight. HF supports exactly this ($C$ plus $h\le0$ in one problem); the DPCC projector has no cost slot
at all.

### 6.3 Why this is a *good* test of HF's central thesis, not just a UAV fix

HF's core claim is that **pathwise** projection is over-restrictive and damages sample quality, while
terminal-only constraint does not. Velocity is where that claim should bite hardest:

> The velocity channel of a generated trajectory is a **finite difference** — a high-frequency functional
> of the sample. Pathwise projection perturbs every intermediate iterate; those perturbations are
> low-amplitude in position but get **differentiated**, so they are amplified in velocity. Terminal-only
> constraint touches the sample once, at the end.

**Testable prediction:** DPCC-projector variants and HardFlow should look similar on *position*
smoothness and diverge sharply on *velocity/acceleration* consistency — HF better by a margin that grows
with derivative order. If true, this is a clean quantitative demonstration of HF's thesis on a metric the
paper never reports (it reports safety, steps, time — never smoothness; see
[`DISCUSSION_foresight_fan_and_smoothness_paradigms.md`](../../Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md)).
It also predicts the U_13 ordering anomaly should be *worse* in velocity than in position — a free
retrospective check on data we already have.

### 6.4 What HF does **not** solve here — four honest limits

1. **Conditioning ≠ constraint.** The 9D network's *prior* remains velocity-blind. §6.2(b) gives the plan
   the right boundary condition, but the U-Net still has no learned notion of momentum, so the NLP has to
   drag the sample further from its nominal — larger $\lVert\widehat x_N-\bar x_N\rVert$, and HF's own
   appendix warns that over-steering produces "spurious samples." **Constraint-based momentum awareness
   is strictly weaker than putting `v` back in the tensor.** The honest experimental design is a
   three-way: 9D+stopgo (current) vs 9D+HF-velocity-rows vs 12D+`v` (E7).
2. **Wall-clock jitter is not a constraint problem.** If a replan takes 145 ms instead of 30.3 ms
   (§8), the drone has moved further than any plan assumed. The constraint fixes *plan* self-consistency,
   not real time. The genuine fix is orthogonal and worth its own note: a constrained plan has a
   meaningful time parameterization, so it can be tracked **time-indexed** (evaluate the reference at
   actual elapsed time) instead of **waypoint-indexed** — which is what makes Option 4's failure mode
   disappear rather than merely shrink.
3. **A feasible plan velocity is not an achieved velocity.** Gap G1 again: the tracker still has a 0.667 s
   time constant (§3.3). Velocity rows without §5.3's closed-loop model just move the 22× optimism from
   position to velocity. **§6.2 must be built on top of D3, not instead of it.**
4. **`p` for $t>0$ is FM-imagined, not measured** (per [`CRITIQUE…` §0.1](../../../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/data_example_anlysis/CRITIQUE_three_layer_absurdity.md)),
   so the velocity rows constrain finite differences of an imagined channel. This is not a new sin — the
   existing `p ← act` row already does exactly this — but it does mean the guarantee is "the plan is
   self-consistent," not "the world is."

### 6.5 The experiment to run

Cheapest first, and note that **step 1 needs no HF port at all** — the velocity rows are linear, so they
drop into the existing DPCC projector via the §5.1 `fd_bound` + per-row-`dt` infrastructure. HF is only
required once the cost term (§6.2c) or the cones (§5.2) enter.

| Step | What | Needs | Cost |
|---|---|---|---|
| 0 | On existing logs: how far is $\big(p_{t+1}-p_t\big)/\Delta t$ in current plans from measured $v$? Quantifies the inconsistency before fixing it. | nothing (offline) | hours |
| 1 | Add the 3 initial-velocity rows + speed bound to the projector; re-run `pid_const_v` with $v_{des}$ read from the **plan's** finite difference instead of $\text{unit}(a)\bar v$ | §5.1 infra | S; **cluster** |
| 2 | Three-way comparison: 9D+`pid_stopgo` vs 9D+velocity-rows vs 12D+`v` (E7 ckpt) | E7 checkpoint still exists? | M; **cluster** |
| 3 | HF arm with the speed cost $C(x)$ + velocity rows; compare velocity-consistency vs DPCC-projector variants (the §6.3 prediction) | Stage 2 of §9 | L; **cluster** |

**Metrics that decide it:** executed speed profile vs planned; $\lVert v_{real}-v_{des}\rVert$; fraction of
steps with $\lVert v\rVert<0.05$ m/s (the stop-and-go signature — should collapse); `max_track_err`;
motor saturation fraction; and steps-to-goal, which is where the payoff should show up if the drone stops
braking 33 times a second.

> **One caveat on framing.** U2 was explicitly designed as *"a baseline to measure the cost of dropping
> velocity"* — i.e. `pid_stopgo` was never meant to be the destination, and it is currently the default in
> both config blocks. Before building anything here, it is worth confirming whether the U2-vs-E7
> measurement was ever actually completed. If 12D+`v` simply wins, the cheapest fix for the *product* is
> to go back to it, and §6 becomes a research question (can constraints substitute for conditioning?)
> rather than an engineering fix. Both are worth doing; they should not be confused with each other.

---

## 7. Is HardFlow's math actually bound to a 1st-order Euler / stop-and-go system? **No — but its 2nd-order experiment is open-loop, so the regime we need is still unclaimed**

*(Added 2026-08-13, answering: "is HF math only located to a 1st-order Euler system, stop-and-go only,
like the avoiding coding — or can it expand to higher order and solve the velocity-speed and
instantaneous-calculation problem we hit in Gen11?")*

**Answer in one line: the first-order-ness and the stop-and-go are both properties of the *avoiding
experiment*, not of HardFlow. The paper's own maze2d experiment is second-order, velocity-in-the-plan,
PD-tracked — and its code is sitting in our `aux_repo` on an unread branch.**

### 7.1 There are two Eulers in this method and they have nothing to do with each other

| | Where it lives | What it discretizes | Is it a limitation? |
|---|---|---|---|
| **Euler #1** — *sampling* | Problem 2: $x_{j+1}=x_j+v^\theta_{t_j}(x_j)\Delta t_j+u_j\Delta t_j$, over $\tau\in[0,1]$ | the **generative ODE**. Has no robot in it at all. | **No.** The paper's appendix replaces it with any one-step map $\Psi_i^\theta$ (Heun shown, measured faster) and states *"all results in the Theoretical Analysis section extend naturally."* |
| **Euler #2** — *physics* | inside $h(x)\le0$: $s_{i+1}=As_i+Ba_i+c$ | the **plant**, in physical time | **Not even Euler.** $(A,B,c)$ is *least-squares fitted to real transitions* — it absorbs whatever the true discrete-time map is, including exact ZOH. Calling it "Euler" is our import, not theirs. |

> **Sharpening of §3.1:** the "explicit Euler" wording is **ours**. `DynamicConstraints`' docstring
> (`projection.py:435–447`) literally specifies $x_{t+1}=x_t+\Delta t\,\dot x_t$ (or trapezoidal). *We*
> hard-code an integrator; HardFlow *fits* a transition. That is a second protocol asymmetry on top of
> the asserted-vs-fitted one from [`FIT_DYNAMICS…` §6b](../FIT_DYNAMICS_what_the_linear_dynamics_model_is.md).

### 7.2 The guarantee makes **no assumption whatsoever** about $h$

The proof of Prop. 1 (Appendix A.1) is four lines, and this is its entirety: at $i=N-1$, the scheduler
boundary conditions $\alpha_1=1,\ \beta_1=0$ collapse Alg. 1's update to

$$ x_N=\alpha_1\widehat{x}_N^{*}+\beta_1(\cdots)=\widehat{x}_N^{*}\ \Longrightarrow\ h(x_N)=h(\widehat{x}_N^{*})\le0 .$$

**That is the whole proof.** It uses: the scheduler's endpoint values, and the fact that the last NLP was
solved feasibly. It does **not** use linearity of $h$, convexity of $h$, any dynamics model, any order,
any relative degree, any smoothness. $h:\mathbb{R}^d\to\mathbb{R}^m$ is an arbitrary black box.

So the algorithm is **order-agnostic by construction**. Everything about "first order" enters through one
choice — *what the experimenter wrote into $h$* — and D3IL avoiding wrote in a single integrator because
that is what a velocity-commanded, quasi-static robot arm **is**.

### 7.3 "First-order recursion" ≠ "first-order physics" — the state-augmentation point

This is likely the crux of the confusion, and it is worth stating flatly:

$$ s_{i+1}=As_i+Ba_i+c \quad\text{is a \emph{first-order recursion in the state vector }} s . $$

State-space form makes **every** system first-order in $s$; the *physical* order is set by what you put
**in** $s$. With $s=(p,v)$ and $a=$ force,

$$ \begin{bmatrix}p_{i+1}\\ v_{i+1}\end{bmatrix}=\begin{bmatrix}I&\Delta t\,I\\ 0&I\end{bmatrix}\begin{bmatrix}p_i\\ v_i\end{bmatrix}+\begin{bmatrix}\tfrac12\Delta t^2 I\\ \Delta t\,I\end{bmatrix}a_i $$

is a **double integrator** — a second-order plant — written as a first-order recursion, and it is *exactly
the shape HardFlow's constraint already has*. Relative degree 3 or 4 follows the same way: augment to
$s=(p,v,a)$ or $(p,v,a,j)$, or (equivalently, and cheaper for us) impose multi-step finite differences
directly on $x$ as in §5.1. **All of it stays linear.** Nothing in HF resists this.

So the D3IL avoiding case is not "HF's math is first order." It is "HF was pointed at a first-order robot."

### 7.4 The receipt — with a large asterisk: maze2d is **open-loop, planned once**

> [!IMPORTANT]
> **Correction to the first draft of this section.** I initially presented maze2d as "our exact problem
> shape." It is not, and the objection that prompted this revision is correct. **maze2d plans exactly
> once per episode and never replans.** The code comment is literal:
> ```python
> pbar = tqdm.tqdm(range(env.max_episode_steps), desc="Episode")
> for t in pbar:
>     # only plan once at t=0
>     if t == 0:
>         conditions[0] = observation
>         action, samples, _, _, info = policy(conditions, batch_size=cfg.batch_size)
>         state_sequence = samples.observations[0]
> ```
> — `origin/maze2d:run/eval.py:341–352`. There is no `replan_steps` key in the maze2d config at all,
> whereas the d3il/avoiding branch has one and uses it (`origin/d3il:run/eval.py:381–403`:
> `if planned_actions is None or action_index >= cfg.replan_steps`). One $H{=}384$ plan is generated at
> $t{=}0$ and PD-tracked open-loop for up to 800 steps.

| | D3IL avoiding | **maze2d** | **Gen11 UAV** |
|---|---|---|---|
| state $s$ | $(p_{cur},p_{des})\in\mathbb{R}^4$ — **no velocity** | $(p,v)\in\mathbb{R}^4$ — **velocity in state** | $(p_{des},p,v)$ 12D; $(p_{des},p)$ 9D |
| action $a$ | desired velocity | **force** | $\Delta p_{des}$ |
| plant order | 1 (single integrator) | **2 (double integrator)**, fully actuated | 2+, **underactuated**, rel. deg. 4 |
| tracker | none — actions executed directly | **PD**, $K_p{=}5,K_d{=}1$ | **cascaded PID** |
| $v_{des}$ source | n/a | **read out of the plan** | $\Delta p_{des}/dt_{fm}$, or $0$, or $\text{unit}(a)\bar v$ |
| horizon $H$ | 16 | **384** | 8 |
| **replanning** | **every 8 steps** | ❗ **never — plan once at $t{=}0$** | ❗ **every step, 33 Hz** |
| environment | static, novel test obstacles | **static, fully known at $t{=}0$**, `fixed_start=True`, goal = $[\,target,0,0\,]$ | static geometry, but 4.8× over real-time budget |
| plan cost | 0.190 s **per replan** (~7/episode) | 4.09 s **once per episode** | 30.3 ms budget **per step** |
| HF result | 1.00 safety, 52.5 steps | 1.00 safety, 0.0 violations, best score | — |

**The structural finding this exposes is more useful than the one I claimed.** HardFlow's two robotics
experiments each have *one* of the two features our UAV needs, and **never both**:

| | replanning (receding horizon) | velocity in the state |
|---|:--:|:--:|
| avoiding | ✅ | ❌ |
| maze2d | ❌ | ✅ |
| **Gen11 UAV** | ✅ **required** | ✅ **required** |

> **So HardFlow has never demonstrated velocity-in-state *together with* replanning.** That is a genuine
> gap in the paper, not just in our reading of it — and it is exactly the regime the UAV lives in. This
> is better news for us than a ready-made template would have been: it is unclaimed ground, and §5.5
> (recursive feasibility across replans) is precisely the theory that regime needs.

The tracking code (`origin/maze2d:run/eval.py:361–371`) is still worth reading closely:

```python
if cfg.controller == "pd":            # proportional-derivative
    px, py, vx, vy = observation
    idx_t   = min(t, len(state_sequence) - 1)      # ← TIME-indexed into the plan
    pdes_t  = state_sequence[idx_t][:2]            # ← position  from the plan
    vdes_t  = state_sequence[idx_t][2:]            # ← VELOCITY  from the plan
    Kp = np.array([1.0, 1.0]) * 5.0
    Kd = np.array([1.0, 1.0]) * 1.0
    action = Kp * (pdes_t - np.array([px, py])) + Kd * (vdes_t - np.array([vx, vy]))
```

with the velocity channel held consistent by the dynamics rows at
`origin/maze2d:hardflow/models_flow/flow_policy.py:299`.

**What survives the correction, and what does not:**

| Claim | Verdict |
|---|---|
| `v_des` is **read from the plan**, never computed by runtime differencing — the "instantaneous calculation" problem is *deleted*, not mitigated | ✅ **survives.** The mechanism is real and transfers: it needs a velocity channel + dynamics rows, not open-loop planning. |
| Velocity as a **generated, dynamically-constrained** channel; HF's constraint machinery handles $(p,v,a)$ | ✅ **survives** — `flow_policy.py:299` on that branch. |
| Stop-and-go is a Gen11 choice, not an HF property | ✅ **survives.** |
| Time-indexed tracking (`idx_t = min(t, …)`) is the fix for replan jitter | ⚠️ **weakened.** It is trivially easy when there is only *one* plan spanning the episode. Under replanning you must additionally stitch across plan boundaries — a problem maze2d never faces. Still the right idea; no longer free. |
| maze2d's 1.00 safety / 0 violations proves HF works here | ❌ **retracted.** It proves HF works for a **one-shot open-loop plan in a fully-known static maze**. That is a strictly easier setting than 33 Hz closed-loop flight. |
| maze2d's 4.09 s is comparable to our budget | ❌ **retracted.** It is **once per episode**, not per step. Only the avoiding number (0.190 s/replan) belongs in the §8 compute comparison — which is what §8 uses. |

> **The actionable find:** `git branch -a` in `/workspaces/aux_repo/HardFlow` shows
> `remotes/origin/{d3il, maze2d, burgers, image}`. We have been reading **`d3il`** — the *first-order,
> no-velocity, no-tracker* branch — and generalising from it to a quadrotor. **`origin/maze2d` is fetched
> and readable right now** (`git show origin/maze2d:<path>`), and it is the better template for Gen11:
> second-order plant, velocity in the state, PD tracker, dynamics constraint over $(p,v,a)$. Reading it
> should precede any UAV port. Gen12/Gen13 planning assumed `d3il` was *the* HardFlow codebase; that
> assumption is worth revisiting.

### 7.5 So: can HF solve the Gen11 velocity problem? The concrete recipe

Mapping maze2d onto Gen11, the recipe is three changes to components we already have — and note that
**maze2d's setup corresponds to E7's 12D config, not the current 9D**, because the network must *emit* a
velocity channel for the tracker to read one:

| # | Change | Gen11 status |
|---|---|---|
| 1 | Velocity **in the tensor** (so the plan has a $v$ channel to read) | **E7 12D already had this.** Dropped in U2. → un-drop it |
| 2 | Velocity **constrained** to be consistent with position: the double-integrator rows | **missing entirely** — `v(9,10,11)` unconstrained (§3.1). Needs per-row `dt` (§5.1) |
| 3 | $v_{des}$ **read from the plan, time-indexed** — not $\Delta p_{des}/dt_{fm}$, not $0$, not $\text{unit}(a)\bar v$ | **missing** — all four Gen11 options compute $v_{des}$ at runtime (§6.1) |

Item 1 is a config revert. Item 2 is the §5.1 infrastructure. Item 3 is a few lines in `rollout_one`.
**None of the three requires HardFlow.** They are the *maze2d recipe* — but note §7.4: maze2d applies it
open-loop, so transplanting it into a **replanning** loop is new ground, not a port. They land in the DPCC projector
just as well — which makes this the cheapest high-value experiment in the whole document, and it is a
sharper version of §6.5 step 2. HF then adds, on top: the cost slot $C(x)$ for speed shaping (§6.2c), and
the nonconvex/SOC constraints (§5.2) the linear projector cannot hold.

### 7.6 Where the maze2d analogy stops — four honest limits

1. **Maze2d is fully actuated.** A force-actuated ball has relative degree 2 and no attitude loop; the
   thrust direction *is* the control. A quadrotor is underactuated with relative degree 4 (§3.2). So
   maze2d validates *"velocity in the plan + PD tracking + dynamics constraint"*, **not** *"this works for
   an underactuated vehicle."* §5.2's cones are still needed.
2. **$H=384$ vs our $H=8$.** Time-indexed tracking is comfortable over a 384-step plan; over 8 steps
   (0.242 s) the plan is exhausted almost immediately and `min(t, len-1)` degenerates to "hold the last
   waypoint." **Time-indexed tracking probably requires growing $H$**, which re-inflates §8's compute
   wall. This tension is real and should be sized before building.
3. **Maze2d replans rarely; we replan every step.** Different regime; the per-replan cost budget is not
   comparable.
4. **Their PD is a single loop with $K_p{=}5,K_d{=}1$ on a point mass.** Ours is a cascade with an SO(3)
   inner loop. The 0.667 s outer-loop time constant of §3.3 has no maze2d counterpart, so the D3
   closed-loop model (§5.3) remains necessary regardless.

### 7.7 So should we just plan once with a big $H$? **No — but the operating point is the real question**

Two questions follow directly from §7.4, and they deserve separate answers.

**Q1: "If the UAV planned once, would the instant-replan problem disappear in Gen11 too?"**
Mechanically yes, and that is exactly *why* maze2d's velocity handling looks so clean — but it disappears
by removing the feature that makes the system a controller. Replanning is not overhead we tolerate; **it
is the entire mechanism by which model error, disturbance, and anything unseen get corrected.** Drop it
and you have open-loop trajectory playback with a PD servo. That is a legitimate design (maze2d does it),
but only under conditions the UAV does not meet:

| maze2d can plan once because… | Gen11 UAV |
|---|---|
| environment fully known at $t{=}0$ — static maze, known target, `fixed_start=True` | geometry known, but the *drone's own response* is not (§3.3) |
| plant is a deterministic point mass; the PD tracks near-exactly | 0.667 s tracker time constant, `max_track_err` 2.07 m |
| nothing new is ever observed mid-episode | E10 will add **cameras** — the whole premise is new observation |
| open-loop error does not accumulate meaningfully | it accumulates for 7.8 s over 256 steps |

**Q2: "So extend to $H{=}256$ and one-shot Visual Aligning?" — no, and your instinct is right.** Three
reasons, in increasing order of severity:

1. **Partial observability.** Visual aligning is image-conditioned. A one-shot plan at $t{=}0$ must
   predict the whole episode from the first frame. maze2d's target is *given as a hard condition*
   (`conditions[H-1] = [*target, 0, 0]`); an image-conditioned task has no such oracle.
2. **Multimodality.** D3IL's defining property is multiple valid behavior modes. One shot = one mode,
   committed at $t{=}0$, with no opportunity to switch when the commitment turns out badly.
3. **It deletes the "PC" from FM-PCC.** *Predictive Control* is receding-horizon by definition. A
   one-shot planner is not a controller and cannot be compared against DPCC on DPCC's own terms — the
   benchmark hierarchy would no longer mean anything.

**But the useful question underneath is real: is Gen11's operating point sane?** Laid out on one axis:

| | $H$ | lookahead | replan cadence | plans/episode |
|---|---|---|---|---|
| **Gen11 UAV** | 8 | **0.24 s** | **every step (33 Hz)** | ~600 |
| HF avoiding | 16 | — | every 8 steps | ~7 |
| HF maze2d | 384 | whole episode | once | 1 |

**Gen11 sits at the extreme of both axes simultaneously — the shortest lookahead *and* the highest replan
rate. That is the worst combination available.** And one number makes it concrete:

> The plan spans **0.24 s**. The tracker's error time constant is **0.667 s** (§3.3). **The lookahead is
> shorter than the tracker's own response time by ~2.7×** — the plan ends before the drone can respond to
> its beginning. We are re-deciding the future 33 times a second while never looking far enough ahead for
> any decision to have taken physical effect.

That reframes several things in this document at once. Moving toward the middle of the axis — e.g.
$H\approx32$–$64$ (1–2 s ≈ 1.5–3 tracker time constants) with replanning every ~8 steps (≈4 Hz) — would:

- cut NLP solves by ~8× → **this, not solver tuning, is what makes §5.2/§5.6 affordable** (§8);
- make lookahead exceed the tracker time constant → plans become physically trackable;
- give time-indexed tracking something to index into (§7.6 limit 2);
- give a terminal set somewhere to live (§5.5 — note maze2d's goal condition $[\,target,0,0\,]$ is
  literally a zero-terminal-velocity constraint, i.e. an implicit stoppability condition);
- **retain replanning**, so reactivity to the unseen is preserved.

**Cost, stated honestly:** $H$ is a model property — changing it means retraining and re-windowing the
dataset, and an FM trained at $H{=}8$ gives no evidence about its behaviour at $H{=}32$. Longer horizons
also enlarge the NLP per solve (it scales with $H\times$transition-dim), partially offsetting the
solve-count saving. **Net effect is an empirical question, and "sweep $(H,\text{replan cadence})$" is
probably the single highest-value experiment in this document** — it is a config-and-retrain sweep with no
new theory, it directly attacks the §8 compute wall, and every other direction here gets cheaper or more
meaningful if it lands well.

### 7.8 Proposal — long horizon + **replan-on-violation** instead of continuous replanning

*(User proposal, 2026-08-13: "design into the new MPC like H256, meet violation — i.e. it happened — then
replan, instead of keeping replan?" With the user's own assessment attached: "this is suboptimal, since
violated then can replan, compared to real MPC — but it feels like it solves the instant-calculation
problem related to velocity." **Both halves of that assessment are correct, and the second half is
correct for a sharper reason than intuition suggests.** Recorded here in full.)*

This is **event-triggered MPC** (as opposed to the time-triggered MPC we run now) — an established
control-theory idea (Heemels / Johansson / Tabuada lineage, with a substantial event-triggered-MPC
literature). It fits FM-PCC unusually well for one reason: event-triggering was invented for regimes where
**computation is the scarce resource**, and our cost asymmetry is far more extreme than in the literature
that motivated it.

> **Context — there are two MPCs in this stack, and conflating them makes MPC look like the villain when
> it isn't.** A classical quadrotor NMPC over the full state $(p,v,q,\omega)$ solves in **sub-ms to a few
> ms** and runs at 50–100+ Hz onboard (acados/RTI-class solvers); velocity is simply a state, supplied by
> the estimator. Our own `MJPCTracker` *is* such an MPC and already handles velocity (`mjx_vel_weight`
> exists precisely to penalise it, "mirrors PID Kd"). The 145 ms is **not** MPC being too slow for
> higher-order control — it is 10–20 U-Net forward passes plus a per-sampling-step NLP. **MPC is not the
> bottleneck; the generative planner is.** Periodic replanning is free in classical MPC and nobody
> event-triggers it; here one solve costs 4.8 replan budgets, which is exactly when triggering starts to
> matter.

#### 7.8.1 Why it is suboptimal — the user's assessment, made precise

Three distinct losses versus time-triggered MPC, in increasing severity:

1. **Staleness.** MPC's value *is* re-optimisation against fresh state. A plan held for $N$ steps is
   suboptimal by construction, and standard MPC suboptimality bounds grow with the inter-solve interval.
2. **Weakened disturbance rejection.** Between triggers only the PD/PID loop rejects disturbance; the
   *plan* does not adapt. With our 0.667 s tracker time constant (§3.3), that is a long time to be riding
   an open-loop reference.
3. **The trigger as literally stated is too late.** "Violation happened → replan" is **reactive**: if the
   drone has already entered the obstacle, the episode is already failed and there is nothing to replan
   *for*. For safety constraints a post-hoc trigger buys nothing. **This is the one part of the proposal
   that must change** — see §7.8.4.

#### 7.8.2 Why the intuition is right anyway: **replan cadence *is* the velocity problem**

This is the part worth keeping, and it is stronger than "it feels like it helps."

Under $H{=}8$ replanning **every 30.3 ms**, each plan is a *fresh, independent sample* from a generative
model, discarded one step later. Successive plans are not tied together by anything — there is no
constraint that plan $k{+}1$ agrees with plan $k$ at the seam. Consequences:

- A velocity read off plan $k$ and off plan $k{+}1$ can differ arbitrarily, because they are different
  draws — even when each plan is *internally* consistent.
- Differencing such a plan at runtime ($v_{des}=\Delta p_{des}/dt_{fm}$) differentiates that
  plan-to-plan jitter. This is Option 4's "unreliable by design" (§6.1), now with a mechanism attached.
- **Therefore `pid_stopgo` ($v_{des}=0$) is not an arbitrary choice — it is the rational response to a
  velocity reference that is noise.** If the reference cannot be trusted, commanding zero is safer than
  commanding garbage. **Stop-and-go is a *symptom of the 33 Hz replan cadence*, not an independent design
  decision.**

Now invert it. A long-horizon, rarely-replanned plan is a **stable object that lives for seconds**, so:

| | 33 Hz replan, $H{=}8$ | long $H$, rare replan |
|---|---|---|
| $v_{des}$ source | runtime finite difference of a plan about to be discarded | **read from a stable plan, time-indexed** |
| depends on $dt_{fm}$? | yes — timing-fragile | **no** |
| plan-to-plan seams | ~33 per second | a handful per episode |
| trustworthy velocity reference? | no → hence $v_{des}=0$ | **yes** |

**That is exactly why maze2d's velocity handling looks so clean (§7.4) — it plans once.** So the user's
intuition is correct and now has a mechanism: *the proposal attacks the root cause of the velocity
problem rather than the symptom.* §6 and §7.7 are therefore **not two problems but one**: the velocity
reference problem and the replan-cadence problem are the same problem viewed from two ends.

#### 7.8.3 The sharper fix — keep fast replanning, constrain the **seam**

The trade-off above looks forced: fast replanning gives good control and an unusable velocity reference;
slow replanning gives a stable reference and sluggish control. **It is not forced.** The seam jitter is
caused by successive plans being unconstrained relative to each other — and constraining a plan's initial
condition is precisely what HardFlow's $s_0$ anchor does (`flow_policy.py:423–439`):

$$ \underbrace{p_0^{(k+1)} = p_{\text{meas}}}_{\text{avoiding has this}},\qquad \underbrace{v_0^{(k+1)} = v_{\text{meas}}}_{\textbf{only if velocity is in }s}$$

With velocity in the anchored state, **every new plan is forced to depart at the velocity the drone
actually has**, so consecutive plans agree at the seam *by construction* and the reference stays
continuous across replans — at 33 Hz or any other rate.

This is exactly the avoiding-vs-maze2d split of §7.4: avoiding anchors position only (no velocity in $s$),
maze2d anchors $(p,v)$. **Gen11's 9D config is in the avoiding camp; E7's 12D was in the maze2d camp but
never had the anchor constraint.** So the ranking is:

| option | velocity reference | control quality | cost |
|---|---|---|---|
| current: 9D, 33 Hz, $v_{des}{=}0$ | unusable → zeroed | poor (fights itself 33×/s) | 4.8× over budget |
| **proposal: long $H$, event-triggered** | **stable** | **degraded (stale)** | much cheaper |
| **velocity-anchored seam, keep fast replan** | **stable** | **best** | unchanged (rows are free) |

**The third row dominates the second** — it gets the proposal's benefit without paying its cost. The
proposal's real contribution is diagnostic: it is what made the seam mechanism visible.

#### 7.8.4 If event-triggering is built anyway, use a **predictive** trigger

| trigger | condition | cost | verdict |
|---|---|---|---|
| violation occurred | $h(p)>0$ | free | ❌ **too late — failure already happened** |
| tracking error | $\lVert p-p_{plan}(t)\rVert>\varepsilon$ | free | ✅ classic; **already logged** (`max_track_err`) |
| **tube exit** | drone about to leave the §5.4 tube | free | ✅ **carries a guarantee** |
| predicted violation | forward-sim the §5.3 linear closed-loop model along the remaining plan | ~free (small matmul) | ✅ cheap lookahead |
| novelty | observation/image changed materially | E10 | ✅ for the visual arm |

**The tube trigger is the one worth building**, because it makes the scheme *rigorous* rather than
heuristic. If obstacles are tightened by the tube radius $\varepsilon$ (§5.4), the plan is provably safe
**as long as the drone remains inside the tube**; triggering on imminent tube exit means safety holds
*continuously between replans*, by construction, and the trigger fires exactly when that guarantee is
about to expire. That is the standard event-triggered-MPC certificate — and it means **this proposal is
the mechanism that makes §5.4 (tube) and §5.5 (recursive feasibility) pay off** instead of remaining
standalone nice-to-haves. Three previously separate directions collapse into one design.

Add a **minimum dwell time** between triggers (guaranteed by the tube's finite escape time) to prevent
chattering / Zeno behaviour.

#### 7.8.5 The arithmetic — why $H{=}256$ specifically is the wrong size

Solve cost scales with $H\times$transition-dim, and maze2d calibrates it: $384\times6=2304$ dims →
**4.09 s per plan**. So UAV $H{=}256\times12=3072$ dims → **order 5 s per solve**. Against a
$634\times30.3\text{ ms}\approx19$ s episode:

| | NLP dims | ~s/solve | solves affordable per episode |
|---|---|---|---|
| now: $H{=}8$, every step | 96 | 0.145 | 634 needed → **4.8× over** |
| $H{=}256$, event-triggered | 3072 | ~5 | **≈3.8 total** |
| **$H{=}32$–$64$, event-triggered** | 384–768 | ~0.3–0.5 | **≈40 — comfortable** |

> **The trap: long horizon and fast reaction are in direct tension.** At $H{=}256$ the trigger fires and
> the response takes ~5 s, during which the drone flies the stale plan — so the trigger cannot actually
> rescue anything. **A long horizon makes triggering less useful, not more.** This reinforces §7.7:
> the operating point is $H\approx32$–$64$, not 256.

Two architectural changes compose with this and are arguably prerequisites:

- **Asynchronous replanning.** The FM currently *blocks* the control step, which is why "145 ms vs
  30.3 ms" reads as fatal. Real drone stacks run the planner in its own thread at its own rate; the
  tracker never waits. This requires the time-indexed plan of §7.4 so a late plan can be spliced in at the
  correct time offset. Independent of HardFlow, and possibly the highest-impact change in this document.
- **Hierarchical two-rate planning** — a long "route" plan (event-triggered, rare) supplying the terminal
  condition for a short "local" plan (fast). Standard robotics architecture; it *dissolves* the
  horizon-vs-reactivity tension rather than trading against it, and the route plan's endpoint is a natural
  home for §5.5's terminal set.

**Honest limitation of event-triggering in general:** it improves *average* compute, not *worst-case
latency* — and triggers tend to fire during aggressive manoeuvres, exactly when a stall is least
affordable. Async replanning is the mitigation.

#### 7.8.6 What to actually build from this

**The cadence ablation, and it is cheap.** Same checkpoint, same scenes, same everything — vary only the
replan cadence (every step → every 8 → every 32 → once) and measure **velocity-reference stability**:
plan-to-plan seam discontinuity $\lVert v^{(k+1)}_0 - v^{(k)}_1\rVert$, executed speed profile, fraction
of steps with $\lVert v\rVert<0.05$ m/s. This isolates how much of the velocity problem is caused by
cadence alone, it needs **no new theory and no retrain** (cadence is eval-side), and it decides whether
§7.8.3's seam constraint or §7.8's slow-replanning is the right lever. **Run before building either.**

> **Verdict.** The proposal is *diagnostically* valuable and *operationally* dominated. It correctly
> identifies that continuous replanning is what makes the velocity reference untrustworthy — a link
> nothing in Gen11's four-option memo (§6.1) saw, because all four options treated $v_{des}$ as a runtime
> computation problem rather than a plan-stability problem. But its own mechanism (slow down, trigger on
> violation) pays for that stability with stale control and an unusable-because-too-late trigger. **Keep
> the diagnosis, replace the remedy:** anchor velocity at the seam (§7.8.3), trigger on the tube rather
> than the violation (§7.8.4), and size the horizon at 32–64 rather than 256 (§7.8.5).

### 7.9 Net effect on this document

- §1 axis **B** should be read as: HF handles 2nd-order *linear, fully-actuated* — **demonstrated**, not
  merely possible. The open ground is narrower and better-defined than §1 implied: *nonlinear,
  underactuated, high relative degree, cascaded tracker.*
- §6's conclusion is unchanged but its emphasis shifts. §6.2(b) — recovering momentum on the **9D**
  checkpoint via finite-difference boundary rows — remains the interesting *research* question ("can
  constraints substitute for conditioning?"). But §7.5 shows the **engineering** answer is simply to put
  velocity back and constrain it, as HF's own second-order experiment does. Do not let the research
  question delay the engineering fix.
- **§5.9's ranking should be re-read in light of §7.7.** The $(H,\text{replan cadence})$ sweep is not in
  that table and arguably outranks everything in it: it is pure config + retrain, it is the only lever
  that changes the compute budget by an order of magnitude, and D1/D2/D4/D5 all become either cheaper or
  more meaningful once the lookahead exceeds the tracker time constant. **Revised entry point: §7.7 sweep
  → §5.3 (closed-loop model) → §5.1 (infrastructure) → §5.4 (tube).**
- **HardFlow has never run velocity-in-state *with* replanning** (§7.4). Any UAV port is therefore
  building something the paper does not cover, and should be scoped and written up as such rather than as
  "porting the maze2d setup."
- **§6 and §7.7–§7.8 are one problem, not two.** §7.8.2 establishes the link: continuous 33 Hz replanning
  is *what makes the velocity reference untrustworthy*, and `pid_stopgo` is the rational response to an
  untrustworthy reference. Every fix in §6 should be read as also being a statement about replan cadence,
  and the §7.8.6 cadence ablation should precede all of them — it is eval-side, needs no retrain, and
  tells us whether to spend effort on the seam constraint (§7.8.3) or on cadence itself.
- **Three directions collapse into one design.** §5.4 (tube), §5.5 (recursive feasibility) and §7.8
  (event-triggering) are not independent: the tube supplies the trigger, the trigger supplies the
  inter-replan guarantee, and the terminal set supplies recursive feasibility across triggers. That
  bundle — not any single one of them — is the paper-shaped contribution.

---

## 8. The wall everything hits: 30.3 ms

| Quantity | Value | Source |
|---|---|---|
| UAV control rate | 33 Hz → **30.3 ms/step** | `config/uav.py:219`, and every `TIMING` log line |
| Replan cadence | **every step** (no `replan_steps`; the FM runs each control step, first action executed) | `eval_fm_uav.py` rollout loop |
| Measured `total_ms` (E7 rollout) | **145.2 ms** vs budget 30.3 → effectively ~7 Hz | `TRACKING_ERROR_Gen11E7.md:135` |
| HardFlow published cost (D3IL) | **190 ms/replan**, $N=10$ steps with the NLP on the last 5 ⇒ ~38 ms/NLP solve | paper Table (Robotic Manipulation), Experiment Details |
| Problem size | HF D3IL: $H{=}16 \times 6 = 96$ dims. UAV: $H{=}8 \times 12 = 96$ dims | — |

**Read this carefully, because it cuts both ways.** The problem sizes are *identical* (96), so HF's
per-solve cost transfers almost directly: **~38 ms per NLP solve against a 30.3 ms total budget.** A naive
HardFlow port to the UAV misses real time by **≈6.3×** on the full 5-solve schedule — and we are already
4.8× over budget *before* adding any NLP.

Three consequences, all of which shape the research rather than merely constraining it:

1. **Convexity is a real-time requirement, not an aesthetic one.** §5.1 (QP) and §5.2 (SOCP) are cheap;
   nonconvex sphere-avoidance rows in a general NLP are not. Prefer formulations that keep the
   per-step problem convex.
2. **Fewer sampling steps is the only order-of-magnitude lever** → D7/iMF is on the critical path.
3. **Honest framing for a writeup:** on the UAV, HardFlow-class methods are currently a *planning-quality*
   result under a relaxed real-time assumption, not a demonstrated 33 Hz controller. Say so; the
   `over_budget` / `total_ms_p95` telemetry already in `behavior_logger.py` is the right instrument for
   reporting it rather than hiding it.

---

## 9. A staged program

**Stage 0 — offline, no new runs, no code changes to the pipeline.**
Take existing Gen11 rollout logs. Compare one-step prediction error of three candidate dynamics models:
(a) today's asserted identity rows, (b) D3a fitted closed-loop LTI, (c) D3b analytic PD closed loop.
Deliverable: a plot and a number. **This alone decides whether §5.3 is worth building**, and it costs
nothing but analysis. *(Analysis-only Python is fine locally per [`docker-no-python-cluster-only`]; any
rollout regeneration is a cluster job.)*

**Stage 1 — projector infrastructure (linear only).**
Per-row `dt` in `DynamicConstraints`; new `fd_bound` family; $v$-chain rows. Wire as new
`constraint_types` entries so the existing `geo_free` / `model_free` / `bounds_free` ablation grammar
keeps working unchanged. Run the standard variant sweep; **run on cluster**.

**Stage 2 — the HF arm for the UAV.**
Sibling pair (`flow_matcher_v3_uav_hardflow/` ↔ `FM_v3_uav_hardflow_test/`) per the repo's copy-modify
convention, porting `hardflow_new_forward` (the black-box-$f(x,t)$ variant — the only portable one, as
Gen12 established; `projection`/`hardflow` embed the U-Net via l4casadi and are architecture-locked).
Constraints from Stage 1 plus the §5.2 cones. Baseline it against `dpcc-c` per
[`da-target-is-best-baseline-variant`].

**Stage 3 — the theory arm.**
D4 tube, then D5 terminal set, with the recursive-feasibility metric. This is where a paper lives.

---

## 10. Open questions to settle before committing

1. **Which tracker is the object of study — PID or MJPC?** D3b/D4/D5 all assume an analyzable closed
   loop. MJPC (`mjpc_tracker.py`) is a sampling optimizer, not an LTI system. *Recommendation: PID for
   the theory arm; MJPC as an empirical robustness check.*
2. **Is the FM's `v` channel even accurate?** It is currently unconstrained and never evaluated. If the
   FM's predicted $v$ is poor, constraining $p$–$v$ consistency (§5.1) may *hurt* before it helps.
   Measure it in Stage 0 — same logs, near-zero marginal cost.
3. **$H=8$ / 0.242 s — is the horizon long enough for a terminal set to mean anything?** Stopping from
   typical plan speeds may take longer than the horizon, in which case D5 needs $H$ to grow, which
   directly re-inflates §8. This tension is real and should be quantified before D5 is attempted.
4. **Do we have $T_{max}$, $\theta_{max}$, $j_{max}$ for the Skydio X2?** Mass and inertia are read from
   the MuJoCo model already; `u_max = max(2·u_hover, 6.0)` in `flight_controller.py` is a *heuristic
   cap*, not a datasheet figure. §5.2 is only as good as these numbers.
5. **Normalizer units for any fitted $(A,B,c)$** — the Gen12 warning applies verbatim.

---

## 11. Cross-references

**HF_Study (this folder's parent)**
- [`FIT_DYNAMICS_what_the_linear_dynamics_model_is.md`](../FIT_DYNAMICS_what_the_linear_dynamics_model_is.md) — what $A,B,c$ is; DPCC-asserts-vs-HF-fits; the protocol asymmetry (§6b)
- [`MAP_Algorithm1_to_AvoidingCode.md`](../MAP_Algorithm1_to_AvoidingCode.md) — Alg. 1 ↔ `flow_policy.py`
- [`H2H_iMF_vs_HardFlow_stepwise.md`](../H2H_iMF_vs_HardFlow_stepwise.md)

**HF_iMF/Research**
- [`ANALYSIS_hardflow_vs_dpcc_planning_structure.md`](../../Research/ANALYSIS_hardflow_vs_dpcc_planning_structure.md) — selection vs projection; batch=1; NFE counts (the §5.7 numbers)
- [`DISCUSSION_foresight_fan_and_smoothness_paradigms.md`](../../Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md) — why HF never discusses smoothness (§5.1/§5.2 make smoothness a *hard constraint*, which is the reply)

**Gen11 (UAV)**
- [`CRITIQUE_three_layer_absurdity.md`](../../../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/data_example_anlysis/CRITIQUE_three_layer_absurdity.md) — §0.3 on `v`/`v_des`; the $p_{des}$-vs-$p$ binding asymmetry
- [`TRACKING_ERROR_Gen11E7.md`](../../../Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/Real_Time_eval_loggging/data_example_anlysis/TRACKING_ERROR_Gen11E7.md) — $e_t$ as the decisive scalar; DPCC's $w_t$
- [`CHANGELOG_uav_dynamics_anchor_to_p.md`](../../../Gen11/Epoch8_UAV_Mjpc_thrust_control/FIX_DYNAMIC_CONSTRAINTS/CHANGELOG_uav_dynamics_anchor_to_p.md) — DC_FIX, the six `deriv` rows
- [`INVESTIGATION_geo_free_model_free_worse_than_diffuser.md`](../../../Gen11/Epoch9_PCC_Constraints/U_13/INVESTIGATION_geo_free_model_free_worse_than_diffuser.md) — dynamics is the load-bearing family
- [`PROBLEM_projection_cost_explosion.md`](../../../Gen11/Epoch9_PCC_Constraints/Fix_15_projection_cost_explosion_guard/PROBLEM_projection_cost_explosion.md) — the 30.3 ms budget
- [`PLAN_E10_uav_visual_mode.md`](../../../Gen11/Epoch10_Visual_UAV/PLAN_E10_uav_visual_mode.md) — the *other* Gen11 expansion axis (vision); orthogonal to this one

**External**
- HardFlow: Li, Alim, Azizan, arXiv 2511.08425v3 — Problem 1–6, Alg. 1, Prop. 1, Appendices
- Mellinger & Kumar 2011 (differential flatness / minimum snap); Lee et al. 2010 (SE(3) geometric control)
- Mayne et al. 2000 (terminal ingredients, recursive feasibility); Xiao & Belta (HOCBF)
- `aux_repo/SafeFlowMPC`, `aux_repo/UAV-Flow` — nearest neighbours for §5.6 / the UAV arm
