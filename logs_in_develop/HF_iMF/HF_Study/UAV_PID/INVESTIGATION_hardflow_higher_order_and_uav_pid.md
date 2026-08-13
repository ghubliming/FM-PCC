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
- Prior HF_Study / Gen11 docs, cited inline in §9.

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
constraint on all of it is the 30.3 ms real-time budget (§6), which HF's published 190 ms/replan misses
by 6.3×.

---

## 1. Disambiguating "2nd/higher order" — three axes, only one is open

| # | Axis | What "order" means | HardFlow's status | Open for us? |
|---|---|---|---|---|
| **A** | **Sampling-time solver order** | Euler vs Heun vs RK on $\dot{x}_\tau = v^\theta_\tau(x_\tau)$, $\tau\in[0,1]$ | **Solved & published.** Appendix "High-Order Solvers and Non-Uniform Time Grids": swap $\Psi_i^\theta$, all theory carries. Heun measured *faster* than default on image editing (38.4 s vs 51.3 s). | **No.** Nothing to invent. It *is* a compute lever (§6). |
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
sampling step, and §6 shows we have no headroom. The levers, in order of leverage:

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

## 6. The wall everything hits: 30.3 ms

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

## 7. A staged program

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

## 8. Open questions to settle before committing

1. **Which tracker is the object of study — PID or MJPC?** D3b/D4/D5 all assume an analyzable closed
   loop. MJPC (`mjpc_tracker.py`) is a sampling optimizer, not an LTI system. *Recommendation: PID for
   the theory arm; MJPC as an empirical robustness check.*
2. **Is the FM's `v` channel even accurate?** It is currently unconstrained and never evaluated. If the
   FM's predicted $v$ is poor, constraining $p$–$v$ consistency (§5.1) may *hurt* before it helps.
   Measure it in Stage 0 — same logs, near-zero marginal cost.
3. **$H=8$ / 0.242 s — is the horizon long enough for a terminal set to mean anything?** Stopping from
   typical plan speeds may take longer than the horizon, in which case D5 needs $H$ to grow, which
   directly re-inflates §6. This tension is real and should be quantified before D5 is attempted.
4. **Do we have $T_{max}$, $\theta_{max}$, $j_{max}$ for the Skydio X2?** Mass and inertia are read from
   the MuJoCo model already; `u_max = max(2·u_hover, 6.0)` in `flight_controller.py` is a *heuristic
   cap*, not a datasheet figure. §5.2 is only as good as these numbers.
5. **Normalizer units for any fitted $(A,B,c)$** — the Gen12 warning applies verbatim.

---

## 9. Cross-references

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
