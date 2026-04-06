# 05 Math Note: Continuous vs Discrete Time Semantics

Date: 2026-04-06
Status: Discussion note (major update focus)
Scope: SafeFlow-style continuous-time query vs older FM-PCC v2 discrete/bin-oriented query

---

## 1) Executive Answer

Yes, this is the major FM-v3 update.

What changed most is not the FM objective family itself, but the time-conditioning semantics of the velocity-field query:
1. Old FM-PCC v2 path: continuous sampled time was quantized to discrete time bins for model query.
2. SafeFlow-style (and FM-v3 target): model is queried directly with continuous time $t \in [0,1]$.

This reduces coupling to an arbitrary bin count and aligns better with standard continuous-time FM formulation.

---

## 2) Code Reference Map

### FM-PCC old discrete/bin-oriented path (v2)

1. Time quantization helper:
   - `flow_matcher_v2/models/diffusion.py` (`_model_timestep_from_continuous`)
2. Training query uses quantized timestep:
   - `flow_matcher_v2/models/diffusion.py` (`p_losses`, variable `t_model`)
3. Inference query constructs continuous ratio then quantizes:
   - `flow_matcher_v2/models/diffusion.py` (`p_sample_loop`, `t_cont -> timesteps`)
4. U-Net forward casts scalar times to integer-like path:
   - `flow_matcher_v2/models/unet1d_temporal_cond.py` (`forward`, `dtype=torch.long` for non-tensor scalar)

### SafeFlow continuous-time path

1. Default flow-step resolution:
   - `SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/PlannerConfig.py` (`flow_steps = 10`)
2. Inference loop uses explicit continuous time ratio:
   - `SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py` (`dt = 1/flow_steps`, `t = flow_step/flow_steps`)
3. Velocity evaluation uses model at continuous `t` then multiplies by `dt`:
   - `SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/FlowMatchingField.py` (`compute_velocity`)

### FM-PCC upgraded path (v3)

1. v3 primary step knob and alias:
   - `flow_matcher_v3/models/diffusion.py` (`flow_steps_v3`, `ode_inference_steps_v3` alias)
2. Training query uses continuous `t` directly:
   - `flow_matcher_v3/models/diffusion.py` (`p_losses`, `v_pred = _predict_velocity(..., t, ...)`)
3. Inference query uses continuous `t_i = i/N` directly:
   - `flow_matcher_v3/models/diffusion.py` (`p_sample_loop`)
4. U-Net forward path explicitly float-typed for time tensor:
   - `flow_matcher_v3/models/unet1d_temporal_cond.py` (`forward`, `timesteps.float()`)

---

## 3) Shared FM Core Math (Unchanged Family)

All versions are in the same FM velocity-matching family:

1. Path interpolation (linear bridge):
$$
x_t = (1-t)x_0 + tx_1, \quad t \in [0,1]
$$
2. Target velocity:
$$
v^*(x_t,t) = x_1 - x_0
$$
3. Training objective (pointwise VF regression):
$$
\mathcal{L}(\theta)=\mathbb{E}_{x_0,x_1,t}\left[\|v_\theta(x_t,t)-v^*(x_t,t)\|^2\right]
$$
4. ODE rollout class (Euler discretization):
$$
x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k)
$$

So the upgrade is mainly in *how time is presented to the model*, not in replacing FM with another objective.

---

## 4) Old v2 Semantics: Continuous Sampled Time, Discrete Queried Time

In v2, train-time sampled time is continuous (Beta sample), but then mapped into bins for model query.

Define bin count $B$ (v2 uses `vf_time_bins_v2`) and quantizer
$$
Q_B(t)=\left\lfloor B\,t \right\rfloor\quad\text{(clamped to }[0,B-1]\text{)}.
$$

Then v2 model query is effectively
$$
v_\theta(x_t, Q_B(t))
$$
instead of directly
$$
v_\theta(x_t,t).
$$

At inference in v2:
1. build continuous ratio $t_i=i/N$ using ODE step count $N$,
2. quantize to bin index via $Q_B(t_i)$,
3. query model with quantized index.

This creates a two-knob coupling:
1. model-time resolution knob $B$ (`vf_time_bins_v2`),
2. solver-step knob $N$ (`ode_inference_steps_v2`).

---

## 5) SafeFlow Semantics: Continuous Model Query by Construction

SafeFlow uses explicit flow steps $N=\text{flow_steps}$ with
$$
\Delta t = 1/N,\quad t_i=i/N,
$$
and queries the model with this continuous $t_i$ directly.

Euler update:
$$
x_{i+1}=x_i+\Delta t\,v_\theta(x_i,t_i).
$$

No intermediate time-bin quantizer is required for core behavior.

---

## 6) FM-v3 Semantics: SafeFlow-Style Continuous Query in FM-PCC Structure

FM-v3 in this repo adopts the same core time semantics:

1. Training:
$$
v_\theta(x_t,t)\text{ with continuous }t\sim\text{Beta}(\alpha,\beta)
$$
2. Inference:
$$
N=\text{flow\_steps\_v3},\quad \Delta t=1/N,\quad t_i=i/N,
$$
$$
x_{i+1}=x_i+\Delta t\,v_\theta(x_i,t_i).
$$

Alias rule:
1. `flow_steps_v3` is primary semantic knob,
2. `ode_inference_steps_v3` is compatibility alias.

---

## 7) What "Continuous" Really Means In Practice

Your concern is correct: runtime always uses finite steps.

"Continuous-time FM" means:
1. the model is defined and queried as $v_\theta(x,t)$ for real-valued $t \in [0,1]$,
2. not that the computer integrates with infinitely many steps.

Actual rollout is still numerical integration with finite $N$:
$$
t_k = k/N,\quad \Delta t = 1/N,\quad x_{k+1}=x_k+\Delta t\,v_\theta(x_k,t_k).
$$

So two truths hold together:
1. model-time semantics are continuous-valued,
2. trajectory generation is piecewise-discrete because ODE solving is numerical.

Why this is still a major upgrade:
1. v3 removes extra time quantization at model input,
2. but it does not remove finite-step ODE discretization (which is always present in practice).

---

## 8) Why Continuous Query Is Theoretical Improvement

Assume $v_\theta(x,t)$ is Lipschitz in time with constant $L_t$:
$$
\|v_\theta(x,t)-v_\theta(x,s)\|\le L_t|t-s|.
$$

If quantized time $Q_B(t)/B$ is used, per-step time-input error is at most $1/B$.
Then velocity-input error from quantization is bounded by
$$
\|v_\theta(x,t)-v_\theta(x,Q_B(t)/B)\|\le L_t/B.
$$

So as $B$ gets finite/small, binning introduces extra approximation error unrelated to Euler truncation itself.
Continuous query removes this specific quantization term from the model-time interface.

Interpretation:
1. old v2 had solver discretization error + time-quantization interface error,
2. v3 keeps solver discretization (as any Euler method does) but removes the explicit bin-interface quantization in core query semantics.

---

## 9) Practical Expectation From This Major Update

From theory alone, expect:
1. Better semantic consistency between training/inference time handling.
2. Less sensitivity to arbitrary time-bin settings.
3. Potentially smoother rollout behavior (especially near transitions in bin boundaries).

Do **not** assume guaranteed huge metric jump solely from this change, because:
1. objective family is unchanged,
2. backbone/training budget may dominate absolute performance.

---

## 10) Minimal Comparison Checklist For Your Tests

When comparing v2 vs v3, keep fixed:
1. same seeds,
2. same environment and projection variants,
3. same action weight,
4. same train budget.

Only vary:
1. time semantics path (v2 quantized-query vs v3 continuous-query),
2. flow/ODE step count sweep after baseline.

Always label outputs with:
1. `Eval ODE=<...>, FlowSteps=<...>, Beta=(alpha,beta)`.

---

## 11) Bottom Line

Major update verdict:
1. Yes, the continuous-time query migration (SafeFlow-style) is the main mathematical upgrade in FM-v3.
2. This is a semantics-level correction/alignment rather than a new objective-family invention.
3. It is the right foundation before any larger architecture/training changes.
