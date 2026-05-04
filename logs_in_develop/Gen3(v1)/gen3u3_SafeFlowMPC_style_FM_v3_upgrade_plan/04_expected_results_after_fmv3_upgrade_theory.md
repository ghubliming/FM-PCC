# 04 Expected Results After FM-v3 Upgrade (Theory)

Date: 2026-04-06
Status: Pre-test expectation note
Depends on: 03_fmv3_coding_execution_record.md

---

## 1) Baseline for Comparison

Use this baseline for first comparison:
1. FM-v2 with Beta-time.
2. Eval ODE = 10.

Reason:
1. FM-v3 changed time-query semantics and naming, not the core FM objective family.
2. So expected changes are mostly in stability/consistency, not a dramatic objective jump.

---

## 2) Train Results We Should Expect

Keywords: similar convergence, smoother curve, no collapse.

Expected:
1. Training loss should decrease normally (no divergence or flatline at high value).
2. Final train loss should be in the same order of magnitude as FM-v2 Beta-time baseline.
3. Test loss should track train loss trend without large instability spikes.
4. A0-related behavior should stay consistent with chosen action weight setting.

Not expected:
1. Massive train-loss drop only from v3 rename/semantics changes.
2. Huge degradation if config keys are mapped correctly.

Red flags:
1. NaN/Inf loss.
2. Loss exploding early and never recovering.
3. Loss plateau far above FM-v2 Beta-time baseline under same seed/budget.

---

## 3) Eval Results We Should Expect

Keywords: parity-plus, smoother rollout, stable constraints.

Expected:
1. Success rate should be close to, or modestly better than, FM-v2 Beta-time with ODE=10.
2. Constraint satisfaction should be at least baseline-level in matched settings.
3. Average step count to success should be similar range as baseline.
4. Trajectory rollout should look smoother/less jittery due to continuous-time query semantics.

Not expected:
1. Guaranteed large jump on every metric without additional model/backbone/training changes.
2. Large regressions when using same seeds and same environment conditions.

Red flags:
1. Consistent success-rate drop across seeds.
2. Increased violations with no compensating gain.
3. Strong sensitivity to tiny ODE/FlowSteps changes.

---

## 4) Parameter-Sweep Expectation (Eval)

Keywords: monotonic quality-cost trend.

For flow_steps_v3 sweep (for example 5, 10, 20):
1. Lower steps: faster but potentially less accurate rollout.
2. Mid steps (10): expected practical default trade-off.
3. Higher steps (20): potentially smoother/better but higher compute cost.

Theory expectation:
1. Quality should improve or saturate as steps increase.
2. Runtime should increase roughly with step count.

---

## 5) Labeling Rule for Clean Interpretation

Always log explicit setting labels:
1. Eval ODE=<value>, FlowSteps=<value>, Beta=(alpha,beta)

And always compare with matched:
1. seed list,
2. environment/trial count,
3. projection variant set,
4. action_weight setting.

---

## 6) Decision Rule After Your Test

Interpretation shortcut:
1. If FM-v3 is within noise band of baseline but smoother/more stable, upgrade is successful.
2. If FM-v3 is clearly better on success + constraints at similar compute, adopt FM-v3 as default.
3. If FM-v3 regresses consistently, inspect parameter wiring and time-query path before broader changes.

---

## 7) Parameter Set To Start With (FM-PCC vs SafeFlow Task Difference)

Why this section:
1. SafeFlowMPC and FM-PCC are related but not identical tasks.
2. We should copy time-semantics principle, not blindly copy all planner/task hyperparameters.

Task-difference reminder:
1. SafeFlowMPC is a robot safety-filtered planning stack with its own horizon/state/action design.
2. FM-PCC here is dataset-driven avoiding task with its own environment, projection settings, and metrics.

Recommended FM-v3 start set (theory-first, FM-PCC task):
1. `flow_steps_v3 = 10`
2. `ode_inference_steps_v3 = 10` (alias, keep equal to `flow_steps_v3`)
3. `time_beta_alpha_v3 = 1.5`
4. `time_beta_beta_v3 = 1.0`
5. `action_weight = 1` for first FM-v3 baseline pass

Why these values:
1. `flow_steps=10` follows SafeFlow default ODE resolution principle and your aligned v2 baseline.
2. Beta `(1.5, 1.0)` keeps your FM-v2 successful time-bias behavior.
3. `action_weight=1` matches current FM-v2/v3 practical tuning direction in this repo.

What not to copy directly from SafeFlow:
1. Do not force-copy SafeFlow horizon/model-width/planner internals as FM-v3 defaults for FM-PCC.
2. Keep FM-PCC task-specific environment/projection settings unchanged for fair comparison.

---

## 8) Is The Best Theory Start Already In Code?

Short answer: Yes, the proposed first-run theoretical set is already wired in code.

Confirmed in FM-PCC config:
1. `flow_matching_v3` has:
	- `flow_steps_v3 = 10`
	- `ode_inference_steps_v3 = 10`
	- `time_beta_alpha_v3 = 1.5`
	- `time_beta_beta_v3 = 1.0`
	- `action_weight = 1`
2. `plan_fm_v3` has:
	- `flow_steps_v3 = 10`
	- `ode_inference_steps_v3 = 10`
	- `time_beta_alpha_v3 = 1.5`
	- `time_beta_beta_v3 = 1.0`

Confirmed in FM-v3 engine behavior:
1. v3 diffusion resolves `flow_steps_v3` as primary with `ode_inference_steps_v3` alias.
2. v3 rollout uses `dt = 1 / flow_steps_v3` and continuous `t = i / flow_steps_v3` query.
3. v3 training loss samples Beta time using `time_beta_alpha_v3`, `time_beta_beta_v3`.

Practical note before you run tests:
1. Keep train/eval on this exact set first.
2. Only after baseline is recorded, sweep `flow_steps_v3` (for example 5, 10, 20).

---

## 9) FM-v2 vs FM-v3 Parameter Comparison (What Changed, Why)

Legend:
1. `CHANGED` = behavior/name changed in v3 path.
2. `SAME` = intentionally kept same for fair comparison.

### Core training/inference knobs

1. `time_beta_alpha_v2=1.5` -> `time_beta_alpha_v3=1.5` (`SAME value`, `CHANGED name`)
	- Why: keep the same beta-time bias behavior, but separate v3 namespace.
2. `time_beta_beta_v2=1.0` -> `time_beta_beta_v3=1.0` (`SAME value`, `CHANGED name`)
	- Why: preserve v2 beta-time baseline while moving to v3-specific config keys.
3. `ode_inference_steps_v2=10` -> `flow_steps_v3=10` + `ode_inference_steps_v3=10` alias (`CHANGED interface`)
	- Why: adopt SafeFlow-style naming/semantics (`flow_steps`) and keep compatibility alias for old call patterns.
4. `vf_time_bins_v2=20` -> removed as core v3 control (`CHANGED behavior`)
	- Why: v3 model query path is continuous-time oriented; discrete VF-bin control is no longer the primary behavior knob.

### Model query semantics

1. v2 model-time query: discretized/bin-oriented path (`CHANGED in v3`)
2. v3 model-time query: continuous `t in [0,1]` path (`CHANGED`)
	- Why: align with SafeFlow FM-style continuous-time querying and reduce discretization coupling.

### ODE rollout semantics

1. v2: Euler rollout with ODE steps (`SAME class of method`)
2. v3: Euler rollout with `flow_steps_v3` primary knob (`SAME method`, `CHANGED canonical knob`)
	- Why: keep inference math stable while making semantics explicit and SafeFlow-consistent.

### A0 action weight and objective family

1. `action_weight=1` in v2 FM baseline -> `action_weight=1` in v3 start set (`SAME`)
	- Why: avoid introducing confounders in first v2-v3 comparison.
2. FM objective family (velocity matching) remains the same (`SAME`)
	- Why: this upgrade targets time-conditioning semantics and interface clarity, not objective replacement.

### Practical interpretation of changes

1. What changed most:
	- time-query interface (bin-centric -> continuous-time-centric),
	- canonical inference-step knob (`ode_inference_steps` -> `flow_steps`).
2. What was intentionally kept same:
	- beta-time values,
	- action weight start value,
	- Euler ODE integration class,
	- FM velocity-matching objective family.
3. Expected effect:
	- improved semantic clarity and potentially smoother/stabler rollout behavior,
	- without expecting a guaranteed dramatic metric jump from this upgrade alone.
