# 01 Implementation Plan: FM-v3 SafeFlow Approach in FM-PCC

Date: 2026-04-06
Status: Revised Minimal Plan
Depends on: `00_evaluation_fmv2_to_fmv3_with_safeflowmpc.md`

---

## 1) Goal

Use the SafeFlowMPC FM approach in v3 engine behavior, inside FM-PCC structure.

Target outcome:
1. SafeFlow-style continuous-time VF training/inference semantics in v3 engine,
2. FM-PCC train/eval tooling and project structure preserved,
3. v2 path remains untouched for rollback and A/B.

---

## 2) What Will Change (Only These)

1. Create `flow_matcher_v3` by copying `flow_matcher_v2`.
2. Modify FM engine inside `flow_matcher_v3` only (v2 code stays untouched).
3. Add FM-v3 experiment keys in config:
   - `flow_matching_v3`
   - `plan_fm_v3`
4. Add FM-v3 script entrypoints by copying v2 scripts and changing experiment keys:
   - `FM_v3_test/train_FM_v3.py`
   - `FM_v3_test/eval_FM_v3.py`
   - `FM_v3_test/load_results_FM_v3.py`
5. Keep explicit v3 parameter names for clarity:
   - `time_beta_alpha_v3`, `time_beta_beta_v3`
   - `flow_steps_v3` (SafeFlow-style inference step knob)
   - `ode_inference_steps_v3` as compatibility alias to `flow_steps_v3`
6. Add one clear result label line in loader output:
   - `Eval ODE=<value>, FlowSteps=<value>, Beta=(alpha,beta)`

---

## 3) What Will NOT Change

1. No backbone redesign.
2. No new training objective.
3. No rollout-based training loss.
4. No large refactor outside v3 engine copy.
5. No behavior change to current v2 path.

---

## 4) SafeFlowMPC Principle Mapping (FM-v3)

Keep the same principle split:
1. Train-time: Beta-shaped time sampling for FM supervision.
2. Model-time: use continuous time conditioning in the v3 engine path.
3. Inference-time: flow-step ODE rollout (`dt = 1 / flow_steps`).

Current truth for v2 baseline:
1. Beta-time is already implemented in v2 (`loss(...)` samples Beta and applies `t = 1 - t`).
2. FM-v3 keeps this and does not re-implement Beta from scratch.

This is principle parity, not codebase merge.

## 4.1) Math Change Map (from -> to)

This section states exactly what changes mathematically from v2 to v3.

### A) VF training objective

From (v2):
1. Beta-time FM supervision is already used.
2. Interpolation path: $x_t = x_0 + t(x_1-x_0)$.
3. Target velocity: $v^* = x_1-x_0$.
4. Loss: $\mathcal{L}=\mathbb{E}[\|v_\theta(x_t,\text{time input})-v^*\|^2]$.

To (v3):
1. Keep the same FM objective family and same Beta-time sampling.
2. Keep the same interpolation path and same target velocity.
3. Change time-input semantics to SafeFlow-style continuous-time query in v3 engine path.

Net effect:
1. Objective stays FM velocity matching.
2. Main math change is time-conditioning interface (discrete-bin-centric -> continuous-time-centric in v3 path).

### B) ODE inference rollout

From (v2):
1. Euler rollout with step count knob.
2. $\Delta t = 1/N$ and loop for $N$ steps.
3. Time query by step ratio.

To (v3):
1. Keep same SafeFlow-style ODE semantics explicitly:
   - $N = \text{flow\_steps\_v3}$,
   - $\Delta t = 1/N$,
   - $t_i = i/N$,
   - $x_{i+1}=x_i+v_\theta(x_i,t_i)\Delta t$.
2. Use `ode_inference_steps_v3` only as compatibility alias to `flow_steps_v3`.

Net effect:
1. ODE math is the same class, but v3 naming/semantics are aligned to SafeFlow convention (`flow_steps`).

### C) What does NOT change mathematically

1. No switch to diffusion denoising objective.
2. No rollout-backprop trajectory loss introduction.
3. No backbone-theory change required by this phase.

## 4.2) Note: What is the FM standard approach?

Short answer: SafeFlow-style time handling is closer to the textbook FM standard.

FM standard (literature-level) usually means:
1. learn continuous-time velocity field $v_\theta(x,t)$,
2. train with time samples in $t\in[0,1]$,
3. generate with ODE solver rollout over continuous time.

How your current FM-PCC v2 fits:
1. same FM objective family (velocity matching),
2. same ODE-style inference class,
3. but model-time input is more discretized/bin-oriented in implementation.

How SafeFlow fits:
1. same FM objective family,
2. continuous-time model query style,
3. flow-step ODE rollout naming/semantics are explicit.

Conclusion for this v3 plan:
1. your v2 method is a valid FM implementation variant,
2. SafeFlow approach is the better target for "standard FM style" alignment,
3. FM-v3 moves to SafeFlow-style time semantics while preserving FM-PCC structure.

### SafeFlowMPC-derived shortlist to add/change in FM-v3

Required:
1. Keep eval ODE default at 10 for v3 parity with SafeFlowMPC planner default (`flow_steps=10`).
2. Keep ODE integration semantics explicit and unchanged:
   - loop count = ODE steps,
   - `dt = 1 / ODE_steps`,
   - query time as `t = i / ODE_steps`.
3. Keep Beta-time sampling as already completed (no rework).
4. Keep U-Net velocity-field training logic structure (pointwise VF supervision), but in v3 use SafeFlow-style continuous time in model query path.
5. Keep ODE inference rollout logic (`dt=1/N`, loop over N, time query by step ratio) explicit in `flow_matcher_v3`.
6. Deprecate v3 dependence on VF-bin quantization for core behavior (keep compatibility only if needed for old interfaces).

Optional but recommended:
1. EMA decision note (FM-PCC-first):
   - Do not switch EMA behavior just to mimic SafeFlowMPC design pattern.
   - Keep current FM-PCC EMA approach as default for FM-v3 unless we have measured evidence that SafeFlow-style EMA usage is better on FM-PCC metrics.
   - ~~If needed later, run an explicit A/B (FM-PCC default EMA path vs SafeFlow-style EMA inference path) before any behavior change.~~

---

## 5) Exact File-Level Edit List

1. Create `flow_matcher_v3/` by copying `flow_matcher_v2/`.
2. Edit `flow_matcher_v3/models/diffusion.py`
   - keep SafeFlowMPC-aligned principles explicit (Beta train-time, ODE rollout-time),
   - switch v3 model-time query path to continuous time semantics,
   - use `flow_steps_v3`/`ode_inference_steps_v3` for inference step loop and `dt`.
3. Edit `flow_matcher_v3/models/unet1d_temporal_cond.py` only if needed
   - ensure time-conditioning path accepts continuous float time in v3 path.
4. Edit `config/avoiding-d3il.py`
   - add `flow_matching_v3` block
   - add `plan_fm_v3` block
   - add `flow_steps_v3`/`ode_inference_steps_v3` fields with clear comments
5. Create `FM_v3_test/train_FM_v3.py`
   - clone from `FM_v2_test/train_FM_v2.py`
   - set parser experiment to `flow_matching_v3`
6. Create `FM_v3_test/eval_FM_v3.py`
   - clone from `FM_v2_test/eval_FM_v2.py`
   - set parser experiment to `plan_fm_v3`
7. Create `FM_v3_test/load_results_FM_v3.py`
   - clone from `FM_v2_test/load_results_FM_v2.py`
   - set parser experiment to `plan_fm_v3`
   - print/store the explicit label line

No other files are in initial scope.

---

## 6) ~~Minimal Acceptance Criteria~~

1. ~~FM-v3 scripts run with new config keys.~~
2. ~~FM-v2 path remains unchanged.~~
3. ~~FM-v3 engine uses SafeFlow-style continuous-time model query path.~~
4. ~~Output tables/logs show explicit eval ODE/FlowSteps label.~~
4. ~~Old result interpretation rule is preserved:~~
   - ~~historical mismatch results are labeled `Eval ODE=20`.~~
5. ~~FM-v3 default eval run uses ODE=10 unless explicitly overridden.~~

---

## 7) Execution Order

1. Config blocks first.
2. Script copies with new experiment keys.
3. Loader labeling update.
4. ~~One smoke run per script (train/eval/load).~~

---

## 8) Immediate Next Action

Implement the seven file tasks in Section 5 exactly, with no extra scope.
