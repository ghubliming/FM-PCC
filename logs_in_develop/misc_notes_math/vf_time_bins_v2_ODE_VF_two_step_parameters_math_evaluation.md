# v2 Math Re-Evaluation (Verified): 2-Parameter Setup from FM-PCC vs SafeFlowMPC

This version explicitly checks the quoted report against code and then updates the recommendation.

---

## 1) Verification of the quoted report (True / Needs nuance)

## A) "FM uses continuous time $t\in[0,1]$" -> TRUE
Both codebases use continuous-time FM logic conceptually, then feed time-conditioned networks during training/inference.

## B) "SafeFlowMPC training discretizes to 50 steps" -> TRUE, but only for one training branch
In SafeFlowMPC `train_imitation_learning_safe.py`:
1. one branch samples from precomputed tensors using `idx_t = (t * 50).int()` (50-step discretized lookup),
2. another branch uses direct continuous interpolation `x_t = x_0 + t(x_1-x_0)` without 50-step table lookup.

So the statement is correct but incomplete: SafeFlowMPC training is mixed-mode.

## C) "SafeFlowMPC inference uses flow_steps=10 (sometimes 7)" -> TRUE
Verified from:
1. `PlannerConfig.py`: default `flow_steps=10`,
2. `SafeFlowMPC.py`: `dt = 1/flow_steps`, loop over `flow_steps`,
3. `inference_global_planner.py`: example override `flow_steps=7`.

## D) "Use this directly for FM-PCC v2" -> PARTIALLY TRUE
It is a useful prior, but FM-PCC avoiding has different inference coupling:
1. projector/constraints each step,
2. different state/action layout,
3. different success metric composition (goal + constraints).

So SafeFlowMPC step counts are a reference prior, not a direct transfer optimum.

---

## 2) Target difference that matters for recommendation

## A) SafeFlowMPC target
Target is safety-filtered robot trajectory generation to goal under explicit safety filter loop.

## B) FM-PCC target on avoiding
Target is feasible trajectory generation under multiple constraints with projector corrections, where evaluation emphasizes:
1. goal success,
2. constraint satisfaction,
3. violation magnitude,
4. runtime.

Implication: FM-PCC may need slightly more conservative ODE step resolution than aggressive SafeFlowMPC fast settings.

---

## 3) FM math meaning for current v2 parameters

The current v2 parameters are:
1. `vf_time_bins_v2`
2. `ode_inference_steps_v2`

### `vf_time_bins_v2`
Defines the integer time-id range seen by the UNet embedding:
$$
k = \mathrm{clip}(\lfloor t\cdot\text{vf\_time\_bins\_v2}\rfloor,0,\text{vf\_time\_bins\_v2}-1)
$$

### `ode_inference_steps_v2`
Defines Euler solver resolution and runtime:
$$
\Delta t=\frac{1}{\text{ode\_inference\_steps\_v2}},\quad
x_{i+1}=x_i+v_\theta(x_i,k_i)\Delta t
$$

Interpretation:
1. keep `vf_time_bins_v2` aligned with training embedding regime,
2. tune `ode_inference_steps_v2` as accuracy-speed knob.

---

## 4) Updated reasonable initial recommendation (after verification)

Given verified SafeFlowMPC references (10 default, 7 fast) and FM-PCC avoiding constraints, the recommended initial v2 pair is:

1. **`vf_time_bins_v2 = 20`**
2. **`ode_inference_steps_v2 = 10`**

Why this is now preferred:
1. `vf_time_bins_v2=20` preserves current v2 model-time embedding regime.
2. `ode_inference_steps_v2=10` follows the published/implemented SafeFlowMPC default step scale, not just anecdotal fast mode.
3. It gives a stronger speed gain than 12 while staying at a known stable reference point from the published code style.
4. For FM-PCC avoiding constraints, 10 is a principled first point before deciding to increase to 12/16 if violation metrics degrade.

---

## 5) Immediate sweep around the new initial point

Start from:
1. `(vf_time_bins_v2, ode_inference_steps_v2) = (20,10)` <- initial default

Then test:
1. `(20,12)`
2. `(20,16)`

Evaluate with:
1. success rate (goal)
2. success rate (goal + constraints)
3. average violations and total violations
4. average computation time per step

Decision objective:
$$
J = \text{SR}_{goal+safe} - \lambda\,\text{viol} - \mu\,\text{time}
$$

---

## 6) Final rewritten answer

After checking the quoted report against code and applying FM-PCC avoiding target differences, the most reasonable **initial** v2 setup is:

1. **`vf_time_bins_v2 = 20`**
2. **`ode_inference_steps_v2 = 10`**

Then locally sweep `ode_inference_steps_v2 in {12,16}` with `vf_time_bins_v2=20` fixed.
