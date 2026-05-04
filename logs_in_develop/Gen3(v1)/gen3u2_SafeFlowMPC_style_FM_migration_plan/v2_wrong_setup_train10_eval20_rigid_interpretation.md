# Rigid Interpretation Note: FM-v2 Wrong Setup (Train ODE=10, Eval ODE=20)

Date: 2026-04-06
Scope: FM-PCC v2 (`flow_matching_v2` / `plan_fm_v2`)

---

## 1) Problem Statement

Intended setup:
1. Train with `ode_inference_steps_v2 = 10`
2. Eval with `ode_inference_steps_v2 = 10`

Actual setup that happened:
1. Train used `flow_matching_v2.ode_inference_steps_v2 = 10`
2. Eval used `plan_fm_v2.ode_inference_steps_v2 = 20`

Question: how to interpret the result rigorously, and whether retraining is required.

---

## 2) Code-Path Ground Truth

### A) Training path
Training script uses experiment `flow_matching_v2`, so it reads the `flow_matching_v2` block in `config/avoiding-d3il.py`.

In `flow_matcher_v2/models/diffusion.py`, training objective is implemented in `loss(...)` -> `p_losses(...)`.
This path samples continuous `t` and computes FM loss from `v_pred` vs `v_target`.

Key point:
- Training loss path does **not** run the inference ODE rollout loop.

### B) Evaluation path
Eval script uses experiment `plan_fm_v2`, so it reads the `plan_fm_v2` block in `config/avoiding-d3il.py`.

In `flow_matcher_v2/models/diffusion.py`, inference uses:
1. `p_mean_variance(...)`: `dt = 1.0 / ode_inference_steps_v2`
2. `p_sample_loop(...)`: number of integration steps = `ode_inference_steps_v2`

Key point:
- Eval metrics are directly affected by `plan_fm_v2.ode_inference_steps_v2`.

---

## 3) Rigid Interpretation of the Wrong Setup

Given train=10 and eval=20:

1. The checkpoint is valid; weights are not corrupted.
2. No retraining is required to correct this mismatch.
3. The obtained eval numbers are valid for **20-step solver inference**.
4. The numbers are **not valid evidence** for 10-step deployment behavior.

So this is a **metadata/config inconsistency**, not a model-training failure.

---

## 4) What Conclusions Are Allowed vs Not Allowed

Allowed conclusions:
1. "This checkpoint evaluated with ODE solver steps=20 achieved metric X."
2. "20-step solver runtime/quality tradeoff is Y for this checkpoint."

Not allowed conclusions:
1. "This checkpoint evaluated at 10 steps achieved X" (if run used 20).
2. Any claim comparing "10-step vs baseline" using this mismatched run as the 10-step point.

---

## 5) Expected Behavioral Shift (Eval 10 -> Eval 20)

Holding checkpoint fixed, increasing eval ODE steps from 10 to 20 typically implies:
1. Smaller Euler step size (`dt` halves).
2. More integration steps (2x loop count).
3. Higher compute time.
4. Often smoother/more stable rollout integration.
5. Possible metric improvement, but not guaranteed.

Thus mismatch can bias reported speed/quality in either direction, but speed is almost always worse at 20.

---

## 6) Recovery Procedure (No Retrain)

1. Keep trained checkpoint unchanged.
2. Set `plan_fm_v2.ode_inference_steps_v2 = 10`.
3. Re-run eval only.
4. Report two rows with same checkpoint:
   - eval steps = 20 (existing run)
   - eval steps = 10 (corrected run)

This gives a clean solver-ablation view with no retraining confound.

---

## 7) Notes on `vf_time_bins_v2`

1. `vf_time_bins_v2` controls time-id embedding granularity (model time representation).
2. `ode_inference_steps_v2` controls numerical integration granularity (solver).
3. For stable attribution, keep `vf_time_bins_v2` fixed (e.g., 20) and sweep only `ode_inference_steps_v2`.

---

## 8) Final Judgment

This was a **configuration mismatch between train and plan blocks**, not a training mistake.
The correct interpretation of your existing result is:

- "FM-v2 checkpoint evaluated with ODE steps = 20"

and not:

- "FM-v2 checkpoint evaluated with ODE steps = 10".

No retraining is required; only eval rerun and result relabeling are required.
