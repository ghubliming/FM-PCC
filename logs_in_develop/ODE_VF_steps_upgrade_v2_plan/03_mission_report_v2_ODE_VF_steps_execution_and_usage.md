# Mission Report: v2 ODE/VF Steps Decoupling Execution

## Mission Status
Completed.

This report documents execution of the approved mission:
1. start implementation of the v2 ODE/VF decoupling plan,
2. keep scope in current v2 path,
3. provide usage instructions.

---

## Scope Compliance Check

### In-scope targets (executed)
1. `flow_matcher_v2/models/diffusion.py`
2. `FM_v2_test/train_FM_Unet_v2.py`
3. `config/avoiding-d3il.py`
4. new mission report in this folder

### Out-of-scope protections (respected)
1. No edits to original `flow_matcher`.
2. No edits to original `flow_matcher_unet_v2`.
3. No edits to DPCC trajectory-selection logic.
4. No edits to projector/constraint formulation.

---

## What Was Implemented

## A) v2 diffusion decoupling logic
Edited file:
1. `flow_matcher_v2/models/diffusion.py`

Implemented:
1. Added constructor arguments:
   - `vf_time_bins_v2=None`
   - `ode_inference_steps_v2=None`

2. Added fallback behavior for compatibility:
   - if missing, both fallback to `n_timesteps` (old behavior).

3. Added model-time mapping helper:
   - continuous `t in [0,1]` -> integer UNet time id in `[0, vf_time_bins_v2-1]`.

4. Decoupled inference ODE integration from training bin count:
   - integration count now uses `ode_inference_steps_v2`
   - Euler step size uses `dt = 1.0 / ode_inference_steps_v2`
   - UNet timestep ids are mapped into trained bin range via `vf_time_bins_v2`

5. Kept training interpolation target structure unchanged:
   - still uses FM interpolation and velocity target,
   - now explicitly maps continuous sampled `t` to model ids via `vf_time_bins_v2` before UNet call.

## B) v2 train wiring
Edited file:
1. `FM_v2_test/train_FM_Unet_v2.py`

Implemented:
1. Passed new args into diffusion config:
   - `vf_time_bins_v2=args.vf_time_bins_v2`
   - `ode_inference_steps_v2=args.ode_inference_steps_v2`

## C) Config parameters (requested approach)
Edited file:
1. `config/avoiding-d3il.py`

Implemented for `flow_matching_v2`:
1. `vf_time_bins_v2: 20`
2. `ode_inference_steps_v2: 20`

Implemented for `plan_fm_v2`:
1. `vf_time_bins_v2: 20`
2. `ode_inference_steps_v2: 20`

---

## Validation Results

Static checks run on edited files:
1. `flow_matcher_v2/models/diffusion.py` -> no errors
2. `FM_v2_test/train_FM_Unet_v2.py` -> no errors
3. `config/avoiding-d3il.py` -> no errors

---

## Code-Math Meaning (Short)

1. ODE solver resolution is now controlled by `ode_inference_steps_v2`:
   - $$x_{i+1} = x_i + v_\theta(x_i, t_i) \cdot \Delta t$$
   - $$\Delta t = 1 / \text{ode\_inference\_steps\_v2}$$

2. UNet time embedding range is controlled by `vf_time_bins_v2`:
   - continuous solver/training time is mapped to integer ids in `[0, vf_time_bins_v2-1]`.

3. This separates:
   - numerical integration granularity,
   - model time-id binning range.

---

## How to Use

## 1) Default behavior (same as before)
In `config/avoiding-d3il.py` keep:
1. `vf_time_bins_v2: 20`
2. `ode_inference_steps_v2: 20`

This reproduces the prior 20-step behavior with explicit decoupling enabled.

## 2) Faster inference test
Keep `vf_time_bins_v2` fixed and lower ODE steps:
1. `vf_time_bins_v2: 20`
2. `ode_inference_steps_v2: 10`

Expected:
1. fewer UNet calls,
2. faster runtime,
3. potentially more integration error.

## 3) Higher-accuracy inference test
Keep `vf_time_bins_v2` fixed and raise ODE steps:
1. `vf_time_bins_v2: 20`
2. `ode_inference_steps_v2: 40`

Expected:
1. more UNet calls,
2. slower runtime,
3. lower Euler discretization error.

## 4) Training and evaluation entry points
Use copied v2 scripts:
1. train: `FM_v2_test/train_FM_Unet_v2.py` with experiment `flow_matching_v2`
2. eval: `FM_v2_test/eval_FM_Unet_v2.py` with experiment `plan_fm_v2`

Both now read the two new parameters from config.

---

## Final Changed Paths
1. `flow_matcher_v2/models/diffusion.py`
2. `FM_v2_test/train_FM_Unet_v2.py`
3. `config/avoiding-d3il.py`
4. `logs_in_develop/ODE_VF_steps_upgrade_v2_plan/03_mission_report_v2_ODE_VF_steps_execution_and_usage.md`

---

## Mission Conclusion
Execution completed successfully under the approved v2-only direction, and usage instructions are now documented in this `03` report.
