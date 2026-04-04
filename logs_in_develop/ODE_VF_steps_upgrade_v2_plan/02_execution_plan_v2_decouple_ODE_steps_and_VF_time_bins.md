# Execution Plan (v2): Decouple ODE Inference Steps from VF Time Bins

## Mission target
Upgrade the current v2 FM path so we can tune:
1. ODE inference steps (runtime speed/accuracy)
2. VF time-bin behavior (model time-conditioning range)

independently.

This plan is scoped for v2 only.

---

## Scope and boundaries

### In scope
1. `flow_matcher_v2/models/diffusion.py`
2. `FM_v2_test/train_FM_Unet_v2.py`
3. `FM_v2_test/eval_FM_Unet_v2.py` (only if needed for argument plumbing)
4. `config/avoiding-d3il.py` (`flow_matching_v2`, `plan_fm_v2`)
5. new mission logs in this folder

### Out of scope
1. original `flow_matcher` and `flow_matcher_unet_v2`
2. DPCC trajectory-selection logic
3. projector/constraint math
4. non-v2 experiment keys

---

## Proposed parameter split (v2)

Add two v2-only parameters:
1. `vf_time_bins_v2`
2. `ode_inference_steps_v2`

Semantics:
1. `vf_time_bins_v2`: trained/expected integer time embedding range used by v2 UNet-facing time ids
2. `ode_inference_steps_v2`: number of Euler integration steps at inference

Backward compatibility rule:
- if either is missing, fallback to current behavior using `n_diffusion_steps`.

### Where these two parameters are set
Set both parameters directly in `config/avoiding-d3il.py` under:
1. `flow_matching_v2` (train-side config)
2. `plan_fm_v2` (inference-side config)

Recommended initial values:
1. `vf_time_bins_v2: 20`
2. `ode_inference_steps_v2: 20`

Then tune only `ode_inference_steps_v2` for speed/accuracy sweeps while keeping `vf_time_bins_v2` fixed.

---

## Technical design

### A) Training behavior
In v2 loss path, keep continuous Beta time sampling as is:
- `t ~ Beta(alpha,beta)`, `t = 1 - t`

For UNet time input in training:
1. convert continuous `t` to integer ids in `[0, vf_time_bins_v2 - 1]`
2. pass these ids to the model time embedding path

Reason:
- keeps model in a known, bounded embedding index range.

### B) Inference behavior
In `p_sample_loop(...)`:
1. iterate `ode_inference_steps_v2` times
2. use `dt = 1.0 / ode_inference_steps_v2`
3. compute continuous solver time `t_cont = i / ode_inference_steps_v2`
4. map to model time id by linear rescale to trained bin range:
   - `t_id = round(t_cont * (vf_time_bins_v2 - 1))`

Reason:
- decouples ODE numerical resolution from embedding range used during training.

### C) Compatibility mode
If config does not provide new keys:
1. set `vf_time_bins_v2 = n_diffusion_steps`
2. set `ode_inference_steps_v2 = n_diffusion_steps`

This preserves today’s behavior exactly.

---

## Implementation steps

1. Add constructor args in v2 diffusion:
- `vf_time_bins_v2=None`
- `ode_inference_steps_v2=None`

2. Resolve effective values in constructor:
- `self.vf_time_bins_v2 = vf_time_bins_v2 or n_timesteps`
- `self.ode_inference_steps_v2 = ode_inference_steps_v2 or n_timesteps`

3. Add helper for time-id mapping:
- float in `[0,1]` -> integer in `[0, vf_time_bins_v2 - 1]`

4. Update training call path to pass mapped integer ids to model.

5. Update inference loop:
- iterations use `ode_inference_steps_v2`
- `dt` uses `ode_inference_steps_v2`
- model timestep ids use mapped bins in trained range.

6. Wire config from train script:
- pass both new params into diffusion config.

7. Add new keys in config blocks:
- `flow_matching_v2`
- `plan_fm_v2`

Required keys to add in both blocks:
1. `vf_time_bins_v2`
2. `ode_inference_steps_v2`

8. Run static checks for edited files.

---

## Validation plan

### Functional checks
1. With defaults (`vf_time_bins_v2=20`, `ode_inference_steps_v2=20`), output path matches current behavior.
2. With `vf_time_bins_v2=20`, `ode_inference_steps_v2=40`, inference runs without out-of-range time ids.
3. With `vf_time_bins_v2=20`, `ode_inference_steps_v2=10`, inference runs with expected speed-up.

### Logging checks
1. Confirm `args.json` records new v2 params.
2. Confirm saved path naming remains unchanged unless explicitly updated.

---

## Risks and mitigations

1. Risk: changing training time-id mapping may shift baseline metrics.
- Mitigation: keep default equal-value mode and run A/B with same seed.

2. Risk: projector threshold logic uses `self.n_timesteps` assumptions.
- Mitigation: normalize projector threshold checks against `ode_inference_steps_v2` in the loop when decoupled mode is active.

3. Risk: hidden assumptions in model time embedding for integer scale.
- Mitigation: always map inference ids to trained range `[0, vf_time_bins_v2-1]`.

---

## Acceptance criteria

1. New folder-level logs exist (this folder).
2. v2 can set ODE steps independently from VF bins.
3. old default behavior remains unchanged when new params are omitted.
4. no changes outside approved v2/config scope.

---

## Suggested first experiment set

1. Baseline: `vf_time_bins_v2=20`, `ode_inference_steps_v2=20`
2. High-accuracy ODE: `vf_time_bins_v2=20`, `ode_inference_steps_v2=40`
3. Fast ODE: `vf_time_bins_v2=20`, `ode_inference_steps_v2=10`

Hold all other settings fixed for clean attribution.
