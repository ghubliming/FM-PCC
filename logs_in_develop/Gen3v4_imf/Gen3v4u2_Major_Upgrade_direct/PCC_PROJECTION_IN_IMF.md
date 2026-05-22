# PCC Projection in iMeanFlow (iMF) — How It Differs from DPCC

**Date**: 2026-05-22

---

## Short answer

**Yes, PCC projection is implemented in iMF.** It lives in `p_sample_loop` inside
`flow_matcher_v3_imeanflow/models/imf_diffusion.py` and is triggered by passing a
`Projector` object through `Policy → iMFDiffusion.forward → p_sample_loop(projector=...)`.

---

## Why the save path has no threshold `T`

DPCC plan paths look like: `H8_T0.5_K10_D...`  
iMF plan paths look like:   `H8_K10_Meuler_D...`  ← **no T**

The `T` abbreviation comes from `args_to_watch_fmv3_ode_plan`:
```python
('diffusion_timestep_threshold', 'T'),
```
The `watch()` function skips keys whose value is `None`. The `plan_fm_v3_imeanflow`
config dict does **not** define `diffusion_timestep_threshold` as a model
hyperparameter, so the key resolves to `None` and is dropped from the path.

This means **all iMF projection variants share the same model checkpoint path** —
the threshold is carried by the `Projector` object at runtime (read from
`config/projection_eval.yaml`), not baked into the training path.

---

## How projection is applied: iMF vs DPCC

| | DPCC (Gaussian Diffusion) | iMF (ODE Flow) |
|---|---|---|
| **Process** | Reverse diffusion, T=20 discrete steps | Forward ODE integration, K=10 flow steps |
| **Threshold meaning** | Apply projection for timesteps ≤ threshold×T | Apply projection for ODE steps ≥ `(1-threshold)×K` |
| **Snapping point** | `t ≤ T_threshold` (near end of denoising) | `loop_idx ≥ snapping_start_idx` (near end of flow) |
| **Code** | `if t < threshold * T: project(x)` | `snapping_start_idx = int((1 - threshold) * flow_steps)` |

Example with `threshold=0.5`, `K=10`:
- `snapping_start_idx = int(0.5 × 10) = 5`
- Projection applies at ODE steps 5, 6, 7, 8, 9 (last half of the flow)

Both modes support:
- **Gradient projection** (`gradient=True`): adds constraint gradient to trajectory in-place
- **SLSQP projection** (`gradient=False`): calls `projector.project(x)` — SLSQP solve per step

---

## Why `diffusion_timestep_threshold` is still needed in `projection_eval.yaml`

Even for iMF runs, `projection_eval.yaml` must define `diffusion_timestep_threshold`
because `config/avoiding-d3il.py` reads it at import time and raises if missing:
```python
if 'diffusion_timestep_threshold' not in _proj_config:
    raise ValueError("CRITICAL: ...")
```
The value is passed to the `Projector` constructor and used inside `p_sample_loop`
via `projector.diffusion_timestep_threshold`.

---

## How to adjust the threshold for iMF

**Yes, fully adjustable.** Edit `config/projection_eval.yaml`:

```yaml
diffusion_timestep_threshold: 0.5   # ← change this
```

Data flow:
```
projection_eval.yaml
  └─ eval_flow_matching_v3_imeanflow.py line 51:
       diffusion_timestep_threshold = config.get('diffusion_timestep_threshold', 0.5)
  └─ Projector(..., diffusion_timestep_threshold=diffusion_timestep_threshold)
  └─ p_sample_loop:
       snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * flow_steps)
```

Effect with `K=10` flow steps:

| `threshold` | `snapping_start_idx` | Projection applies at steps | Projection window |
|-------------|---------------------|-----------------------------|-------------------|
| `0.1` | 9 | step 9 only | last 10% — very late snap |
| `0.5` | 5 | steps 5–9 | last half |
| `0.9` | 1 | steps 1–9 | almost all steps |
| `1.0` | 0 | steps 0–9 | every step |

### Save path — fixed

`diffusion_timestep_threshold: _yaml_threshold` is now added to the
`plan_fm_v3_imeanflow` config dict in `config/avoiding-d3il.py`, so `watch()`
includes `T` in the exp_name. iMF paths now look like:
`H8_K10_Meuler_T0.5_D...` — threshold sweeps go to separate folders, same as DPCC.

---

## Summary

- iMF **does** do PCC — projection logic is in `p_sample_loop`, applied during ODE steps
- The path **omits** the threshold because the model config doesn't store it (only the projector does)
- Semantics are mirrored: DPCC projects near the end of denoising; iMF projects near the end of the ODE flow — both controlled by the same threshold value from `projection_eval.yaml`
- **To adjust**: change `diffusion_timestep_threshold` in `projection_eval.yaml` — but be aware iMF runs with different thresholds will write to the same output path
