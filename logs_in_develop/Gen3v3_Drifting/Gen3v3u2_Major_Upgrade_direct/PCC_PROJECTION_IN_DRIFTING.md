# PCC Projection in Drifting (FM-D) — and 3-Way Comparison

**Date**: 2026-05-22  
**iMF reference**: [`../../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/PCC_PROJECTION_IN_IMF.md`](../../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/PCC_PROJECTION_IN_IMF.md)

---

## How Drifting handles PCC

**Yes, PCC projection is implemented in Drifting.** It lives in
`flow_matcher_v3_drifting/models/diffusion.py → GaussianDiffusion.p_sample_loop`,
with the same snapping formula as iMF:

```python
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
```

Projection is applied near the **end of forward ODE integration** (near t=1, near data),
identical to iMF. Both gradient and SLSQP modes are supported.

**Key difference from iMF**: the velocity is predicted by a `DriftAugmentedUNet1D`
— a standard flow-matching UNet plus a learned *drift field* that biases trajectories
toward constraint-satisfying regions **during generation**, before any explicit
projection is applied. This means the base trajectory is already "leaning toward"
valid regions before the SLSQP snap.

### Save path

`plan_fm_v3_drifting` already has `'diffusion_timestep_threshold': _yaml_threshold`
in its config dict — `T` was already encoded in the path. No fix needed.

### How to adjust the threshold

Same as iMF: change `diffusion_timestep_threshold` in `config/projection_eval.yaml`.
It flows through to `Projector(diffusion_timestep_threshold=...)` and into
`p_sample_loop` via `projector.diffusion_timestep_threshold`.

---

## 3-Way Comparison: DPCC vs iMF vs Drifting

| | **DPCC** | **iMF** | **Drifting (FM-D)** |
|---|---|---|---|
| **Model class** | `GaussianDiffusion` (diffuser) | `iMFDiffusion` | `GaussianDiffusion` (drifting, FM-style) |
| **Integration direction** | Reverse: noise ← data | Forward: noise → data | Forward: noise → data |
| **Integration steps** | T=20 discrete timesteps | K=10 ODE steps | K=10 ODE steps |
| **Velocity / update rule** | ε-prediction + DDPM posterior | Mean flow velocity | Mean flow velocity **+ drift field** |
| **Drift augmentation** | ✗ | ✗ | ✓ `DriftAugmentedUNet1D` biases toward feasible region during sampling |
| **ODE backend** | N/A (DDPM reverse chain) | `legacy_euler` only | `legacy_euler` or `torchdiffeq` |
| **Projection gate formula** | `t < T × threshold` (low t-index = near data in reverse chain) | `loop_idx ≥ K × (1 − threshold)` | `loop_idx ≥ K × (1 − threshold)` — **identical to iMF** |
| **Projection timing intuition** | Last `threshold × T` denoising steps | Last `threshold × K` ODE steps | Last `threshold × K` ODE steps |
| **`T` in save path** | Always encoded | Fixed (added `_yaml_threshold` 2026-05-22) | Always encoded |
| **Threshold knob** | `diffusion_timestep_threshold` in `projection_eval.yaml` | Same | Same |
| **Projection modes** | Gradient + SLSQP | Gradient + SLSQP | Gradient + SLSQP |

### Key insight

- **DPCC** projects during reverse denoising — the model is already noise-to-data
  trained; projection redirects the trajectory at each denoising step.
- **iMF** projects during the final ODE steps with no additional structural bias —
  pure flow + late-stage snap.
- **Drifting** projects during the final ODE steps just like iMF, but the
  velocity field itself already has a learned drift component that nudges the
  trajectory toward feasible regions *before* SLSQP is called. In theory this
  means SLSQP has less correction work to do.

All three read the same `diffusion_timestep_threshold` from `projection_eval.yaml`
and pass it through the same `Projector` interface — the threshold semantics are
directly comparable across all three models.
