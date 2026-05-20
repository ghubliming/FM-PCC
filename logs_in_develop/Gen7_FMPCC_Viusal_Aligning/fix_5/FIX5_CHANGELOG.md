# Fix 5 — `flow_steps_v3` Slurm Override Not Propagating to Model

**Date:** 2026-05-20
**Branch:** update_into_FM
**File modified:** `fm_visual_aligning_test/eval_fm_visual_aligning.py`

---

## Bug

When a Slurm job passed `--flow_steps_v3 N` (e.g. `N=1` for a fast smoke-test run),
the output directory was named with `K1` but the model still ran with `flow_steps_v3=100`
(the value baked into the checkpoint at training time).

**Root cause:** `diffusion_model.flow_steps_v3` is set from `diffusion_config.pkl` when
the checkpoint loads. The eval script read it for logging but never wrote `args.flow_steps_v3`
back to the model object. The Slurm override existed only in `args` — it never reached the
ODE integration loop in `p_sample_loop()`, which reads from `self.flow_steps_v3`.

```
# Before fix — diagnostic print showed checkpoint value, not Slurm arg:
[ eval ] FM flow_steps_v3 = 100  (Euler ODE integration steps 0→1)
# Directory name: .../K1/...   ← named from args, ran with 100 steps
```

This is safe to override at inference for Flow Matching because `flow_steps_v3` controls
only the Euler ODE step count (`dt = 1.0 / flow_steps_v3`). It is not tied to any buffer
or noise schedule. Fewer steps = coarser discretization, more steps = finer. No
architectural or checkpoint compatibility concern.

---

## Fix

After `clip_denoised` is forced, write the args value back to the model:

```python
_args_flow = getattr(args, 'flow_steps_v3', None)
if _args_flow is not None:
    diffusion_model.flow_steps_v3 = int(_args_flow)
    diffusion_model.ode_inference_steps_v3 = int(_args_flow)
```

The diagnostic print now appends `[overridden from args]` or `[checkpoint default]`
so the log makes the source of the value explicit.

---

## Reference pattern

`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` line 138-139
does this correctly:

```python
fm_model.flow_steps_v3 = int(getattr(args, 'flow_steps_v3', getattr(fm_model, 'flow_steps_v3', 10)))
fm_model.ode_inference_steps_v3 = int(getattr(args, 'ode_inference_steps_v3', ...))
```

Fix 5 aligns the visual-aligning eval script with that pattern.

---

## Not applicable to Gen6V4 (DDPM)

Gen6V4 (`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`) uses a DDPM reverse
chain. Its `n_timesteps` attribute is inseparable from the `betas`, `alphas_cumprod`, and
all derived buffers that were registered at checkpoint creation time. Overriding `n_timesteps`
at inference without recomputing those buffers would index them incorrectly or corrupt the
denoising chain. The existing mismatch warning (lines 799–804) is the correct behaviour —
it alerts the user but does not modify the model.
