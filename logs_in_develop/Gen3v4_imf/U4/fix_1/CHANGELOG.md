# U4 Fix 1 — Train-script crash: `ode_inference_steps_v3` + missing imfv2 params

**Date:** 2026-06-14
**Slurm job:** 21545 (`train_imf`, node `i6-gpu-1`)
**Log:** `temp/new_IMF/slurm log`
**Status:** Fixed, syntax-clean.

---

## Error observed

```
AttributeError: 'Namespace' object has no attribute 'ode_inference_steps_v3'
  File "FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py", line 184
```

Job died at 14:04:57 — 5 seconds after start, before any training step.

---

## Root cause (2 bugs in `train_flow_matching_v3_imeanflow.py`)

### Bug 1 — hard attribute access on a commented-out config key (line 184)
```python
# BEFORE (broken):
ode_inference_steps_v3=args.ode_inference_steps_v3,

# AFTER (fixed):
ode_inference_steps_v3=getattr(args, 'ode_inference_steps_v3', getattr(args, 'flow_steps_v3', 10)),
```
When `ode_inference_steps_v3` was commented out of the config (it is dead in training — see
[U4 CHANGELOG §ODE note](../CHANGELOG.md)), `args` no longer carried the key, so a bare attribute
access crashed immediately. `getattr` with the `flow_steps_v3` fallback is correct: the model
`__init__` already resolves `flow_steps_v3 → ode_inference_steps_v3` on line 73 anyway.

### Bug 2 — 5 imfv2 params never forwarded to `diffusion_config`
The `diffusion_config` block only passed the legacy params; the 5 U4 additions from
`iMeanFlowODE.__init__` were never wired through the training script:

```python
# ADDED (all safe-defaulted via getattr):
imf_objective=getattr(args, 'imf_objective', 'fm_equivalent'),
meanflow_r_equals_t_frac=getattr(args, 'meanflow_r_equals_t_frac', 0.25),
meanflow_adaptive_p=getattr(args, 'meanflow_adaptive_p', 0.5),
meanflow_adaptive_c=getattr(args, 'meanflow_adaptive_c', 1e-3),
meanflow_aux_weight=getattr(args, 'meanflow_aux_weight', 0.0),
```

Without Bug 2, setting `'imf_objective': 'meanflow_jvp'` in the config would have been silently
ignored — the model would have fallen back to `fm_equivalent` regardless. (Bug 1 crashed first,
masking Bug 2.)

---

## File changed

| File | Change |
|------|--------|
| `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` | Line 184: `getattr` for `ode_inference_steps_v3`. Lines 185–189 (new): forward 5 imfv2 params. |

---

## Good news from the log

- Folder naming works: `...aw10_objmeanflow_jvp/6` — the new `args_to_watch_fmv3_imf_train` with
  `('imf_objective', 'obj')` is resolving correctly.
- Dataset loaded, model config saved — everything before `diffusion_config()` is healthy.

---

## Next run

Re-submit `Slurm_Codes/sbatch/iMF/imf_pipeline.sh` (or `train_imf.sh` directly). The fix is
syntax-verified; no other changes needed. Still on Step 1 of
[U5/NEXT_STEPS.md](../../U5/NEXT_STEPS.md): watch for forward-AD/JVP errors once training
actually starts.
