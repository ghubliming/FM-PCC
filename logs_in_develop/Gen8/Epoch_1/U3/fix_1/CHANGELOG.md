# U3 Fix 1 — Train-script: missing imfv2 params (preemptive mirror of Gen3v4 U4 Fix 1)

**Date:** 2026-06-14
**Triggered by:** Gen3v4 U4 Fix 1 crash ([CHANGELOG](../../../../Gen3v4_imf/U4/fix_1/CHANGELOG.md))
**Status:** Fixed preemptively (no crash observed — Gen8 would have failed silently).

---

## What was wrong

`imf_visual_aligning_test/train_imf_visual_aligning.py` did not forward the 5 U3 imfv2 params to
`diffusion_config`. Unlike Gen3v4, it never hard-accessed `args.ode_inference_steps_v3`, so there
would have been **no crash** — but `'imf_objective': 'meanflow_jvp'` in the config would have been
**silently ignored**: the model would have fallen back to `fm_equivalent` and trained the wrong
objective with no warning.

---

## Fix

```python
# ADDED to diffusion_config block in train_imf_visual_aligning.py:
imf_objective=getattr(args, 'imf_objective', 'fm_equivalent'),
meanflow_r_equals_t_frac=getattr(args, 'meanflow_r_equals_t_frac', 0.25),
meanflow_adaptive_p=getattr(args, 'meanflow_adaptive_p', 0.5),
meanflow_adaptive_c=getattr(args, 'meanflow_adaptive_c', 1e-3),
meanflow_aux_weight=getattr(args, 'meanflow_aux_weight', 0.0),
```

All `getattr` with safe defaults — zero risk to existing `fm_equivalent` runs.

---

## File changed

| File | Change |
|------|--------|
| `imf_visual_aligning_test/train_imf_visual_aligning.py` | Lines added after `loss_schedule`: forward 5 imfv2 params to `diffusion_config`. |

---

## Difference from Gen3v4 Fix 1

Gen3v4 had two bugs (crash + silent). Gen8 only had the silent one — its `diffusion_config` block
already used `getattr` throughout and never hard-accessed `ode_inference_steps_v3`.
