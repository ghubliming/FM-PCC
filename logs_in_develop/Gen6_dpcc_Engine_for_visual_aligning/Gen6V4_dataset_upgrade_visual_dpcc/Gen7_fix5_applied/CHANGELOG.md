# Gen7 Fix 5 — Applicability to Gen6V4

**Date:** 2026-05-20
**Branch:** update_into_FM
**Primary record:** `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/fix_5/FIX5_CHANGELOG.md`

---

## Summary

Fix 5 corrects a bug where `--flow_steps_v3 N` Slurm overrides changed the output
directory name but not the actual ODE step count used at inference.

**Fix 5 was applied to Gen7 only.**
**Fix 5 is NOT applicable to Gen6V4.**

---

## Why Gen6V4 is unaffected

Gen6V4 uses a DDPM reverse diffusion chain, not a Flow Matching ODE. The equivalent
attribute is `n_timesteps`, but it cannot be safely overridden at inference time:

| Model | Step-count attribute | Override safe? | Reason |
|---|---|---|---|
| Gen7 FM | `flow_steps_v3` | **Yes** | Controls only Euler `dt = 1/N`; no tied buffers |
| Gen6V4 DDPM | `n_timesteps` | **No** | Baked into `betas`, `alphas_cumprod`, and all derived noise-schedule buffers at checkpoint creation |

Overriding `diffusion_model.n_timesteps` at inference without recomputing all those
buffers would either corrupt the denoising chain (wrong beta values) or trigger an
out-of-bounds index if the new value exceeds the buffer length.

Gen6V4's eval script already logs a warning when `n_timesteps` (checkpoint) differs
from `n_diffusion_steps` (config args). That warning-only behaviour is correct and
was not changed.

---

## No code changes in Gen6V4

`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — **unchanged**.
