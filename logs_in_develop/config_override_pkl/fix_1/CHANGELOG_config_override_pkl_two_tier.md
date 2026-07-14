# CHANGELOG — config-override-pkl **fix_1**: two-tier reconcile (INFO vs WARNING)

**Date:** 2026-07-14 · **Scope:** all 9 live eval loaders + Gen3v4 imf plan config
**Follows:** `../CHANGELOG_config_overrides_pkl.md` (the original "config overrides pkl" mechanism).

## Problem (design intent vs behavior)

The mechanism's intent: **preserve training-time params (pkl), compare the eval config against them, override where it makes sense, and message the user.** But v1 printed `[WARNING]` for **every** key that differed — and train-block vs plan-block differ **by design** (train = how it was trained; plan = how to sample). So on a straight train→eval pipeline with **zero** user changes, warnings fired on every run → the signal was pure noise and couldn't flag a *real* problem (wrong checkpoint / architecture mismatch).

Also clarified (user's worry): the override is **real**, not cosmetic — it mutates `diffusion_config._dict[k]` *before* construction, and the model uses the eval value. Proof in the U10 eval log: pkl `flow_steps_v3=10` but the eval folder is `H8_K2` and weights are EMA — the config values won. (The "pkl silently wins / zombie" behavior was the *pre-fix* repo default — exactly the bug behind the un-disableable CFG explosion — and is gone.)

## Fix — two tiers, by key role

In every loader, before instantiation:

- **SAMPLING knobs** (operating point, safe to change at eval) → eval config **OVERRIDES** the pkl, logged as **`[ config->pkl ] INFO`**. Allowlist:
  `flow_steps_v3, ode_inference_steps_v3, ode_solver_{backend,method,rtol,atol,step_size}_v3, meanflow_cfg_omega, meanflow_cfg_t_min, meanflow_cfg_t_max, condition_guidance_w, clip_denoised, diffusion_timestep_threshold`.
- **Identity / architecture keys** (must match the checkpoint) → the **pkl value is KEPT** (protects the `state_dict`), and a loud **`[ config->pkl ] WARNING`** fires only if the eval config disagrees. This is now the *real* "something's wrong — wrong checkpoint or config" signal (e.g. `imf_objective`, `dual_head`, `interval_cfg`, `imf_backbone`, `horizon`, DiT dims, `p_mean/p_std`, `returns_condition`).

Net: a clean pipeline prints a few **INFO** lines (expected operating-point changes) and **no WARNING**. A WARNING now means a genuine train↔eval identity divergence worth stopping for.

## Files changed (10)

| File | Change |
|---|---|
| `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | merge block → two-tier |
| `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | " |
| `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | " |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | " (top-level `override_args`) |
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | " |
| `imf_visual_aligning_test/eval_imf_visual_aligning.py` | " |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | " |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | " |
| `flow_matcher_v3_uav/utils/serialization.py` (`load_diffusion`) | " |
| `config/avoiding-d3il.py` (`plan_fm_v3_imeanflow`) | `returns_condition: False → True` to MATCH the pkl (it's an identity key now, and inert — neutralized by `condition_guidance_w=0`), so it no longer false-warns every eval |

## Why keep-pkl for identity keys (not override)
Overriding an architecture key (e.g. `returns_condition` for DPCC builds a returns MLP; `dual_head`/dims change tensor shapes) would build a model the checkpoint weights can't load → crash or silent wrongness. Keeping the pkl value is the safe default; the WARNING tells the user to fix the config to match the checkpoint (or retrain). Sampling knobs never touch the `state_dict`, so overriding them is safe and is the whole point.

## Verification
- ✅ `py_compile` on all 10 files.
- ✅ Behavior sim against this run's pkl vs plan config: 5 sampling knobs → INFO+applied; matched identity keys (`imf_objective`, `horizon`, `dual_head`, …) → silent; a deliberately-mismatched identity key → WARNING + pkl kept. Result model config had `omega=1.0, flow_steps=2, condition_guidance_w=0` (eval) and identity keys from the checkpoint.
- ⏳ Cluster: re-run the imf eval — expect only the 5 `INFO` lines above, **no WARNING**.

## Notes
- Legacy per-key post-load stamps (Gen7 Fix5 `flow_steps_v3`, D1 `clip_denoised`, the imf solver stamps) are left in place — all are sampling keys, so they now agree with the tier-1 override (redundant, harmless).
- Gen9 `_warn_pkl_config_mismatch` banner still covers the name-aliased `n_diffusion_steps ↔ n_timesteps` case (not a same-named key, so outside this loop).
- Uncommitted (user commits manually).
