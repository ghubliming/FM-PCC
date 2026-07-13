# CONFIG-OVERRIDES-PKL — eval config precedence fix across all live generations

**Date:** 2026-07-13 · **Scope:** repo-wide (all `working on` generations per MASTER_TEST_HISTORY Master Trace Map)
**Trigger:** Gen3v4_imf U9 kill-table investigation (`logs_in_develop/Gen3v4_imf/U9/debug_notes/INVESTIGATION_new_vs_upstreams_KILL_TABLE.md`) — the eval-time CFG poisoning could not be disabled from the config because pickled train-time values silently won at eval.

---

## The bug (was repo-wide, by design-drift)

Every eval reconstructs the diffusion wrapper from the checkpoint's pickled `diffusion_config.pkl` and instantiates it with the **pickled kwargs verbatim**. Editing the `.py` config plan block changed **nothing** except the output folder name. The repo *knew*: Gen7's own "Fix 5" comment ("Without this, the checkpoint's baked-in value is always used"), Gen7 "D1" (`clip_denoised`), and Gen9's `_warn_pkl_config_mismatch` banner ("[ eval pkl values ] (these win over the .py plan config) … *** MISMATCH — patch pkl or retrain ***") — each generation patched one victim key at a time instead of fixing the precedence.

## The fix (uniform semantics)

In every live eval loader, **immediately before `diffusion = diffusion_config(model)`**:

> Every pickled diffusion kwarg whose name also exists in the parsed config args is **overwritten by the config value**, and a console line
> `[WARNING] config-overrides-pkl: '<key>': <pkl> (pkl) -> <config> (config)`
> is printed for every value that actually changes. Keys equal in both stay silent. Config keys that are not diffusion kwargs (policy, batch_size, …) are ignored.

The block is copied per-generation (copy-modify sibling convention — no shared-code refactor). Nested loaders take `args` from the closure; top-level loaders got a new optional `override_args=None` parameter (backward-compatible) passed at the call site.

## Files touched (11)

| # | Gen | File | Change |
|---|-----|------|--------|
| 1 | Gen3v4 (iMF) | `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` | merge block in nested `load_diffusion_with_override` (before instantiation) |
| 2 | Gen3v4 (iMF) | `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | same |
| 3 | Gen3v3 (Drifting) | `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | same |
| 4 | Gen7 (FM Visual) | `fm_visual_aligning_test/eval_fm_visual_aligning.py` | `override_args` param + merge block + call site; note added that this generalizes Fix 5 / D1 |
| 5 | Gen6V4 (DPCC Visual) | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | `override_args` param + merge block + call site |
| 6 | Gen8 (iMF Visual) | `imf_visual_aligning_test/eval_imf_visual_aligning.py` | `override_args` param + merge block + call site |
| 7 | Gen9 (FM Avoiding) | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | `override_args` param + merge block + call site; `_warn_pkl_config_mismatch` banner text updated (old text claimed pkl wins) |
| 8 | Gen9 (DPCC Avoiding) | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | same as #7 |
| 9 | Gen11 (UAV) | `flow_matcher_v3_uav/utils/serialization.py` | `override_args` param + merge block in `load_diffusion` (UAV eval loads via utils, not an inline loader) |
| 10 | Gen11 (UAV) | `FM_v3_uav_test/eval_fm_uav.py` | call site passes `override_args=args` |
| 11 | Gen3v4 config | `config/avoiding-d3il.py` → `plan_fm_v3_imeanflow` | kill-switch values: `meanflow_cfg_omega 4.0→0.0`, `t_min→0.0`, `t_max→1.0`, **new** `condition_guidance_w: 0.0`, **new** `returns_condition: False` (keys must exist in the plan block for the override to reach them) |

Old post-load stamps (imf eval solver lines, Gen7 Fix 5/D1) are left in place — after the merge they are consistent no-ops (belt-and-suspenders).

**Not touched (finished/legacy/archived):** Gen0 `scripts/eval.py`, `FM_test`, `FM_v2_test`, `FM_Unet_v2_test`, `FM_v3_test`, `FM_v3_ode_selectable_test` (finished), `ddpm_encdec_vision_test`, `fm_encdec_vision_test`, `diffuser_test`, both `(legacy_based_on_visual_aligning)` folders, `Archived_Codes/`. Extend the same block there if any is revived.

## Verification

- All 11 files pass `python3 -m py_compile` locally.
- The merge block was behavior-simulated with a mock of the failing bbdit checkpoint's pickled dict + today's plan args: gates close, equal keys silent, non-diffusion plan keys ignored (scratchpad `merge_sim.py`, output in session log).
- **Runtime validation must happen on cluster** (no torch/env here).

## ⚠️ Known limits & behavior changes to watch on first re-runs

1. **Name-aliased keys are NOT covered**: DPCC's plan key `n_diffusion_steps` maps to pickled kwarg `n_timesteps` — different names, so the pkl still wins there (deliberate: changing the diffusion step count vs a trained schedule breaks buffer shapes / checkpoint loading). Gen9's legacy banner still surfaces those.
2. **Architecture-shaped kwargs**: if a plan block ever carries a same-named architecture kwarg (`horizon`, …) that differs from training, the override applies and `trainer.load()` will fail with a shape mismatch — the `[WARNING] config-overrides-pkl` line printed right before is the diagnosis. Keep plan blocks matched to checkpoints, as before.
3. **Pre-existing train/plan divergences now become live overrides.** Scan flagged in `config/avoiding-d3il-visual.py` (Gen9): `clip_denoised` True(train)/False(plan) and `action_weight` 10/1 — on the next Gen9 eval these will warn and apply the plan value. `clip_denoised=False` at eval matches the D1/DPCC-reference convention (likely intended); `action_weight` only affects the loss (unused at eval; the buffer is restored from the checkpoint). **Check the first eval log of every generation for unexpected `[WARNING] config-overrides-pkl` lines.**
4. `model_config` / `dataset_config` / `trainer_config` remain pkl-authoritative (architecture & data must match the checkpoint).

## What to do next (the U9 test)

1. Sync to cluster; make sure `plan_fm_v3_imeanflow` has `imf_backbone: 'dit'` (+ matching `dit_*`) for the failing bbdit checkpoint.
2. Run the imf eval. Expect exactly these console lines:
   `condition_guidance_w: 1.2 -> 0.0`, `returns_condition: True -> False`, `meanflow_cfg_omega: 4.0 -> 0.0`, `meanflow_cfg_t_min: 0.4 -> 0.0`, `meanflow_cfg_t_max: 0.6 -> 1.0`.
3. `diffuser` variant bounded → kill chain confirmed (checkpoint was healthy). Still exploding → gates exonerated; next suspects per kill table: 1-NFE reconstruction check, `eval_use_ema: True`.
