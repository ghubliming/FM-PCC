# Gen7 Fix 8 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Source MD Reference**: [upgrade_8/CHANGELOG_FIX8.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/upgrade_8/CHANGELOG_FIX8.md) · [upgrade_8/PLAN_FIX8_MPC_RECOVERY.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/upgrade_8/PLAN_FIX8_MPC_RECOVERY.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

## Summary

Fix 8 recovers the full DPCC MPC inference logic that was broken by commit `7b14333`. All six changes below apply to `eval_visual_aligning_dpcc.py` symmetrically with `eval_fm_visual_aligning.py`.

## P1 — Remove magic `batch_size = 6`
Removed hardcoded `batch_size = 6` override that killed the config-driven candidate pool. `diffuser` variant now uses batch=1; all projected variants use `getattr(args, 'batch_size', 4)`.

## P2 — Restore trajectory-selection — exact DPCC logic (corrected)
Exact `dpcc/scripts/eval.py` logic: `trajectory_selection = 'random'`; `if 'dpcc-t' in variant → temporal_consistency`; `if 'dpcc-c' in variant → minimum_projection_cost`. `'random'` = index 0 (deterministic). Initial Fix 8 used wrong generic `-c`/`-t` substring matching — corrected.

## P3 — Fix `'random'` non-determinism
`elif trajectory_selection == 'random': np.random.randint(B)` replaced with `else: which = 0`. Matches DPCC reference where `'random'` always resolves to index 0 (deterministic).

## P4 — Full candidate storage
New `curr_rollout_all_candidates` / `curr_rollout_selected_idx` accumulators (cleared on `reset()`) capture `(B, H, 3)` action trajectories and winning index at every replan step. Propagated through `update_rollout_info()` into `master_rollout_history`.

## P5/P6 — MPC Foresight PNG (per-rollout + aggregate)
Both PNG sites now read `all_candidates` and `selected_idx` from rollout history. Non-selected candidates rendered as thin lightblue lines; selected candidate as bold royalblue. Title shows candidate count per step.

## Config — `projection_variants` corrected to exact DPCC reference
`config/visual_aligning_eval.yaml` now contains the exact 17-variant list from `dpcc/config/projection_eval.yaml`: `dpcc-r/c/t` + tightened (6), `diffuser`/`gradient`/`post_processing`/`model_free` baselines + tightened (7), `dpcc-c-tightened-dt*` ablations (4). Removed custom-invented `-c`/`-t` baseline suffixes (initial Fix 8 error).

## D1 — `clip_denoised` made config-driven
`eval_visual_aligning_dpcc.py`: hardcoded `= False` replaced with `getattr(args, 'clip_denoised', False)`. `plan_visual_aligning_dpcc` in config now has `clip_denoised: False` (default; ablate-only). DPCC-only.

## D4/B1 — Initial-state row coefficient reverted to original
`diffuser_visual_aligning/sampling/projection.py`: `[DANGEROUS_FLAG_B1_SCALING]` applied at `build_matrices`, `project`, `compute_gradient`. Reverted to `mat_fix_initial[0, x_idx] = 1` and `b[...] = s_0[x_idx]`. Upgraded code preserved in comments with upgrade rationale and original code comment-block above the live code.

## D7/A4 — Per-sample anchor reverted to original
`diffuser_visual_aligning/sampling/projection.py`: `[DANGEROUS_FLAG_A4_PER_SAMPLE_ANCHOR]` applied at `project` and `compute_gradient`. Reverted to `s_0 = trajectory_reshaped[0, ...]` outside the batch loop. Same comment structure: upgrade reason, upgraded code block, original code comment-block, live original code.
