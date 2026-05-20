# Gen6V4 — GEN7 Fix 3 Applied

**Date:** 2026-05-20
**Branch:** update_into_FM
**Primary record:** `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_3/FIX3_CHANGELOG.md`

---

## Summary

Fix 3 was designed for Gen7 diagnostics but applies equally to Gen6V4 because both
models share `VisualAgentWrapper` (eval agent) and `Aligning_Sim` (sim loop).
All 6 issues were applied to `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`
at the same time as the Gen7 file.

---

## Changes applied to Gen6V4 eval

### `d3il/simulation/aligning_sim.py`
- Added `agent.record_step_info(info)` hook after every `env.step()` call (both
  visual and non-visual loops). Hook is guarded by `hasattr` — zero impact on
  agents that don't implement it.

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

| Change | What was added |
|---|---|
| `record_step_info()` | Accumulates `mean_distance` per step from `env.step()` info dict |
| `curr_rollout_dist_to_target` | Per-rollout distance-to-box time series |
| `curr_rollout_act_magnitudes` | Per-rollout action magnitude time series |
| PNG `[0,3]` panel | Distance to Target over Steps (replaces empty 4th slot in expanded grid) |
| PNG `[1,2]` panel | Action Magnitude per Step — replaces nothing, grid expanded 2×3 → 2×4 |
| PNG `[1,3]` panel | End-effector velocity (moved from `[1,2]`) |
| GIF step counter | `s{N}` yellow text overlay — confirms left-panel-frozen is scene-static not render bug |
| `[ DIAG obs ]` | des_c_pos / c_pos / obs_6d_norm logged at first replan |
| `[ DIAG img ]` | bp and inhand image std + shape logged at first replan; WARNING if std < 0.01 |
| `[ DIAG replan=N ]` | One line every 50 replans: norm\|a0\|, denorm\|a0\|, direction unit vector |

---

## Why this matters for Gen6V4 retrospectively

The Gen6V4 run that prompted these changes showed:
- Healthy first-replan DIAG (normalized a0 magnitude 0.24, physical step 0.22 mm)
- Mode 0 at episode end (robot reached the box)
- But Success: False, Final Mean Distance: 0.091 m

The new Distance-to-Target panel (`[0,3]`) will immediately show whether the
policy was making steady progress toward the box or oscillating in place.
The Action Magnitude panel (`[1,2]`) will show whether ~0.22 mm/step was
consistent throughout or collapsed mid-episode.

These two panels are the minimum needed to distinguish
"policy is working but slow" from "policy reached the box and got stuck".

---

## No retraining required

These are eval-script and sim-loop changes only. No model weights, config values,
or training code were modified.
