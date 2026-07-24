# [DC_FIX] Master Changelog: Dynamics Constraint Bug Fix

**Date:** 2026-07-01
**Flag:** `DC_FIX` — grep to retrieve all touched locations: `grep -rn "DC_FIX" .`
**Changelogs:** `logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/FIX_DYNAMIC_CONSTRAINTS/`

---

## Summary

All FM-PCC eval ports were missing half of the required dynamics constraint rows.
DPCC avoiding (ground truth) constrains **both** `p` and `p_des` with the same action — 4 rows for 2D, 6 rows for 3D.
Every ported eval script constrained only 1 of the 2 real channels, leaving the other free to hallucinate.

---

## Principle

| Task dims | Real channels | Required rows | Bug state |
|-----------|---------------|---------------|-----------|
| 2D, 2 ch  | `p` + `p_des` | 4             | DPCC baseline (correct) |
| 3D, 2 ch  | `c_pos` + `des_c_pos` | 6   | all ports had 3 |
| 3D UAV, 2 ch | `p` + `p_des` | 6          | had 3 (either/or) |

---

## Fixes Applied

### `fm_visual_aligning_test/eval_fm_visual_aligning.py`
- Added `des_c_pos(3,4,5)` rows alongside existing `c_pos(6,7,8)` rows → 6 total
- [Changelog](../../../../../Epoch8_UAV_Mjpc_thrust_control/FIX_DYNAMIC_CONSTRAINTS/CHANGELOG_fm_visual_aligning.md)

### `imf_visual_aligning_test/eval_imf_visual_aligning.py`
- Same change
- [Changelog](../../../../../Epoch8_UAV_Mjpc_thrust_control/FIX_DYNAMIC_CONSTRAINTS/CHANGELOG_imf_visual_aligning.md)

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`
- Same change
- [Changelog](../../../../../Epoch8_UAV_Mjpc_thrust_control/FIX_DYNAMIC_CONSTRAINTS/CHANGELOG_diffuser_visual_aligning.md)

### `FM_v3_uav_test/eval_fm_uav.py`
- Replaced `if anchor_to_p / else` with always-6 rows (both `p_des(3,4,5)` and `p(6,7,8)`)
- `anchor_to_p` (cond_on_p) **DEPRECATED** for constraint selection — was a workaround for this bug
- `anchor_to_p` retained only for rollout integration behavior (`p_des = p + action` vs `p_des += action`)
- [Changelog](../../../../../Epoch8_UAV_Mjpc_thrust_control/FIX_DYNAMIC_CONSTRAINTS/CHANGELOG_uav_dynamics_anchor_to_p.md)

---

## Not Changed (correct)

| Module | Why |
|--------|-----|
| `flow_matcher_v3_uav/utils/constraints_helpers.py` | DPCC ground truth — 4/6 rows correct |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | calls `formulate_dynamics_constraints` — correct |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | same |
| `fm_encdec_vision_test/eval_fm_encdec_vision.py` | 6D traj, 1 real channel, 3 rows correct |
| `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | same |

---

## anchor_to_p / cond_on_p Deprecation Note

`anchor_to_p=True` (cond_on_p mode) was introduced in Gen11 Fix 5 to ground the projector to the real drone position instead of drifted `p_des`. The mechanism was: choose `p(6,7,8)` rows instead of `p_des(3,4,5)` rows. This choice was only necessary because the full 6-row constraint was never built — the projector had to pick one channel or the other.

With both channels now always anchored, `anchor_to_p` has no role in constraint construction. Experiments that ran with `anchor_to_p=True` were effectively running with only `p` anchored and `p_des` floating — a different (and weaker) constraint than intended.
