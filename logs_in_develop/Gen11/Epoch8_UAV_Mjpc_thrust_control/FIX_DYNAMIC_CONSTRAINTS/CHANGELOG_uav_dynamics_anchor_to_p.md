# [DC_FIX] Changelog: UAV dynamics constraint — restore both channels + deprecate anchor_to_p (cond_on_p)

**File:** `FM_v3_uav_test/eval_fm_uav.py`
**Lines:** 206–214 (constraint block) + 414–415 (rollout comment) + 177–188 (docstring)

## Bug
`anchor_to_p` if/else forced an either/or choice: only one of the two real position channels was ever constrained. One channel always floated free.

- `anchor_to_p=True`  → only `p(6,7,8)` constrained; `p_des(3,4,5)` floats
- `anchor_to_p=False` → only `p_des(3,4,5)` constrained; `p(6,7,8)` floats

## Root Cause of anchor_to_p
`anchor_to_p` (cond_on_p mode) was introduced as a workaround precisely because the constraint was broken. The intent was to "ground" the projector to the real drone position by choosing `p` over `p_des`. But the real fix is to anchor both — the if/else was treating a symptom of the missing rows, not the cause.

## Fix

### Constraint block
Replaced if/else with always-6 rows:
```python
# BEFORE (3 rows, either/or)
if anchor_to_p:
    constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]
else:
    constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]

# AFTER (6 rows always — both channels anchored)
constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]  # p_des ← act
constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]  # p     ← act
```

### anchor_to_p / cond_on_p — DEPRECATED for constraint selection
`anchor_to_p` is kept in the rollout integration path only (`p_des = p + action` vs `p_des += action`). It no longer controls which constraint rows are built. Comments added at:
- `setup_dpcc_projector` docstring
- constraint block
- rollout integration line

## Traj layout
`[act(0,1,2) | p_des(3,4,5) | p(6,7,8) | v(9,10,11)]`
- `p_des(3,4,5)`: commanded setpoint
- `p(6,7,8)`: actual drone position from `data.qpos[:3]`
Both real. Both must be anchored.
