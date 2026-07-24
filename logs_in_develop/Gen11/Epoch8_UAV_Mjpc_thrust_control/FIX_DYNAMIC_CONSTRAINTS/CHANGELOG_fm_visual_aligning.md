# [DC_FIX] Changelog: fm_visual_aligning dynamics constraint — missing des channel

**File:** `fm_visual_aligning_test/eval_fm_visual_aligning.py`
**Lines:** 125–128 → 125–135

## Bug
Only `c_pos(6,7,8)` was constrained (3 rows). `des_c_pos(3,4,5)` was left unanchored, allowing the projector to hallucinate the initial commanded position.

Traj layout: `[act(0,1,2) | des_c_pos(3,4,5) | real c_pos(6,7,8)]`
Both channels are real (C4 fix already in predict()).

## Fix
Added 3 rows for `des_c_pos(3,4,5)` before the existing `c_pos(6,7,8)` rows → 6 rows total.

```python
# BEFORE (3 rows — des floats)
constraint_list.append(('deriv', [6, 0]))
constraint_list.append(('deriv', [7, 1]))
constraint_list.append(('deriv', [8, 2]))

# AFTER (6 rows — both anchored)
constraint_list.append(('deriv', [3, 0]))
constraint_list.append(('deriv', [4, 1]))
constraint_list.append(('deriv', [5, 2]))
constraint_list.append(('deriv', [6, 0]))
constraint_list.append(('deriv', [7, 1]))
constraint_list.append(('deriv', [8, 2]))
```

## Principle
DPCC avoiding (2D) uses 4 rows for 2 real channels. 3D with 2 real channels requires 6.
