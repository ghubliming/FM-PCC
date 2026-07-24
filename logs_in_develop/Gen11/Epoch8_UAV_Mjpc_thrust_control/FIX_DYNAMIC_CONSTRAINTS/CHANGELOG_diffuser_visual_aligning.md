# [DC_FIX] Changelog: diffuser_visual_aligning (DPCC baseline) dynamics constraint — missing des channel

**File:** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`
**Lines:** 121–124 → 121–131

## Bug
Same pattern: only `c_pos(6,7,8)` constrained, `des_c_pos(3,4,5)` unanchored.
C4 fix is present in this file's predict() — both channels are real.

## Fix
Added 3 rows for `des_c_pos(3,4,5)` → 6 rows total.

```python
# AFTER (6 rows — both anchored)
constraint_list.append(('deriv', [3, 0]))
constraint_list.append(('deriv', [4, 1]))
constraint_list.append(('deriv', [5, 2]))
constraint_list.append(('deriv', [6, 0]))
constraint_list.append(('deriv', [7, 1]))
constraint_list.append(('deriv', [8, 2]))
```
