# [DC_FIX] Changelog: imf_visual_aligning dynamics constraint — missing des channel

**File:** `imf_visual_aligning_test/eval_imf_visual_aligning.py`
**Lines:** 125–128 → 125–135

## Bug
Identical to fix1: only `c_pos(6,7,8)` constrained, `des_c_pos(3,4,5)` unanchored.

## Fix
Same as fix1 — added 3 rows for `des_c_pos(3,4,5)` → 6 rows total.

```python
# AFTER (6 rows — both anchored)
constraint_list.append(('deriv', [3, 0]))
constraint_list.append(('deriv', [4, 1]))
constraint_list.append(('deriv', [5, 2]))
constraint_list.append(('deriv', [6, 0]))
constraint_list.append(('deriv', [7, 1]))
constraint_list.append(('deriv', [8, 2]))
```
