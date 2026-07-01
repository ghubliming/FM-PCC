# Fix Plan: Dynamics Constraint Bug — Missing Channel

## Principle (DPCC ground truth)

Every **real** predicted position channel needs its own constraint rows.
- 2D, 2 channels → **4 rows**
- 3D, 2 channels → **6 rows**

DPCC avoiding (2D) constrains both `p(4,5)` and `p_des(2,3)` with the same velocity — 4 rows.
All FM-PCC ports have **only half** those rows. One channel always floats free.

---

## Obs Layouts (verified from code)

| Task | Traj dim | Layout | C4 fix |
|------|----------|--------|--------|
| Visual aligning (9D) | 9 | `[act(0,1,2) \| des_c_pos(3,4,5) \| real c_pos(6,7,8)]` | ✅ all 3 aligning files |
| Visual avoiding (6D) | 6 | `[act(0,1) \| des_pos(2,3) \| c_pos(4,5)]` | ❌ visual path only has des |
| UAV (9D/12D) | 9/12 | `[act(0,1,2) \| p_des(3,4,5) \| real p(6,7,8) \| (vel)]` | real p via qpos[:3] |
| encdec_vision (6D) | 6 | `[act(0,1,2) \| des(3,4,5)]` | N/A — 1 channel only |

---

## Defected Modules

### 1. `fm_visual_aligning_test/eval_fm_visual_aligning.py` — line 125–128
**Bug:** 3 rows on real `c_pos(6,7,8)` only. `des(3,4,5)` unanchored.
```python
# CURRENT
constraint_list.append(('deriv', [6, 0]))
constraint_list.append(('deriv', [7, 1]))
constraint_list.append(('deriv', [8, 2]))

# FIX — add des rows (6 rows total)
constraint_list.append(('deriv', [3, 0]))
constraint_list.append(('deriv', [4, 1]))
constraint_list.append(('deriv', [5, 2]))
constraint_list.append(('deriv', [6, 0]))
constraint_list.append(('deriv', [7, 1]))
constraint_list.append(('deriv', [8, 2]))
```

---

### 2. `imf_visual_aligning_test/eval_imf_visual_aligning.py` — line 125–128
**Same bug, same fix as above.** C4 fix already present. Add 3 rows for `des(3,4,5)`.

---

### 3. `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — line 121–124
**Same bug, same fix.** C4 fix already present. Add 3 rows for `des(3,4,5)`.

---

### 4. `FM_v3_uav_test/eval_fm_uav.py` — line 206–214
**Bug:** `anchor_to_p` if/else forces either/or — one channel always unanchored.
```python
# CURRENT (broken — one channel always missing)
if anchor_to_p:
    constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]
else:
    constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]

# FIX — always 6 rows, both channels
constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]  # p_des ← a
constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]  # p     ← a
```
`anchor_to_p` flag is kept for the **rollout integration only** (`p_des = p + action` vs `p_des += action`).

---

## Correct — Do Not Touch

| Module | Rows | Reason |
|--------|------|--------|
| `flow_matcher_v3_uav/utils/constraints_helpers.py` avoiding | 4 (2D×2) | DPCC baseline, ground truth |
| `flow_matcher_v3_uav/utils/constraints_helpers.py` 3D avoiding | 6 (3D×2) | correct |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | via helper | calls `formulate_dynamics_constraints` — correct |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | via helper | same |
| `fm_encdec_vision_test/eval_fm_encdec_vision.py` | 3 | 6D traj, 1 real channel only |
| `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py` | 3 | same |

Note: `fm_visual_avoiding_test (legacy_based_on_visual_aligning)/` and `diffuser_visual_avoiding_test (legacy_based_on_visual_aligning)/` have the 2-row bug but are legacy/inactive.

---

## Fix Order

1. **Aligning ×3** — `eval_fm_visual_aligning.py`, `eval_imf_visual_aligning.py`, `eval_visual_aligning_dpcc.py` — each is +3 lines, no infra changes
2. **UAV** — `eval_fm_uav.py` — replace if/else with always-6 rows
