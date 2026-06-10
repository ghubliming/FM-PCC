# Gen11 E4 U6 — Changelog

**Date:** 2026-06-10  
**Branch:** update_into_FM  
**Implements:** Gen11E4U6_Plan_Fable.md (C1 + C2 + C3; C4 and C5 are run-strategy, no code)

---

## C1 — Revert s_curve diagonal to x=±0.5 (`trajectories.py`)

**Problem (U5 Step 5 bug):** Moving hover/diagonal endpoints from x=±0.5 to x=∓0.7
put the first 0.2 m of Seg B inside the walled section of corridor 1
(`seg1_wall_pos` occupies x ∈ [−3.0, −0.5]). The diagonal y-movement inside the
walled section brought the rotor within contact range of `seg1_wall_pos`:
first contact at x = −0.577 m (clearance = 0.31 m = rotor_reach); by x = −0.5
(wall exit), clearance was only 0.221 m. This caused 8 direct contact rejects and
13 knock-down floor crashes in F5 — identical 100% rejection rate to F4 but from a
completely different geometric mechanism.

**Change (`uav_expert_data_collect/trajectories.py`, `s_curve_scene_path`):**

Hover and diagonal endpoints fully reverted to x = ±0.5:

| Phase | U5 (wrong) | U6 (restored) |
|-------|-----------|---------------|
| Seg A end | (−0.7, y1, z) | (−0.5, y1, z) |
| Hover 1 | (−0.7, y1, z) | (−0.5, y1, z) |
| Seg B | (−0.7,y1) → (+0.7,y2) | (−0.5,y1) → (+0.5,y2) |
| Hover 2 | (+0.7, y2, z) | (+0.5, y2, z) |
| Seg C start | (+0.7, y2, z) | (+0.5, y2, z) |
| d_a / d_c | 2.5 m | **2.7 m** |
| d_b x-span | 1.4 m | **1.0 m** |

Rationale for full revert (not bridge-segment design): F4 ran hovers at x=±0.5 with
zero contact rejects in that position — the "wall end-face risk" never materialized.
The diagnostic is: the diagonal must start at the gap boundary (x=−0.5), where walls
end, so no y-movement occurs inside the walled section.

**Kept from U5:** Seg B 2× time weight (`d_total = d_a + 2.0*d_b + d_c`) which halves
peak lateral velocity. The d_b value is updated to reflect the shorter 1.0 m x-span
(≈ 1.89 m diagonal vs 2.13 m in U5).

---

## C2 — Exact-scale thrust-priority allocation (`flight_controller.py`)

**Problem (U5 Step 4 bug):** The binary search (halve scale 10 times) over-cut torque
authority by up to 2×. Example: if the exact feasible scale is 0.49, the binary search
converges to 0.25 — giving the attitude loop only half the authority it could have had.
F4 homotopy table shows cross-channel homotopies (L,R,L) and (R,L,R) were healthy
(~60–67% success) before U5, then collapsed to ~5–10% — the only change affecting
them was the binary search removing attitude authority on the lateral diagonals.

**Changes (`uav_env_test/flight_controller.py`, `CascadedPID`):**

`__init__`: two new fields initialised:
```python
self.last_raw_saturated = False   # True if raw M_inv @ wrench exceeded bounds
self.last_torque_scale  = 1.0     # actual torque scale applied (1.0 if no saturation)
```

`compute`: replaced binary search with analytic exact scale + torque floor:
```python
# Before (U5 binary search — over-cuts)
scale = 1.0
for _ in range(10):
    u_try = thrust_cmd + scale * torque_comp
    if u_try.max() <= u_max and u_try.min() >= u_min: break
    scale *= 0.5

# After (U6 exact scale + floor)
caps = [(u_max - thrust_cmd)/tc  for tc in torque_comp if tc > 1e-9] + \
       [(thrust_cmd - u_min)/(-tc) for tc in torque_comp if tc < -1e-9]
scale = max(min(1.0, min(caps)) if caps else 1.0, 0.5)
```

- **Exact scale**: `min(caps)` is the largest multiplier that keeps all motors in
  [u_min, u_max]. No under-shooting.
- **Torque floor 0.5**: `max(scale, 0.5)` — attitude authority is never reduced
  below 50%. If the exact scale is < 0.5, accept slight thrust corruption (the final
  `np.clip`) rather than zeroing attitude control. F4 proved partial-torque clipping
  is survivable; U5 proved near-zero torque is not.
- `self.last_raw_saturated` and `self.last_torque_scale` written on every call so
  `generator.py` can count true saturation events (see C3).

---

## C3 — Fix saturation telemetry (`generator.py`)

**Problem (U5 Step 1 flaw):** `n_clip` checked whether the *output* `u` touched
`u_max` or `u_min`. With U5's thrust-priority allocation, the output was held strictly
inside bounds by design — so `clip=0.0%` appeared even when saturation had occurred
and torques were being scaled down. This made the U5 diagnostic ("saturation
eliminated") unverifiable.

**Change (`uav_expert_data_collect/generator.py`, `run_trial` inner loop):**

```python
# Before (U5 — output-boundary check)
n_clip += int(np.any(u >= pid.u_max - 1e-6) or np.any(u <= pid.u_min + 1e-6))

# After (U6 — pre-allocation raw demand flag)
n_clip += int(pid.last_raw_saturated)
```

`pid.last_raw_saturated` is set by `CascadedPID.compute` before any allocation
adjustment — it is `True` whenever `M_inv @ wrench` would have exceeded motor bounds.
`clip=X%` in reject lines now means "X% of steps required saturation handling",
regardless of how the controller resolved it.

---

## C4 — Pillars run strategy (no code change)

After C2 restores attitude authority, run pillars with all 4 homotopies. Expected:
cross-channel homotopies (L,R,L) and (R,L,R) return to F4 baseline (~60–67% success).
Decision rule:
- Overall < 30% rejection on 20-trial smoke → run full 500 trials.
- Cross-channel still > 50% rejection → restrict main run to (L,L,L)/(R,R,R) and
  collect cross-channel in separate dedicated 300-trial runs.

## C5 — Kept unchanged

- z range [0.90, 1.30] (Step 2)
- Reject histogram and per-reject logging (Step 1, now with correct clip telemetry)
- empty / corridor config untouched

---

## Files changed

| File | Change |
|------|--------|
| `uav_expert_data_collect/trajectories.py` | C1: revert diagonal to x=±0.5, update d_a/d_b/d_c |
| `uav_env_test/flight_controller.py` | C2: exact-scale allocation, torque floor 0.5, saturation fields |
| `uav_expert_data_collect/generator.py` | C3: `n_clip` uses `pid.last_raw_saturated` |

---

## Validation plan

```bash
# s_curve smoke (expect < 30%)
python uav_expert_data_collect/collect.py --scene s_curve --n-trials 20 --seed 100

# pillars smoke (expect < 30% overall, no homotopy > 50%)
python uav_expert_data_collect/collect.py --scene pillars --n-trials 20 --seed 100

# then full 500 each
```

Check: `clip%` in reject lines should now be non-zero for genuine saturation events
(especially pillars cross-channel diagonals). If s_curve still > 30%: read histogram
before changing anything. If pillars cross-channel still > 50%: follow C4 split-run.
