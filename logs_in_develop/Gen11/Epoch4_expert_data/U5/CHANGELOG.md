# Gen11 E4 U5 — Changelog

**Date:** 2026-06-10  
**Branch:** update_into_FM  
**Implements:** U5/CODING_PLAN.md (all 5 steps)

---

## Step 1 — Instrument reject reasons (`generator.py`, `collect.py`)

**Problem:** both `contact_frac > limit` and `min_z < Z_FLOOR_MARGIN` returned bare `None`, indistinguishable in logs. Multiple debug cycles (U3→U4) wasted because we could not confirm which check fired.

**Changes:**

`uav_expert_data_collect/generator.py`:
- `run_trial` inner loop: added `n_clip` counter — increments when any motor in returned `u` is at `u_max` or `u_min` saturation limit.
- Contact-reject path: now returns `{'rejected': True, 'reason': 'contact', 'contact_frac': ..., 'min_z': ..., 'motor_clip_frac': ...}` instead of `None`.
- Floor-reject path: now returns `{'rejected': True, 'reason': 'floor', 'min_z': ..., 'contact_frac': ..., 'motor_clip_frac': ...}` instead of `None`.
- Success return: added `motor_clip_frac` field.
- Docstring updated.

`uav_expert_data_collect/collect.py`:
- Added `from collections import Counter`.
- `reject_counter = Counter()` tracks reason → count per run.
- Rejection check changed from `rollout is None` to `rollout is None or rollout.get('rejected')`.
- Each rejection prints: `[ collect ] REJECT #N  reason=floor  min_z=0.542  clip=12.3%`
- ABORT message now includes reject histogram: `HISTOGRAM: floor=18  contact=3`.
- DONE message now includes reject histogram.
- `run_summary.json` now includes `reject_histogram` field.

---

## Step 2 — Altitude headroom (`generator.py`)

**Problem:** s_curve start `z ~ U(0.70, 1.10)`. Observed floor-reject z-dip is 0.30–0.45 m. With z_start=0.70 m and `Z_FLOOR_MARGIN=0.50`, the drone reliably drops below the reject threshold.

**Change:**

`uav_expert_data_collect/generator.py` line 124:
```
z = float(rng.uniform(0.70, 1.10))   →   z = float(rng.uniform(0.90, 1.30))
```

+0.20 m floor headroom. Walls are 1.5 m tall — [0.90, 1.30] remains inside the corridor.  
Note: the `empty` scene's per-segment z still uses the old range (intentional — that scene has no floor-reject problem).

---

## Step 3 — Slow the diagonal (`trajectories.py`)

**Problem:** Seg B (diagonal gap crossing) received time proportional to its Euclidean length (~1.89 m), equal priority with the straight segments. Peak lateral velocity ≈ 1.41 m/s caused attitude-loop overshoot with 0.84 m lateral swing — the root-cause chain for motor saturation.

**Change:**

`uav_expert_data_collect/trajectories.py` `s_curve_scene_path`:
```python
# Before
d_total = d_a + d_b + d_c
t_b = T_move * d_b / d_total

# After
d_total = d_a + 2.0 * d_b + d_c
t_b = T_move * 2.0 * d_b / d_total
```

Seg B gets 2× the time proportionally. Peak lateral accel/velocity is roughly halved; overshoot scales quadratically so the saturation trigger is reduced ~4×.

---

## Step 4 — Thrust-priority allocation in CascadedPID (`flight_controller.py`)

**Problem:** `u = np.clip(u, u_min, u_max)` clipped ALL four motors independently. When attitude torques saturated any motor, the clip destroyed the collective thrust component → drone lost altitude → floor reject. This was the primary physical mechanism for all s_curve rejections (and a contributor to pillars end-of-episode speed spikes).

**Change:**

`uav_env_test/flight_controller.py` `CascadedPID.compute`:
```python
# Before
u = self.M_inv @ wrench
u = np.clip(u, self.u_min, self.u_max)

# After
u = self.M_inv @ wrench
if u.max() > self.u_max or u.min() < self.u_min:
    thrust_cmd  = u.mean()            # collective thrust preserved
    torque_comp = u - thrust_cmd      # torque offset scaled down
    scale = 1.0
    for _ in range(10):               # binary search: halve scale until within bounds
        u_try = thrust_cmd + scale * torque_comp
        if u_try.max() <= self.u_max and u_try.min() >= self.u_min:
            break
        scale *= 0.5
    u = np.clip(thrust_cmd + scale * torque_comp, self.u_min, self.u_max)
```

When torques saturate, the algorithm reduces torque authority uniformly while keeping `thrust_cmd` (mean of unsaturated solution) intact. The final `np.clip` is a safety net for the degenerate case where thrust itself exceeds limits.  
Side effect: resolves the pillars end-of-episode speed spikes (Fix C closes for free).

---

## Step 5 — Relocate hover pauses (`trajectories.py`)

**Problem:** U4 Fix A placed hover waypoints at x=±0.5, exactly on the wall end-face plane, with only 0.14 m lateral margin. F4 confirmed these pauses had no effect on rejection rate (junction v≈0 per cosine profile already). Parking the drone at the wall end-face is a contact risk.

**Change:**

`uav_expert_data_collect/trajectories.py` `s_curve_scene_path`:

Hover waypoints and segment endpoints moved from x=±0.5 to x=∓0.7 (0.2 m inside each corridor):

| Phase | Before | After |
|-------|--------|-------|
| Seg A end | (-0.5, y1, z) | (-0.7, y1, z) |
| Hover 1 | (-0.5, y1, z) | (-0.7, y1, z) |
| Seg B start | (-0.5, y1, z) | (-0.7, y1, z) |
| Seg B end | (+0.5, y2, z) | (+0.7, y2, z) |
| Hover 2 | (+0.5, y2, z) | (+0.7, y2, z) |
| Seg C start | (+0.5, y2, z) | (+0.7, y2, z) |
| d_a / d_c | 2.7 m | 2.5 m |
| d_b (x-span) | 1.0 m | 1.4 m |

Docstring updated with U5 rationale.

---

## Validation plan

Run 20-trial smoke test on s_curve:

```bash
python uav_expert_data_collect/collect.py --scene s_curve --n-trials 20 --seed 100
```

Expected: reject histogram printed; rejection rate < 60% (Step 1+2 gate).  
If < 30%: run full 500-trial collection.  
If still > 30%: Steps 3 and 4 may need tuning — check histogram to see if `floor` or `contact` dominates, and inspect `motor_clip_frac` values.

---

## Files changed

| File | Step |
|------|------|
| `uav_expert_data_collect/generator.py` | 1, 2 |
| `uav_expert_data_collect/collect.py` | 1 |
| `uav_expert_data_collect/trajectories.py` | 3, 5 |
| `uav_env_test/flight_controller.py` | 4 |
