# Gen11 Epoch 4 — Fix_3: s_curve persistent rejection from high lateral PID demand

**Date**: 2026-06-04  
**Triggered by**: `temp/Gen11E4 outputs/2/outptus` — jobs 21220–21221  
**Parent**: [`../Fix_2/CHANGELOG.md`](../Fix_2/CHANGELOG.md)

---

## Results that triggered this fix

| Scene | Saved | Rejected | vs Fix_2 |
|---|---|---|---|
| **s_curve** | **8/500** | **61.9%** | ❌ improved (90.5%→61.9%) but still aborting |
| pillars | 477/500 | 4.6% | ✅ fixed — no change needed |

---

## Fix_3.1 — s_curve: lower tanh steepness + longer duration

**Files**: `uav_expert_data_collect/trajectories.py`, `uav_expert_data_collect/generator.py`

### Root cause

Fix_2's tanh trajectory (`k=3.66`) kept the reference path geometrically safe (clearance 0.46 m at wall ends), but the peak lateral speed at `x=0` was:

```
dy/dt_max = y_amp × k × v_x = 0.8 × 3.66 × 0.4 = 1.17 m/s
```

The PID (Kp_pos=[4,4,8]) cannot track this cleanly. Accumulated lateral position error occasionally pushed the drone body into a wall face, causing 62% rejection. The 8/21 passes (38%) confirm the trajectory geometry is correct — the problem is purely the tracking demand.

### Fix

Two parameter changes:

**1. Lower k: `3.66` → `2.0`**  
Reduces peak lateral speed while keeping adequate wall clearance:

| Parameter | k=3.66 (Fix_2) | k=2.0 (Fix_3) |
|---|---|---|
| y at x=±0.5 | ±0.760 m (clearance 0.46 m) | ±0.609 m (clearance 0.31 m) |
| Peak lateral speed (T=22s) | 1.06 m/s | 0.58 m/s |
| Peak lateral accel | 1.32 m/s² | 0.44 m/s² |

**2. Longer duration: `[16,22]s` → `[22,30]s`**  
Lower `v_x` further reduces lateral speed and acceleration:

| T | v_x | Peak lateral v (k=2.0) |
|---|---|---|
| 16 s | 0.40 m/s | 0.64 m/s |
| 22 s | 0.29 m/s | 0.47 m/s |
| 30 s | 0.21 m/s | 0.34 m/s |

At `T=22–30s`, peak lateral speed **0.34–0.47 m/s** — directly comparable to the corridor scene (mean 0.72 m/s, 87% pass rate).

```python
# trajectories.py — before:
k = 3.66

# after:
k = 2.0

# generator.py — before:
dur = float(rng.uniform(16.0, 22.0))

# after:
dur = float(rng.uniform(22.0, 30.0))
```

---

## Files touched

| File | Change |
|---|---|
| `uav_expert_data_collect/trajectories.py` | `s_curve_scene_path` tanh steepness `k`: `3.66` → `2.0` |
| `uav_expert_data_collect/generator.py` | s_curve duration range: `[16,22]` → `[22,30]` s |

---

## Expected after fix

| Scene | Expected |
|---|---|
| s_curve | < 15% rejection |
| pillars | Unchanged — 4.6% rejection |

Re-run s_curve only:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve 500
```
