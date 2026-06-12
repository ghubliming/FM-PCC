# Gen11 Epoch 4 — Fix_2: s_curve persistent failure + pillars centre-class collision

**Date**: 2026-06-04  
**Triggered by**: `temp/Gen11E4 outputs/1/outputs` — jobs 21212–21215  
**Parent**: [`../Fix_1/CHANGELOG.md`](../Fix_1/CHANGELOG.md)

---

## Results that triggered this fix

| Scene | Saved | Rejected | vs Fix_1 |
|---|---|---|---|
| empty | 500/500 | 0% | ✅ unchanged |
| corridor | 436/500 | 12.8% | ✅ unchanged |
| **s_curve** | **2/500** | **90.5%** | ❌ barely improved from 100% |
| **pillars** | **10/500** | **54.5%** | ❌ still aborting; only 2 of 4 homotopy classes produced data |

Fix_1.4 (noise augmentation) confirmed working: empty action norm dropped from `0.047` → `0.0116 m/step` ✅.

---

## Fix_2.1 — s_curve: replace piecewise path with single continuous tanh trajectory

**File**: `uav_expert_data_collect/trajectories.py` — `s_curve_scene_path`

### Root cause

Fix_1 added interior waypoints and longer duration but **did not eliminate zero-velocity stops**. `s_curve_path` (piecewise `traverse_line`) enforces `v=0` at every joint. With stops at the wall-end waypoints (`x=±0.5`), any PID position error at rest caused body contact with wall geometry. The 90.5% rejection rate confirmed the stops themselves are the problem, not the speed.

### Fix

Replaced the piecewise factory entirely with a **single analytic tanh trajectory**. The drone maintains non-zero velocity throughout:

```
y(x) = y_mid + y_amp · tanh(k·x)
```

- `k = 3.66` — calibrated so 95% of the lateral shift happens within `x ∈ (−0.5, +0.5)` (the open gap between the two corridor segments, no walls)
- `x ∈ [−3.2, −0.5]`: drone deep inside seg1 at `y ≈ −0.8` (tanh saturated)
- `x ∈ [−0.5, +0.5]`: smooth transition from `y=−0.8` to `y=+0.8`
- `x ∈ [+0.5, +3.2]`: deep inside seg2 at `y ≈ +0.8` (tanh saturated)

Velocity and acceleration are derived analytically via chain rule — no finite-difference approximation. The PID receives accurate feed-forward terms throughout.

```python
# Before: piecewise stops (Fix_1 variant, still fragile)
wps = [(-3.2,-0.8,z), (-1.5,-0.8,z), (-0.5,-0.8,z), (0.5,0.8,z), (1.5,0.8,z), (3.2,0.8,z)]
return s_curve_path(wps, duration/5.0)

# After: single continuous function, no stops
th    = np.tanh(3.66 * x)
y     = y_mid + y_amp * th
dy_dt = y_amp * 3.66 * (1 - th²) * v_x          # analytic derivative
```

---

## Fix_2.2 — pillars: fix amplitude for centre-pass homotopy classes

**File**: `uav_expert_data_collect/generator.py` — `_build_traj_and_init` pillars branch

### Root cause

`(L,R,L)` was mapped to amplitude `+0.55` and `(R,L,R)` to `−0.55`. The pillar column edges are at `y = ±(0.6 − 0.12) = ±0.48`. Amplitude `0.55 > 0.48` — the weave swung through the pillar body on every cycle. Both classes produced 0 usable episodes.

Validator showed only `{'(L,L,L)': 5, '(R,R,R)': 5}` in the 10 saved episodes — confirming the ±1.0 outer-swing classes were the only ones that cleared the pillars.

### Fix

Set amplitude to `0.0` for both mixed classes. With amplitude=0, `weave` produces a straight line at `y=0` — the centre of the pillar field. Clearance from `y=0` to nearest pillar edge (`y=±0.48`) is 0.48 m on each side, well clear of the 0.12 m radius pillar.

```python
# Before:
'(L,R,L)':  0.55,   # inside pillar zone — 100% rejection
'(R,L,R)': -0.55,   # inside pillar zone — 100% rejection

# After:
'(L,R,L)':  0.0,    # straight centre at y=0, clearance 0.48 m
'(R,L,R)':  0.0,    # same centre line
```

*Note*: `(L,R,L)` and `(R,L,R)` now produce identical straight-centre trajectories. They remain as separate labels for forward-compatibility but are geometrically indistinct at this epoch. True homotopy differentiation for these classes requires a more sophisticated path planner (future epoch scope).

---

## Files touched

| File | Change |
|---|---|
| `uav_expert_data_collect/trajectories.py` | `s_curve_scene_path`: piecewise stops → single continuous tanh trajectory |
| `uav_expert_data_collect/generator.py` | Pillars amplitude map: `(L,R,L)` `0.55`→`0.0`, `(R,L,R)` `-0.55`→`0.0` |

---

## Expected after fix

| Scene | Expected outcome |
|---|---|
| empty | Unchanged — 0% rejection |
| corridor | Unchanged — ~13% rejection |
| s_curve | < 10% rejection (continuous path, no stops near walls) |
| pillars | < 15% rejection; all 4 homotopy labels produce episodes |

Re-run s_curve and pillars only:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  500
```
