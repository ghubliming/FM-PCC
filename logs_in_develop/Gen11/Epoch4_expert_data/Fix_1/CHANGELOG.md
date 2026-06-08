# Gen11 Epoch 4 — Fix_1: s_curve / pillars 100% rejection + noise corruption

**Date**: 2026-06-04  
**Triggered by**: `temp/Gen11E4 outputs/SLURM OUTPUTS` — jobs 21207–21209  
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md)

---

## Results that triggered this fix

| Scene | Saved | Rejected | Root cause |
|---|---|---|---|
| empty | 500/500 | 0% | ✅ — no fix needed |
| corridor | 436/500 | 12.8% | ✅ — acceptable |
| **s_curve** | **0/500** | **100%** | Zero-velocity stops near wall boundaries |
| **pillars** | **1/500** | **95%** | Zero-velocity stops near pillar pairs |
| *(all scenes)* | — | — | Action deltas noise-dominated (silent) |

---

## Fix_1.1 — s_curve: 100% rejection from PID overshoot at wall boundaries

**File**: `uav_expert_data_collect/trajectories.py` — `s_curve_scene_path`

### Root cause

The original 4-waypoint path used `s_curve_path` (piecewise `traverse_line`), which enforces **zero velocity at every waypoint junction**. Waypoints 2 and 3 were placed exactly at the corridor wall ends (`x = ±0.5`). The drone decelerated to a dead stop at these points — with any PID tracking error, it oscillated near the wall edge and tripped the obstacle contact check on every trial.

### Fix

Added two interior waypoints so the drone spends more time well inside each corridor before attempting the gap crossing. Increased segment count from 3 to 5, reducing peak speed per segment and PID overshoot.

```python
# Before (4 waypoints, 3 legs):
wps = [(-3.2, -0.8, z), (-0.5, -0.8, z), (0.5, 0.8, z), (3.2, 0.8, z)]
seg_dur = duration / 3.0

# After (6 waypoints, 5 legs):
wps = [
    (-3.2, -0.8, z),
    (-1.5, -0.8, z),   # ← new: well inside seg1
    (-0.5, -0.8, z),   # at seg1 exit (wall end)
    ( 0.5,  0.8, z),   # at seg2 entry (wall start)
    ( 1.5,  0.8, z),   # ← new: well inside seg2
    ( 3.2,  0.8, z),
]
seg_dur = duration / 5.0
```

---

## Fix_1.2 — s_curve: longer episode duration

**File**: `uav_expert_data_collect/generator.py` — `_build_traj_and_init` s_curve branch

### Root cause

Duration range was `[10, 16]` s over 3 legs → each leg 3.3–5.3 s for 2.7 m → peak cosine speed up to **1.06 m/s**. Fast approach to zero-velocity stops near walls amplifies overshoot.

### Fix

Increased to `[16, 22]` s over 5 legs (Fix_1.1) → each leg 3.2–4.4 s for 1.4 m average → peak speed ≈ **0.50 m/s**. Also reduced y_jitter from `±0.05` to `±0.04` m to keep larger clearance from walls.

```python
# Before:
dur = float(rng.uniform(10.0, 16.0))
y_jitter = float(rng.uniform(-0.05, 0.05))

# After:
dur = float(rng.uniform(16.0, 22.0))
y_jitter = float(rng.uniform(-0.04, 0.04))
```

---

## Fix_1.3 — pillars: replace stop-and-go path with continuous sinusoidal weave

**File**: `uav_expert_data_collect/generator.py` — `_build_traj_and_init` pillars branch

### Root cause

`pillar_path` used `s_curve_path` through 8 waypoints, stopping near each pillar pair. Even a small PID tracking error at a stop next to a pillar (clearance ≈ 0.20 m at `_Y_L = −0.92`) triggered a contact. 95% of trials rejected.

### Fix

Replaced with the existing **`weave` factory** from `uav_env_test/trajectories.py` — a continuous sinusoidal trajectory that never stops mid-flight. Homotopy classes are mapped to amplitude sign:

| Homotopy | Amplitude | Path shape |
|---|---|---|
| `(L,L,L)` | −1.0 | Negative-y swing — outside column A |
| `(L,R,L)` | +0.55 | Positive-y swing — through centre-right |
| `(R,L,R)` | −0.55 | Negative-y swing — through centre-left |
| `(R,R,R)` | +1.0 | Positive-y swing — outside column B |

```python
# Before: trajs.pillar_path(seq, z, dur)  →  s_curve_path through 8 waypoints

# After: continuous weave — never stops near pillars
_amp_map = {'(L,L,L)': -1.0, '(L,R,L)': 0.55, '(R,L,R)': -0.55, '(R,R,R)': 1.0}
traj_fn = trajs.weave(
    x_range=(-3.2, 3.2), y_amplitude=_amp_map[homotopy],
    period=4.0, altitude=z, duration=dur,
)
```

---

## Fix_1.4 — Noise augmentation corrupts action deltas (silent data bug)

**File**: `uav_expert_data_collect/dataset_writer.py` — `rollout_to_episode`

### Root cause

Per-step **independent** noise was added to every target position. `actions = np.diff(targets)` then computes the difference of adjacent noisy steps:

```
Var(targets[t+1] - targets[t]) = σ² + σ² = 2σ²
std(action noise) = √2 · 0.02 = 0.028 m/step
```

The actual trajectory signal is only ~0.012 m/step at 0.4 m/s / 33 Hz. Noise is **2.3× the signal** — FM would learn to predict random walk, not trajectories.

Evidence from validator: `action_delta_norm mean=0.047 m/step` vs expected `0.012 m/step`.

### Fix

One **constant offset** per episode. Shifting the entire trajectory rigidly thickens the data manifold as intended, while leaving `Δp_des = targets[t+1] − targets[t]` unchanged.

```python
# Before (per-step independent noise):
targets = targets + rng.normal(0.0, noise_sigma, targets.shape)

# After (one constant offset per episode):
offset = rng.normal(0.0, noise_sigma, (1, 3))   # shape (1,3) broadcasts over T
targets = targets + offset
```

---

## Files touched

| File | Change |
|---|---|
| `uav_expert_data_collect/trajectories.py` | `s_curve_scene_path`: 4→6 waypoints, 3→5 legs |
| `uav_expert_data_collect/generator.py` | s_curve duration `[10,16]`→`[16,22]` s, jitter `±0.05`→`±0.04`; pillars: `pillar_path` → `weave` with amplitude map |
| `uav_expert_data_collect/dataset_writer.py` | Noise augmentation: per-step → per-episode constant offset |

---

## Expected after fix

| Scene | Expected outcome |
|---|---|
| empty | Unchanged — 0% rejection, ~0.39 m/s |
| corridor | Unchanged — ~13% rejection, homotopy balanced |
| s_curve | Rejection rate < 15% |
| pillars | Rejection rate < 20% |
| Action Δp_des norm | ~0.012 m/step (matches 0.4 m/s at 33 Hz) — noise no longer dominates |

Re-run with same commands:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  500
```
Empty and corridor do **not** need re-collection.
