# Gen11 Epoch4 U3 Fix_2 — Floor Crash Contamination

**Date:** 2026-06-09  
**Discovered via:** E5 U3 GIF smoke-test investigation  
**Code fix applied:** `generator.py` — `Z_FLOOR_MARGIN = 0.50` + `min_z` rejection in `run_trial`  
**Status:** Fix coded — requires E4 U4 re-collection to take effect

---

## 1. Investigation Trail

E5 U3 smoke test submitted 3 pillar GIF pairs (state-injection + physics replay). Both
GIFs showed identical "move → stop → strongly forward lean → continue" behaviour.
Initial hypothesis: physics replay was broken.

**Finding A — Both GIFs being identical is correct.**  
The 3 smoke-test episodes are all `L_L_L`, a clean homotopy. Physics replay correctly
reproduces collection (same seed → same trajectory → same PID → same positions). For
no-contact episodes there is nothing to differentiate the two views.

**Finding B — Stop-and-lean is expected trajectory behaviour.**  
`pillar_path` uses `traverse_line` for each of 7 segments. `traverse_line` uses a
cosine velocity profile: `v(0) = v(T) = 0`. Drone fully decelerates to zero at every
waypoint boundary, then pitches forward to re-accelerate. Data confirms: 1.5% of frames
have `|v| < 0.05 m/s` (brief but real stops). Both GIFs faithfully show this. Not a bug.

**Finding C — While inspecting pkl data, a critical dataset quality issue was discovered.**

---

## 2. Root Cause

`generator.py:101–104`:

```python
def _is_obstacle_contact(model, contact):
    """True if neither geom is the floor — counts as a drone/obstacle collision."""
    n1 = model.geom(contact.geom1).name
    n2 = model.geom(contact.geom2).name
    return n1 != 'floor' and n2 != 'floor'
```

This function **explicitly returns False for floor contacts**. MuJoCo records
drone-floor contact as a geom named `'floor'`, so every floor crash returns `n_hit=0`
and `contact_fraction=0.0`. The episode passes the rejection filter as if it flew
perfectly.

**When does this cause crashes?**  
For homotopies that require large lateral displacement in a short segment, the PID loses
altitude control:

- **pillars L_R_L / R_L_R**: must cross y=−1.11 → y=+1.11 (Δy=2.22 m) in the
  x=−1.5→−0.5 span. Cosine velocity profile drives the drone aggressively sideways
  while altitude controller lags → descent into the floor.
- **s_curve**: diagonal gap crossing y=−0.8 → y=+0.8 (Δy=1.6 m). Same mechanism.

`empty` and `corridor` have no large lateral manoeuvres → zero contamination.

---

## 3. Contamination Scale

Data source: `temp/Gen11E5U3` (E4 U3 full dataset, 1829 episodes).

### Pillars (473 episodes)

| Homotopy | n | z<0.20 | z<0.40 | max_spd>2 m/s | max_spd>3 m/s |
|---|---|---|---|---|---|
| L_L_L | 125 | 0 (0%) | 6 (5%) | 19 (15%) | 19 (15%) |
| L_R_L | 115 | 19 (17%) | 30 (26%) | 87 **(76%)** | 83 (72%) |
| R_L_R | 108 | 19 (18%) | 28 (26%) | 70 **(65%)** | 65 (60%) |
| R_R_R | 125 | 1 (1%) | 3 (2%) | 12 (10%) | 12 (10%) |
| **Total** | **473** | **39 (8%)** | **67 (14%)** | **188 (40%)** | **179 (38%)** |

Speed range: `p95 = 6.59 m/s`, `max = 7.66 m/s` (vs clean L_L_L episode peak ~1.1 m/s).  
Several episodes have `min_z < 0` — drone penetrated below the floor plane.

### Other scenes

| Scene | n_saved | max_spd>2 m/s | Notes |
|---|---|---|---|
| empty | 500 | 0 (0%) | Straight-line traversals only — clean |
| corridor | 500 | 0 (0%) | No large lateral crossing — clean |
| s_curve | 356 | 304 **(85%)** | All 5 consecutive episodes sampled had z<0.40 |

**s_curve sample (first 5 consecutive episodes):**
```
s_curve_default_0000001: z=[0.028, 0.927]  max_spd=2.61
s_curve_default_0000002: z=[0.021, 0.884]  max_spd=2.54
s_curve_default_0000003: z=[0.015, 0.778]  max_spd=2.77
s_curve_default_0000004: z=[0.005, 1.149]  max_spd=2.51
s_curve_default_0000005: z=[0.010, 1.023]  max_spd=4.60
```

### Overall E4 U3 dataset quality

| Scene | Saved | Contaminated | Clean estimate |
|---|---|---|---|
| empty | 500 | 0 | ~500 |
| corridor | 500 | 0 | ~500 |
| pillars | 473 | ~188 (40%) | ~285 |
| s_curve | 356 | ~304 (85%) | ~52 |
| **Total** | **1829** | **~492 (27%)** | **~1337** |

---

## 4. Impact on Training

Contaminated episodes contain:
- **Velocity spikes up to 7.66 m/s** (6× clean peak) in the `obs` velocity column
- **z-positions below 0.40 m** (some below 0) — positions the drone should never occupy
- **contact_fraction = 0** in metadata — episodes appear safe when they are crashes

Training on these teaches the FM that floor-proximal, high-speed trajectories are valid
expert demonstrations → unsafe rollouts at evaluation time.

---

## 5. Fix

### Fix A — `generator.py`: add floor-z rejection in `run_trial`

Add after the simulation loop (after `contact_frac` is computed, `generator.py:236`):

```python
Z_FLOOR_MARGIN = 0.50   # metres; normal hover z=0.7–1.1m; floor at z=0

# ... existing contact_frac check ...
min_z = min(s['p'][2] for s in steps)
if min_z < Z_FLOOR_MARGIN:
    return None   # reject — floor crash
```

0.50 m threshold: rotor radius ≈ 0.14 m → body must be ≥ 0.14 m above floor for zero
contact; 0.50 m gives 0.36 m PID-recovery margin above that.

### Fix B — Re-collect affected scenes (E4 U4)

With Fix A in place, re-run collection for **pillars** and **s_curve** only.
`empty` and `corridor` are clean — do not re-collect.

### Fix C — Post-hoc filter (optional, for training before E4 U4)

```python
obs = np.array(ep['obs'])
p   = obs[:, 3:6]
v   = obs[:, 6:9]
if p[:,2].min() < 0.50 or np.linalg.norm(v, axis=1).max() > 2.5:
    os.remove(pkl_path)   # remove corrupted episode
```

Reduces usable dataset to ~1337 episodes but makes it safe to train on immediately.

---

## 6. What NOT to Change

- `_is_obstacle_contact`: the floor exclusion is correct for counting obstacle contacts.
  Do not change it. The z-check is a separate, independent rejection criterion.
- Trajectory geometry (L_R_L / R_L_R channels, waypoints): correct routing. The issue is
  PID altitude control during aggressive crossing, not the route design.

---

## 7. Next Steps

| Step | Action | Urgency |
|---|---|---|
| 1 | Apply Fix A to `generator.py` | Before E4 U4 |
| 2 | (Optional) Run post-hoc filter on E4 U3 dataset | Before E6 training if re-collection delayed |
| 3 | Submit E4 U4: pillars + s_curve re-collection | After Fix A verified |
| 4 | Verify E4 U4: `min_z > 0.50` across all saved episodes | During/after E4 U4 SLURM |

---

## 8. Files Referenced

| File | Role |
|---|---|
| `uav_expert_data_collect/generator.py:101` | `_is_obstacle_contact` — floor exclusion (do not change) |
| `uav_expert_data_collect/generator.py:236` | `run_trial` — where Fix A z-check goes |
| `uav_expert_data_collect/dataset_writer.py:53` | obs layout `[p_des(3) \| p(3) \| v(3)]` |
| `temp/Gen11E5U3/uav_expert_data/` | E4 U3 dataset inspected in this analysis |
| `logs_in_develop/Gen11/Epoch5_visual_and_validation/U3/` | E5 U3 smoke test that triggered discovery |
