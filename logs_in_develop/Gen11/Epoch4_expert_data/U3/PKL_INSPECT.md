# Dataset Inspection Without GIFs — pkl Direct Read Methodology

**Date:** 2026-06-09  
**Context:** Developed during E5 U3 smoke-test investigation that discovered Fix_2 floor crash bug.  
**Value:** Faster and more precise than visual GIF inspection. Catches numerical anomalies
(velocity spikes, altitude crashes) that are hard to see in a 96×96 pixel GIF.

---

## pkl Structure

Every saved episode is a dict with these keys (see `dataset_writer.py:53`):

```
episode_id  : str     e.g. 'pillars_L_L_L_pid_default_0000000'
scene       : str
homotopy    : str
controller  : str
dt          : float   0.03 s  (dataset rate = 33 Hz; NOT physics dt)
obs         : (T, 9)  float32  [p_des(3) | p(3) | v(3)]
actions     : (T-1,3) float32  Δp_des per step
targets     : (T, 3)  float32  noisy p_des (not used here)
q           : (T, 4)  float32  actual quaternion (attitude)
obstacles   : list[dict]
metadata    : dict    {start_pos, total_time, dt_physics, contact_fraction, ...}
```

**obs column layout** (9 columns total):

| Columns | Field | Meaning |
|---|---|---|
| 0:3 | `p_des` | Desired position from trajectory function |
| 3:6 | `p` | Actual drone position (world frame, metres) |
| 6:9 | `v` | Actual drone velocity (world frame, m/s) |

---

## Inspection Snippets

### 1. Single episode — speed + altitude profile

```python
import pickle, numpy as np

ep  = pickle.load(open('logs/uav_expert_data/pillars/L_L_L/pillars_L_L_L_pid_default_0000000.pkl', 'rb'))
obs = np.array(ep['obs'])
p   = obs[:, 3:6]   # actual position
v   = obs[:, 6:9]   # actual velocity
spd = np.linalg.norm(v, axis=1)
dt  = ep['dt']      # 0.03 s

print(f"Duration: {ep['metadata']['total_time']:.1f}s  Steps: {len(obs)}")
print(f"z range:  [{p[:,2].min():.3f}, {p[:,2].max():.3f}] m")
print(f"speed:    mean={spd.mean():.3f}  max={spd.max():.3f} m/s")
print(f"near-zero (|v|<0.05): {(spd<0.05).sum()}/{len(spd)} = {100*(spd<0.05).mean():.1f}%")

# Tabular speed profile sampled every 0.5 s
stride = max(1, int(0.5 / dt))
print(f"\n{'t(s)':>5}  {'x':>6}  {'y':>6}  {'z':>6}  {'spd':>7}")
for i in range(0, len(obs), stride):
    print(f"{i*dt:5.1f}  {p[i,0]:6.2f}  {p[i,1]:6.2f}  {p[i,2]:6.3f}  {spd[i]:7.4f}")
```

### 2. Dataset-wide scan — find floor crashes and speed anomalies

```python
import pickle, numpy as np, os

data_dir = 'logs/uav_expert_data/pillars'
results  = []

for homotopy in sorted(os.listdir(data_dir)):
    hp = os.path.join(data_dir, homotopy)
    if not os.path.isdir(hp): continue
    for fn in sorted(os.listdir(hp)):
        if not fn.endswith('.pkl'): continue
        ep  = pickle.load(open(os.path.join(hp, fn), 'rb'))
        obs = np.array(ep['obs'])
        p   = obs[:, 3:6]
        v   = obs[:, 6:9]
        spd = np.linalg.norm(v, axis=1)
        results.append({
            'ep_id':    ep['episode_id'],
            'homotopy': homotopy,
            'min_z':    p[:,2].min(),
            'max_spd':  spd.max(),
            'cf':       ep['metadata']['contact_fraction'],
        })

# Summary by homotopy
from collections import defaultdict
by_homo = defaultdict(list)
for r in results: by_homo[r['homotopy']].append(r)

for h, rs in sorted(by_homo.items()):
    min_zs  = np.array([r['min_z']   for r in rs])
    spds    = np.array([r['max_spd'] for r in rs])
    print(f"{h:<8}: n={len(rs)}  z<0.40={( min_zs<0.40).sum()}  spd>2={(spds>2.0).sum()}")

# Print worst offenders
print("\nWorst by max_spd:")
for r in sorted(results, key=lambda x: x['max_spd'], reverse=True)[:10]:
    print(f"  {r['ep_id']}  min_z={r['min_z']:.3f}  max_spd={r['max_spd']:.3f}  cf={r['cf']:.4f}")
```

### 3. Quick sanity check after re-collection (verify Fix_2 is working)

```python
import pickle, numpy as np, glob

bad = 0; total = 0
for f in glob.glob('logs/uav_expert_data/pillars/**/*.pkl', recursive=True):
    ep  = pickle.load(open(f, 'rb'))
    obs = np.array(ep['obs'])
    p   = obs[:, 3:6]
    v   = obs[:, 6:9]
    if p[:,2].min() < 0.50 or np.linalg.norm(v, axis=1).max() > 2.5:
        bad += 1
    total += 1
print(f"Bad: {bad}/{total}")   # Expected 0 after Fix_2
```

---

## What Each Metric Catches

| Metric | Threshold | What it means |
|---|---|---|
| `p[:,2].min()` (min z) | < 0.50 m | Floor crash — PID lost altitude control |
| `np.linalg.norm(v,axis=1).max()` (max speed) | > 2.5 m/s | Crash or physics instability (clean episodes peak ~1.1 m/s) |
| `(spd < 0.05).sum() / T` (near-zero fraction) | — | Waypoint stops — expected for `traverse_line` design (~1–2%) |
| `ep['metadata']['contact_fraction']` | > threshold | Obstacle contact — already filtered at collection time |

---

## What This Caught (Fix_2)

Running snippet 2 on `temp/Gen11E5U3` revealed:
- **188/473 pillar episodes** had `max_spd > 2.0 m/s` — dominated by L_R_L (76%) and R_L_R (65%)
- **304/356 s_curve episodes** had `max_spd > 2.0 m/s` — 85% contamination
- All with `contact_fraction = 0.0` — completely invisible to the existing filter
- Root cause: `generator.py:_is_obstacle_contact` excludes floor contacts by design

See `Fix_2/ANALYSIS.md` for the full breakdown.

---

## Why This Is Faster Than GIFs

| Method | Time per episode | Detects crash? | Detects speed spike? | Quantitative? |
|---|---|---|---|---|
| GIF visual inspection | ~10–30 s | Maybe (hard at 96px) | No | No |
| pkl direct read | < 0.1 s | Yes (`min_z`) | Yes (`max_spd`) | Yes |

For a 473-episode dataset, GIF inspection would take hours. The dataset-wide scan (snippet 2)
runs in under 5 seconds and produces an exact count and episode list.
