# Gen11 Epoch 4 — U3 Plan: Full Scene & Homotopy Audit + Re-collection

**Date**: 2026-06-08  
**Based on**: E4 U2 (`CLOSURE.md` — 1769 episodes, obs=(T,9), 12D FM tensor)  
**Status**: Planning  
**Investigation that opened this**: `../../Epoch5_visual_and_validation/U2/INVESTIGATE_pillar_passthrough.md`

---

## Why U3 is needed

The GIF review that found the pillar pass-through prompted a full geometry audit of every
scene and homotopy class. The audit found problems beyond pillars: corridor L/R channels
are also in contact with walls by design, and two of the four pillar homotopy classes have
completely wrong labels (the weave trajectory flies through the centre, labelled as L/R).

Additionally, the `q=(T,4)` field coded in E5 U2 D-prep needs a re-collection run to
populate it across all scenes.

---

## Full scene and homotopy audit

### Key drone geometry (from `quadrotor_modified.xml`)

Rotor collision ellipsoids (`class="rotor"`, ARE physics collision geoms):
- Positions: `±(0.14, 0.18)` m from COM in body frame
- Radius: `0.13 m` in x/y
- **Maximum y reach from COM**: `0.18 + 0.13 = 0.31 m`

The drone's physical collision footprint in y is **±0.31 m** from the COM centre.
Any clearance calculation must use this number, not just the body dimensions.

---

### Scene: empty (homotopy N/A)

- Random straight-line trajectories, no obstacles.
- **Status**: ✅ No issues. No change.

---

### Scene: corridor (homotopy C, L, R)

**Geometry**: walls at `y = ±0.45` (inner surface). Clear interior: `y ∈ (−0.45, +0.45)`.  
**Current channels** (`trajectories.py`): `L = −0.18`, `C = 0.0`, `R = +0.18`.  
**Current y_jitter**: `±0.05 m` added per episode in `generator.py`.

**Clearance analysis** (worst case = channel_y ± jitter_max):

| Homotopy | Channel y | Worst y | Rotor wall edge | Wall surface | Clearance |
|---|---|---|---|---|---|
| C | 0.00 | ±0.05 | ±(0.05+0.31)=±0.36 | ±0.45 | **+0.09 m OK** |
| L | −0.18 | −0.23 | −0.23−0.31=−0.54 | −0.45 | **−0.09 m CONTACT** |
| R | +0.18 | +0.23 | +0.23+0.31=+0.54 | +0.45 | **−0.09 m CONTACT** |

At worst jitter, corridor L/R have **9 cm rotor penetration into the wall**. This is real
physics contact — it was explicitly accepted in E4 U2 Fix_1 as "inherent to the
homotopy." That decision is revisited here:

- **The old argument**: L homotopy means flying close to the left wall; contact is realistic.
- **The U3 argument**: If we want clean training data for the FM, the FM should learn "fly
  to the left" — not "fly left and hit the wall." The homotopy signal should be a
  position preference, not a wall-contact instruction.

**Recommended fix**: move channels inward and remove y_jitter from L/R.

Safe channel for no contact (wall at ±0.45, rotor reach 0.31 m, target 2 cm margin):
```
y_L_safe = −0.45 + 0.31 + 0.02 = −0.12 m   (was −0.18)
y_R_safe = +0.45 − 0.31 − 0.02 = +0.12 m   (was +0.18)
```

At y = ±0.12, no jitter:
- Rotor wall edge: ±(0.12 + 0.31) = ±0.43 m
- Wall surface: ±0.45 m
- Clearance: **+0.02 m** (2 cm) ✅

L and C channels are now separated by 12 cm in y. The FM observes `p_des.y` directly
(it's in obs columns 0:3), so a 12 cm distinction is clearly learnable.

**Decision on jitter**: Remove y_jitter for corridor L/R in U3. The jitter was added to
thicken the data manifold, but it is the main cause of contact at the extreme values. The
obs noise (`noise_sigma=0.02`) already provides stochastic variation in the action targets.

---

### Scene: s_curve (homotopy default)

**Geometry**: two narrow corridors at y ≈ ±0.80, each with 0.90 m interior width.  
Wall inner surfaces: seg1 at y = (−1.25, −0.35), seg2 at y = (+0.35, +1.25).  
Drone path: y = −0.80 (seg1), y = +0.80 (seg2).

**Clearance analysis**:
- At y = −0.80: rotor edge toward near wall = −0.80 − 0.31 = −1.11. Near wall at −1.25. Clearance = **+0.14 m OK**.
- At y = −0.80: rotor edge toward far wall = −0.80 + 0.31 = −0.49. Far wall at −0.35. Clearance = **+0.14 m OK**.

Both clearances = 14 cm. End-face grazes at the diagonal transition (x = ±0.5) are
structural — they happen at the corners where the corridor direction changes, not from
incorrect channel positioning.

**Status**: ✅ No geometry issues. Contact threshold 0.08 (Fix_4 rationale) unchanged.

---

### Scene: pillars — full redesign required

**Geometry**: 6 cylinders, radius 0.12 m.  
Column A: `y = −0.60`, columns at `x ∈ {−2.0, 0.0, +2.0}`.  
Column B: `y = +0.60`, columns at `x ∈ {−2.0, 0.0, +2.0}`.

**Rotor contact zone** (from investigation `INVESTIGATE_pillar_passthrough.md`):  
Drone COM y values that cause rotor-pillar contact with column A: **y ∈ (−0.55, −0.97)**.  
The only safe zones on the left side: `y > −0.55` (centre gap) or `y < −0.97` (far left).

**Minimum safe channel** (column A, 8 cm margin above zero-contact):
```
y_L_safe = −0.60 − 0.12 − 0.31 − 0.08 = −1.11 m
y_R_safe = +0.60 + 0.12 + 0.31 + 0.08 = +1.11 m
```
At y = ±1.11: clearance = **+0.108 m (10.8 cm)** — enough margin for PID tracking error.

#### Homotopy class audit

| Class | Current trajectory | Phase correct? | Contact? | Fix needed |
|---|---|---|---|---|
| `(L,L,L)` | `weave` amp=−1.0 | ❌ ~50% T values wrong side | ❌ ~54% T values contact | Redesign |
| `(R,R,R)` | `weave` amp=+1.0 | ❌ Symmetric | ❌ Symmetric | Redesign |
| `(L,R,L)` | `weave` amp=0.0 (centre!) | ❌ **Always wrong** — flies straight at y=0 | ✅ No contact (19cm clear) | **Label is completely wrong** |
| `(R,L,R)` | `weave` amp=0.0 (centre!) | ❌ **Always wrong** | ✅ No contact | **Label is completely wrong** |

For `(L,R,L)` and `(R,L,R)`: Fix_2 in E4 set amplitude to 0 to avoid 100% rejection.
The drone now flies straight through the centre, labelled as a weave homotopy. Every
episode in these two classes is mislabelled — the FM learns "this is an L,R,L episode"
while the trajectory is actually a C,C,C path. This is a critical data quality problem.

#### Correct pillar_path design

Replace `weave` with an explicit 4-waypoint `pillar_path`. Waypoints are positioned so:

1. The approach to the first pillar is complete **before** reaching `x = −2.0`.
2. All three pillar x positions are within the constant-channel segment (no transition there).
3. Crossing transitions for (L,R,L)/(R,L,R) pass through `y = 0` exactly at `x = 0.0`
   (midpoint between columns A2 and B2, clearance 19 cm).

**4-waypoint scheme**:
```
Entry: (x=−3.2, y=0.0)          ← drone starts here
Way-1: (x=−2.5, y=chan_1)       ← approach COMPLETE 0.5m before first pillar
Way-2: (x=+2.5, y=chan_2)       ← exit starts AFTER last pillar
Exit:  (x=+3.2, y=0.0)
```

`chan_1` and `chan_2` per homotopy class:

| Class | Way-1 y | Way-2 y | y at A1 (x=−2.0) | y at A2 (x=0.0) | y at A3 (x=+2.0) |
|---|---|---|---|---|---|
| (L,L,L) | _Y_L=−1.11 | _Y_L=−1.11 | −1.11 (clr +10.8cm ✅) | −1.11 (clr +10.8cm ✅) | −1.11 (clr +10.8cm ✅) |
| (R,R,R) | _Y_R=+1.11 | _Y_R=+1.11 | +1.11 (clr +10.8cm ✅) | +1.11 (clr +10.8cm ✅) | +1.11 (clr +10.8cm ✅) |
| (L,R,L) | _Y_L=−1.11 | _Y_R=+1.11 | ≈−1.02 (clr +7cm ✅) | ≈0.0 (clr +19cm ✅) | ≈+1.02 (clr +7cm ✅) |
| (R,L,R) | _Y_R=+1.11 | _Y_L=−1.11 | ≈+1.02 (clr +7cm ✅) | ≈0.0 (clr +19cm ✅) | ≈−1.02 (clr +7cm ✅) |

For (L,R,L)/(R,L,R): the cosine blend from Way-1 to Way-2 (5m span, midpoint at x=0)
places the drone at y ≈ 0 exactly at A2 — the safest possible crossing point.
At A1 and A3 (10% and 90% along the Way-1→Way-2 segment), y ≈ ±1.02 — 7 cm clearance.

**Trajectory continuity**: using `traverse_line` with cosine profile means lateral velocity
goes to zero at each waypoint junction. The zero-lateral-velocity at Way-1 (x=−2.5) puts
the drone in the channel before the first pillar: safe. The brief hover at Way-2 (x=+2.5)
is after the last pillar: safe. No waypoint is at a pillar x position.

**Contact threshold**: tighten to `0.001` (reject any meaningful contact).

---

## Summary of all changes

### Change A — `trajectories.py`: fix constants and rewrite `pillar_path`

```python
# ── Corridor ─────────────────────────────────────────────────────────────────
CORRIDOR_CHANNELS = {
    'L':  -0.12,   # was -0.18; rotor clearance 2cm from left wall at channel centre
    'C':   0.0,    # unchanged
    'R':  +0.12,   # was +0.18
}

# ── Pillars ───────────────────────────────────────────────────────────────────
PILLAR_ROTOR_REACH = 0.31   # max y from COM to rotor ellipsoid edge
PILLAR_SAFETY      = 0.08   # 8cm margin — sufficient for PID tracking error
_Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_ROTOR_REACH - PILLAR_SAFETY  # = -1.11
_Y_R = PILLAR_Y_B + PILLAR_RADIUS + PILLAR_ROTOR_REACH + PILLAR_SAFETY  # = +1.11

def pillar_path(homotopy_seq, altitude, duration, x_start=-3.2, x_end=3.2, yaw=0.0):
    """4-waypoint explicit pillar path.
    Way-1 at x=-2.5 (approach complete before first pillar).
    Way-2 at x=+2.5 (exit starts after last pillar).
    No waypoints AT pillar x-positions — zero-velocity stops never happen near pillars.
    """
    y_map = {'L': _Y_L, 'R': _Y_R}
    chan_1 = y_map[homotopy_seq[0]]   # dominant side entering
    chan_2 = y_map[homotopy_seq[2]]   # dominant side exiting
    z = float(altitude)
    T = float(duration)

    # Equal time per segment (adjust if needed for velocity tuning)
    x_points = [x_start, -2.5, +2.5, x_end]
    y_points  = [0.0,    chan_1, chan_2, 0.0]
    seg_durs  = [T * (x_points[i+1] - x_points[i]) / (x_end - x_start)
                 for i in range(3)]

    segs = [traverse_line((x_points[i], y_points[i], z),
                          (x_points[i+1], y_points[i+1], z),
                          seg_durs[i], yaw)
            for i in range(3)]

    def traj(t):
        for i, seg in enumerate(segs):
            t_start = sum(seg_durs[:i])
            if t < t_start + seg_durs[i] or i == len(segs) - 1:
                return seg(t - t_start)

    return traj
```

### Change B — `generator.py`: revert pillars to `pillar_path`, fix corridor jitter

```python
# ── Corridor ─────────────────────────────────────────────────────────────────
# BEFORE:
y_jitter = float(rng.uniform(-0.05, 0.05))
y_bias   = trajs.CORRIDOR_CHANNELS[homotopy] + y_jitter

# AFTER (U3):
# Remove jitter for L/R — channel is already tight (2cm wall clearance).
# Jitter was main source of wall contact; obs noise provides enough variation.
if homotopy == 'C':
    y_jitter = float(rng.uniform(-0.03, 0.03))   # keep mild jitter for centre
else:
    y_jitter = 0.0
y_bias = trajs.CORRIDOR_CHANNELS[homotopy] + y_jitter

# ── Pillars ───────────────────────────────────────────────────────────────────
# BEFORE (Fix_1 weave):
_amp_map = {'(L,L,L)': -1.0, '(L,R,L)': 0.0, '(R,L,R)': 0.0, '(R,R,R)': 1.0}
amp = _amp_map[homotopy]
traj_fn = trajs.weave(x_range=(-3.2, 3.2), y_amplitude=amp, period=4.0, ...)

# AFTER (U3 explicit path):
_homotopy_seq = {
    '(L,L,L)': ['L', 'L', 'L'],
    '(L,R,L)': ['L', 'R', 'L'],
    '(R,L,R)': ['R', 'L', 'R'],
    '(R,R,R)': ['R', 'R', 'R'],
}
traj_fn = trajs.pillar_path(_homotopy_seq[homotopy], altitude=z, duration=dur)
```

### Change C — `generator.py`: update contact thresholds

```python
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,    # unchanged
    'corridor': 0.02,    # unchanged — 2cm clearance leaves PID-error contact possible
    's_curve':  0.08,    # unchanged — end-face clips structural (Fix_4)
    'pillars':  0.001,   # U3: tightened from 0.02 — reject any meaningful contact
}
```

### Change D — `q` field (already coded, no further changes)

`generator.py` and `dataset_writer.py` were updated in E5 U2 D-prep. U3 collection
automatically includes `q=(T,4)` quaternion in all pickles.

---

## Execution plan

### Phase 1 — Code changes (local, no cluster)

- [ ] `trajectories.py`: update `CORRIDOR_CHANNELS`, add `PILLAR_ROTOR_REACH`/`PILLAR_SAFETY`/`_Y_L`/`_Y_R`, rewrite `pillar_path`
- [ ] `generator.py`: remove corridor L/R jitter, revert pillar case to `pillar_path`, update `SCENE_MAX_CONTACT_FRACTION`
- [ ] Unit test: run geometry math for all 4 pillar classes, confirm clearance ≥ 5 cm at all pillar x positions

### Phase 2 — Smoke test (cluster)

```bash
# Corridor: 10 episodes each L/C/R — confirm contact_fraction ~0
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 30

# Pillars: 20 episodes each class — confirm contact_fraction=0 and correct sides
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars 80
```

Verification:
```python
import pickle, glob, numpy as np

for path in sorted(glob.glob('logs/uav_expert_data/pillars/**/*.pkl', recursive=True)):
    ep = pickle.load(open(path, 'rb'))
    p  = ep['obs'][:, 3:6]
    cf = ep['metadata']['contact_fraction']
    ys = [p[np.argmin(np.abs(p[:, 0] - xp)), 1] for xp in [-2.0, 0.0, 2.0]]
    htpy = ep['homotopy']

    # Check each class
    if htpy == '(L,L,L)':
        side_ok = all(y < -0.60 - 0.12 for y in ys)   # all left of column A
    elif htpy == '(R,R,R)':
        side_ok = all(y > 0.60 + 0.12 for y in ys)
    elif htpy == '(L,R,L)':
        side_ok = (ys[0] < -0.72 and ys[2] < -0.72 and ys[1] > +0.72)
    elif htpy == '(R,L,R)':
        side_ok = (ys[0] > +0.72 and ys[2] > +0.72 and ys[1] < -0.72)

    status = '✅' if cf == 0 and side_ok else '❌'
    print(f'{status} {ep["episode_id"]}  cf={cf:.4f}  y={[f"{y:+.2f}" for y in ys]}  side_ok={side_ok}')
```

### Phase 3 — Full re-collection (all 4 scenes)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty    500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  600
# pillars raised to 600 to ensure ≥500 after strict threshold filtering
```

### Phase 4 — Full dataset verification

```python
for scene in ['empty', 'corridor', 's_curve', 'pillars']:
    eps = [pickle.load(open(p, 'rb'))
           for p in glob.glob(f'logs/uav_expert_data/{scene}/**/*.pkl', recursive=True)]
    cfs     = [e['metadata']['contact_fraction'] for e in eps]
    q_ok    = all('q' in e for e in eps)
    cf_pos  = sum(c > 0 for c in cfs)
    print(f'{scene}: n={len(eps)}  q_field={q_ok}  '
          f'cf>0: {cf_pos}/{len(eps)} ({100*cf_pos/len(eps):.0f}%)')
```

**Pass criteria**:
- pillars: `cf > 0` count = 0 (zero contact), all homotopy labels physically correct
- corridor L/R: `cf > 0` count low (PID tracking error only, not by design)
- all scenes: `q` field present
- all scenes: episode counts ≥ 450

---

## Expected changes vs U2

| Item | E4 U2 | E4 U3 |
|---|---|---|
| Corridor L/R channel | y=±0.18 ± 0.05 jitter | y=±0.12, no jitter |
| Corridor contact | ~12% wall contact by design | ~0% (PID error only) |
| Pillars (L,L,L)/(R,R,R) | weave: 54% T values have contact | pillar_path: 0% contact |
| Pillars (L,R,L)/(R,L,R) | weave at y=0: WRONG LABELS | pillar_path: correct weave path |
| q field | Absent | ✅ All scenes |
| obs shape | (T, 9) | (T, 9) — unchanged |
| FM tensor | 12D | 12D — unchanged |
| Total episodes | 1769 | ~1770 ± 50 |

---

## Cross-references

| Document | Content |
|---|---|
| [`../U2/CLOSURE.md`](../U2/CLOSURE.md) | U2 final state — what U3 is based on |
| [`../../Epoch5_visual_and_validation/U2/INVESTIGATE_pillar_passthrough.md`](../../Epoch5_visual_and_validation/U2/INVESTIGATE_pillar_passthrough.md) | Full geometry audit, phase misalignment proof |
| `uav_expert_data_collect/trajectories.py` | `CORRIDOR_CHANNELS`, `pillar_path`, `_Y_L/_Y_R` |
| `uav_expert_data_collect/generator.py` | Corridor jitter, pillar weave, contact thresholds |
