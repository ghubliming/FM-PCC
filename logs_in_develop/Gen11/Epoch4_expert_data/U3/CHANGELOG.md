# Gen11 Epoch 4 — U3 Changelog

**Date**: 2026-06-08  
**Based on**: E4 U2  
**Files changed**: `uav_expert_data_collect/trajectories.py`, `uav_expert_data_collect/generator.py`  
**Usage**: Same as U2 — no command changes needed.

---

## Why U3

E5 U2 GIFs revealed drones flying through pillars. Investigation (see
`../../../Gen11/Epoch5_visual_and_validation/U2/INVESTIGATE_pillar_passthrough.md`)
found three root-cause bugs in the E4 trajectory design:

1. **Pillar channel too narrow** — `_Y_L = -0.92` placed the drone inside the rotor
   contact zone (`y ∈ (-0.55, -0.97)`). The fix in Fix_1 (switching to `weave`) avoided
   stops at y=-0.92 but introduced new problems.
2. **weave phase misalignment** — `y(t) = A·sin(2π·t/period)` with random T ∈ [10,16]s
   means the phase at each pillar x-position is T-dependent. For ~54% of T values in that
   range, the drone contacts a pillar or passes the wrong side.
3. **Wrong homotopy labels for (L,R,L)/(R,L,R)** — Fix_2 set weave amplitude to 0.0 for
   these classes (to avoid pillar zone). Amplitude=0 = straight centre line at y=0 on
   every episode. 100% of (L,R,L) and (R,L,R) labels are wrong.

U3 fixes all three bugs. The corridor L/R channels also had a contact problem (rotor 9 cm
inside wall at worst jitter) — fixed in the same pass.

---

## Change A — `uav_expert_data_collect/trajectories.py`

### A1: Pillar channel constants

```diff
-PILLAR_MARGIN   = 0.20
-_Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_MARGIN   # ≈ -0.92
-_Y_R = PILLAR_Y_B + PILLAR_RADIUS + PILLAR_MARGIN   # ≈ +0.92
+PILLAR_ROTOR_REACH = 0.31   # max y-distance from COM to rotor ellipsoid edge
+PILLAR_SAFETY      = 0.08   # 8 cm clearance above zero-contact
+_Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_ROTOR_REACH - PILLAR_SAFETY  # = -1.11
+_Y_R = PILLAR_Y_B + PILLAR_RADIUS + PILLAR_ROTOR_REACH + PILLAR_SAFETY  # = +1.11
```

Old margin (0.20 m) only cleared the pillar cylinder radius. Rotor ellipsoids extend 0.31 m
from COM in the y-direction. New formula accounts for rotor reach + 8 cm safety → 10.8 cm
clearance between rotor edge and pillar surface.

### A2: Corridor channel constants

```diff
-CORRIDOR_CHANNELS = {'L': -0.18, 'C': 0.0, 'R': +0.18}
+CORRIDOR_CHANNELS = {'L': -0.12, 'C': 0.0, 'R': +0.12}
```

At ±0.18 with ±0.05 jitter: worst-case y=-0.23, rotor edge at -0.54, wall inner face at
-0.45 → rotor 9 cm inside wall (contact). At ±0.12 with no L/R jitter: rotor at -0.43 vs
wall at -0.45 → 2 cm clearance.

### A3: `pillar_path` rewrite (5-waypoint scheme)

Replaced the old 8-waypoint `s_curve_path`-based implementation with a clean 5-waypoint
`traverse_line`-based design:

- **Waypoints at**: `x_start`, `-2.0`, `0.0`, `+2.0`, `x_end`
- **y at each pillar position**: exactly `_Y_L` or `_Y_R` per the homotopy sequence
- **Time allocation**: proportional to x-distance per segment
- **No sinusoidal phase dependence**: correct channel assignment is deterministic

This correctly routes every homotopy:
- `(L,L,L)`: passes left of all 3 pillar pairs (y=-1.11 at x=-2,0,+2)
- `(R,R,R)`: passes right of all 3 (y=+1.11)
- `(L,R,L)`: left at x=-2, right at x=0, left at x=+2 (explicit S-path)
- `(R,L,R)`: right at x=-2, left at x=0, right at x=+2

---

## Change B — `uav_expert_data_collect/generator.py`

### B1: Corridor jitter — L/R removed, C reduced

```diff
-    y_jitter = float(rng.uniform(-0.05, 0.05))
+    y_jitter = float(rng.uniform(-0.03, 0.03)) if homotopy == 'C' else 0.0
```

L/R jitter was the proximate cause of rotor-wall contact. C jitter tightened from ±0.05
to ±0.03 (still provides diversity without approaching the wall).

### B2: Pillars — weave → pillar_path

```diff
-    traj_fn = trajs.weave(
-        x_range=(-3.2, 3.2), y_amplitude=amp,
-        period=4.0, altitude=z, duration=dur,
-    )
+    seq = _seq_map[homotopy]
+    traj_fn = trajs.pillar_path(seq, altitude=z, duration=dur)
```

The `_seq_map` explicitly maps each homotopy string to its 3-element waypoint sequence.
`weave` is no longer used for the pillars scene.

### B3: Pillar contact threshold

```diff
-'pillars':  0.02,
+'pillars':  0.001,
```

With 10.8 cm rotor clearance, any contact during collection is a trajectory bug and the
episode should be rejected. The threshold 0.001 (0.1%) allows for rounding/init noise
while hard-rejecting genuine contact.

---

## Scenes unaffected

| Scene | Status | Reason |
|---|---|---|
| empty | Unchanged | No obstacle geometry; existing design is fine |
| corridor C | Jitter only (±0.05 → ±0.03) | Channel at y=0 cannot contact walls |
| s_curve | Unchanged | 14 cm rotor clearance; end-face grazes are structural |

---

## Usage

Same commands as U2. No argument changes.

```bash
# Collect all scenes (example — same as U2 collect.sh)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh 500 empty ""
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh 500 corridor ""
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh 500 s_curve ""
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh 500 pillars ""
```

After collection, verify with a quick contact fraction check:

```python
import pickle, glob, numpy as np

for f in glob.glob('logs/uav_expert_data/pillars/**/*.pkl', recursive=True):
    ep = pickle.load(open(f, 'rb'))
    cf = ep.get('contact_fraction', 0.0)
    if cf > 0.001:
        print(f'CONTACT {cf:.4f}  {f}')
```

Expected: no output (zero contact episodes accepted under new threshold).
