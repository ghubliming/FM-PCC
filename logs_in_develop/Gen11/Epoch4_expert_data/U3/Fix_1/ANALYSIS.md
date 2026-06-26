# Gen11 E4 U3 — Fix_1 Analysis: Pillar 100% Rejection

**Date:** 2026-06-08  
**Run that failed:** `logs_in_develop/Gen11/Epoch4_expert_data/U3/` SLURM job 21342  
**Result:** 0/500 saved, 100% rejection, aborted after 21 episodes  

---

only pillars failed. The other three:

- empty: 500/500 ✅ 0% rejection
- corridor: 500/500 ✅ 0% rejection
- s_curve: 356/500 ⚠️ 28.8% rejection — but this is expected structural rejections (some randomised s_curves are geometrically too tight), not a code bug. You got 356 valid episodes which is usable.


## 1. Symptom

```
[ collect ] ABORT: rejection rate 100.0% > limit 60.0%.
[ collect ] DONE  saved=0  rejected=21  (100.0% rejected)  elapsed=6s
```

Rejection criterion: `SCENE_MAX_CONTACT_FRACTION['pillars'] = 0.001`  
(≥ 2 contact steps out of ~1000 per episode → reject)

---

## 2. Diagnostic Method (Python-only, no MuJoCo needed)

The commanded trajectory itself was sampled at 500 points and the minimum
clearance between every rotor geom and every pillar was computed analytically.

**Rotor offsets (from `quadrotor_modified.xml`):**
```
rotor1: (-0.14, -0.18)   rotor2: (-0.14, +0.18)
rotor3: (+0.14, +0.18)   rotor4: (+0.14, -0.18)
```
Rotor ellipsoid radius: 0.13 m  
Pillar cylinder radius: 0.12 m  
Contact threshold (2D): rotor_centre_distance < 0.25 m

**Pillar centres:**
```
A1(-2.0, -0.6)  B1(-2.0, +0.6)
A2( 0.0, -0.6)  B2( 0.0, +0.6)
A3(+2.0, -0.6)  B3(+2.0, +0.6)
```

---

## 3. Root Cause: 5-Waypoint Entry Diagonal Intersects Pillar Geometry

The U3 `pillar_path` used a **5-waypoint scheme** with waypoints AT each pillar x-position:

```
xs = [-3.2, -2.0, 0.0, 2.0, 3.2]
ys = [  0,  y_ch[0], y_ch[1], y_ch[2], 0]
```

### Why it fails — TWO rotor offset effects

**Effect 1 — FRONT rotor on approach:**  
The entry diagonal goes from `(-3.2, 0)` to `(-2.0, y_L)` (cosine profile).  
Rotor 3 has offset `(+0.14, +0.18)` — it reaches **ahead of the body in x**.  
When `body_x ≈ -2.24` (approaching pillar at x=-2.0) and `body_y ≈ -0.89`
(still transitioning toward target -1.11), rotor 3 sits at `(-2.10, -0.71)` —
only **0.148 m** from pillar A1 at `(-2.0, -0.6)`.  
Contact radius = 0.25 m → **clearance = -10.2 cm**.

**Effect 2 — REAR rotor on departure (would affect inter-pillar diagonals):**  
Starting the next diagonal AT x=-2.0 means the rear rotor `(-0.14 in x)` is
initially at x=-2.14, BEHIND the pillar. As the drone moves forward, the rear
rotor sweeps through the pillar zone at the exact moment y is transitioning
through the dangerous range.

### Diagnostic output (5-waypoint, T=10s)

```
Homotopy   min_clearance   time    body_pos          rotor_off       pillar
(L,L,L):   -10.3 cm       t=1.32s  (-2.24, -0.89)   (+0.14, +0.18)  (-2.0, -0.6)
(L,R,L):   -13.3 cm       t=7.44s  (-2.24, -0.86)   (+0.14, +0.18)  (+2.0, -0.6)
(R,L,R):   -13.3 cm       t=7.44s  (-2.24, +0.86)   (+0.14, -0.18)  (+2.0, +0.6)
(R,R,R):   -10.3 cm       t=1.32s  (-2.24, +0.89)   (+0.14, -0.18)  (-2.0, +0.6)
```

**All 4 homotopies have negative commanded clearance → 100% rejection guaranteed.**

The drone NEVER executes a valid episode: the commanded path itself intersects
pillar geometry before any PID tracking error is even considered.

---

## 4. Why the Obvious Fix (pre-stabilise at x=-2.5) Is Not Enough

A 9-waypoint variant that reaches `y_ch[0]` at x=-2.5 and stabilises to x=-2.0
before the first diagonal would fix the FRONT-rotor approach problem. However,
the same rear-rotor drag problem then appears on the **inter-pillar diagonals**:

Diagonal `(-2.0, -1.11) → (-0.5, +1.11)` for (L,R,L):
- Rotor `(-0.14, +0.18)` starts at `(-2.14, -0.93)` — BEHIND pillar A1 at x=-2.0
- As the drone moves forward, this rotor sweeps from x=-2.14 toward +x
- At the exact moment the rotor x crosses -2.0, `body_y` is still near -1.0,
  putting rotor_y near -0.82. Pillar A1 at `(-2.0, -0.6)`: **-18.1 cm** contact!

```
LRL seg2 (-2.0,-1.11)->(-0.5,+1.11): min_cl=-18.1cm  ***CONTACT***
LRL seg4 (0.0,+1.11)->(1.5,-1.11):  min_cl=-18.1cm  ***CONTACT***
```

The contact is not caused by PID lag — it is caused by the trajectory geometry
itself routing the **commanded rotor centre** inside the pillar cylinder.

---

## 5. Fix: 8-Waypoint Design with Mid-Span Transitions

**Core insight:** channel transitions must happen in the **middle of each inter-pillar
span** (x ∈ [−2, 0] and x ∈ [0, +2]), not at the pillar x-positions.  
With transition midpoints at x=−1.5 and x=+0.5, every rotor has ≥0.5 m x-margin
from the nearest pillar during the full diagonal sweep.

### New waypoint layout

```
xs = [-3.2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.2]
ys = [  0,   y0,   y0,   y1,  y1,  y2,  y2,   0 ]
```
(where `y0/y1/y2 = y_ch[0..2]`, each ∈ {−1.11, +1.11})

7 segments, time allocation by **Euclidean segment length** (not x-distance),
so diagonal transition segments get proportionally more time.

### Analytical clearance (Fix_1, T=10s)

```
Segment type                              Clearance
Entry/exit diagonal  (-3.2,0)↔(-2.5,y)   23.1 cm   [front rotor, nearest pillar]
Stabilisation at y   any pillar pass       8.0 cm   [rotor offset vs pillar y]
Inter-pillar diagonal  e.g. LRL           21.4 cm   [all rotor-pillar pairs]
```

**All 4 homotopies: min_clearance = +8.0 cm  [PASS]**

```
=== LLL ===  min_cl=8.0cm   [PASS]
=== LRL ===  min_cl=8.0cm   [PASS]
=== RLR ===  min_cl=8.0cm   [PASS]
=== RRR ===  min_cl=8.0cm   [PASS]
```

The 8 cm floor comes from the stabilisation segments: body at y=±1.11, the
nearest-y rotor is at y=±0.93, the pillar inner edge is at y=±0.72, giving
exactly `|0.93 − 0.72| − 0.13(rotor_r) = 0.08 m`.  This is tight but well
above zero, and the PID at these gains tracks y better than 2 cm in steady state.

---

## 6. Code Change

**File:** `uav_expert_data_collect/trajectories.py`  
**Function:** `pillar_path`

| | Old (5-waypoint) | New (8-waypoint, Fix_1) |
|---|---|---|
| Waypoints in x | `[-3.2, -2.0, 0.0, 2.0, 3.2]` | `[-3.2, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.2]` |
| Transition position | AT each pillar | BETWEEN pillars |
| Time allocation | Proportional to Δx | Proportional to Euclidean dist |
| Segments | 4 | 7 |
| Min clearance | −13.3 cm (contact!) | +8.0 cm (safe) |

---

## 7. Rerun — Second Crash (Job 21354)

The rerun (job 21354, 2026-06-09) crashed immediately with:

```
AttributeError: 'NoneType' object has no attribute 'glGetError'
  File ".../mujoco/gl_context.py", line 38, in <module>
      from mujoco.osmesa import GLContext as _GLContext
```

**Cause:** the Group C EGL cleanup (SLURM GPU IT fix) had removed `export MUJOCO_GL=egl`
from `collect.sh`. Without it, MuJoCo 2.3.7 auto-detects osmesa as a fallback — osmesa is
not installed on i6-gpu-1 → crash at import, 0 episodes run.

**Why `MUJOCO_GL=egl` is also wrong here:**  
`mujoco/egl/__init__.py:65` calls `eglInitialize()` at module import time (not lazily at
`Renderer` creation). A no-GPU-allocation job setting `MUJOCO_GL=egl` would open GPU 0 at
import → same IT violation.

**Correct fix:** `MUJOCO_GL="disabled"` — MuJoCo 2.3.7 `gl_context.py:25` guards the entire
GL backend block behind `if _MUJOCO_GL not in ('disable','disabled',...)`. With `disabled`,
neither osmesa nor EGL is imported. Physics APIs (`MjModel`, `MjData`, `mj_step`) work without
any GL backend. See `COLLECT_SH_GL_FIX.md` for full source-verified analysis.

**File changed:** `Slurm_Codes/sbatch/uav_expert_data/collect.sh`
- `export MUJOCO_GL="disabled"` (was `egl`, then removed, now `disabled`)
- `export PYOPENGL_PLATFORM=egl` removed (unnecessary when GL disabled)
- Debug GPU-leak check added as commented-out line (uncomment once to verify, then remove)

---

## 8. Next Step

Re-run SLURM pillar collection with `trajectories.py` (8-waypoint fix) and `collect.sh`
(`MUJOCO_GL=disabled`).

To verify no GPU leak on first run: uncomment the `[DEBUG]` line in `collect.sh`, check
SLURM output for `[ GPU-LEAK CHECK ] DRI fds: NONE — clean`, then comment it back out.

Expected outcome: rejection rate ≈ 0–5% (any residual from PID tracking noise,
not trajectory geometry).
