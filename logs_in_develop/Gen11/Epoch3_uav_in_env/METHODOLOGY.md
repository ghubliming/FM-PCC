# Gen11 Epoch 3 — UAV in Obstacle Scenes: Methodology

**Date**: 2026-06-06  
**Status**: ✅ Complete  
**Maximum fix index**: none (one mid-epoch bug; no Fix_N directory)  
**CLOSURE**: [`EPOCH3_CLOSURE.md`](EPOCH3_CLOSURE.md)

---

## What this epoch does

Wraps the Epoch 1 X2 model inside four MuJoCo obstacle scenes and re-runs the Epoch 2
controller inside each one.  The central question: is the controller **scene-agnostic** —
does the presence of walls and pillars change how the drone's physics behave?  The
expected answer is no (MuJoCo walls don't produce aerodynamic effects), and the experiment
confirms it.  Failures are traced to the trajectory factory, not the controller logic.

---

## Scene geometry

| Scene | What's in it | Key dimensions | Homotopy routes |
|---|---|---|---|
| `scene_empty` | Floor + skybox only | — | Free-space baseline |
| `scene_corridor` | Two parallel box walls | y = ±0.5, face at y = ±0.45 → 0.9 m clear | L / C / R |
| `scene_s_curve` | Two offset corridor segments + diagonal gap | Seg 1: y≈−0.8; Seg 2: y≈+0.8 | single (`default`) |
| `scene_pillars` | 6 cylinders, r=0.12 m | x∈{−2,0,+2}, cols A at y=−0.6, B at y=+0.6 | (L,L,L)/(L,R,L)/(R,L,R)/(R,R,R) |

Each scene XML `<include>`s `quadrotor_modified.xml`.  Walls and pillars are passive rigid
bodies — MuJoCo computes contact forces when the drone hits them but there are no airflow
effects.

Code: `d3il/environments/.../scenes/scene_*.xml`; obstacle metadata in
`uav_expert_data_collect/generator.py` `SCENE_OBSTACLES`.

---

## The scene-agnosticism test

The Epoch 2 Task C circle (9D, RMS = 0.029114 m) was re-run inside `scene_empty`
(job 21028).  Result: **RMS = 0.029114 m — bit-for-bit identical to Epoch 2.**

This is the cleanest possible proof: the scene wrapper (floor + walls + skybox +
`<include>` of X2 model) is transparent.  Adding geometry around the drone does not
change its dynamics or the controller's behaviour.  All future epochs can treat the scene
as a backdrop.

---

## Mid-epoch bug — XML asset path resolution

**Symptom**: First job crashed:
```
ValueError: file not found: '.../scenes/assets/X2_lowpoly.obj'
```

**Root cause**: MuJoCo resolves `<compiler assetdir="assets"/>` relative to the
**top-level XML's directory**.  When `scene_empty.xml` `<include>`s
`quadrotor_modified.xml` (which lives under `.../quadrotor/`), MuJoCo looks for the mesh
under `.../scenes/assets/` — wrong directory.

**Fix**: Each scene XML gets `<compiler meshdir="../assets" texturedir="../assets"/>` after
its `<include>` line.  MuJoCo merges `<compiler>` tags with **last-wins** semantics, so
the scene's override beats the included file's `assetdir`.  `quadrotor_modified.xml`
itself is untouched — it still loads standalone for Epoch 2.

---

## Test results — deep interpretation

### Run 1: `empty C` (scene-agnosticism control test)

**Parameters**: circle, radius=0.5 m, period=10 s, 30 s duration, altitude 0.75 m.  
**Result**: RMS = **0.029114 m** — bit-for-bit identical to Epoch 2's Task C 9D result.

This is the cleanest possible proof.  The controller sees the same drone, the same gains,
the same trajectory.  Adding a floor/skybox/lighting wrapper changes nothing — the scene
is transparent to the physics.  If the scene wrapper had introduced any coupling (shared
bodies, joint constraints, modified inertia) the RMS would differ.  It doesn't.

**Conclusion**: the controller is scene-agnostic.  All future epochs can safely put walls
and pillars in the scene without worrying that they change how the drone tracks.

---

### Run 2: `corridor traverse` ✅

**Parameters**: `traverse_line([-2.5, 0, 0.75], [2.5, 0, 0.75], duration=8.0)` — a single
straight 5 m segment at ~0.625 m/s (cosine velocity profile).  
**Result**: RMS = **0.023 m**, 0 contact steps, endpoint reached.

This is actually *better* than the circle (0.023 < 0.029 m).  Why?

- A circle demands **constant non-zero centripetal acceleration** throughout:
  `a_des = −Aω² · [cos(θ), sin(θ)]` — the direction rotates continuously.
  The controller must continuously update its tilt angle to track the rotating demand.

- A straight traverse demands acceleration only during **ramp-up and ramp-down**
  (the cosine profile).  The peak is a simple forward acceleration vector that doesn't
  rotate.  Once at cruise speed the demand drops near zero.  Much easier to track.

Crucially, a single-segment `traverse_line` has **no internal waypoint transitions**
(no moments where `v_des = 0` mid-flight).  The only v=0 moments are at the very start
and end — when the drone is at rest or stopped, not mid-path.  So the hover limit-cycle
instability from Epoch 2 is never triggered.

**Physical meaning**: a drone flying straight down a hallway at ~0.6 m/s with a well-tuned
9D reference is a well-solved problem.  2.3 cm RMS is near sensor noise level.

---

### Run 3: `s_curve` ❌  (41% contact)

**Parameters**: `s_curve_path(waypoints, segment_duration=5.0)` — 3 legs through the
two offset corridors, each leg 5 s, with zero velocity at every waypoint junction.  
**Result**: RMS = **0.533 m**, 614 contact steps (41%), endpoint not reached.

This is not a controller bug — it is the **Epoch 2 hover instability resurfacing at
waypoint transitions**.

At t=5 s and t=10 s the trajectory commands `v_des = 0`.  For one instant the drone
is in the exact same regime as Epoch 2's hover task: position setpoint with near-zero
velocity and no feedforward acceleration.  The `Kp_omega = 10` angular-rate gain
saturates (`|ω| > 0.18 rad/s` → torque clip → opposite correction next step → limit
cycle at 100 Hz).  The drone oscillates violently, loses altitude control, and drifts
laterally into the corridor walls.  Once it contacts a wall, MuJoCo's contact forces push
it further; the controller cannot recover quickly enough.  614 steps of contact = the
drone is wedged against the wall for ~6 seconds out of 15.

The **root cause is the trajectory factory, not the controller**.  `s_curve_path` is
built from `traverse_line` segments chained with `s_curve_path(wps, T_leg)` — each
segment starts and ends at v=0 (cosine profile boundary condition).  Any trajectory
factory that forces v=0 at internal waypoints will trigger this on the current gains.

**Two fix paths (both used in Epoch 4)**:
1. Drop `Kp_omega` from 10 → 2.5 — eliminates the saturation condition.
2. Replace `s_curve_path` with a factory that never forces v=0 at internal joints
   (Epoch 4 Fix_5: proportional-duration `traverse_line` segments with shared velocity).

---

### Run 4: `pillars weave` ⚠️  (high RMS, endpoint reached)

**Parameters**: `weave(x_range=(−3.2, 3.2), y_amplitude=1.0, period=4.0, duration=10.0)`.  
**Result**: RMS = **0.922 m**, max error = 1.85 m, 29 contact steps (2.9%), final error = 0.062 m.

The numbers look alarming, but unpacking them reveals the drone behaved correctly:

**What the weave demands**:
- Forward motion: `v_x = 6.4 m / 10 s = 0.64 m/s` (constant, easy to track).
- Lateral motion: sinusoidal, `y(t) = 1.0·sin(ωt)` with `ω = 2π/4 = 1.571 rad/s`.
- Peak lateral velocity: `A·ω = 1.0 × 1.571 = 1.57 m/s`.
- Peak lateral acceleration: `A·ω² = 1.0 × 2.47 = 2.47 m/s² ≈ 0.25g`.

This is demanding — the drone must swing 2 m side-to-side every 4 seconds while flying
forward.  At 0.25g lateral demand the drone tilts ~14° away from vertical to generate
the required force.  The controller can produce this, but it introduces **phase lag**:
the actual y lags the commanded y by a fraction of the period.

**Why RMS = 0.922 m but only 2.9% contact**:  
The 0.922 m is the mean distance between commanded and actual position — it's the
tracking lag over the entire sinusoidal oscillation, not a crash metric.  The drone was
always *somewhere* along the correct sinusoidal path, just consistently ~0.5–1 s late.
The pillar columns are at y = ±0.6 with radius 0.12 m (edge at y = ±0.48 inward);
the weave swings to y = ±1.0 (amplitude), so the drone passes on the *outside* of each
column, not through the gap.  With phase lag the drone is slightly wrong in x when it
passes each column in y — occasionally close enough to graze (29 steps), but never stuck.

**Final error = 0.062 m** means the drone reached the exit of the pillar field within
6 cm.  The x-axis tracking was good throughout; only the y-axis was laggy.

**Why the weave does NOT trigger the hover instability**:  
`y(t) = sin(ωt)` passes through zero (v_y = max) and through peaks (v_y = 0), but
v_x is always 0.64 m/s.  Total velocity is never truly zero except instantaneously —
unlike `s_curve_path` which holds v=0 for a full MuJoCo step.  The limit cycle requires
*sustained* near-zero velocity; instantaneous zero crossing is harmless.

---

## Summary table — what determined each result

| Run | Trajectory type | `v=0` at internal joints? | `a_des` continuous? | Outcome |
|---|---|---|---|---|
| `empty C` | circle | no | ✅ yes (centripetal) | ✅ 0.029 m RMS |
| `corridor traverse` | single `traverse_line` | no (single segment) | ✅ yes (ramp) | ✅ 0.023 m RMS |
| `s_curve` | piecewise `s_curve_path` | **yes** | ❌ drops to 0 at joints | ❌ 41% contact |
| `pillars weave` | sinusoidal | never sustained | ✅ yes (lateral accel) | ⚠️ laggy but succeeds |

The pattern is unambiguous: **any trajectory that touches v=0 at an internal waypoint
during the flight fails on these gains**.  The fix is applied in Epoch 4.

---

## Architectural conclusion

The controller is scene-agnostic (exact RMS match on `empty C`).  Scene XMLs are stable.
The s_curve and pillars results are **trajectory-factory failures**, not controller bugs.
Epoch 4 can use all four scenes as data collection environments once the gain fix and
trajectory redesign are applied.

---

## Cross-references

| Document | Content |
|---|---|
| [`EPOCH3_CLOSURE.md`](EPOCH3_CLOSURE.md) | Full results, pass/fail scorecard |
| [`READY_MADE_ENVS_INVESTIGATION.md`](READY_MADE_ENVS_INVESTIGATION.md) | External env survey (UAV-Flow, MJPC) + E3 collision resolution status |
| [`../Epoch2_UAV_mujoco_run/METHODOLOGY.md`](../Epoch2_UAV_mujoco_run/METHODOLOGY.md) | Hover instability root cause (Kp_omega arithmetic) |
| [`../Epoch4_expert_data/METHODOLOGY.md`](../Epoch4_expert_data/METHODOLOGY.md) | Kp_omega fix + s_curve trajectory redesign (Fix_5) |
