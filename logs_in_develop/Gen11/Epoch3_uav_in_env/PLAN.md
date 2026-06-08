# Gen11 Epoch 3 — UAV in a Real Env: Plan

**Date**: 2026-05-31
**Branch**: `update_into_FM`
**Status**: Plan only — no code yet.
**Predecessor**: [`../Epoch2_UAV_mujoco_run/EPOCH2_CLOSURE.md`](../Epoch2_UAV_mujoco_run/EPOCH2_CLOSURE.md)
**Roadmap context**: [`../path_temp_initial.md`](../path_temp_initial.md) — this epoch executes step 2 ("Define the MuJoCo obstacle world").

---

## 1. Goal & Scope

**Goal:** put the X2 in a *real* MuJoCo scene (floor, walls, lighting,
obstacles) and verify the Epoch 2 controller still tracks trajectories
there — so the GIFs are actually legible and we have a foundation for
DPCC/FM work that needs an environment to operate in.

**In scope:**
- New scene XML(s) wrapping the Epoch 1 `quadrotor_modified.xml` with
  floor + lighting + skybox + obstacle geoms.
- 2–3 canonical layouts: empty room, corridor, S-curve (and optionally
  pillar field).
- Re-run trajectory tests in each scene, confirm controller still works
  with the scene present.
- Optional: stub function that returns obstacle half-space / signed-distance
  list from MuJoCo state (precursor for DPCC integration in Epoch 4+).

**Out of scope (explicitly deferred):**
- No DPCC projector wiring.
- No FM policy. Trajectories remain hand-coded as in Epoch 2.
- No learned obstacle avoidance — if a hand-coded trajectory clips a wall,
  that's information, not a failure to fix in Epoch 3.
- No modification to Epoch 2 code (`uav_naive_test/`, its SLURM script,
  its docs). Epoch 2 stays bit-for-bit identical.
- No edits to D3IL, `config/`, FM-PCC stacks.

---

## 2. The "Copy, Don't Rewrite" Discipline

Per user direction: **copy Epoch 2 files into a new directory, modify
there.** No rewriting from scratch. Preserves the validated controller
math and prevents drift between Epoch 2 evidence and Epoch 3 work.

| Source (Epoch 2, untouched) | Target (Epoch 3, new) | Treatment |
|---|---|---|
| `uav_naive_test/__init__.py` | `uav_env_test/__init__.py` | `cp` verbatim |
| `uav_naive_test/flight_controller.py` | `uav_env_test/flight_controller.py` | `cp` verbatim — same X2, same dynamics, controller works as-is |
| `uav_naive_test/trajectories.py` | `uav_env_test/trajectories.py` | `cp`, then **add** new factories for env-specific paths (corridor flythrough, S-curve, etc.). Existing `hover/step/circle` kept intact. |
| `uav_naive_test/smoke_load.py` | `uav_env_test/smoke_load_env.py` | `cp`, then change XML path to point at the new scene wrapper |
| `uav_naive_test/run_naive.py` | `uav_env_test/run_env.py` | `cp`, then: (a) accept `--scene {empty,corridor,s_curve,pillars}` arg, (b) load the matching scene wrapper XML, (c) log to `logs/uav_env/<scene>_<task>/` |
| `uav_naive_test/README.md` | `uav_env_test/README.md` | `cp`, then update pointers to Epoch 3 docs |
| `Slurm_Codes/sbatch/uav_naive/run_naive.sh` | `Slurm_Codes/sbatch/uav_env/run_env.sh` | `cp`, then change script path + dispatch on scene argument |

Net: **6 files copied** (with small surgical edits), **2-4 new files**
created (the scene wrappers).

---

## 3. New Files (Scene Wrappers + Optional SDF Helper)

Each scene wrapper is a thin MuJoCo XML that:
1. `<include file="../../d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml"/>`
   — pulls in the verified X2 model from Epoch 1.
2. Adds a `<visual>` block with skybox + headlight settings.
3. Adds a `<worldbody>` with a floor plane, several lights, and (per
   scene) obstacle geoms.
4. Adds a `<sensor>` block with a contact / collision flag so the driver
   can detect strikes without a vision pass.

```
d3il/environments/d3il/models/mj/robot/quadrotor/scenes/    ← NEW dir
├── scene_empty.xml          ← floor + skybox + lights only. Baseline visual sanity.
├── scene_corridor.xml       ← scene_empty + 2 parallel walls (4 m long, 1 m apart, 1.5 m tall)
├── scene_s_curve.xml        ← scene_empty + 2 offset corridor segments
└── scene_pillars.xml        ← scene_empty + 4–6 vertical cylinder obstacles
```

All four wrappers are hand-written here (small — each ~30-50 lines of XML)
because Menagerie's own `scene.xml` is a useful reference but doesn't
include obstacles. No mesh files needed — boxes and cylinders are
primitive geoms.

**Optional helper** (decide before Phase ε):
```
uav_env_test/obstacles.py     ← stub returning per-scene obstacle list
                                 [{type, center, half-extents/radius}, ...]
                                 — usable as input to a DPCC halfspace
                                 builder in Epoch 4
```

This is **read-only** w.r.t. MuJoCo state — just hard-coded geometry
matching the XML. The "extract from MjModel at runtime" version is
more elegant but defer until DPCC actually needs it.

---

## 4. Target File Tree (after Epoch 3 lands)

```
# Epoch 2 — UNTOUCHED
uav_naive_test/                        ← unchanged
Slurm_Codes/sbatch/uav_naive/          ← unchanged
logs/uav_naive/                        ← unchanged (Epoch 2 evidence)

# Epoch 3 — NEW
uav_env_test/                          ← NEW, code lives here
├── __init__.py
├── flight_controller.py               ← cp from uav_naive_test (verbatim)
├── trajectories.py                    ← cp + new factories
├── smoke_load_env.py                  ← cp + scene-path swap
├── run_env.py                         ← cp + scene-dispatch
├── obstacles.py                       ← NEW (optional SDF stub)
└── README.md

d3il/environments/d3il/models/mj/robot/quadrotor/scenes/   ← NEW
├── scene_empty.xml
├── scene_corridor.xml
├── scene_s_curve.xml
└── scene_pillars.xml

Slurm_Codes/sbatch/uav_env/            ← NEW
└── run_env.sh                         ← cp from uav_naive + scene-dispatch

logs/uav_env/                          ← NEW (runtime, gitignored)
├── empty_circle_9D/{log.json, metrics.txt, rollout.gif, ...}
├── corridor_traverse/{...}
├── s_curve/{...}
└── pillars_weave/{...}

logs_in_develop/Gen11/Epoch_3_uav_in_env/   ← NEW (docs)
├── PLAN.md                            ← this file
├── EXECUTION_PLAN.md                  ← (next, after this plan is approved)
├── RUNBOOK.md                         ← (during phases)
└── CHANGELOG.md                       ← (closure)
```

---

## 5. Phased Execution (parallel to Epoch 2's α–η)

| Phase | Deliverable | Pass condition |
|---|---|---|
| 3-α | Copy Epoch 2 files into `uav_env_test/`, syntax-check | Files in place, py/bash syntax clean |
| 3-β | Write `scene_empty.xml` (floor + lights + skybox, no obstacles) | `smoke_load_env.py --scene empty` runs on Slurm, prints `nq nv nu`, drone falls onto the floor (terminal `qpos_z ≈ 0.0` instead of free-fall) |
| 3-γ | Re-run **Task C circle 9D** in `scene_empty` | RMS within 2× of Epoch 2's 0.029 m result (floor + walls shouldn't change controller behavior). GIF now shows drone + floor + sky — legible. |
| 3-δ | Write `scene_corridor.xml` + new `traverse_corridor` trajectory factory | `run_env.py --scene corridor --task traverse` produces a GIF where the drone flies between the walls without contact |
| 3-ε | Write `scene_s_curve.xml` + matching trajectory | Same — drone threads the curve |
| 3-ζ | Write `scene_pillars.xml` + weaving trajectory | Same — drone weaves through pillars |
| 3-η | (Optional) `obstacles.py` SDF stub | Returns the correct list of obstacles per scene; ready for DPCC consumption |
| 3-θ | Closure changelog | What landed, what passed, GIFs preserved |

**3-α → 3-γ are the must-haves.** 3-δ through 3-ζ are scope expansion;
ship as many as time allows. 3-η defers to Epoch 4 if not done here.

---

## 6. New Trajectory Factories (in `trajectories.py`)

Additions on top of the Epoch 2 set (`hover_at`, `step_to`, `circle`):

| New factory | Purpose | Scene |
|---|---|---|
| `traverse_line(p_start, p_end, duration, max_v, max_a)` | Smooth point-to-point with bounded velocity/acceleration. Trapezoidal velocity profile in 1D, projected. | corridor |
| `s_curve_path(waypoints, segment_durations)` | Piecewise-linear interpolated through waypoints with quintic blends between segments | s_curve |
| `weave(centers, radius, period)` | Sinusoidal weave between two columns of pillars | pillars |

All return `(p, v, a, yaw)` per Epoch 2 convention. 9D format only (Epoch
2 closure locked this in).

---

## 7. Scene XML Sketch (illustrative — actual XML to be hand-written in 3-β)

```xml
<!-- scene_empty.xml -->
<mujoco model="UAV Empty Scene">
  <include file="../quadrotor_modified.xml"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="32" height="32"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5"/>
  </asset>

  <worldbody>
    <light name="overhead" pos="0 0 5" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 10 0.05" material="groundplane"/>
  </worldbody>
</mujoco>
```

`scene_corridor.xml` adds 2 wall box geoms; `scene_pillars.xml` adds 4-6
cylinder geoms; `scene_s_curve.xml` adds 4 wall segments. All ~30 lines
each. Hand-written from MuJoCo XML reference — no LLM-synthesized model
content beyond the boilerplate above (which is straight from Menagerie's
own `scene.xml` template and the MuJoCo docs).

---

## 8. Pass / Fail Criteria for Epoch 3 as a Whole

**Must pass for Epoch 3 to close:**
1. ✅ All 6 copied files land in `uav_env_test/` with syntax clean.
2. ✅ `scene_empty` smoke load: drone settles onto the floor (no NaN, no infinite fall).
3. ✅ Task C circle 9D in `scene_empty`: RMS within 2× of Epoch 2 baseline (0.029 m × 2 = 0.058 m ceiling).
4. ✅ GIFs are visually legible (floor + drone + sky visible).
5. ✅ At least **one** obstacle scene (corridor / s_curve / pillars) runs end-to-end with the drone visibly traversing it.

**Should pass (scope expansion):**
6. All three obstacle scenes work.
7. Obstacle stub `obstacles.py` returns geometry usable by a future DPCC builder.

**Closure threshold:** all 5 must-pass items met. If 5/5 hit, Epoch 3
closes regardless of whether the should-pass items landed.

---

## 9. Estimated Effort

| Phase | Time |
|---|---|
| 3-α copy + syntax check | 20 min |
| 3-β scene_empty + smoke load | 30 min |
| 3-γ re-run circle in empty scene | 20 min (Slurm wait + verify) |
| 3-δ corridor scene + trajectory | 1 h |
| 3-ε s_curve scene + trajectory | 1 h |
| 3-ζ pillars scene + trajectory | 1 h |
| 3-η obstacles stub (optional) | 1 h |
| 3-θ closure changelog | 30 min |
| **Total to all-scenes-pass** | **~5-6 h** |
| Minimum-viable Epoch 3 (α + β + γ + one obstacle) | **~2-3 h** |

---

## 10. Decisions Needed Before Phase α

| # | Question | Default if not specified |
|---|---|---|
| D1 | Implement all 3 obstacle scenes or just corridor for now? | All 3 (small XML each, low risk) |
| D2 | Include `obstacles.py` SDF stub in Epoch 3 or defer to Epoch 4 (where DPCC actually needs it)? | Defer — Epoch 4 owns the consumer |
| D3 | Camera POV for the GIFs — keep Epoch 2's `track` cam (chases the drone) or add a fixed cinematic angle per scene? | Keep `track` for tracking shots; optionally add a 2nd fixed cam later |
| D4 | Scene size — 10 m × 10 m floor enough, or go bigger for the s-curve? | 10×10 (existing trajectories all stay within ±2 m of origin) |
| D5 | Should `run_env.py` warn / error on contact (drone hits a wall), or just record it silently in the log? | Warn only; never abort. Contact is information, not a fatal error. |

---

## 11. Bottom Line

Epoch 3 = **Epoch 2 controller, copied verbatim, dropped into 2-4
hand-written MuJoCo scenes, re-run on the same Slurm pipeline.** No new
control theory, no learning, no rewriting. Just env construction + visual
validation.

If Phase 3-γ shows Task C 9D RMS within 2× of Epoch 2's 0.029 m,
**we've proven the controller is scene-agnostic** and any subsequent
work (DPCC obstacle avoidance, FM policy outputs threaded through the
env) has a stable substrate to build on.

Ready to draft EXECUTION_PLAN.md and start Phase 3-α on greenlight.
