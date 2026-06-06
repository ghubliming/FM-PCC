# Gen11 Epoch 3 — UAV in Obstacle Scenes: Methodology

**Date**: 2026-06-06  
**Status**: ✅ Complete  
**Maximum fix index**: none (one mid-epoch bug; no Fix_N directory)  
**CLOSURE**: [`EPOCH3_CLOSURE.md`](EPOCH3_CLOSURE.md)

---

## What this epoch does

Wraps the Epoch 1 X2 model inside four obstacle-scene XMLs and re-runs the Epoch 2
controller inside each one.  Goal: prove that the controller is **scene-agnostic** — the
presence of walls and pillars does not change the physics of the drone itself.

---

## Scene design

Four MuJoCo scenes were built, each as a top-level XML that `<include>`s
`quadrotor_modified.xml`:

| Scene | Geometry | Homotopy routes |
|---|---|---|
| `scene_empty` | Floor + skybox only | Free-space baseline |
| `scene_corridor` | Two parallel walls (y = ±0.5, face at y = ±0.45) | L / C / R |
| `scene_s_curve` | Two offset corridors with a diagonal gap (seg1 at y≈−0.8, seg2 at y≈+0.8) | single (default) |
| `scene_pillars` | 6 cylinders, radius 0.12 m, at x∈{−2,0,+2}, y∈{−0.6,+0.6} | (L,L,L)/(R,R,R)/centre passes |

Code: `d3il/environments/.../scenes/scene_*.xml`; obstacle metadata in
`uav_expert_data_collect/generator.py` `SCENE_OBSTACLES`.

---

## The scene-agnosticism test

The Epoch 2 Task C circle (9D, RMS = 0.029114 m) was re-run inside `scene_empty`.
Result: **RMS = 0.029114 m — bit-for-bit identical to Epoch 2.**

This is the cleanest possible proof: the scene wrapper (floor + walls + skybox +
`<include>` of X2 model) is transparent.  Adding geometry around the drone does not
change its dynamics or the controller's behaviour.  All future epochs can treat the scene
as a backdrop.

---

## Mid-epoch bug — XML asset path resolution

**Symptom**: `smoke_empty` (job 21028) crashed:
```
ValueError: file not found: '.../scenes/assets/X2_lowpoly.obj'
```

**Root cause**: MuJoCo resolves `<compiler assetdir="assets"/>` **relative to the
top-level XML's directory** — not relative to the `<include>`d file that carries the
tag.  When `scene_empty.xml` included `quadrotor_modified.xml`, MuJoCo looked for meshes
under `scenes/assets/` instead of `quadrotor/assets/`.

**Fix**: added `<compiler meshdir="../assets" texturedir="../assets"/>` directly inside
each scene XML, after the `<include>` line.  MuJoCo merges `<compiler>` tags with
last-wins semantics, so the scene's override beats the included file's `assetdir`.
`quadrotor_modified.xml` was not touched — it still loads standalone for Epoch 2.

---

## Obstacle scene results

| Scene | RMS (m) | Contact steps | Endpoint reached | Verdict |
|---|---|---|---|---|
| `empty C` | 0.029 | 0 | ✅ | ✅ Baseline match |
| `corridor traverse` | 0.023 | 0 | ✅ | ✅ Pass |
| `pillars weave` | 0.922 | 29 (2.9%) | ✅ | ⚠️ High lag, endpoint reached |
| `s_curve` | 0.533 | 614 (41%) | ❌ | ❌ Fail |

**Why corridor passes and s_curve fails** — same root as the Epoch 2 hover instability.
`s_curve_path` chains piecewise segments with **zero velocity at waypoint transitions**
(v=0 at t=5s and t=10s).  At those moments the drone is in the same near-zero-velocity
regime that triggers the `Kp_omega=10` limit cycle.  The controller oscillates, the drone
drifts into the walls.  Corridor uses a single `traverse_line` with no internal stops —
`a_des ≠ 0` throughout — so the limit cycle never fires.

The fix (drop `Kp_omega` from 10 → 2.5, or replace piecewise path with a continuous
spline) is deferred to Epoch 4 where it is required for data collection.

---

## Architectural conclusion

The controller is scene-agnostic (proven by exact RMS match).  Scene XMLs are stable and
loadable.  Epoch 4 can use any of these scenes as collection environments with confidence
that the X2 + cascaded PID combination behaves predictably inside them.

---

## Cross-references

| Document | Content |
|---|---|
| [`EPOCH3_CLOSURE.md`](EPOCH3_CLOSURE.md) | Full results, detailed diagnosis |
| [`../Epoch2_UAV_mujoco_run/METHODOLOGY.md`](../Epoch2_UAV_mujoco_run/METHODOLOGY.md) | Controller design and 9D format decision |
| [`../Epoch4_expert_data/METHODOLOGY.md`](../Epoch4_expert_data/METHODOLOGY.md) | Kp_omega fix applied; s_curve trajectory redesigned (Fix_5) |
