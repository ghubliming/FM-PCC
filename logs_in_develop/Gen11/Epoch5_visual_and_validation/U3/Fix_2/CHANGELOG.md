# Gen11 Epoch 5 U3 Fix_2 — Two residual bugs after E4 completion

**Date:** 2026-06-10  
**Triggered by:** E4 dataset complete (U7 F7: all 4 scenes pass gate) → E5 GIF runs unblocked  
**Status:** ✅ Fixed

---

## Bug 1 — `collect_camera_images.py` still used old chase camera (`_TRACK_CAM_NAME = 'track'`)

### Symptom

U3 Fix_1 updated `generate_trajectory_gifs.py` and `generate_physics_gifs.py` to use the
correct nose-mounted FPV camera (`fpv`), but `collect_camera_images.py` was missed. If WS-A
had been run, every image in `track-cam/` would be a 3rd-person chase view from 1 m behind
the drone — the exact wrong camera — mislabelled as FPV.

### Fix

`uav_expert_data_collect/collect_camera_images.py` line 60:

```diff
- _TRACK_CAM_NAME = 'track'
+ _TRACK_CAM_NAME = 'fpv'
```

Comment updated to match.

---

## Bug 2 — `run_env.py` task `'s_curve'` still used the infeasible diagonal

### Symptom

`uav_env_test/run_env.py` task `'s_curve'` waypoints included the old diagonal leg
`(-0.5, -0.8, 0.75) → (0.5, +0.8, 0.75)` which passes **0.019 m inside** the rotor-contact
zone of both gap-side wall corners (corner A=(−0.5,−0.25), corner B=(+0.5,+0.25)).

E4 U7 F7 confirmed: running the task evaluator with `--task s_curve` produced
**40.9% contact fraction** and **1.563 m final position error** — not a controller failure,
purely the wrong path. The U7 Z-route fix (E4 `uav_expert_data_collect/trajectories.py`)
was never propagated to the task evaluator.

### Fix

`uav_env_test/run_env.py` `build_task()` `name == 's_curve'` block:

**Before (3-leg, diagonal):**
```python
waypoints=[(-3.0, -0.8, 0.75),
           (-0.5, -0.8, 0.75),
           ( 0.5,  0.8, 0.75),   # infeasible diagonal
           ( 3.0,  0.8, 0.75)],
segment_duration=5.0,            # 3 × 5 s = 15 s
```

**After (5-leg Z-route):**
```python
waypoints=[(-3.0, -0.8, 0.75),
           (-0.5, -0.8, 0.75),
           ( 0.0, -0.8, 0.75),   # Leg B1: pure-x to centerline
           ( 0.0,  0.8, 0.75),   # Leg B2: pure-y at x=0
           ( 0.5,  0.8, 0.75),   # Leg B3: pure-x to corridor 2
           ( 3.0,  0.8, 0.75)],
segment_duration=3.0,            # 5 × 3 s = 15 s (duration unchanged)
```

Z-route clearances: Leg B2 at x=0 runs 0.50 m from both corners (rotor reach=0.31 m).

`_resolve_camera()` also updated to prefer `'fpv'` over `'track'` so the task eval
rollout GIF uses the correct FPV camera:

```diff
- for i in range(model.ncam):
-     if model.camera(i).name == 'track':
-         return 'track'
+ for name in ('fpv', 'track'):
+     for i in range(model.ncam):
+         if model.camera(i).name == name:
+             return name
```

---

## Files changed

| File | Change |
|------|--------|
| `uav_expert_data_collect/collect_camera_images.py` | `_TRACK_CAM_NAME`: `'track'` → `'fpv'` |
| `uav_env_test/run_env.py` | s_curve waypoints: 5-leg Z-route; `_resolve_camera`: prefers `'fpv'` |

---

## Expected outcome

After these fixes, rerunning `run_env.py --scene s_curve --task s_curve` should show:
- contact_fraction ≈ 0 (matching the E4 collection result of 0% rejection)
- final_pos_err < 0.1 m (clean arrival at (3.0, 0.8, 0.75))
- rollout GIF rendered from nose-mounted FPV camera

WS-A camera images will now correctly capture the forward-facing FPV view, consistent
with `generate_trajectory_gifs.py` and `generate_physics_gifs.py`.
