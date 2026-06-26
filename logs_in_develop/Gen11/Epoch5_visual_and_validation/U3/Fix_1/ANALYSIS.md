# Gen11 Epoch5 U3 Fix_1 — FPV Camera is Actually a Chase Camera

**Date:** 2026-06-09  
**Symptom:** The "FPV" panel in both GIFs looks like a 3rd-person view from behind
the drone, not an on-board forward-facing camera.  
**Root cause:** `track` camera in `quadrotor_modified.xml` is positioned 1 m BEHIND
the drone body, not at the nose.  
**Status:** Fix planned — XML + two GIF scripts

---

## 1. Root Cause

All four scenes (`empty`, `corridor`, `s_curve`, `pillars`) include the same model:

```
d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_*.xml
  └── <include file="../quadrotor_modified.xml"/>
```

Current camera definition in `quadrotor_modified.xml:35`:

```xml
<camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2"/>
```

**Breaking down the current camera:**

| Attribute | Value | Meaning |
|---|---|---|
| `pos` | `-1 0 .5` | **1 m behind** body origin, 0.5 m above — body frame |
| `xyaxes` | `0 -1 0 1 0 2` | cam-x = (0,−1,0); cam-y = (1,0,2) unnormalised |
| `mode` | (default = `fixed`) | Body-fixed, rotates with the drone |

**Look-direction derivation:**

```
cam_x = (0, −1, 0)            ← body rightward
cam_y = (1, 0, 2) / √5        ← forward + 2×up (unnormalised)

cam_z = cam_x × cam_y         ← MuJoCo camera looks in −cam_z
      = (0,−1,0) × (0.447, 0, 0.894)
      = (−0.894, 0, 0.447)

look = −cam_z = (+0.894, 0, −0.447)  ← forward, slightly DOWN in body frame
```

The camera sits **1 m behind** the drone, **0.5 m above**, looking **forward and
slightly downward**. That is exactly a "rear-chase" 3rd-person camera, not FPV.
The drone's tail section always appears in the lower half of the frame.

---

## 2. Correct FPV Definition

A true FPV (first-person view) camera:
- Mounted near the nose of the drone (body +x direction)
- Looks exactly forward along the drone's +x axis
- Rotates rigidly with the body (`mode="fixed"`)

**Desired geometry:**

```
look direction  = +x_body = (1, 0, 0)
camera up (y)   = +z_body = (0, 0, 1)
camera right (x) = cam_up × look
                 = (0,0,1) × (−1,0,0) = (0, −1, 0)    ← body rightward = cam right
```

Verify cam_z = cam_x × cam_y = (0,−1,0) × (0,0,1) = (−1, 0, 0)  
look = −cam_z = (+1, 0, 0) ✓

**Resulting camera XML:**

```xml
<camera name="fpv" pos="0.1 0 0.06" xyaxes="0 -1 0 0 0 1" mode="fixed"/>
```

| Attribute | Value | Meaning |
|---|---|---|
| `pos` | `0.1 0 0.06` | 10 cm forward from body origin, 6 cm above — nose-mount position |
| `xyaxes` | `0 -1 0 0 0 1` | cam-x = body right (0,−1,0); cam-y = body up (0,0,1) |
| `mode` | `fixed` | Body-fixed, rotates with drone — camera pitches/rolls with the drone |

The `mode="fixed"` means when the drone pitches forward to accelerate, the view tilts
forward too — this is authentic FPV behaviour.

---

## 3. Changes Required

### Change A — `quadrotor_modified.xml`

**File:** `d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml:35`

```diff
- <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2"/>
+ <camera name="track" pos="-1 0 .5" xyaxes="0 -1 0 1 0 2"/>
+ <camera name="fpv"   pos="0.1 0 0.06" xyaxes="0 -1 0 0 0 1" mode="fixed"/>
```

Keep `track` in place — do not remove it. Add `fpv` alongside it. This preserves
any existing code that references `track`.

### Change B — `generate_trajectory_gifs.py`

**File:** `uav_expert_data_collect/generate_trajectory_gifs.py`

```diff
- _TRACK_CAM_NAME = 'track'
+ _TRACK_CAM_NAME = 'fpv'
```

Also update the overlay label (line ~176):
```diff
- _burn_overlay(track_bgr, 'FPV')
+ _burn_overlay(track_bgr, 'FPV-onboard')
```

### Change C — `generate_physics_gifs.py`

**File:** `uav_expert_data_collect/generate_physics_gifs.py`

```diff
- _TRACK_CAM_NAME   = 'track'
+ _TRACK_CAM_NAME   = 'fpv'
```

The overlay label `'FPV-physics'` is already accurate — no change needed there.

---

## 4. Expected Visual Difference

| | Current (`track`) | Fixed (`fpv`) |
|---|---|---|
| Camera position | 1 m behind drone, 0.5 m above | At drone nose, 0.1 m forward |
| View content | Drone body visible from rear | World visible from drone nose — no drone body |
| When drone pitches forward | View tilts forward (body-fixed) | View tilts forward — authentic FPV lean |
| During corridor L/R | Drone appears to pan left/right | Environment rushes in from center |
| During pillar approach | Drone body + pillar in background | Pillar approaches in centre of frame |

The FPV view is the most informative for evaluating trajectory quality: obstacles appear
as they would from onboard sensors, making it easy to judge proximity and approach angle.

---

## 5. Scope

Only `quadrotor_modified.xml` needs the XML change. All 4 scenes include it.
`quadrotor.xml` (original, not used by any scene) can be updated in parallel but
is not required for GIF generation.

---

## 6. Next Step

Implement the 3 changes above, re-run the smoke test (3 pillar episodes), compare the
new FPV panel against the old `track` view.

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh 3 pillars "" 3
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh 3 pillars "" 3
```
