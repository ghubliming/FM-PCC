# Success, Mode & Angle — What Do They Mean?

**Added**: 2026-05-28  
**Applies to**: `eval_visual_aligning_dpcc.py`, `eval_fm_visual_aligning.py`  
**Env source**: `d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py`

---

## Success (`success: True / False`)

### Definition

```python
# _check_early_termination() in aligning.py
box_goal_pos_dist = np.linalg.norm(box_pos_3d - target_pos_3d)           # metres
box_goal_rot_dist = rotation_distance(box_quat, target_quat) / np.pi     # dimensionless [0, 1]

success = (box_goal_pos_dist <= 0.018) and (box_goal_rot_dist <= 0.048)
```

| Threshold | Value | Interpretation |
|---|---|---|
| `pos_min_dist` | 0.018 m (1.8 cm) | Box centre must be within 1.8 cm of target centre (3D Euclidean) |
| `rot_min_dist` | 0.048 | Rotational error < 0.048π rad ≈ **8.6°** |

**Both** conditions must hold simultaneously at the same timestep.

`rotation_distance(p, q) = 2 * arccos(|p · q|)` — geodesic angle between two quaternions (0 = identical, π = 180° apart).

### Termination behaviour

When `success=True` at step N:
- `_check_early_termination()` sets `self.terminated = True`
- `is_finished()` returns `True` → `done = True`
- The `while not done` loop in `aligning_sim.py` exits
- The final `info` contains `success=True`

So: `Success: False` means the rollout ran to the **max step limit (400 steps)** without ever meeting both thresholds simultaneously.  The box never got within 1.8 cm + 8.6° of the target before time ran out.

### What "Final Mean Distance" measures

```python
mean_distance = 0.5 * (box_goal_pos_dist_metres + box_goal_rot_dist_normalised)
```

This mixes metres and a normalised rotation scalar — it is the D3IL paper's standard combined metric, not a pure distance.  The success boundary corresponds to `mean_distance ≤ 0.5 * (0.018 + 0.048) = 0.033`.

- `mean_distance ≈ 0.000` → box perfectly at target  
- `mean_distance ≈ 0.033` → on the edge of success (one or both thresholds barely met)  
- `mean_distance > 0.033` → failed rollout; box still far from target  
- `mean_distance ≈ 0.5` → 1 m away OR 180° rotated (or some combination)

---

## Mode (`mode: 0` or `mode: 1`)

### Definition

```python
# check_mode() in aligning.py
robot_box_dist = np.linalg.norm(box_pos_xy - robot_pos_xy)

mode = 0 if robot_box_dist < 0.051 else 1   # threshold: 5.1 cm
```

| Value | Meaning |
|---|---|
| `0` | Robot end-effector is **in contact** with (or very close to) the box at the final step (dist < 5.1 cm) |
| `1` | Robot end-effector has **moved away** from the box at the final step (dist ≥ 5.1 cm) |

**Why Mode 0 is typical:** The robot pushes the box throughout the episode.  At the last step it is almost always still touching or right next to the box, so `mode=0` is the expected output for most rollouts regardless of success.

### Mode probability in W&B

In `aligning_sim.py`, `mode_probs` is computed only over **successful** rollouts:

```python
mode_probs[c, :] = [
    fraction of successful rollouts where mode == 0,   # pushed while in contact
    fraction of successful rollouts where mode == 1    # released / approach from outside
]
```

A mode_probs distribution close to [1, 0] means all successful pushes happened while the robot maintained contact throughout.

---

## Final Box Angle (added UF-16)

### How the init angle is stored

The aligning env samples box state as `[x, y, angle_deg]` (not a true 3D position — `pos[2]` is the rotation angle in degrees, not height).  The z-height is fixed by the MuJoCo model.

### How the final angle is derived

At rollout end, `aligning_sim.py` reads the live MuJoCo quat:

```python
final_box_quat = env.scene.get_obj_quat(env.push_box)  # [w, x, y, z]
```

The aligning task only rotates around Z.  The Z Euler angle is extracted with the standard formula:

```python
angle_z_rad = arctan2(2*(w*z + x*y),  1 - 2*(y² + z²))
angle_deg   = degrees(angle_z_rad)
```

**Verification:** for a pure Z-rotation by θ the quat is `[cos(θ/2), 0, 0, sin(θ/2)]`, and the formula yields `arctan2(sin θ, cos θ) = θ` exactly.  The formula is correct.

### Console output

```
  - Box  init XY=(0.581, -0.204)  angle=-43.5°
  - Target   XY=(0.498,  0.333)  angle=-58.3°
  - Init XY dist (box→target): 0.5374 m
  - Box  final XY=(0.501,  0.330)  angle=-57.1°  (dist_to_target: 0.0043 m)
```

`dist_to_target` in the final line is the **2D XY distance** from the final box centre to the target centre (metres).  This is a pure position metric; it does not include the rotational component.  Use `mean_distance` for the combined position+rotation score.

---

## Quick reference

| Console field | Source | Healthy value |
|---|---|---|
| `Success status` | `_check_early_termination`: pos ≤ 1.8 cm AND rot ≤ 8.6° | `True` |
| `Final Mean Distance` | `0.5*(3D_pos_m + rot/π)` | < 0.033 for success |
| `Environment Mode` | Robot–box contact at last step (< 5.1 cm = mode 0) | 0 or 1 (both normal) |
| `Box final XY / angle` | Live MuJoCo state at rollout end | Close to target XY / angle |
| `dist_to_target` (final) | 2D only, no rotation | ≈ 0.000–0.020 m for success |

---

## No bugs found

The success / mode / angle logic in the D3IL env and eval scripts is correct as-is:

- `done` and `success=True` are set on the **same** step when success conditions are met.
- `mode=0` at the final step is expected (robot in contact during push).
- The Z-angle formula is exact for pure Z-rotation quaternions.
