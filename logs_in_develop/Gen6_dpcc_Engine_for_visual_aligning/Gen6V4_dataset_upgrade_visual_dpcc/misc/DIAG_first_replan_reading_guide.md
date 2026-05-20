# Reading `[ DIAG first-replan ]` Output — Healthy vs Catastrophic

**Context:** Gen6V4 `visual_aligning_dpcc` — DPCC inference diagnostic printed at the first replanning step.

---

## What the diagnostic captures

At the first timestep where the controller replans, the eval harness prints:

| Field | Meaning |
|---|---|
| `normalized a0` | The first action in the predicted horizon, **in normalizer space** (trained on dataset statistics). Range should be roughly `[-1, 1]`. |
| `\|mag\|` (normalized) | L2 magnitude of `a0` in normalizer space. Healthy: `< 1.5`. |
| `denormalized a0` | The same action after reversing the `LimitsNormalizer` — real physical delta `[dx, dy, dz]` in **metres**. |
| `\|mag\|` (denormalized) | Physical step size. For this robot, healthy single-step deltas are roughly `0.0001 – 0.003 m`. |
| `horizon act range` | Min and max across all `H × 3` normalized action values in the full horizon. Healthy: within `[-1.5, 1.5]`. |
| `per-step normalized acts` | Full `H=8` action sequence. Healthy: smooth progression, no wild oscillation. |

---

## Healthy output — annotated

```
[ DIAG first-replan ] normalized   a0 = [-0.0225  0.0145 -0.235 ]  |mag| = 0.2365
[ DIAG first-replan ] denormalized a0 = [-1.9e-04  1.2e-04 -2.0e-05]  |mag| = 0.000224 m
[ DIAG first-replan ] horizon act (normalized) range: [-0.7778, 0.9952]
[ DIAG first-replan ] per-step normalized acts (H=8):
  step  0: [-0.0225  0.0145 -0.235 ]
  step  1: [-0.0324  0.901  -0.2361]
  step  2: [-0.1016  0.9935 -0.222 ]
  step  3: [-0.2906  0.9534 -0.2248]
  step  4: [-0.7473  0.9952 -0.2407]
  step  5: [-0.7778  0.9295 -0.2271]
  step  6: [-0.7223  0.9241 -0.239 ]
  step  7: [-0.6207  0.8142 -0.2292]
```

**Observations:**
- `a0` normalized magnitude = `0.24` — comfortably in-distribution.
- Physical step = `0.000224 m` (0.22 mm) — appropriate sub-mm delta for this task.
- Full horizon range `[-0.78, 1.00]` — entirely within the trained normalizer domain.
- Per-step acts show a **coherent trajectory**: `y` ramps from `0.01 → 0.81` over 8 steps (controller executing a y-sweep), `z` stays stable `~-0.23` (consistent depth), `x` drifts gradually negative. This is a plausible, smooth robot path.
- `z` component is nearly constant across all 8 steps — the model has learned to hold depth while executing lateral motion.

---

## Catastrophic output — annotated

```
[ DIAG first-replan ] normalized   a0 = [ -24.2305 -188.876  -180.7739]  |mag| = 262.5651
[ DIAG first-replan ] denormalized a0 = [-0.00833 -0.00833 -0.00833]  |mag| = 0.014434 m
[ DIAG first-replan ] horizon act (normalized) range: [-346.3870, 385.7436]
[ DIAG first-replan ] per-step normalized acts (H=8):
  step  0: [ -24.2305 -188.876  -180.7739]
  step  1: [ 228.5538  192.3963 -251.888 ]
  step  2: [  85.5093 -320.3506   -5.3699]
  ...
  step  6: [ 385.7436  186.6204 -346.387 ]
  step  7: [ 319.5519   93.861  -237.1826]
```

**Observations:**
- `a0` normalized magnitude = `262.6` — **~1100× larger than a healthy value**. Completely outside the normalizer's trained range.
- Denormalized `a0 = [-0.00833 -0.00833 -0.00833]` — **all three dims identical**. This is the `LimitsNormalizer` hard-clamping: every value is so far out of range that it saturates to the same limit in each dimension. The physical output is meaningless (the actual predictions are garbage; the clamp hides that).
- Full horizon range `[-346, +385]` — **~350× the trained range**. The model is producing white noise in action space.
- Per-step acts oscillate with no coherent structure — random-looking sign flips at every step. A real policy would never produce this; it indicates **the model weights are either untrained (random init) or the wrong checkpoint was loaded**.

---

## Root cause of the catastrophic case

The save path in the catastrophic output contains `steps400` in the **plan** directory:
```
.../H8_K100_T0.1_D..._VTrue_steps400/...
```
while the training directory always used `steps1000`:
```
.../H8_K100_D..._aw10_VTrue_steps1000/...
```

This mismatch arose because `max_episode_length=400` (an **eval-time-only** parameter — how many env steps to run per episode) was included in `args_to_watch_fm_visual_plan`, which drove the plan `exp_name` and therefore the `diffusion_loadpath` directory name.

**The load path resolved to a directory that does not match where the model was saved** → the checkpoint was either not found (→ random weights) or a stale/wrong checkpoint was loaded.

This was the bug fixed in **Fix 2 Change 2** for Gen7: `max_episode_length` was removed from `args_to_watch_fm_visual_plan` so the plan dir name no longer diverges from the training dir name.

---

## Quick sanity checklist when reading `[ DIAG first-replan ]`

| Check | Healthy range | Red flag |
|---|---|---|
| `\|mag\|` normalized | `< 1.5` | `> 10` → model is OOD or wrong checkpoint |
| `horizon act range` | `[-2, 2]` | `> ±10` → normalizer overflow |
| Per-step trajectory | Smooth / monotone per-dim | Wild sign-flips every step → random weights |
| All denorm dims equal | Should differ per axis | All same value → LimitsNormalizer hard-clamp |
| Physical `\|mag\|` | `0.0001 – 0.005 m` | `> 0.01 m` per step → saturated / garbage |

---

## `Environment Mode` — what it means

`Environment Mode` is **not** a scenario configuration index. It is a live behavioral state computed by `aligning.py::check_mode()` at every `env.step()` call and read from the final step's `info` dict:

```python
# d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py
def check_mode(self):
    robot_box_dist = np.linalg.norm(box_pos[:2] - robot_pos[:2])
    if robot_box_dist < self.robot_box_dist:   # within contact threshold
        mode = 0   # robot is near / engaging the box
    else:
        mode = 1   # robot is far from the box
    return mode, mean_distance
```

| Mode | Meaning |
|---|---|
| `0` | Robot ended the episode **close to / in contact with the box** — it engaged with the task |
| `1` | Robot ended the episode **far from the box** — it never approached |
| `-1` | Not set (should not appear in eval output) |

### Correct run → Mode 0

The robot reached the box and was physically near it at episode end. It didn't complete the alignment (`Success: False`, `Mean Distance: 0.091 m`) but the controller was doing the right gross motor behavior — it moved toward and engaged the object. This is a **policy performance issue**, not a pipeline failure.

### Catastrophic run → Mode 1

The robot never got close to the box. The garbage actions from the wrong/random checkpoint sent the end-effector to completely wrong positions in the workspace, so `robot_box_dist` never dropped below the contact threshold. This is a **direct behavioral consequence** of the normalized values being 100-300× out of range — the physically-clamped actions (`-0.00833` in all dims) drove the robot somewhere useless.

### Summary

Mode is a **result indicator** about where the robot ended up, not a cause:

```
Random weights → garbage actions → robot goes wrong way → Mode 1
Correct weights → coherent actions → robot reaches box → Mode 0
```

Mode 0 is necessary but not sufficient for success. A robot can be near the box (`Mode 0`) and still fail to align it within the episode budget — which is exactly what the correct run shows.

---

## Why the correct run still shows `Success: False`

`Final Mean Distance: 0.091 m` at 400 steps means the box wasn't close enough to the target position/orientation within the episode budget. The model was trained on only seed 6 (`--seeds 6`), which likely underfit. The pipeline is fundamentally working (Mode 0, healthy diagnostics) — this is a training data / capacity issue, not a bug.
