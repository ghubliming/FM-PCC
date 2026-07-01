# Diagnostic Report: Missing Dynamics Constraints in FM-PCC Evaluation

This report documents a critical discrepancy discovered between the original **DPCC** (Diffusion Planner with Constraint Correction) architecture and the newly adapted **FM-PCC** evaluation scripts for the UAV and Visual Aligning tasks. 

Specifically, the evaluation scripts have inadvertently dropped the rigid anchoring logic for desired positions ($p_{des}$), allowing the optimizer to hallucinate unanchored trajectories.

---

## 1. The Original DPCC Baseline (The "Ground Truth")

In the original DPCC `avoiding` environment, the robot's state includes both the real position ($p$) and the desired position/waypoint ($p_{des}$). 

In `diffuser/utils/constraints_helpers.py`, the dynamic constraints were strictly defined to force **both** $p$ and $p_{des}$ to follow the commanded velocity ($v$) via explicit Euler integration:

```python
# Original DPCC: BOTH p and p_des are mathematically anchored and integrated
dynamic_constraints = [
    ('deriv', np.array([act_obs_indices['x'], act_obs_indices['vx']])),
    ('deriv', np.array([act_obs_indices['y'], act_obs_indices['vy']])),
    ('deriv', np.array([act_obs_indices['x_des'], act_obs_indices['vx']])),
    ('deriv', np.array([act_obs_indices['y_des'], act_obs_indices['vy']])),
]
```

### Why this matters:
Because `deriv` constraints dictate which variables the `Projector` anchors at $t=0$, submitting all four constraints guarantees that:
1. $p_0$ is pinned to the real physical sensor.
2. $p_{des,0}$ is pinned to the real commanded waypoint sensor.
3. Both trajectories step forward in perfectly synchronized parallel tracks.

---

## 2. The Visual Aligning Task Problem

**File:** `/workspaces/FM-PCC/fm_visual_aligning_test/eval_fm_visual_aligning.py`

In the visual aligning evaluation script, the generic helper was bypassed in favor of hardcoded indices. However, the author **only** constrained the real robot positions (indices `6, 7, 8`), completely omitting the desired positions (`3, 4, 5`).

```python
# Current Visual Aligning Code (Lines 125-128)
if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
    constraint_list.append(('deriv', [6, 0])) # x = x + dt * dx
    constraint_list.append(('deriv', [7, 1])) # y = y + dt * dy
    constraint_list.append(('deriv', [8, 2])) # z = z + dt * dz
    # MISSING: des_x, des_y, des_z
```

### The Consequence:
Because the `des` dimensions are missing from the constraints list, **the projector never anchors $p_{des}$ at $t=0$**. The diffusion model is free to "float" or hallucinate the initial desired waypoint position, which can cause severe tracking glitches when the trajectory is handed to the low-level controller.

---

## 3. The UAV Task Problem

**File:** `/workspaces/FM-PCC/FM_v3_uav_test/eval_fm_uav.py`

In the UAV script, the author recognized the existence of $p$ and $p_{des}$ but introduced an `anchor_to_p` boolean flag that forces an **either/or choice**.

```python
# Current UAV Code (Lines 206-214)
if anchor_to_p:
    # Constraints ONLY real p
    constraint_list += [('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2])]
else:
    # Constraints ONLY desired p
    constraint_list += [('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])]
```

### The Consequence:
This boolean flag breaks the DPCC baseline logic. By forcing an `if/else` choice, the script ensures that **one of the trajectory streams is always left unanchored**. If `anchor_to_p` is True, the waypoint tracking floats. If it's False, the real drone position floats. 

---

## 4. Required Fixes

To restore parity with the rigorous DPCC baseline, both scripts must be updated to apply dynamics constraints to **both** $p$ and $p_{des}$ simultaneously.

### Fix for Visual Aligning (`eval_fm_visual_aligning.py`)
Add the missing indices 3, 4, and 5 to the constraint list:
```python
if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
    # Anchor Real Position
    constraint_list.append(('deriv', [6, 0]))
    constraint_list.append(('deriv', [7, 1]))
    constraint_list.append(('deriv', [8, 2]))
    # Anchor Desired Position
    constraint_list.append(('deriv', [3, 0]))
    constraint_list.append(('deriv', [4, 1]))
    constraint_list.append(('deriv', [5, 2]))
```

### Fix for UAV (`eval_fm_uav.py`)
Remove the `anchor_to_p` branching logic and apply all constraints together:
```python
if 'dynamics' in config.get('constraint_types', []) and 'model_free' not in variant:
    constraint_list += [
        # Anchor Real p
        ('deriv', [6, 0]), ('deriv', [7, 1]), ('deriv', [8, 2]),
        # Anchor Desired p_des
        ('deriv', [3, 0]), ('deriv', [4, 1]), ('deriv', [5, 2])
    ]
```
