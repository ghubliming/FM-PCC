# Gen11 Epoch 4 — Expert Data Collection: Methodology

**Date**: 2026-06-06  
**Status**: ✅ Complete — 1769 episodes across 4 scenes  
**Maximum fix index**: Fix_5  
**CLOSURE**: [`CLOSURE.md`](CLOSURE.md)

---

## 0. Purpose

Epoch 4 generates the **training dataset** for FM-PCC on the UAV task.  It is Stage 1 of a
two-stage pipeline: this epoch collects *state-only* data headlessly; Stage 2 (Epoch 5)
replays those states to render cameras.  Separating the stages means every expensive
MuJoCo physics rollout only runs once, and camera capture can be re-run (different
resolutions, added cameras) without re-flying the trajectories.

---

## 1. Why a scripted PID controller is "expert"

**Real-world meaning**: In imitation learning the word *expert* means "a policy that
demonstrates the desired behaviour" — not "a policy that is optimal".  For the research
question ("can FM-PCC plan obstacle-avoiding flight?"), a well-tuned PID controller that
reliably navigates the scene without contact is a legitimate demonstrator.  Human
teleop or MJPC (a racing-focused solver) would answer different questions.

**Math**: The PID runs a cascaded loop:
- Outer loop (position): `F_des = Kp_pos·(p_des − p) + Kd_pos·(v_des − v) + a_des`
  — a PD controller on 3D position + feedforward acceleration.
- Inner loop (attitude): `τ = Kp_omega·(ω_des − ω)` — a P controller on angular velocity
  to keep the drone level (zero roll/pitch target).

The outer loop commands a desired force vector; the inner loop converts it to four rotor
thrusts.  The net effect: the drone tracks a smooth 3D reference trajectory with bounded
overshoot. `Kp_omega = [2.5, 2.5, 1.0]` (fixed in Decision 2 before any data collection
— original `[10, 10, 2]` caused limit-cycle oscillation on tight turns).

**Code location**: `uav_env_test/flight_controller.py` (`CascadedPID`), called from
`generator.py:_make_pid()` (line 102–107).

---

## 2. Reference trajectory design (what the drone is told to follow)

The PID needs a time-varying setpoint `(p_des(t), v_des(t), a_des(t))`.  Each trial
generates one **trajectory function** `traj(t) → (p, v, a, yaw)`.

### 2.1 Four scenes, four trajectory strategies

| Scene | Real-world task | Trajectory factory | Homotopy classes |
|---|---|---|---|
| `empty` | Free-space point-to-point | `traverse_line` (cosine velocity profile) | N/A — random start+end |
| `corridor` | Navigate a straight hallway | `traverse_line` with lateral bias | L / C / R (left/centre/right channel) |
| `s_curve` | Navigate two offset corridors with a diagonal gap crossing | 3-segment piecewise `traverse_line` | `default` (only one topological route) |
| `pillars` | Thread through 3 pairs of cylindrical pillars | Sinusoidal `weave` factory | (L,L,L) / (L,R,L) / (R,L,R) / (R,R,R) — per-pair side choice |

**Why cosine velocity profile?**  A straight-line traverse at constant speed would demand
infinite jerk at start/stop.  `traverse_line` uses a minimum-jerk cosine ramp:
`v(t) = v_peak · sin²(πt/T)`, so position follows a smooth S-curve in time.  The drone
never has to instantaneously accelerate, which keeps inner-loop tracking errors small and
prevents contact during the start/stop transients.

**Why homotopy classes?** FM-PCC must learn a *multimodal* distribution — the drone can
pass left or right of an obstacle.  Labelling each episode with its homotopy class ensures
the dataset covers all modes rather than collapsing to the most common path.  During
inference, the homotopy is selected by the planner; FM samples from the corresponding
conditional distribution. Code: `generator.py` `HOMOTOPY_CLASSES` dict (lines 65–70),
`_build_traj_and_init()` dispatches to the scene's trajectory factory (lines 110–165).

### 2.2 Randomisation per trial

To prevent overfitting to a single trajectory, each trial is seeded by an integer and
randomises:
- **Altitude** `z ~ U(0.70, 1.10)` m — covers the operating envelope without hitting floor or ceiling.
- **Duration** sampled per scene (e.g. corridor `U(6, 10)` s, s_curve `U(16, 22)` s).
- **Lateral jitter** `U(−0.05, +0.05)` m — slightly varies the channel centre to thicken the manifold.

Code: `rng = np.random.default_rng(seed)` at `generator.py:run_trial()` line 191.
All randomisation uses this seeded RNG so trials are deterministically reproducible.

---

## 3. Physics simulation loop

**MuJoCo at 100 Hz**: The model runs at `dt_physics = 0.01 s` (100 Hz).  Each step:
1. Evaluate `traj(k · dt)` to get `(p_des, v_des, a_des, yaw_des)`.
2. Read current state: `p = data.qpos[:3]`, `v = data.qvel[:3]`, `q = data.qpos[3:7]` (quaternion), `ω = data.qvel[3:6]`.
3. Compute rotor thrusts `u = pid.compute(p, q, v, ω, p_des, v_des, a_des, yaw_des)`.
4. Apply `data.ctrl[:4] = u`, then `mujoco.mj_step(model, data)` — advances physics by `dt`.
5. Record `{p, v, p_des}` **before** the step (position at time `k·dt`, not `(k+1)·dt`).

Code: `generator.py:run_trial()` main loop lines 211–228.

**Why record before the step?** The convention matches D3IL: observation at time `t` is the
state the controller *sees* when choosing the action at time `t`, not the state it arrives
at.

---

## 4. Contact filter — rejecting "bad" episodes

After the rollout completes, the fraction of steps with obstacle contact is computed:

```
contact_fraction = n_hit / n_step
```

where `n_hit` counts steps where MuJoCo reports at least one contact between two non-floor
geoms (`_is_obstacle_contact`, lines 95–99):

```
n1 != 'floor'  AND  n2 != 'floor'
```

Episodes exceeding the per-scene threshold are **discarded** (return `None`):

| Scene | Threshold | Rationale |
|---|---|---|
| empty | 2% | No obstacles — any contact is a floor graze |
| corridor | 2% | Up to 4 contact steps in 200-step episode allowed |
| pillars | 2% | Same |
| s_curve | **8%** | Narrow wall end-faces at x=±0.5 cause brief grazes even on valid diagonal crossings — Fix_4 |

**Real-world meaning**: The 2% threshold accepts episodes where the drone briefly touches
an obstacle (e.g. due to a momentary PID overshoot at speed) but recovers immediately.
This is consistent with real UAV flight where instantaneous skin contact at low relative
velocity is non-catastrophic.  However, it also means training data includes positions
*at* the wall surface — an important caveat for Epoch 5 visual inspection (see
`INVESTIGATION_wall_contact_gifs.md`).

Code: `generator.py` lines 230–233, `SCENE_MAX_CONTACT_FRACTION` dict lines 85–90.

---

## 5. From 100 Hz physics to 33 Hz dataset — downsampling

**Why downsample?** FM-PCC operates over a finite planning horizon `H`.  At 100 Hz a
1-second look-ahead is 100 steps; at 33 Hz it is 33 steps.  Shorter sequences train faster
and the slower rate matches the response bandwidth of the PID (the drone cannot
meaningfully respond to commands at 100 Hz).

**How**: keep every 3rd physics step (`stride = round(1 / (dt_physics × 33)) = 3`).
Dataset `dt = 3 × 0.01 = 0.030 s ≈ 33 Hz`.

Code: `dataset_writer.py:_downsample()` lines 34–36, called at line 49.

---

## 6. Noise injection on targets — thickening the data manifold

**Problem**: A scripted PID always sends the drone to the same trajectory for a given
seed.  The dataset manifold is one-dimensional (a single curve per episode).  FM needs
a distribution, not a delta function.

**Fix (Decision 1 of Fix_1.4)**: Add a single constant Gaussian offset per episode to
`targets` (the commanded positions), not per-step independent noise:

```
offset ~ N(0, σ²·I₃)    σ = 0.02 m
targets_noisy = targets + offset   # broadcast — same offset for all T steps
```

**Why constant, not per-step?** Per-step noise makes the *actions* (which are target
deltas) noise-dominated:  `actions = diff(targets + per_step_noise)` adds
`N(0, 2σ²)` noise to each action, with std ≈ 0.028 m/step vs. signal ≈ 0.012 m/step —
SNR < 1, unusable.  A constant offset shifts the whole trajectory rigidly: `diff(targets + c) = diff(targets)`, so actions are unaffected.  The effect is to generate many
"parallel copies" of the same trajectory in the data manifold, each shifted by a small
random 3D translation.

Code: `dataset_writer.py:rollout_to_episode()` lines 57–67.

---

## 7. Action convention — position-delta

**Central design choice (Decision 1, AUDIT Risk 3):**

```
actions[t] = targets[t+1] − targets[t]   # Δp_des ∈ ℝ³
```

- `actions` shape: `(T-1, 3)` float32
- `targets` shape: `(T, 3)` float32 — kept for debugging, NOT fed to FM

**Why delta, not absolute?** Three independent lineages converge on this convention:
1. **D3IL**: `vel_state = des_c_pos[1:] - des_c_pos[:-1]` — the action is the commanded
   velocity of the end-effector.
2. **UAV-Flow**: `_transform_to_local_frame` → body-frame Δpose per step.
3. **FM-PCC theory**: the flow field `v_θ(x, t)` integrates along a trajectory in state
   space; feeding absolute positions would require the model to learn the full state rather
   than the *increment*.

In code, actions represent "how much should the desired position move this step" — essentially
a commanded velocity at the dataset frequency.  At 33 Hz and 0.4 m/s the typical action
norm is 0.012 m/step.

Code: `dataset_writer.py` line 68: `actions = np.diff(targets, axis=0)`.

---

## 8. Output structure — folders, filenames, and pickle contents

### 8.1 Folder tree

```
logs/uav_expert_data/
│
├── empty/
│   └── N_A/                        ← homotopy "N/A" → folder "N_A"
│       ├── empty_N_A_pid_default_0000001.pkl
│       ├── empty_N_A_pid_default_0000002.pkl
│       └── …  (500 episodes)
│
├── corridor/
│   ├── C/                          ← Centre channel (drone flies down the middle)
│   │   ├── corridor_C_pid_default_0000001.pkl
│   │   └── …
│   ├── L/                          ← Left channel (drone hugs left wall)
│   │   ├── corridor_L_pid_default_0000056.pkl
│   │   └── …
│   └── R/                          ← Right channel (drone hugs right wall)
│       └── …
│       (436 episodes total: C≈167, L≈139, R≈130)
│
├── s_curve/
│   └── default/                    ← Only one topological route
│       └── …  (356 episodes)
│
└── pillars/
    ├── L_L_L/                      ← Go left of pillar 1, left of 2, left of 3
    ├── L_R_L/                      ← Left, right, left
    ├── R_L_R/                      ← Right, left, right
    └── R_R_R/                      ← Right of all 3 pillars
        (477 episodes total: ~112–125 per class)
```

**Filename format**: `{scene}_{homotopy_safe}_{controller}_{7-digit-counter}.pkl`  
Example: `corridor_L_pid_default_0000056.pkl`

**`homotopy_safe`** is the folder-safe version of the homotopy label:

| Raw homotopy | Folder name | Meaning |
|---|---|---|
| `'N/A'` | `N_A` | No obstacle — empty scene |
| `'C'` | `C` | Centre channel |
| `'L'` | `L` | Left channel |
| `'R'` | `R` | Right channel |
| `'default'` | `default` | Only route (s_curve) |
| `'(L,L,L)'` | `L_L_L` | Parentheses and commas stripped |

---

### 8.2 What L / C / R mean physically

**Corridor**: the corridor is a straight hallway with two walls.  The drone can fly through
three distinct lateral channels:

```
  ┌─────────────────────────────────────────┐  ← wall (y = +0.45 m)
  │   L channel   │  C channel  │ R channel │
  │  (y ≈ −0.22)  │  (y ≈ 0.0)  │ (y ≈+0.22)│
  └─────────────────────────────────────────┘  ← wall (y = −0.45 m)
  → flight direction: x
```

- **C** (centre): flies straight down the middle.  No wall contact.
- **L** (left): flies near the **negative-y** wall (left when looking forward along +x).
  Brief wall contact is physically unavoidable — L and R homotopies always have
  `contact_fraction > 0` (see E4 U2 Fix_1 CLOSURE for why the threshold was not tightened).
- **R** (right): mirrors L on the **positive-y** wall.

The FM must learn all three — at inference the planner selects a homotopy and the FM
samples from that conditional distribution.

**Pillars**: three pairs of cylindrical pillars along the flight path.  For each pair the
drone can pass left or right, giving `2³ = 8` combinations.  Only the four
topologically distinct ones are collected (the others are mirror images and would be
redundant given the scene's bilateral symmetry):

| Label | Per-pillar side | Physical path |
|---|---|---|
| `(L,L,L)` | Left, Left, Left | S-shape to the negative-y side |
| `(L,R,L)` | Left, Right, Left | Weave |
| `(R,L,R)` | Right, Left, Right | Weave (mirror) |
| `(R,R,R)` | Right, Right, Right | S-shape to the positive-y side |

---

### 8.3 What is inside each PKL — field-by-field

Load an episode:
```python
import pickle
ep = pickle.load(open('logs/uav_expert_data/corridor/L/corridor_L_pid_default_0000056.pkl', 'rb'))
ep.keys()
# dict_keys(['episode_id', 'scene', 'homotopy', 'controller', 'dt',
#            'obs', 'actions', 'targets', 'q', 'obstacles', 'metadata'])
```

| Field | Shape / Type | Content |
|---|---|---|
| `episode_id` | str | `'corridor_L_pid_default_0000056'` — unique identifier, matches filename |
| `scene` | str | `'corridor'` — which scene XML was loaded |
| `homotopy` | str | `'L'` — raw label (not folder-safe) |
| `controller` | str | `'pid_default'` — which PID gain set was used |
| `dt` | float | `≈ 0.030 s` — time between consecutive obs rows (33 Hz) |
| `obs` | `(T, 9)` float32 | State at each timestep — **see §8.4** |
| `actions` | `(T-1, 3)` float32 | Position deltas — **see §8.5** |
| `targets` | `(T, 3)` float32 | Noisy absolute p_des (debug only — do not feed to FM) |
| `q` | `(T, 4)` float32 | Actual body quaternion `[w, x, y, z]` — attitude for rendering (E4 U3+) |
| `obstacles` | list[dict] | Scene geometry for DPCC — **see §8.6** |
| `metadata` | dict | Run statistics — **see §8.7** |

---

### 8.4 `obs` — the observation tensor `(T, 9)`

After E4 U2, obs is 9D. Each row is the drone state at one timestep:

```
obs[t] = [ p_des_x,  p_des_y,  p_des_z,    ← columns 0:3  — commanded position (unnoisy)
           p_x,      p_y,      p_z,         ← columns 3:6  — actual position (world frame, m)
           v_x,      v_y,      v_z ]        ← columns 6:9  — actual velocity (world frame, m/s)
```

- **`p_des` (columns 0:3)**: the position the PID was trying to reach at timestep `t`.
  Unnoisy — this is the exact trajectory-function output `traj(t·dt).p`.  Added in E4 U2
  to give the FM a goal signal (DPCC_OBS_DEVIATION §Deviation 2).

- **`p` (columns 3:6)**: the drone's actual COM position in world frame (metres).
  `p[0]` = forward (x), `p[1]` = lateral (y), `p[2]` = altitude (z).
  Typical values: x ∈ [0, 5] m, y ∈ [−0.45, +0.45] m (corridor), z ∈ [0.7, 1.1] m.

- **`v` (columns 6:9)**: linear velocity in world frame (m/s).  Zero at episode start.
  Corridor mean speed: 0.716 m/s.  `obs[0, 6:9]` is always `[0, 0, 0]`.

```python
obs = ep['obs']                 # (T, 9)
p_des = obs[:, :3]              # (T, 3)  commanded positions
p     = obs[:, 3:6]             # (T, 3)  actual positions
v     = obs[:, 6:9]             # (T, 3)  velocities
print(f'Episode length: T={len(obs)} steps = {len(obs)*ep["dt"]:.1f} s')
print(f'Mean speed: {np.linalg.norm(v, axis=1).mean():.3f} m/s')
```

**FM tensor**: the FM dataloader stacks `[actions ‖ obs]` to form a 12D column per step:
`D = [Δp_des(3) ‖ p_des(3), p(3), v(3)]` — shape `(T-1, 12)` (one fewer row because
actions are forward differences of targets).

---

### 8.5 `actions` — position deltas `(T-1, 3)`

```
actions[t] = targets[t+1] − targets[t]   →   Δp_des at step t   (m)
```

This is the **commanded velocity** at the dataset frequency (33 Hz).  Typical norm: 0.012
m/step for empty, 0.021 m/step for corridor.

`actions` has one fewer row than `obs` because it is a forward difference.  The FM
predicts a horizon of `H` future actions from the current obs.

```python
actions = ep['actions']         # (T-1, 3)
print(f'Action norm mean: {np.linalg.norm(actions, axis=1).mean():.4f} m/step')
print(f'Actions range: [{actions.min():.3f}, {actions.max():.3f}] m/step')
```

---

### 8.6 `obstacles` — scene geometry for DPCC

A list of dicts, one per obstacle geom.  Used by the DPCC safety filter to construct
half-space constraints.

```python
ep['obstacles']
# e.g. for corridor:
# [{'type': 'box', 'name': 'wall_left',  'center': [2.5, -0.45, 0.5], 'half_extents': [2.5, 0.05, 0.5]},
#  {'type': 'box', 'name': 'wall_right', 'center': [2.5,  0.45, 0.5], 'half_extents': [2.5, 0.05, 0.5]}]
```

Keys per obstacle:

| Key | Type | Meaning |
|---|---|---|
| `type` | str | `'box'` or `'cylinder'` |
| `name` | str | MuJoCo geom name |
| `center` | list[float] | World-frame centre `[x, y, z]` |
| `half_extents` | list[float] | Box half-sizes `[dx, dy, dz]` (box only) |
| `radius` | float | Cylinder radius (cylinder only) |

---

### 8.7 `metadata` — run statistics

```python
ep['metadata']
# {'start_pos':        [0.1, -0.21, 0.9],   ← initial drone position (m)
#  'total_time':       6.23,                 ← episode duration (s)
#  'dt_physics':       0.01,                 ← physics timestep (100 Hz)
#  'contact_fraction': 0.0138,               ← fraction of steps with wall contact
#  'controller_gains': 'pid_default',
#  'noise_sigma':      0.02}                 ← std of target offset noise (m)
```

`contact_fraction` is the key quality metric — episodes above the per-scene threshold
were discarded by the contact filter (§4).  The value here is the fraction of the
*accepted* episode that had contact.

---

### 8.8 `q` — body quaternion `(T, 4)` *(E4 U3+)*

The actual drone body orientation at each timestep, stored as `[w, x, y, z]`:

```python
q = ep['q']                     # (T, 4)
print(f'q at hover start: {q[0]}')   # ≈ [1, 0, 0, 0] — level
print(f'q at mid-speed:   {q[T//2]}') # slightly nose-down during fast forward flight
```

`q[0]` is always close to identity (drone starts level).  During forward flight the drone
pitches forward to generate thrust: at 0.7 m/s the pitch is ~10° nose-down.  This field
is used by Epoch 5 WS-A/B to render the drone with correct tilt (attitude-aware rendering,
Change D in E5 U2).

**Note**: this field is absent (`episode.get('q', None)` returns `None`) in pickles
collected before E4 U3.  WS-A/B fall back to identity quaternion (level) in that case.

Code: `dataset_writer.py` lines 52–55 (`q` built from `s['q']` step dicts), `generator.py`
line 222 (`q = data.qpos[3:7].copy()`).

---

## 9. PID gain variants — diversity in control style

Three gain settings provide data diversity in how the drone responds to its trajectory:

| Variant | `Kp_pos` scale | `Kd_pos` scale | Behaviour |
|---|---|---|---|
| `pid_default` | ×1.0 | ×1.0 | Baseline tracking |
| `pid_high_gain` | ×1.2 | ×1.0 | Tighter tracking, more aggressive corrections |
| `pid_low_gain` | ×0.8 | ×0.9 | Softer tracking, more lag |

Each episode is labelled with its gain variant in the `controller` field.  At training
time the FM model can condition on controller type (heterogeneous-demonstrator setting) or
treat all variants as the same distribution.

*Note*: Only `pid_default` was collected in the final runs; high/low variants are reserved
for Epoch 6.

Code: `generator.py` `GAIN_VARIANTS` (lines 73–77), `_make_pid()` (lines 102–107).

---

## 10. Fix history summary

| Fix | Scene | Problem | Principle of fix |
|---|---|---|---|
| Fix_1.1–1.2 | s_curve | Piecewise stops → contacts at wall ends | More waypoints + longer duration |
| Fix_1.3 | pillars | Stops near pillars → 95% rejection | Replace piecewise with continuous sinusoidal `weave` |
| Fix_1.4 | **all** | Per-step noise → actions noise-dominated | Constant-per-episode noise offset (see §6) |
| Fix_2.1 | s_curve | Persistent 90.5% rejection | Tanh continuous trajectory (no velocity zeros) |
| Fix_2.2 | pillars | Centre-pass amplitude inside pillar zone | Amplitude → 0.0 (straight centre-line) |
| Fix_3 | s_curve | 61.9% rejection remained | Lowered tanh k → made worse (81.8%); reverted in Fix_4 |
| Fix_4.1 | s_curve | Lower k → path closer to walls | Revert k=3.66 |
| Fix_4.2 | s_curve | End-face grazes at 2% threshold | Per-scene threshold: s_curve → 8% (see §4) |
| Fix_5.1 | s_curve | Tanh peak speed 1.17 m/s during gap crossing → 47.6% rejection | Replace tanh with 3-segment piecewise, duration proportional to distance |
| Fix_5.2 | s_curve | Seed variance caused job abort at 21 trials | Abort limit 0.30 → 0.60 |

---

## 11. Final dataset statistics

| Scene | Episodes | Rejection | Speed mean | Action Δp norm mean | Ep length |
|---|---|---|---|---|---|
| empty | 500 | 0% | 0.387 m/s | 0.0116 m/step | 190 steps |
| corridor | 436 | 12.8% | 0.716 m/s | 0.0205 m/step | 274 steps |
| pillars | 477 | 4.6% | 0.417 m/s | 0.0241 m/step | 442 steps |
| s_curve | 356 | 28.8% | 0.560 m/s | 0.0114 m/step | 641 steps |
| **Total** | **1769** | — | — | — | — |

---

## 12. What this epoch does NOT provide

- ❌ Camera images / visual observations — Stage 2 (Epoch 5 WS-A)
- ❌ Visual GIF inspection — Stage 2 (Epoch 5 WS-B)
- ❌ FM model training — Epoch 6
- ❌ On-policy correction (DAgger) — Epoch 7+
- ❌ `pid_high_gain` / `pid_low_gain` data — planned but not collected

---

## Cross-references

| Document | Content |
|---|---|
| [`CLOSURE.md`](CLOSURE.md) | Final stats, fix log, confirmed decisions |
| [`EPOCH4_EXECUTION_PLAN.md`](EPOCH4_EXECUTION_PLAN.md) | Design decisions (Action, PID, Schema) |
| [`../Epoch5_visual_and_validation/METHODOLOGY.md`](../Epoch5_visual_and_validation/METHODOLOGY.md) | How Epoch 5 replays these pickles |
| [`../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md`](../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md) | Impact of 2% contact threshold on visual inspection |
