# Data Flow: FM Inference → DPCC Projection → PID → MuJoCo

**Scope**: Answers "what actually gets fed to MuJoCo, and what is the 12D tensor for?"

---

## 1. Training Data Layout (12D transition)

Every sample in the dataset is a 12D transition vector:

```
transition[12] = [ action(3) | obs(9) ]
               = [ Δx  Δy  Δz | p_des_x p_des_y p_des_z | p_x p_y p_z | v_x v_y v_z ]
  dims:            0   1   2      3       4       5          6   7   8     9   10  11
```

- `action (0,1,2)` = `Δp_des` — 3D commanded position increment the expert PID issued each step
- `p_des (3,4,5)` = current commanded position (before this action is applied)
- `p (6,7,8)` = real drone position (`mujoco data.qpos[:3]`)
- `v (9,10,11)` = real drone velocity (`mujoco data.qvel[:3]`)

The FM is trained on **windows of H=8 consecutive transitions** → one training sample = `H×12D`.

---

## 2. FM Inference Input (Conditioning Only 9D)

At eval time the FM receives **only the current observation** as conditioning:

```python
# FM_v3_uav_test/eval_fm_uav.py  rollout_one() L343-346
obs = np.concatenate([p_des, p, v]).astype(np.float32)   # 9D raw
action, traj = policy({0: obs}, batch_size=batch_size, horizon=horizon)
```

`{0: obs}` means "pin timestep-0 of the denoised trajectory to this obs; denoise all other steps
freely." The FM does NOT receive any action as input — only the 9D current state.

Inside `Policy.__call__`:

```python
# flow_matcher_v3_uav/sampling/policies.py  L116-119
conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')
```

The 9D obs is normalized before being passed to the FM model.

---

## 3. FM Output — Full H×12D Batch

The FM denoises a **batch** of candidate trajectories:

```python
# flow_matcher_v3_uav/sampling/policies.py  L52-54
samples, infos = self.model(conditions, ..., horizon=horizon, ...)
trajectories = utils.to_np(samples)
# shape: [batch_size × H × 12]  =  [4 × 8 × 12]
```

Each step in the H×8 horizon is a full 12D transition: `[action(3) | p_des(3) | p(3) | v(3)]`.
The FM predicts the expected evolution of BOTH the commanded action AND the full state over H steps.
This is the "trajectory" or "foresight."

---

## 4. What Goes to MuJoCo/PID — Only 3 Numbers

From the H×12D tensor, **only `action[0]` (3D) is executed**:

```python
# flow_matcher_v3_uav/sampling/policies.py  L88-92
actions = trajectories[:, :, :self.action_dim]          # dims 0,1,2  →  [batch × H × 3]
actions = self.normalizer.unnormalize(actions, 'actions')
action  = actions[which_trajectory, 0]                  # step 0, chosen candidate  →  (3,)
```

Then in `rollout_one`:

```python
# FM_v3_uav_test/eval_fm_uav.py  L369-383
# fix_5 anchor-p (anchor_to_p=True):
p_des = p + action          # grounded to real position

# default free-running (anchor_to_p=False):
p_des = p_des + action

v_des = action / dt_fm      # velocity setpoint derived from Δp_des / (1/33 s)

u = pid.compute(p, q, v, om, p_des, v_des)   # → 4D MuJoCo control (thrust + attitude)
```

**The PID receives `p_des` (3D) and `v_des` (3D). MuJoCo runs `decim` physics sub-steps
(L314: `decim = max(1, int(round(1.0 / (dt * DATASET_HZ))))`) under that setpoint.
That is the entire interface to the physics engine.**

### What the rest of the H×12D tensor is used for

| Tensor field | Used for |
|---|---|
| `traj.observations` (H×9D unnormalized) | Trajectory selection: temporal consistency / min-projection-cost comparison |
| `traj.observations` | Logging / visualization ("FM foresight plan") |
| `traj.actions[1..7]` (H-1 future actions) | Logged as `fm_horizon=` in `.log` files only |
| Predicted `p[0..7]`, `v[0..7]`, `p_des[0..7]` | **Not sent to MuJoCo** |

---

## 5. DPCC Projector — Modifies the Trajectory Before Extraction

When a DPCC variant is active, the projector intercepts the H×12D tensor **before** the
`action[0]` extraction and returns a modified tensor:

```
FM samples  →  Projector (QP)  →  modified H×12D  →  extract action[0]  →  PID
```

The projector is built in:
```python
# FM_v3_uav_test/eval_fm_uav.py  setup_dpcc_projector()  L149-217
return Projector(
    horizon=8, transition_dim=12, action_dim=3,
    constraint_list=constraint_list,
    normalizer=ProjectorNormalizer(obs_normalizer, act_normalizer),
    variant='states_actions',
    dt=1.0,          # action IS Δp_des (not a rate) → Euler dt=1.0
    ...
)
```

The Projector class lives in `flow_matcher_v3_uav/sampling/projection.py` (adapted from
`diffuser_visual_aligning/sampling/projection.py` — the original DPCC engine; no public D3IL
URL found in this repo, but it is noted as D3IL-derived in the design doc
`DESIGN_fix5_anchor_p_integration.md` §15).

It solves a constrained QP (SLSQP via scipy or gradient step):

```
minimise   ||trajectory - FM_output||²     (stay close to FM prediction)
subject to  dynamics equality constraint   (deriv)
            + any spatial constraints      (bounds / halfspace / obstacles — placeholders)
```

After projection, `action[0]` is extracted exactly as in Section 4.
The projector's only effect on control is that `action[0]` is nudged to be consistent
with a kinematically plausible H-step rollout.

---

## 6. The `deriv` Dynamics Constraint — Full Explanation

### 6.1 First: action and p_des are the SAME physical quantity

The 12D transition stores BOTH `action` (dims 0-2) AND `p_des` (dims 3-5).
They refer to the same physical quantity — the commanded position — but at different times:

```
action[t]   = Δp_des[t]  =  p_des[t+1] − p_des[t]    ← the INCREMENT taken at step t
p_des[t]    = absolute commanded position at step t
```

So `action` = the **difference** between consecutive `p_des` values.
Or equivalently: `p_des[t+1] = p_des[t] + action[t]`.

**The FM predicts both columns independently.** It outputs action[t] (dims 0-2) AND
p_des[t+1] (dims 3-5 at the next row) — but without any coupling between them. They can be
**inconsistent**: the FM might predict action[t]=0.1 but also predict p_des[t+1]=1.5 when
p_des[t]=1.0 (which implies action=0.5, not 0.1). The 12D tensor the FM outputs has NO
built-in guarantee that `p_des[t+1] - p_des[t] == action[t]`.

The `deriv` constraint ENFORCES this consistency.

---

### 6.2 What the constraint matrix says

```python
# flow_matcher_v3_uav/sampling/projection.py  DynamicConstraints.build_matrices()  L344-401
# ('deriv', [x_idx, dx_idx])  enforces:
#     x[t+1] = x[t] + dt * dx[t]    for t = 0 .. H-2
# with dt=1.0 for UAV (action IS already Δp, not a rate):
#     p_des_x[t+1] = p_des_x[t] + action_x[t]
```

Concretely, for default mode (`anchor_to_p=False`, `eval_fm_uav.py L186-187`):

```
('deriv', [3, 0])  →  p_des_x[t+1] = p_des_x[t] + action_x[t]    for t=0..6  (H-1 rows)
('deriv', [4, 1])  →  p_des_y[t+1] = p_des_y[t] + action_y[t]
('deriv', [5, 2])  →  p_des_z[t+1] = p_des_z[t] + action_z[t]
```

For fix_5 (`anchor_to_p=True`, `eval_fm_uav.py L181-184`) — bind to real p instead:

```
('deriv', [6, 0])  →  p_x[t+1] = p_x[t] + action_x[t]
('deriv', [7, 1])  →  p_y[t+1] = p_y[t] + action_y[t]
('deriv', [8, 2])  →  p_z[t+1] = p_z[t] + action_z[t]
```

Additionally, `skip_initial_state=True` (`projection.py L99-108`) **pins the first row**:
`p_des_x[0] = current observed p_des_x` (or `p_x[0] = current observed p_x` for fix_5).
This anchors the whole projected trajectory to the real current state.

---

### 6.3 What the QP actually does — concrete example

The QP objective (`projection.py L133`):
```
minimise   0.5 * ||z_projected − z_FM||²    (stay close to FM prediction)
subject to  A z = b                          (equality: deriv constraints, pinned initial state)
            C z ≤ d                          (inequality: spatial — placeholder here)
```
where `z` is the FULL flattened H×12D trajectory vector (shape 96).

**Concrete inconsistency example (x-axis only, H=2 for clarity):**

```
FM predicts (unnormalized, x-axis):
  step 0:  action_x = 0.10,  p_des_x = 1.00
  step 1:  action_x = 0.05,  p_des_x = 1.50   ← INCONSISTENT: 1.00 + 0.10 ≠ 1.50

Constraint says:
  pin:      p_des_x[0] = 1.00  (from real obs)
  enforce:  p_des_x[1] = p_des_x[0] + action_x[0] = 1.00 + action_x[0]

QP result (equal weights, identity Q):
  Both action_x[0] and p_des_x[1] are adjusted to meet halfway:
  → action_x[0] ≈ 0.30,  p_des_x[1] ≈ 1.30
  (minimises (0.10−0.30)² + (1.50−1.30)² = 0.04 + 0.04 subject to 1.30=1.00+0.30 ✓)
```

The QP moves **both columns** (action AND p_des) simultaneously to find the closest consistent
point. It does NOT just adjust one or the other.

**The executed control effect**: `action[0]` from the projected trajectory is used:
```python
# policies.py L92
action = actions[which_trajectory, 0]   # ← QP-corrected action[0]
```
In the example above, the drone receives `action_x = 0.30` instead of the FM's original `0.10`.
The projector has pushed the action toward what is consistent with the FM's predicted future
trajectory (`p_des[1]=1.50` implies a bigger step was needed).

---

### 6.4 Why this helps — the multi-step coherence argument

Without projection, `action[0]` is whatever the FM's denoising happened to produce in dims 0-2
at step 0. The FM's predicted `p_des[1]` (dims 3-5 at step 1) might be inconsistent with that
action, meaning the FM's own "foresight" of where the drone will be next step disagrees with
the action it chose.

With projection, the executed `action[0]` is guaranteed to be consistent with the FM's
predicted multi-step trajectory. If the FM predicted a sensible path, the projector makes the
first action coherent with that path. If the FM predicted a bad path (e.g., going underground),
spatial constraints (halfspace, bounds — currently placeholders) would catch it here.

---

### 6.5 This is NOT real drone dynamics

The deriv constraint encodes **naive Euler** — essentially "the drone moves exactly where commanded
each step." No mass, drag, motor dynamics, or PID lag.

For a robot arm (D3IL origin), this is accurate: the arm servo tracks commanded position
instantly. For the UAV with PID lag, the real drone position `p` lags behind `p_des`.

- **Default**: constraint operates in commanded space. Action[0] is consistent with the FM's
  prediction of `p_des[1]` — but `p_des[1]` may be far from where the drone actually ends up.
- **fix_5**: constraint binds to `p` column instead. Action[0] is consistent with the FM's
  prediction of where real `p` should be at step 1 — more physically meaningful, since PID
  control error closes the gap from actual position, not commanded position.

---

## 7. The Full H=8 Tensor — What Each Cell Is and Where It Goes

### 7.1 The tensor grid

The FM outputs `[batch=4 × H=8 × dim=12]`. Pick one candidate (e.g. `which_trajectory=0`).
That gives one `[8 × 12]` matrix. Each row is one FM step; each column is one dim:

```
         ┌─── action (Δp_des) ───┐  ┌──────── p_des ─────────┐  ┌──────── p (real) ──────┐  ┌──────── v (real) ──────┐
         dim 0   dim 1   dim 2     dim 3   dim 4   dim 5        dim 6   dim 7   dim 8        dim 9   dim 10  dim 11
         Δx      Δy      Δz        p_des_x p_des_y p_des_z      p_x     p_y     p_z          v_x     v_y     v_z
step 0 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd0  FM_yd0  FM_zd0  |  FM_x0   FM_y0   FM_z0   |  FM_vx0  FM_vy0  FM_vz0  ]
step 1 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd1  FM_yd1  FM_zd1  |  FM_x1   FM_y1   FM_z1   |  FM_vx1  FM_vy1  FM_vz1  ]
step 2 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd2  FM_yd2  FM_zd2  |  FM_x2   FM_y2   FM_z2   |  FM_vx2  FM_vy2  FM_vz2  ]
step 3 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd3  FM_yd3  FM_zd3  |  FM_x3   FM_y3   FM_z3   |  FM_vx3  FM_vy3  FM_vz3  ]
step 4 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd4  FM_yd4  FM_zd4  |  FM_x4   FM_y4   FM_z4   |  FM_vx4  FM_vy4  FM_vz4  ]
step 5 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd5  FM_yd5  FM_zd5  |  FM_x5   FM_y5   FM_z5   |  FM_vx5  FM_vy5  FM_vz5  ]
step 6 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd6  FM_yd6  FM_zd6  |  FM_x6   FM_y6   FM_z6   |  FM_vx6  FM_vy6  FM_vz6  ]
step 7 [ FM_Δx  FM_Δy  FM_Δz  |  FM_xd7  FM_yd7  FM_zd7  |  FM_x7   FM_y7   FM_z7   |  FM_vx7  FM_vy7  FM_vz7  ]
          ↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑
          ONLY step 0 dims 0-2 go to PID.  Everything else = FM foresight.
```

### 7.2 What each group does at runtime

| Group | Dims | Step 0 used for | Steps 1-7 used for |
|---|---|---|---|
| **action** `Δp_des` | 0-2 | **Executed**: `p_des = p + Δp_des`, `v_des = Δp_des / dt_fm` → PID | deriv constraint + logged as `fm_horizon` |
| **p_des** (commanded) | 3-5 | Pinned to current `p_des` (skip_initial_state) by projector | deriv constraint enforces `p_des[t+1] = p_des[t] + action[t]`; traj-selection |
| **p** (real pos) | 6-8 | FM context (conditioned on real `p` via obs) | fix_5 deriv binding; traj-selection; logging |
| **v** (vel) | 9-11 | **FM-predicted v (V3)**: traj-selection only. NOTE: real v (V1) reaches the FM as input obs context and the PID as damping — see §7.3 | FM-predicted v: traj-selection; logging. No deriv constraint binds these dims |

### 7.3 Velocity — the complete picture (correcting "never used")

> ⚠ Earlier drafts said "velocity is never used / a zombie." **That was wrong and is the
> source of the confusion.** Velocity is used heavily — but there are THREE distinct velocity
> entities and only ONE of them is unused for control. Below is the authoritative breakdown.

There are **three different things all called "velocity":**

| # | Name | Where it comes from | Where it is used | Status |
|---|---|---|---|---|
| **V1** | **Real velocity** `v = data.qvel[:3]` | MuJoCo physics (measured) | (a) FM input obs context, (b) **PID damping feedback** | **CRITICAL** |
| **V2** | **Desired velocity** `v_des` | expert: spline derivative; eval: `action/dt_fm` | PID reference (the setpoint the D-term tracks) | **USED** |
| **V3** | **FM-predicted velocity** (tensor dims 9-11) | FM denoising output | traj-selection + logging only | **not used for control** |

Only **V3** (the FM's predicted future velocity column) is unused by the controller.
**V1 and V2 are both essential** — the PID cannot work without them.

#### Why the PID *must* have velocity (the actual control law)

```python
# uav_env_test/flight_controller.py  CascadedPID.compute()  L88-91
e_p   = p - p_des                                    # position error
e_v   = v - v_des                                    # velocity error  ← V1 − V2
a_cmd = -self.Kp_pos * e_p - self.Kd_pos * e_v + a_des   # PD + feed-forward
```

This is a **PD controller**. The `-Kd_pos * e_v` term IS the velocity feedback —
it is the **damping**. Remove the real velocity `v` (V1) and there is no derivative term →
no damping → the drone overshoots and oscillates into instability. So velocity is not just
used, it is half of the entire control law (the "D" in PD).

- `v` (V1, real) = the measured rate, supplies the damping.
- `v_des` (V2, reference) = the rate the controller is trying to match.
- `e_v = v - v_des` = velocity error → scaled by `Kd_pos` → contributes to commanded accel.

#### Where V1 and V2 enter the eval loop

```python
# FM_v3_uav_test/eval_fm_uav.py  rollout_one()

v = data.qvel[:3].copy()              # L342 — V1 real vel → into obs (FM context)
obs = np.concatenate([p_des, p, v])   # L343 — V1 is dims 9-11 of the FM INPUT

action, traj = policy({0: obs}, ...)  # FM call

v_des = action / dt_fm                # L375 — V2 derived from FM position delta

for _ in range(decim):
    v = data.qvel[:3].copy()          # L380 — V1 re-read every physics sub-step
    u = pid.compute(p, q, v, om, p_des, v_des)   # L383 — V1 (real) + V2 (des) both used
```

Note V1 is read **twice**: once to build the FM obs (L342) and again, fresh, inside every
physics sub-step (L380) so the PID always damps against the latest measured velocity.

#### So what about V3 (the FM-predicted velocity column)?

V3 lives in tensor dims 9-11 of every output step. It is:
- **NOT extracted** for control (only `action`, dims 0-2, is extracted — see §4).
- **NOT bound by any constraint** — the deriv dynamics constraint only couples position↔action
  (dims 0-2 ↔ 3-5 or 6-8). No constraint row touches dims 9-11, so in the projection QP the
  velocity column just rides along, kept at its FM value by the identity cost.
- **USED for trajectory selection** — `temporal_consistency` / `minimum_projection_cost`
  compare the full 9D predicted obs (which includes V3) across the 4 candidates
  (`policies.py L62-76`).
- **USED for logging** — written to the foresight plan.

So V3 is "inert in the control path" but not literally dead: it shapes which candidate is
chosen and appears in logs.

### 7.4 "The action is a 1st-order model — where does velocity go in it?"

This is the heart of the confusion. The DPCC dynamics model is **position-only kinematics**:

```
p[t+1] = p[t] + action[t]          # 1st-order, action = position delta. NO velocity term.
```

A **full 2nd-order** model would chain velocity in:

```
p[t+1] = p[t] + dt * v[t]          # position integrates velocity
v[t+1] = v[t] + dt * a[t]          # velocity integrates acceleration (action = accel)
```

The current scheme **collapses** this chain: the action IS the position delta directly, so the
velocity integration step is bypassed inside the FM/DPCC math. That is why velocity dims 9-11
have no constraint — **the dynamics model the projector enforces simply does not contain velocity.**

Velocity is reintroduced **outside** the FM/DPCC math, entirely inside the PID:
- `v_des` (V2) is reconstructed by finite-differencing the position command: `action / dt_fm`.
- `v` (V1) is the real measured rate fed to the PD damping term.

**In one sentence:** the FM/projector reason in position-delta space (1st-order kinematics,
no velocity); the velocity *physics* lives only in the cascaded PID, where real velocity (V1)
provides damping and a finite-difference of the commanded position (V2) provides the reference.

### 7.5 "Does the FM predicting velocity make sense?"

Yes, for two reasons — but it is currently **redundant for control**:

1. **Generative coherence**: the FM models the joint distribution of the full obs
   `[p_des | p | v]`. To produce realistic trajectories it must predict velocity consistent
   with the position changes it predicts. Dropping v from the obs would make the model blind
   to the drone's momentum.
2. **Conditioning**: at inference the FM is conditioned on the *real* current velocity (V1 in
   the obs), so it can pick actions appropriate to how fast the drone is already moving.

What is redundant: the FM's *predicted future* velocity (V3) is never used as the PID's `v_des`.
We instead derive `v_des` from the position delta. A future improvement (noted, not done) would
feed V3 directly as `v_des` for a smoother, less finite-difference-noisy velocity reference.

### 7.6 Robot-arm (IK / D3IL) contrast

| | D3IL robot arm | UAV (this work) |
|---|---|---|
| Action semantics | joint/end-effector **velocity** command | **position delta** `Δp_des` |
| Fed to | IK / low-level servo (tracks instantly) | cascaded PID (position→attitude→thrust) |
| Velocity role | action *is* the velocity | velocity is separate: V1 damping + V2 derived reference |
| Tracking lag | ~0 (servo) | nonzero (drone mass/inertia) → motivates fix_5 |

In D3IL the action and the velocity are the same object, so "where is velocity" never arises.
For the UAV the action is a position delta and velocity is a separate physical quantity handled
by the PID — which is exactly why this caused confusion.

---

## 8. Design Rationale & How to Justify It for Publication

This section answers the three hard questions:
1. Why is velocity in the state but **not a decision variable like the action**?
2. Why must the PID be fed `v_des` (desired velocity) at all?
3. The DPCC model is **1st-order, no velocity** — yet the FM predicts velocity and the PID
   uses 2nd-order dynamics. Is this a "Frankenstein," and how do we justify it in a paper?

### 8.1 Decision variables vs. state/feedback variables

The 12D transition mixes two fundamentally different kinds of quantity:

| Kind | Dims | What it is | Constrained? |
|---|---|---|---|
| **Decision variable** | action 0-2 (and its integral p_des 3-5) | what the planner *chooses* | YES — deriv constraint enforces self-consistency |
| **State / feedback** | p 6-8, v 9-11 | what the world *reports* (or the model predicts about it) | NO constraint on v; p only constrained in fix_5 |

The deriv constraint exists to make the planner's **decisions** self-consistent: the chosen
action sequence must integrate to the chosen position sequence (`p_des[t+1] = p_des[t]+action[t]`).
There is no analogous "decision" the planner makes about velocity — velocity is something the
drone *has*, not something the planner *commands*. That is why velocity sits in the transition
(the FM needs it for prediction + conditioning) but has no constraint binding it. **It is not
"missing from the transition" — it is present but plays a different role than the action.**

> The thing that *looks* missing is a **predicted/stored `v_des`**. The planner never stores a
> velocity command; it stores only the position-delta action and reconstructs `v_des = action/dt`
> at runtime. See 8.3 for whether that should change.

### 8.2 Why the PID needs all four: real `p`, real `v` (from MuJoCo) AND `p_des`, `v_des`

The outer-loop control law is (`flight_controller.py L89-91`):

```
a_cmd =  −Kp·e_p  −  Kd·e_v  +  a_des
      =  −Kp·(p − p_des)  −  Kd·(v − v_des)  +  a_des
            └── feedback ── reference ──┘
```

It has **four** position/velocity inputs, and they come in two error pairs. The structural
rule of any closed-loop PD controller:

> **An error term needs BOTH sides: the measurement (feedback) and the setpoint (reference).**
> `error = feedback − reference`. Drop either side and the loop breaks.

```
e_p = p − p_des     needs  p      (real, from MuJoCo — WHERE WE ARE)
                    and    p_des  (from planner — WHERE WE WANT TO BE)

e_v = v − v_des     needs  v      (real, from MuJoCo — HOW FAST WE ARE MOVING)
                    and    v_des  (derived from action — HOW FAST WE SHOULD MOVE)
```

#### Failure modes if you drop one side

**(a) Drop the real feedback (`p`, `v`) — keep only references → OPEN LOOP:**
```
a_cmd = −Kp·(0 − p_des) − Kd·(0 − v_des) = Kp·p_des + Kd·v_des
```
The command no longer depends on what the drone is actually doing. Any disturbance, model
error, or gust is never corrected → the drone drifts off and never recovers. This is pure
feed-forward, and a quadrotor is open-loop unstable — it falls.

**(b) Drop the references (`p_des`, `v_des`) — keep only real feedback → WRONG TARGET:**
```
a_cmd = −Kp·p − Kd·v
```
This is a stable spring-damper to the **origin at rest** (`p=0, v=0`). The drone flies to the
world origin and stops — it has no knowledge of the goal. Useless for tracking.

**(c) Drop only `v_des` (keep real `v`) → SLUGGISH / HIGH-LAG:**
```
a_cmd = −Kp·(p − p_des) − Kd·v
```
The damping term `−Kd·v` now fights **all** motion — it drives the drone toward *zero velocity
everywhere*. To move at all, the position error `(p − p_des)` must grow large enough to overpower
the damping. The drone crawls, always lagging behind a moving setpoint, and overshoots when the
setpoint stops. `v_des` is the **feed-forward reference**: it tells the controller "you are
*supposed* to be moving at this rate," so the damping only penalizes *deviation from intended
motion*, not the intended motion itself.

#### Why real `v` specifically (from MuJoCo, not a model)

The damping `−Kd·(v − v_des)` only stabilizes if `v` is the **true measured** velocity —
MuJoCo's `data.qvel[:3]`, which includes every disturbance, contact, and aerodynamic effect.
Substituting a *modeled* or *FM-predicted* velocity would inject model error into the damping
term and degrade (or destabilize) the loop. That is precisely why the eval loop re-reads
`v = data.qvel[:3]` fresh inside every physics sub-step (`eval_fm_uav.py L380`), and why the
FM-predicted velocity (V3) is **not** substituted here.

#### Summary

| Input | Symbol | Source | Role | Drop it → |
|---|---|---|---|---|
| Real position | `p` | MuJoCo `qpos[:3]` | position feedback | open loop, drift/fall |
| Commanded position | `p_des` | planner (`p + action`) | position reference | flies to origin |
| Real velocity | `v` | MuJoCo `qvel[:3]` | velocity feedback (damping) | undamped, oscillates/falls |
| Desired velocity | `v_des` | `action / dt_fm` | velocity reference (feed-forward) | sluggish, high-lag |

The expert PID gets the analytic `v_des` from the trajectory spline (`trajectories.py` returns
`p, v, a`); eval must synthesize `v_des = action/dt_fm` to feed the *same* controller. The
controller structure is fixed — the planner must supply both position and velocity references,
and MuJoCo supplies both real feedbacks.

### 8.3 The "Frankenstein" question — and three ways to justify it

**The tension, stated honestly:** the FM predicts the full 12D state (including velocity), the
DPCC projector enforces only a **1st-order position kinematic** model (`p[t+1]=p[t]+action`,
no velocity), and the PID realizes the command on the **true 2nd-order** drone dynamics. Three
different system models in one stack. A reviewer will ask: is this principled or ad hoc?

It is principled. Here are three framings, strongest first.

#### Framing A — Standard hierarchical control (timescale separation) ✅ recommended

This is the textbook robotics stack, not an invention:

```
 FM + DPCC   →  outer loop / PLANNER   (33 Hz)  — kinematic feasibility, obstacle avoidance
 Cascaded PID → inner loop / TRACKER  (~500 Hz) — dynamic feasibility, stabilization
```

- The planner reasons about **where to go** (geometry, collisions, waypoint continuity). For
  that, a kinematic model is the *correct* abstraction — adding full rotor dynamics to the
  planner buys nothing for path-level decisions and costs real-time feasibility.
- The tracker reasons about **how to physically get there** (mass, inertia, thrust limits,
  damping). It owns the 2nd-order dynamics.
- This separation is exactly what Diffuser / Decision-Diffuser / and DPCC itself do: they plan
  in state space and delegate execution to a low-level controller or inverse-dynamics model.
  Our work inherits that contract; the UAV's low-level controller is the cascaded PID.

**Justification sentence for the paper:** *"We adopt the standard planner/tracker hierarchy:
the flow-matching policy with DPCC projection acts as a kinematic motion planner at 33 Hz,
while a cascaded PID tracker enforces the platform's 2nd-order dynamics at the physics rate.
Dynamic feasibility is guaranteed by the tracker within its operating envelope, freeing the
planner to use a lightweight first-order model that is solvable in real time."*

#### Framing B — The 1st-order model IS a single-integrator (canonical planning abstraction)

The action is a position delta over `dt_fm`, so `action/dt_fm` is literally a **velocity
command**, and the constraint

```
p[t+1] = p[t] + action[t] = p[t] + v_cmd[t]·dt_fm
```

is exactly forward-Euler integration of a **single integrator** `ṗ = v_cmd`. The single
integrator (velocity-controlled point mass) is *the* canonical model for trajectory planning in
robotics and multi-agent literature. So the planner is not using a broken dynamics model — it is
using the single-integrator abstraction, with the PID providing the realization on the true
double-integrator-plus-attitude plant. The measured velocity in the state is feedback for that
realization, not a planning decision variable.

#### Framing C — Real-time tradeoff (the quantitative backstop)

Full 6-DOF nonlinear MPC at 33 Hz (30.3 ms budget) is infeasible with an NLP solver
(50–500 ms/solve — see `DESIGN_fix5_anchor_p_integration.md` §15). The 1st-order QP projects in
~milliseconds. The first-order model is a deliberate real-time choice, not an oversight.

### 8.4 Three-way comparison: SafeFlowMPC (full dynamics) vs DPCC (1st-order) vs UAV-mix

The three approaches differ in **where the system dynamics live** and **how many models** the
stack uses. This is the clearest way to position our work.

#### The three dynamics models, side by side

**SafeFlowMPC direction — full nonlinear dynamics *inside* the projection** (what they claim
for their robot): the projection/MPC constrains the generated trajectory onto the platform's
true dynamics manifold:

```
p[t+1] = p[t] + v[t]·dt
v[t+1] = v[t] + (R[t]·f_thrust[t]/m − g − drag(v[t]))·dt        ← full rigid-body + rotor
R[t+1] = R[t]·exp(ω[t]·dt)                                       (SO(3) integration)
ω[t+1] = ω[t] + J⁻¹(τ[t] − ω[t]×J·ω[t])·dt
```
One unified model: the planner's output is *already* dynamically feasible — no separate tracker
needed in principle. Requires an NLP solver and a known `m, J, drag` model.

**DPCC original — single-integrator (1st-order), robot-arm domain:**
```
p[t+1] = p[t] + action[t]        ← Euler, position only.  No v, no R, no ω.
```
The arm servo tracks commanded position essentially perfectly, so commanded ≈ real and **no
separate dynamics model is needed** — the kinematic plan is directly executable.

**UAV-mix (this work) — single-integrator plan + 2nd-order tracker:**
```
PLANNER (FM+DPCC):  p[t+1] = p[t] + action[t]      ← same single integrator as DPCC
TRACKER  (PID):     a_cmd = −Kp·(p−p_des) − Kd·(v−v_des)   then  R,ω,thrust as in MuJoCo
```
The drone does *not* track perfectly (it has mass/inertia/lag), so we re-introduce the dynamics
in a **separate** low-level PID tracker rather than inside the projection.

#### Comparison table

| Aspect | SafeFlowMPC (full dyn.) | DPCC original (arm) | UAV-mix (this work) |
|---|---|---|---|
| Model *in projection* | full nonlinear rigid-body+rotor | single integrator (Euler pos) | single integrator (Euler pos) |
| States constrained | p, v, R, ω | p only | p only (v in state but **unconstrained**) |
| Solver | NLP (nonlinear) | convex QP | convex QP |
| Solve time | ~50–500 ms | ~1–5 ms | ~1–5 ms |
| Real-time @ 33 Hz (30 ms) | marginal → impossible | yes | yes |
| Needs `m, J, drag` model | yes | no | no (PID has gains, not a dyn. model) |
| Velocity role | decision variable (constrained) | not modeled | measured feedback + derived ref (PID) |
| Where dynamics live | **inside projection** | nowhere (perfect arm) | **inside the PID tracker** |
| # of system models | 1 (unified) | 1 (kinematic suffices) | 2 (kinematic plan + dyn. tracker) |
| Dynamic feasibility | guaranteed by projection | trivially (arm tracks) | guaranteed by tracker within envelope |

#### The one-sentence positioning

- **SafeFlowMPC unifies**: one model does planning *and* dynamics, paying NLP cost.
- **DPCC (arm) needs only one** because the plant tracks perfectly — kinematics = execution.
- **UAV-mix separates**: kinematic planning (cheap, real-time QP) + a dynamics-aware tracker.
  This is the only one of the three that has **two models to reconcile** — which is exactly the
  "Frankenstein" worry. The reconciliation is the planner/tracker hierarchy (§8.3-A), and the
  residual mismatch (drone lags the plan) is what **fix_5 / `anchor_to_p`** addresses by binding
  the kinematic constraint to the *real* position the tracker actually achieved.

So the UAV-mix is not "DPCC done wrong" — it is "DPCC's single-integrator planner, kept for
real-time feasibility, plus the dynamics layer the arm never needed, factored into the tracker."
SafeFlowMPC sits at the other extreme (everything in projection); we sit at the standard-robotics
middle (separation of concerns). The honest trade we report: we give up the projection's
guarantee of dynamic feasibility in exchange for real-time speed, and recover dynamic feasibility
through the tracker.

### 8.5 If a reviewer demands a *consistent* model (the 2nd-order alternative)

If we want the FM/DPCC to actually contain velocity (no separation argument needed), the
principled upgrade is a **double-integrator** constraint chain. Redefine the action as
acceleration and add two coupled deriv constraints:

```
p[t+1] = p[t] + dt · v[t]          # deriv on (p, v)
v[t+1] = v[t] + dt · a[t]          # deriv on (v, action=accel)
```

Then:
- velocity becomes a constrained decision variable (no longer a free passenger),
- `v_des` is read directly from the predicted `v` column (V3 stops being inert — it becomes V2),
- the projector enforces dynamic feasibility to first order, matching the velocity in the state.

**Cost:** requires retraining with `action = acceleration` and re-tuning the PID to accept the
new reference. This is the natural "consistency" ablation to offer as future work or as a second
experimental condition. For *publish-now*, Framing A + B is the honest and standard justification;
the 2nd-order model is the "we also tried / future work" upgrade.

### 8.6 Is the dynamics "learned by the NN" or analytical? (IK vs flatness vs MPC)

> ⚠ **Verify against the SafeFlowMPC paper before citing specifics.** The architectural
> distinctions below are sound regardless; the exact SafeFlowMPC internals should be confirmed.

A common confusion: "does SafeFlowMPC compute dynamics through the neural network?"
**No — in model-predictive / projection-based methods the dynamics are an *analytical* model,
not learned.** Two separate things are happening:

1. **The NN (flow matching) proposes** candidate trajectories. It learns the *distribution of
   good trajectories* from data — it does **not** compute physics.
2. **The projection/MPC enforces** an *analytical* dynamics model as constraints (the `f(x,u)`
   equations of motion). This is hand-written physics with known `m, J, drag`, **not** an NN.

So "SafeFlowMPC builds the full dynamics" means it writes the **analytical** rigid-body+rotor
equations as projection constraints. The NN proposes; the analytical model disposes. (A variant
exists where a *learned surrogate* dynamics model replaces the analytical one — §8.5 option 2 —
but that is a different design and not the default.)

#### Where "IK" fits — it doesn't, for drones

IK (inverse kinematics) is a **manipulator** concept: given a desired end-effector pose, solve
for joint angles. Drones have no joints to solve — so there is no IK in a drone stack. The drone
analogs of "turn a desired pose into commands" are:

| Approach | How desired position → motor commands | Used by |
|---|---|---|
| **Cascaded PID** (ours) | feedback control: position→accel→attitude→thrust | this work, expert data |
| **Differential flatness** | algebraic inversion (quadrotor is differentially flat: p + yaw → all states/inputs) | flatness-based planners |
| **Full dynamics MPC** | optimize over the analytical model online | SafeFlowMPC direction |

So when D3IL "feeds the action to IK," that is the arm case. Our drone has **no IK** — it has a
cascaded PID tracker. SafeFlowMPC's "full dynamics" replaces the *need* for a separate tracker by
solving the dynamics inside the optimizer. None of the three uses the NN to compute physics.

#### "So the NN computes all the orders, then an IK-style solver processes it again?"

This is *almost* right — the two-stage picture is correct, but the second stage is a **projection
(constrained optimization)**, not an IK solver. Let me make the two stages precise.

**Stage 1 — NN proposes the full-state trajectory ("all the orders").**
The flow-matching vector field outputs a trajectory in the chosen state representation. If the
state includes higher-order terms (position *and* velocity, possibly acceleration), the NN does
predict all of them — so yes, "the NN gives all the orders that are in the state."
- *State-only* formulation: NN outputs states only; the control inputs are **not** in the NN
  output and must be recovered later.
- *State-action* formulation (**ours** — 12D includes the action): NN outputs states **and**
  the action together, but with no guarantee they are mutually consistent.

**Stage 2 — Projection makes it dynamically feasible (NOT IK).**
The projector solves:
```
ẑ = argmin_z  ½‖z − z_NN‖²     subject to   dynamics(z) = 0   (+ safety/obstacle)
```
This is a **nearest-feasible-point projection**, an optimization. It is not inverse kinematics.
The difference matters:
| | Inverse Kinematics | Projection (DPCC / SafeFlowMPC) |
|---|---|---|
| Question | "what joints achieve this pose?" | "what is the nearest *dynamically valid* trajectory to the NN's guess?" |
| Method | geometric/iterative inversion | constrained QP/NLP minimizing distance |
| Output | joint configuration | corrected full trajectory (states [+actions]) |

**Stage 3 — recover the control inputs (this is the only place an "IK-style inversion" appears).**
Once you have a dynamically-feasible state trajectory, you still need the motor commands.
- In a **full-dynamics** method, the dynamics constraint already *couples states to inputs*, so
  the projected solution **contains the inputs** (or they are read off by inverting `f(x,u)`).
  For a quadrotor this inversion is **differential flatness** — the genuine drone analog of IK
  (algebraic recovery of inputs from the position/yaw trajectory). **This is the grain of truth
  in your "IK-style dynamic solver" intuition** — but it is flatness, and it only exists when the
  full dynamics are in the model.
- In **our UAV-mix**, the projection enforces only 1st-order position kinematics, so it does
  **not** recover inputs. We extract `action[0]`, hand it to the **PID**, and the PID recovers
  the motor thrusts via *feedback control* (not inversion). The PID is our "Stage 3," and it is
  feedback, not an IK/flatness solver.

**Putting it together — your sentence, corrected:**

> *In SafeFlowMPC: the NN computes the full-state trajectory (all orders in the state), then a
> **projection** (constrained optimization, not IK) snaps it onto the analytical dynamics
> manifold, and because the dynamics couple states to inputs, the motor commands fall out (via
> differential-flatness inversion — the closest thing to "IK" for a drone).*
>
> *In our UAV-mix: the NN computes the trajectory, a **projection** enforces only 1st-order
> position kinematics, and the motor commands are produced separately by the **PID tracker via
> feedback** — there is no dynamics inversion in our projection at all.*

So: NN = proposer (all stages), projection = feasibility optimizer (not IK), input-recovery =
flatness inversion (SafeFlowMPC) **or** PID feedback (ours). The "IK-style solver" only literally
exists, in flatness form, when the full dynamics live in the projection.

### 8.7 Why velocity is MANDATORY — the drone is a 2nd-order plant (control theory)

This is the crisp, non-hand-wavy answer to "why isn't first-order enough / why does it feel weird."

**The confusion is conflating two different things:**
- the **plan's** model (1st-order / single integrator) — an *abstraction*, and
- the **plant's** model (the real drone, 2nd-order) — the *physics*.

First-order is enough **to describe the plan**. It is *not* enough **to stabilize the plant**,
because the real drone is a second-order system and that is a hard fact of physics, not a choice:

```
You do NOT command position. You command motor thrusts → force.
Newton:  F = m·a   →   you control ACCELERATION.
         a integrates to velocity:   v̇ = a
         v integrates to position:   ṗ = v
So the path from "what you command" (force/accel) to "what you want" (position) passes
THROUGH velocity. Velocity is the intermediate state you cannot skip.
```

#### The rigorous reason: a double integrator cannot be stabilized by position feedback alone

Model the drone (per axis) as a double integrator: `p̈ = u`. In the Laplace domain the plant is
`P(s) = 1/s²`.

- **Position-only (proportional) feedback** `u = −Kp·(p − p_des)`:
  closed-loop characteristic equation `s² + Kp = 0` → poles at `s = ±j·√Kp` — **purely
  imaginary**. The system **oscillates forever** (marginally stable); with any lag it goes
  unstable. *First-order/position-only feedback literally cannot hold a drone.*

- **Add velocity (derivative) feedback** `u = −Kp·(p − p_des) − Kd·(v − v_des)`:
  characteristic equation `s² + Kd·s + Kp = 0` → poles move into the **left half-plane** for
  `Kd > 0` → **asymptotically stable** (damped). The `Kd·v` term is the damping that pulls the
  oscillation down.

**Conclusion:** velocity feedback is *mathematically required* to stabilize a 2nd-order plant.
This is why the PID *must* have velocity — it is not weird, it is the textbook double-integrator
result. The planner can stay first-order (it only emits references); the tracker must be
second-order (it stabilizes the real plant), and second-order stabilization needs velocity.

#### So "why is velocity in the state at all?"

Because the FM is conditioned on the full state to *predict good references*, and because the
tracker needs *measured* velocity for the `Kd` term. The planner's model ignoring velocity does
not make the plant's velocity disappear — it just delegates handling it to the tracker.

### 8.8 SafeFlowMPC *can* build the full model — should WE?

Short answer: **No for "publish now"; the linear double-integrator (§8.5) is the right
middle-ground if we want any dynamics in the projection; full nonlinear dynamics is overkill
here.** Reasoning:

| Factor | Full nonlinear dynamics (SafeFlowMPC) | Our PID-as-dynamics-layer | 2nd-order linear (§8.5) |
|---|---|---|---|
| Real-time @ 33 Hz | NLP 50–500 ms → **misses budget** | QP ~ms ✓ | QP ~ms ✓ |
| Needs accurate `m, J, drag` | **yes** → sim2real gap | no (PID uses true MuJoCo feedback) | partial (needs `dt`, not full plant) |
| Dynamic feasibility in plan | guaranteed | not in plan (tracker handles it) | first-order guaranteed |
| Engineering effort | high (NLP, model ID, tuning) | already done | moderate (retrain action=accel) |
| Novelty | high | low | medium |

**When first-order planning genuinely fails** (and you'd actually need dynamics in the plan):
high-speed flight near obstacles, where the plan must respect **braking distance** — a
first-order planner can command "stop here" at a point the 2nd-order drone physically cannot
stop at in time. At the speeds in our scenes (corridor/pillars, low-speed waypoint following),
the PID's envelope covers the gap and first-order planning is adequate.

**Recommendation:**
1. **Publish now** with the hierarchical framing (§8.3-A, §8.6). It is standard and defensible.
2. Offer the **2nd-order linear double-integrator** (§8.5) as an ablation — it folds velocity
   into the projection *without* NLP, directly answering "why not put velocity in the model."
3. Treat **full nonlinear dynamics** as explicit future work, justified only if you move to a
   high-speed / tight-obstacle regime where braking dynamics must enter the plan. Do **not**
   adopt it just for consistency — it trades real-time feasibility and adds a sim2real model
   dependency the PID currently avoids.

The deciding principle: **put dynamics in the projection only when a constraint you care about
(e.g., collision at speed) depends on them.** Otherwise the tracker is the cheaper, exact place
to enforce dynamics — it already uses the plant's true response via feedback.

### 8.9 Recommended paper framing (one paragraph)

> *We deliberately separate kinematic planning from dynamic tracking. The flow-matching policy
> with DPCC projection is a real-time kinematic planner modeling the UAV as a single integrator
> (`p[t+1]=p[t]+action`), conditioned on the full measured state (position and velocity) so it
> can anticipate momentum. A cascaded PID tracker closes the loop on the true second-order
> dynamics, using measured velocity for damping and a velocity reference derived from the planned
> position increment. This hierarchy keeps projection within the 33 Hz real-time budget — full
> nonlinear MPC is infeasible at this rate — while the tracker guarantees dynamic feasibility.
> Predicted velocity is retained in the generative model for trajectory coherence and candidate
> selection; folding it into the projection as an explicit double-integrator constraint is a
> consistency extension we leave to future work.*

---

## 9. Summary Table

| Stage | Data | Shape | Code reference |
|---|---|---|---|
| FM input (conditioning) | `[p_des \| p \| v]` normalized | `(9,)` | `eval_fm_uav.py L343-346` |
| FM output (batch) | `[action \| p_des \| p \| v]` per step | `[4 × 8 × 12]` | `policies.py L52-54` |
| Projector I/O | same H×12D (QP modifies in place) | `[4 × 8 × 12]` | `projection.py L337-401` |
| Extracted action | `action[0]` = Δp_des step 0, chosen candidate | `(3,)` | `policies.py L88-92` |
| PID setpoint | `p_des` (V cmd), `v_des` (V2, derived) | `(3,)+(3,)` | `eval_fm_uav.py L369-383` |
| PID feedback | real `p`, real `v` (V1) | `(3,)+(3,)` | `eval_fm_uav.py L379-383`, PD law `flight_controller.py L88-91` |
| MuJoCo control | thrust + attitude `u` | `(4,)` | `eval_fm_uav.py L383` |
| FM-predicted v (V3) dims 9-11 | traj-selection + logging | — | never reaches PID / MuJoCo |

**One-line answer**: From the *FM output tensor*, only `action[0]` (3D Δp_des) enters the
physics. But velocity is NOT unused — real velocity (V1, from MuJoCo) is both FM input context
AND the PID's damping feedback, and `v_des` (V2, derived from the action) is the PID reference.
The only velocity that is inert in the control path is the FM's *predicted* velocity column (V3,
tensor dims 9-11), which only affects candidate selection and logging. See §7.3–7.6.
