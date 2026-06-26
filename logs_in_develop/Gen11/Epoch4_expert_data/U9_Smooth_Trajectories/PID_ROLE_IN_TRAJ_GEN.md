# PID Role in Gen11 E4 U9 Trajectory Generation

**Document scope:** Gen11, Epoch 4, U9 Smooth Trajectories  
**Date written:** 2026-06-23  
**Author note:** Covers both the mathematical role of the cascaded PID and the exact
source-file + line locations where each concept lives in the codebase.

---

## FAQ — Three Questions Answered First

### Q1: Did we build the PID, or does MuJoCo already have one?

**We built it.** `CascadedPID` in `uav_env_test/flight_controller.py` is entirely our code.

MuJoCo is a *forward-dynamics physics engine*. Its job is:

> Give me forces/torques at the actuators → I integrate Newton's laws → I tell you the new positions and velocities.

`data.ctrl[:4]` in MuJoCo expects **raw motor thrust values in Newtons** — nothing more.
MuJoCo has absolutely no concept of "fly to position XYZ". It does not contain any
flight controller, position controller, or attitude controller. We had to implement
the full cascaded PID stack ourselves.

---

### Q2: Is "generate geometric path → PID sim → collect sim data" a good method?

**Yes — it is the standard approach for UAV expert data collection without a real robot.**

The method works in two clean layers:

| Layer | What it does | Who knows about state? |
|-------|-------------|----------------------|
| Geometric planner (`blended_path`) | Defines *what* the drone should do: the reference curve `p_des(t)`, `v_des(t)`, `a_des(t)` | Stateless — pure math, no idea where drone is |
| PID + MuJoCo | Simulates *how* the drone physically follows that curve, including inertia, motor limits, attitude coupling, and tracking error | Full state access: `p, v, q, ω` every step |

The data you collect is physically realistic because:
- The MuJoCo model (mass, inertia, rotor geometry, contacts) matches the real Skydio X2 spec
- Tracking error (`p − p_des`) reflects real quadrotor dynamics, not idealized motion
- Motor saturation and contact are simulated, so bad episodes look bad and get rejected

The alternative — hand-flying a real drone through corridors and pillars — is expensive,
slow, and extremely dangerous near obstacles. Sim + PID is the right tool here.

---

### Q3: Why must it be PID? Why not just "command the UAV in MuJoCo" directly?

**Because there is no such thing as a direct position command in a physics simulator.**

MuJoCo's actuators for this UAV are 4 motor thrust scalars. To move the drone from its
current position to a desired position you must answer the physics question:

```
"What 4 thrust values, applied right now, will cause the drone to accelerate
 toward p_des while staying level, counteracting gravity, and not spinning?"
```

Answering that question requires all four of these steps in order:

```
1. What net force do I need?
   → depends on position error, velocity error, gravity, inertia

2. What attitude (body tilt) do I need to aim that force correctly?
   → depends on current orientation vs. desired thrust direction

3. What torques do I need to reach that attitude?
   → depends on current attitude error, angular velocity, gyroscopic coupling

4. How do I split (total_thrust, τ_x, τ_y, τ_z) across 4 motors?
   → depends on motor arm geometry from the MuJoCo XML
```

**That four-step computation IS the PID** (steps 1→2 = outer position loop,
steps 2→3 = inner attitude loop, step 4 = allocation matrix inversion).

You could replace PID with MPC, LQR, or geometric control — but you **cannot skip the
computation**. There is no "just tell the drone to go somewhere" in a Newtonian simulator.
A physics engine only accepts forces; all position-level intent must be converted to forces
first, and a controller is the thing that does that conversion.

---

## TL;DR — Why PID Exists and What It Produces in the Data

### The concrete problem

We need **expert demonstration data** for imitation learning.
A training sample is one physics step:

```
obs  = [ p(xyz),  v(v_xyz),  q(quaternion) ]   ← actual drone state
act  = [ u0, u1, u2, u3 ]                       ← motor thrusts  (the "action")
ref  = [ p_des(xyz_des) ]                        ← what the planner wanted
```

**The trajectory planner (`blended_path`, `traverse_line`) only produces `p_des`, `v_des`, `a_des`.
It is a math function. It does not move the drone. It has no idea where the drone actually is.**

To get `p`, `v`, and `act` you must physically simulate the drone — which requires
motor commands. PID is the thing that computes those commands.

### What would happen without PID

| Without PID | Consequence |
|-------------|-------------|
| No motor thrusts `u` | MuJoCo has nothing to step with → drone falls under gravity |
| No actual flight | `p` and `v` are never updated by tracking behaviour |
| No `act` column | Dataset is empty — there is nothing for the policy to imitate |

**In other words: without PID there is no dataset, just a geometric curve floating in space.**

### What PID does, step by step, every physics tick

```
Step k:

  Planner:   p_des, v_des, a_des = traj_fn(k * dt)     # purely geometric
  State:     p, v, q, ω          = read from MuJoCo data  # actual drone
  
  PID:       u = CascadedPID.compute(p, q, v, ω,
                                     p_des, v_des, a_des, yaw_des)
             #  ↑ this is the only place xyz_des touches the drone
  
  MuJoCo:    data.ctrl[:4] = u
             mj_step(model, data)         # physics advances
             
  Record:    steps.append({
               'p':     p,           # xyz actual
               'v':     v,           # v_xyz actual
               'p_des': p_des,       # xyz desired (from planner)
               'q':     q,           # orientation (for attitude rendering)
             })
             # NOTE: u is NOT saved here — it is regenerated at training time
             # from the obs. The 'action' in the FM-PCC sense is p_des itself
             # (the target given to the policy), not the raw motor thrust.
```

### So what role does PID play in the variables you see in training?

| Variable | Who produces it | PID's role |
|----------|----------------|-----------|
| `p_des` (xyz_des) | trajectory planner — independent of PID | PID consumes this as its setpoint |
| `p` (xyz) | MuJoCo physics, stepped by `u` | **PID is the sole reason `p` tracks `p_des`** rather than drifting randomly |
| `v` (v_xyz) | MuJoCo physics | Same — if PID saturates, `v` diverges from `v_des` |
| `u` (motor thrusts) | **PID output** | PID is literally the function that maps (state, reference) → action |
| `q` (quaternion) | MuJoCo physics, driven by PID torques | PID inner loop holds the attitude needed to point thrust in the right direction |

### The causal chain in one line

```
xyz_des  →[PID error + feedforward]→  motor u  →[MuJoCo]→  xyz, v_xyz
```

Remove PID → the chain breaks at the first arrow → you have `xyz_des` and nothing else.

---

## 1. High-Level Picture: What PID Does in This Pipeline

In Gen11 E4, *trajectory generation* and *trajectory tracking* are two distinct
phases that are tightly coupled:

```
┌─────────────────────────────────┐
│ Trajectory planner              │
│  blended_path / traverse_line   │
│  → p_des(t), v_des(t), a_des(t) │   ← reference signal (open-loop geometry)
└────────────────┬────────────────┘
                 │ feed-forward + error
                 ▼
┌─────────────────────────────────┐
│ CascadedPID                     │
│  outer loop: position PD        │
│  inner loop: attitude SO(3)     │
│  → motor thrusts u[0:4]         │   ← closed-loop actuation
└────────────────┬────────────────┘
                 │ u applied
                 ▼
        MuJoCo physics step
                 │
                 └──→  recorded (p, v, p_des, q) → training dataset
```

The trajectory planner is *purely geometric*: it produces the reference path
`(p_des, v_des, a_des)` as a function of time but applies no forces.  The PID
is the *only actuator* that connects the reference to the simulated physics.
Everything in U9 that changed — blended corners, larger blend radius, centripetal
feedforward — exists to keep the PID *inside its actuator budget* while tracking
a continuously-moving reference.

---

## 2. Mathematical Formulation

### 2.1 Outer Loop — Position PD with Feed-forward

Given the reference position `p_des ∈ ℝ³`, velocity `v_des ∈ ℝ³`, and
acceleration `a_des ∈ ℝ³` from the trajectory planner, the PD law computes a
commanded acceleration:

```
e_p = p − p_des          (position error)
e_v = v − v_des          (velocity error)

a_cmd = −Kp_pos ⊙ e_p  −  Kd_pos ⊙ e_v  +  a_des
```

| Symbol | Default gain | Role |
|--------|-------------|------|
| `Kp_pos` | `[4, 4, 8]` (x/y/z) | Proportional position stiffness |
| `Kd_pos` | `[3, 3, 4]` (x/y/z) | Derivative velocity damping |
| `a_des` | from trajectory | Feed-forward removes steady-state lag |

The required world-frame thrust vector is then:

```
F_world = m · (a_cmd + g·ẑ)          g = 9.81 m/s²
```

This converts the kinematic command into a *force demand* that must be realised
by the rotors.  Without the `a_des` feed-forward term the drone would always
lag behind the moving reference; with it, the PD correction only needs to
cancel residual errors.

**Why `a_des` matters for U9:** The `blended_path` primitive returns the
centripetal acceleration at every fillet arc:

```
a_centripetal = ṡ²/r · (−r̂)          r̂ = unit vector toward arc centre
```

This term is included in `a_des` so the outer loop receives correct feed-forward
during corner traversal.  Without it, the PID would have to *discover* the lateral
force requirement from position error alone — introducing a tracking lag that
pushes the drone toward the obstacle on the inside of every turn.

### 2.2 Force → Desired Attitude

The force vector `F_world` defines the desired body-z axis (thrust direction):

```
b₃_des = F_world / ‖F_world‖
```

Combined with the desired yaw `yaw_des`, the desired rotation matrix `R_des ∈ SO(3)`
is constructed via a Gram-Schmidt-like orthogonalisation:

```
x_c    = [cos(yaw_des), sin(yaw_des), 0]ᵀ        (candidate body-x)
b₂_des = norm(b₃_des × x_c)                       (body-y, orthogonal to b₃_des)
b₁_des = b₂_des × b₃_des                           (body-x, right-hand completed)
R_des  = [b₁_des | b₂_des | b₃_des]               (3×3 rotation, columns)
```

### 2.3 Inner Loop — SO(3) Attitude PD (Lee 2010)

The attitude error is computed as the skew-symmetric part of the orientation
mismatch:

```
E   = ½ (R_des^T R − R^T R_des)         (skew-symmetric error matrix)
e_R = vee(E) = [E₂₁, E₀₂, E₁₀]ᵀ       (axis-angle-like error vector)
```

Gyroscopic compensation prevents the spinning rotors from coupling into the
attitude error:

```
gyro = ω_body × (J ⊙ ω_body)           J = inertia diagonal
τ    = −Kp_att ⊙ e_R  −  Kp_ω ⊙ ω_body  +  gyro
```

| Symbol | Default gain | Role |
|--------|-------------|------|
| `Kp_att` | `[70, 70, 4]` (roll/pitch/yaw) | Attitude stiffness |
| `Kp_omega` | `[2.5, 2.5, 1.0]` | Gyroscopic damping |

### 2.4 Thrust Scalar

The total thrust is the projection of the force demand onto the **current**
(not desired) body-z axis — this accounts for the attitude lag:

```
b₃    = R[:, 2]                (current body-z)
T     = F_world · b₃           (total thrust scalar)
T     = max(T, T_floor)        T_floor = 0.1·m·g  (prevents free-fall)
```

### 2.5 Motor Allocation

The wrench `[T, τₓ, τᵧ, τ_z]` is mapped to four individual motor thrusts via
the allocation matrix `M` built from the MuJoCo model geometry:

```
u = M⁻¹ · [T, τₓ, τᵧ, τ_z]ᵀ
```

Each column of `M` is `[1, r_y_i, −r_x_i, κ_i]` where `(r_x_i, r_y_i)` is
the motor site position in body frame and `κ_i` is the yaw torque coefficient
from the MuJoCo actuator gear.

### 2.6 Thrust-Priority Saturation Recovery

When `u` exceeds `[u_min, u_max]`, a **thrust-priority** scheme scales the
torque contribution to bring every motor into bounds while preserving total
thrust:

```
thrust_cmd  = mean(u)
torque_comp = u − thrust_cmd

scale = clamp(min_i { (u_max − thrust_cmd)/torque_comp_i | torque_comp_i > 0,
                       (thrust_cmd − u_min)/|torque_comp_i| | torque_comp_i < 0 },
              0.5, 1.0)

u_final = clip(thrust_cmd + scale·torque_comp, u_min, u_max)
```

The `0.5` floor on `scale` ensures attitude authority is *never more than
halved*, even under severe saturation.  This is critical for U9 because the
LRL/RLR fillets at `r=0.30` demanded `8.6 m/s²` lateral accel, which saturated
motors at 29–78% of steps, leaving insufficient torque to hold attitude → contact.

---

## 3. How the PID Shapes the Dataset Quality

Because the raw rollout `(p, v, p_des, q)` **is** the training dataset, the
PID's tracking quality directly determines what the flow-matching model learns:

| PID behaviour | Effect on dataset |
|---------------|-----------------|
| Good tracking (`‖e_p‖ small`) | `p_des − p` is a small, consistent signal; model learns tight corridor-following |
| Motor saturation | Drone drifts, contacts obstacle → episode **rejected** (not saved) |
| Large tracking lag (no `a_des` feedforward) | Noisy, lag-biased `p_des` data → model learns to anticipate drift, not the geometry |
| Gain variant diversity (`pid_default` / `pid_high_gain` / `pid_low_gain`) | Injects natural tracking-error variation → training-time robustness |

### 3.1 Gain Variants

Three gain presets are sampled across collected episodes:

| Variant | `kp_scale` | `kd_scale` | Purpose |
|---------|-----------|-----------|---------|
| `pid_default` | 1.0 | 1.0 | Baseline |
| `pid_high_gain` | 1.2 | 1.0 | Tighter tracking, more motor use |
| `pid_low_gain` | 0.8 | 0.9 | Softer, larger but smoother error |

Only `pid_default` and `pid_high_gain` are sampled by default in `sample_trial_specs`.

---

## 4. U9-Specific PID Considerations

### 4.1 Why U8 Stop-and-Go Was Bad for the PID

The pre-U9 trajectory chain used `traverse_line` per segment, which forces
`v = 0` at every waypoint.  The PID receives `v_des = 0` and `a_des = 0` at
every joint, so the actual drone overshoots the waypoint and the PID must damp
out a velocity excursion before the next segment's cosine ramp-up begins.  This
results in:

- High position error peaks at joints → contact risk near obstacles
- Oscillatory dataset entries during deceleration
- Numerically, zero velocity at joints is a degenerate training target:
  the model sees many near-identical `(p≈p_des, v≈0)` pairs

### 4.2 What U9 Changed for the PID

U9 replaced per-segment chains with `blended_path` (one global cosine speed
profile):

```
s(t)      = L · ½(1 − cos(πt/T))         (arc-length along blended path)
ṡ(t)      = L · ½π/T · sin(πt/T)         (tangential speed)
s̈(t)      = L · ½(π/T)² · cos(πt/T)      (tangential accel)
```

On fillet arcs the returned acceleration includes centripetal:

```
a_des = s̈ · tangent  +  ṡ²/r · (−r̂)
```

The PID now has correct feed-forward *through corners*, so it sees near-zero
error at every point on the path (vs. a large spike at each old waypoint joint).

### 4.3 The Fillet Saturation Problem and the Fix_1 Radius Increase

Peak centripetal acceleration at a fillet scales as:

```
a_peak = ṡ_peak² / r = (π·L / 2T)² / r
```

At `r = 0.30 m`, `T = 10 s`, LRL/RLR pillars path (`L ≈ 14 m`):

```
a_peak ≈ (π × 14 / 20)² / 0.30 ≈ 8.6 m/s²   (≈ 0.88 g lateral)
```

The PID outer loop translates this lateral demand into `F_world`, which tilts
`b₃_des` away from vertical.  The allocation matrix must then produce the
required torque *simultaneously* with total thrust — at 8.6 m/s² this
exceeded the `u_max` budget for 29–78% of steps in the failed episodes.

Fix_1 raised `BLEND_RADIUS = 0.30 → 0.45 m`:

```
a_peak ≈ (π × 14 / 20)² / 0.45 ≈ 5.7 m/s²   (≈ 0.58 g lateral)  ✅
```

The 34% reduction in peak acceleration brought the demand inside the PID's
thrust-priority budget, eliminating saturation on the mixed-homotopy fillets.

---

## 5. Exact Code Locations

### 5.1 `uav_env_test/flight_controller.py` — The PID Implementation

| Line(s) | What |
|---------|------|
| **L1–15** | Module docstring: cascaded structure overview (position PD → attitude SO(3) → allocation) |
| **L64–66** | `Kp_pos = [4, 4, 8]`, `Kd_pos = [3, 3, 4]` — outer loop position gains |
| **L69–70** | `Kp_att = [70, 70, 4]`, `Kp_omega = [2.5, 2.5, 1.0]` — inner loop attitude gains |
| **L60–62** | `u_hover`, `u_max`, `u_min` — motor bounds used by saturation recovery |
| **L73** | `thrust_floor = 0.1·m·g` — prevents free-fall during recovery |
| **L76–77** | `last_raw_saturated`, `last_torque_scale` — saturation telemetry read by `generator.py` |
| **L88–91** | **Outer loop:** `e_p`, `e_v`, `a_cmd` computation (position PD + feedforward) |
| **L94** | `F_world = m·(a_cmd + g·ẑ)` — force demand |
| **L101** | `b3_des = F_world / ‖F_world‖` — desired thrust direction |
| **L104–114** | Gram-Schmidt: `b₂_des`, `b₁_des`, `R_des` construction from `yaw_des` |
| **L117** | `R = quat_to_rot(q)` — current rotation from MuJoCo quaternion |
| **L120–122** | **Inner loop:** SO(3) error `E`, `e_R` (Lee 2010 vee-map) |
| **L124–125** | Gyroscopic compensation `gyro`, torque `τ` |
| **L128–131** | Thrust scalar `T = F_world · b₃`, thrust floor clamp |
| **L134–135** | Allocation `u = M⁻¹ · wrench` |
| **L142–156** | **Thrust-priority saturation recovery** with 0.5 torque floor (U6 C2) |

### 5.2 `uav_env_test/trajectories.py` — Trajectory Primitives (Feed-forward Source)

| Line(s) | What |
|---------|------|
| **L65–89** | `traverse_line` — cosine speed profile, single segment; `a_des` is purely tangential |
| **L114–226** | `blended_path` — U9 smooth primitive; returns centripetal `a_des = ṡ²/r · curv` at fillets |
| **L215–216** | Arc-length parameterisation: `s(t)` and `ṡ(t)` from global cosine profile |
| **L217** | `s̈(t)` — tangential acceleration from global cosine |
| **L221–223** | Position `p`, velocity `v = ṡ·tang`, acceleration `a = s̈·tang + ṡ²·curv` returned per step |
| **L183–188** | `_arc` closure: computes `(position, tangent, curvature = −r̂/r)` for arc elements |

### 5.3 `uav_expert_data_collect/trajectories.py` — Scene Path Factories

| Line(s) | What |
|---------|------|
| **L32** | `BLEND_RADIUS = 0.45` — the Fix_1 tuned value (was 0.30); controls `ṡ²/r` peak |
| **L45** | `PILLAR_SAFETY = 0.08` — 8 cm clearance above zero-contact, sized for PID tracking error |
| **L66–109** | `pillar_path` — calls `blended_path(wps, BLEND_RADIUS, T, yaw)` |
| **L123–171** | `s_curve_scene_path` — calls `blended_path(wps, BLEND_RADIUS, T, yaw)` |
| **L174–176** | `empty_path` — single `traverse_line`, no PID saturation concern |

### 5.4 `uav_expert_data_collect/generator.py` — Rollout Loop (PID Called Here)

| Line(s) | What |
|---------|------|
| **L23** | `from uav_env_test.flight_controller import CascadedPID` — import |
| **L72–77** | `GAIN_VARIANTS` dict — the three `kp_scale` / `kd_scale` multiplier presets |
| **L114–119** | `_make_pid(model, gain_variant)` — instantiates `CascadedPID`, scales `Kp_pos` / `Kd_pos` |
| **L221** | `pid = _make_pid(model, gain_variant)` — PID created once per episode |
| **L229–244** | **Main step loop:** `traj_fn(t)` → `pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)` → `data.ctrl[:4] = u` → `mj_step` |
| **L231** | `p_des, v_des, a_des, yaw_des = traj_fn(t)` — trajectory called every physics step |
| **L238** | `u = pid.compute(...)` — the single call site where PID produces motor thrusts |
| **L242** | `n_clip += int(pid.last_raw_saturated)` — saturation telemetry logged per step |
| **L250–251** | `steps.append({'p', 'v', 'p_des', 'q'})` — raw log saved per step |
| **L253–254** | `contact_frac`, `motor_clip_frac` — rejection metrics |
| **L260–269** | Reject episode if `contact_frac > limit` or `min_z < Z_FLOOR_MARGIN` |
| **L281** | `motor_clip_frac` returned in accepted episode dict — visible in dataset metadata |

---

## 6. Data Flow Summary

```
traj_fn(t)                     generator.py L231
    │ p_des, v_des, a_des, yaw_des
    ▼
pid.compute(p, q, v, om,       generator.py L238
            p_des, v_des, a_des, yaw_des)
    │
    ├─ outer loop:  e_p, e_v, a_cmd        flight_controller.py L88–91
    ├─ F_world = m(a_cmd + g·ẑ)            flight_controller.py L94
    ├─ b₃_des, R_des (attitude target)     flight_controller.py L101–114
    ├─ SO(3) error e_R, torque τ           flight_controller.py L120–125
    ├─ thrust T, wrench                    flight_controller.py L128–134
    ├─ allocation u = M⁻¹·wrench           flight_controller.py L135
    └─ thrust-priority clamp (if sat.)     flight_controller.py L142–156
    │ u[0:4]
    ▼
data.ctrl[:4] = u              generator.py L243
    │
    ▼
mujoco.mj_step(model, data)    generator.py L244
    │
    ├─ new p, v, q read        generator.py L233–236
    ├─ contact check           generator.py L246–248
    └─ append to steps[]       generator.py L250–251

After all steps:
    contact_frac, motor_clip_frac, min_z → accept/reject gate
    Accepted episode → serialised as training data
```

---

## 7. Why U9 Was Necessary: PID Budget Perspective

The original U8 "stop-and-go" chain had the PID handle large velocity
discontinuities at every waypoint.  U9 replaced this with a continuous
path but introduced a new PID stress: high centripetal acceleration at fillets.

The trajectory of the problem across iterations:

| Iteration | Root cause | PID symptom | Fix |
|-----------|-----------|-------------|-----|
| U8 and earlier | `traverse_line` forces `v=0` at joints | Position error spikes at each joint; high contact fraction near obstacles | U9: `blended_path` eliminates internal stops |
| U9 initial (`r=0.30`) | Fillet `a_peak = 8.6 m/s²` for LRL/RLR | Motor saturation 29–78% of steps → attitude loss → contact (45.2% rejection) | Fix_1: `BLEND_RADIUS = 0.30 → 0.45` |
| U9 Fix_1 (`r=0.45`) | `a_peak = 5.7 m/s²` — within budget | `motor_clip_frac` acceptable; contact rate < 30% | ✅ Dataset collected |

The progression shows the PID's motor saturation threshold as the *binding constraint*
on trajectory design: everything in U9 (blend radius choice, duration floor, centripetal
feedforward) is ultimately set by the question "can the PID track this without saturating?"

---

## 8. Quick Reference: Gain Values in Code

```python
# uav_env_test/flight_controller.py
# CascadedPID.__init__

# Outer loop (world frame)
self.Kp_pos = np.array([4.0, 4.0, 8.0])   # x, y, z — higher z for altitude hold
self.Kd_pos = np.array([3.0, 3.0, 4.0])   # velocity damping

# Inner loop (body frame)
self.Kp_att   = np.array([70.0, 70.0, 4.0])   # roll/pitch stiff, yaw softer
self.Kp_omega = np.array([ 2.5,  2.5, 1.0])   # gyroscopic damping

# Motor bounds
self.u_hover = m * g / 4        # ~N per motor at hover
self.u_max   = max(2 * u_hover, 6.0)
self.u_min   = 0.0
self.thrust_floor = 0.1 * m * g

# Gain variants applied in generator.py _make_pid():
# 'pid_default':   kp_scale=1.0  kd_scale=1.0
# 'pid_high_gain': kp_scale=1.2  kd_scale=1.0
# 'pid_low_gain':  kp_scale=0.8  kd_scale=0.9
```
