# MuJoCo MPC UAV (Skydio X2) — Model, Math, Real-World Mapping, and Environments

**Audience:** anyone deciding whether/how to use this model for trajectory
planning, control, or imitation-learning experiments.
**Format:** math + prose. No code.

---

## 1. Identity at a Glance

| Property | Value |
|---|---|
| Name in MuJoCo MPC | "Quadrotor Racing" task |
| Underlying robot | **Skydio X2** — a real-world commercial autonomous drone (Skydio Inc., US) |
| Modeling source | [MuJoCo Menagerie / skydio_x2](https://github.com/google-deepmind/mujoco_menagerie/tree/main/skydio_x2) |
| Modifications by MJPC | One-line quaternion init; removal of redundant MJPC-irrelevant sensors and keyframes (cosmetic / scope hygiene) |
| Physics model | 6-DOF rigid body with 4 propeller-thrust actuators |
| State dimension | 13 (3 position + 4 quaternion + 3 linear velocity + 3 angular velocity) |
| Control dimension | 4 (one scalar thrust per propeller, in Newtons) |
| Body mass | ≈ 0.645 kg (chassis-only in Menagerie XML; effective hover-mass ≈ 1.32 kg with arms+props) |
| Hover thrust (per motor) | ≈ 3.25 N (matches MJPC residual: `(m_body + m_arms) · g / 4`) |

---

## 2. The Mathematical Model (Skydio X2 as Rigid Body + 4 Motors)

### 2.1 State vector

The simulator carries a 13-dimensional state:

```
s = [ p ; q ; v ; ω ]
       ∈ ℝ³ × S³ × ℝ³ × ℝ³
```

- **p ∈ ℝ³** — body centre-of-mass position in the world frame.
- **q ∈ S³** — unit quaternion `[w, x, y, z]` representing body-to-world
  rotation. `‖q‖ = 1` is preserved by MuJoCo's quaternion integrator.
- **v ∈ ℝ³** — linear velocity of the COM, expressed in the **world** frame
  (`framelinvel` sensor convention).
- **ω ∈ ℝ³** — angular velocity, **world** frame (`frameangvel`).

The free joint contributes 6 generalized DOF (qpos has 7 components — 3
position + 4 quaternion — and qvel has 6, with the quaternion's
constraint folded in).

### 2.2 Continuous-time dynamics

Writing `R(q)` for the rotation matrix induced by `q`, `m` for body mass,
`g` for gravitational acceleration (9.81 m/s² in `-ẑ_world`), and `I` for
the body inertia tensor (diagonal in body frame, expressed in world frame
as `R I_b R^T`):

```
ṗ = v
q̇ = ½ · q ⊗ [0, ω_body]            (quaternion derivative; ω_body = R^T ω)
m·v̇ = R(q) · F_body − m·g·ẑ_world − F_drag
I·ω̇ = τ_body − ω × (I·ω)
```

`F_body` and `τ_body` are the net body-frame force and torque produced
by the 4 motors (next subsection). `F_drag` covers MuJoCo's residual
damping — small, not aerodynamically realistic; see §5.2.

### 2.3 Actuator model — the four motors

Let `u = [u₁, u₂, u₃, u₄] ∈ ℝ⁴` be the control input (one thrust scalar
per motor, in Newtons; bounded informally to roughly `[0, 5+]` per motor
but **not** clipped at the model level — the policy is responsible).

Each motor `i` is mounted at a fixed **site** in the body frame, denoted
`p_site_i ∈ ℝ³` (body coordinates). Motors generate two effects:

**1. Lift force along body `+ẑ_body`:**

```
F_i_body = uᵢ · ẑ_body                 (applied at site i)
```

The net body-frame thrust is `F_body = (Σᵢ uᵢ) · ẑ_body`.

**2. Roll and pitch torques (from arm lever arm × lift):**

```
τ_lift,body = Σᵢ p_site_i × (uᵢ · ẑ_body)
```

For the X2's cross-arm geometry, motors 1 and 3 sit on one diagonal,
motors 2 and 4 on the other. Differential thrust (e.g. `u₁ > u₃` with
`u₂, u₄` constant) produces a body-frame torque that rotates the drone
about the perpendicular axis. This is how pitch and roll are commanded.

**3. Yaw reaction torques (propeller drag):**

Each spinning propeller induces a reactive yaw torque about its own
spin axis, encoded in the MJPC actuator `gear` specification as
`κᵢ · uᵢ`, with the X2 using:

```
κ₁ = −0.0201   (CW prop)
κ₂ = +0.0201   (CCW prop)
κ₃ = +0.0201   (CCW prop)
κ₄ = −0.0201   (CW prop)
```

(Units: N·m per N of thrust — i.e. dimensionless aerodynamic constant.)

So the net body-frame yaw torque is:

```
τ_yaw,body = (Σᵢ κᵢ · uᵢ) · ẑ_body
```

In hover all four `uᵢ` are equal, the κᵢ alternate sign, and `τ_yaw = 0`.
Yaw is commanded by breaking that symmetry — e.g. boost `u₂ + u₃`
relative to `u₁ + u₄`.

**Total body-frame torque:**

```
τ_body = τ_lift,body + τ_yaw,body
```

### 2.4 Sensor model (what a controller / policy gets to read)

The MJPC task declares five sensors on the X2 body. All are **noise-free**
(MuJoCo's default), all available every simulation tick:

| Sensor | Symbol | Frame | Dimension |
|---|---|---|---|
| `framepos` "position" | p | world | 3 |
| `framequat` "orientation" | q | world | 4 |
| `framelinvel` "linear_velocity" | v | world | 3 |
| `frameangvel` "angular_velocity" | ω | world | 3 |
| `framepos` "trace0" | same as p, for trajectory visualisation | world | 3 |

The base Skydio X2 XML also defines a gyro, accelerometer, and IMU
quaternion at a body site, but the MJPC patch **removes** those (they
duplicate sensors already given on the body). For real-world deployment
those IMU sensors would be the *only* primary measurements; sim hides
that complication.

---

## 3. The MJPC Racing Task (what wraps the model)

The model itself is just physics. The "task" layered around it has three
parts: geometry, residual, transition.

### 3.1 Waypoint geometry

The task defines **11 waypoints** `{w₁, …, w₁₁} ⊂ ℝ³` at fixed world
positions, plus a 12th implicit "home" returning to the origin region.
A **`goal` mocap body** (not subject to physics — it's a teleportable
visual marker) is placed at the *current* target waypoint. The drone's
job is to chase `goal` through the sequence in order.

The waypoint coordinates (in metres) trace a loop roughly 7 m long with
altitudes between 0.75 m and 2.25 m — small enough to fit in an indoor
room or volumetric capture stage.

### 3.2 Racing gates

**8 visual gates** are added as static `<body>`/`<geom>` groups
(`gates.xml`). They have collision geometry, so an unlucky drone *can*
strike them, but the task residual does **not** explicitly penalize gate
collisions — gates are scene dressing that incidentally constrains the
feasible space. (In a derived task you could add halfspace constraints
per gate; MJPC's residual does not.)

### 3.3 The residual function

The MJPC cost is the squared norm of a **residual vector** `r ∈ ℝ¹³`,
computed each simulation step. Four contiguous blocks:

```
r(s, u) = ┌  p − p_goal                  ┐  ← block (a): 3 components
         │  v                             │  ← block (b): 3 components
         │  ω                             │  ← block (c): 3 components
         └  u − u_hover · 1₄              ┘  ← block (d): 4 components
```

Where:
- `p_goal` is the current waypoint (read from the `goal` mocap).
- `v`, `ω` are world-frame velocities — the task penalizes any motion
  near the goal, not just goal-reaching. Effectively a *position-hold*
  controller masquerading as a waypoint chaser.
- `u_hover ≈ 3.25 N` is the per-motor thrust that exactly supports the
  drone's weight; `1₄ = [1, 1, 1, 1]` so block (d) penalises **deviation
  from hover**.
- The orientation (`q`) is **not** in this residual. There is an
  "Orientation" sensor declared with dim 2 but it is not summed into the
  C++ residual; it appears reserved for a future extension.

### 3.4 Weighting and aggregation

MJPC's `<user>` sensor declarations attach per-block scalar weights:

```
w_pos = 25.0        (×3 components)
w_vel = 1.25        (×3)
w_ang = 1.25        (×3)
w_ctrl = 1e-3       (×4)
w_orient = 0.0      (declared but inactive)
```

The instantaneous cost at state-control pair `(s, u)` is:

```
ℓ(s, u) = w_pos · ‖p − p_goal‖²
        + w_vel · ‖v‖²
        + w_ang · ‖ω‖²
        + w_ctrl · ‖u − u_hover·1₄‖²
```

Position error dominates by 20× over velocity / angular-rate damping;
control deviation is essentially a numerical-stability term (prevents
runaway thrust commands during planning).

### 3.5 Waypoint transition function

The "task transition" is a tiny finite-state machine:

```
let i = current waypoint index ∈ {1, …, 11}
let p = current drone position (from sensor)
let p_goal = position of mocap "goal" (= w_i by construction)

if  ‖p − p_goal‖ ≤ 0.5 m :
    i  ←  (i mod 11) + 1
    place mocap "goal" at  w_i
```

A GUI override (`mode > 0`) can pin the system to any specific waypoint
for debugging. The loop topology (mod 11) means the racing task is
infinite — there is no terminal "done" signal in MJPC's formulation.

---

## 4. The Planner that Consumes the Residual

The model and residual are inert until a planner exercises them. MJPC's
default planner for this task is **Predictive Sampling** (a relative of
MPPI / CEM). Conceptually each replan iteration:

1. **Sample** `N = 32` candidate control sequences over a planning horizon
   of `H = 0.5 s` at `Δt = 0.01 s` (so 50 steps). Controls are encoded as
   cubic splines with 5 knots, with Gaussian exploration noise (σ = 0.3)
   added to the current best trajectory.
2. **Roll out** each candidate inside a *copy* of the MuJoCo model,
   producing a state sequence under the dynamics of §2.
3. **Score** each candidate by integrating `ℓ(s_k, u_k)` over the horizon.
4. **Aggregate** — pick the lowest-cost trajectory (or a softmax-weighted
   mean, depending on planner config) as the new nominal.
5. **Execute** the first control of the nominal on the real system; shift
   the horizon and repeat at the next tick.

Key numbers from the task XML's `<custom>` block:

| Parameter | Value | Interpretation |
|---|---|---|
| `agent_horizon` | 0.5 s | Look-ahead window |
| `agent_timestep` | 0.01 s | Planning discretization |
| `sampling_trajectories` | 32 | Population size per replan |
| `sampling_spline_points` | 5 | Control-sequence dimensionality |
| `sampling_exploration` | 0.3 | Exploration noise std |
| `sampling_processed_noise_passes` | 3 | Low-pass filter passes on samples |

These hyperparameters are **MJPC-specific**: they encode the assumption
that a sampling planner will be used. They have no meaning outside MJPC.

---

## 5. Real-World Mapping

### 5.1 The Skydio X2 hardware

The Skydio X2 is a real commercial drone (the model in MuJoCo Menagerie
is calibrated against publicly-available X2 dimensions and mass
specs). Real-world characteristics:

- Compact form factor, ~30 cm rotor-to-rotor.
- Autonomous obstacle-avoidance flight using 6 onboard cameras + Skydio's
  proprietary AI Pilot — *not* simulated by this model.
- 4 fixed-pitch propellers in cross configuration; thrust ∈ roughly
  [0, 6] N per motor — matches the MuJoCo gear spec.
- Maximum lateral speed ~50 km/h, hover endurance ~30 min on a charged
  battery.
- Primary use cases: ISR (intelligence/surveillance/reconnaissance) for
  US DoD, industrial inspection, search-and-rescue.

The MuJoCo model captures the **rigid-body flight dynamics** of the X2.
It does **not** model: the perception stack, battery dynamics,
controller latency, motor spin-up dynamics, or any of the autonomy
software.

### 5.2 Sim-vs-real gap

For anyone considering this model as a precursor to real X2 flight:

| Effect | Modelled? | Notes |
|---|---|---|
| Newtonian rigid-body dynamics | ✅ | High fidelity |
| Cross-arm thrust → roll/pitch torque | ✅ | Exact via lever-arm geometry |
| Yaw drag torque (CW/CCW props) | ✅ | Linear in thrust (`κ · u`) |
| Gravity | ✅ | Constant 9.81 m/s² |
| Propeller aerodynamic drag (varies with `‖v‖`) | ❌ | MuJoCo treats props as pure point forces; no airfoil model |
| Ground effect (extra lift near ground) | ❌ | Important <1 rotor diameter altitude |
| Motor first-order lag (spin-up time ~50 ms) | ❌ | MuJoCo applies thrust instantly |
| Battery sag (max thrust drops with state-of-charge) | ❌ | Not in scope of MuJoCo |
| Sensor noise (IMU bias, gyro drift) | ❌ | Sensors return ground truth |
| Wind / atmospheric disturbance | ❌ | No fluid simulation |

These are the standard caveats for any MuJoCo-based UAV simulation. A
policy trained purely in this sim will **not** fly a real X2 without
domain randomisation, model-mismatch hardening, or finetuning on hardware.

---

## 6. Environments the Model Can Live In

The XML + mesh are portable. Three host environments differ in what they
*do* with the model.

### 6.1 MuJoCo Menagerie (standalone)

The base model in Menagerie's `skydio_x2/` directory exists as a
reference implementation. It's loadable by any MuJoCo binding (Python,
C, mujoco_mpc, dm_control, IsaacLab) and ships with:
- The full sensor block (gyro, accelerometer, IMU quat).
- A `hover` keyframe at `qpos = [0, 0, 0.3, 1, 0, 0, 0]`,
  `ctrl = [3.25, 3.25, 3.25, 3.25]`.

This is the smallest-surface form of the X2 — pure physics, no task.

### 6.2 MuJoCo MPC racing task (`mujoco_mpc/mjpc/tasks/quadrotor/`)

What §3 and §4 above describe in full. Adds:
- 11 waypoints + 8 gates (geometry).
- 4-block residual function (cost shaping).
- Waypoint-advance transition function.
- Predictive-sampling planner hyperparameters.

The point of the MJPC version is to demonstrate that an MPPI-style
planner can fly the X2 through a racing course **online**, computing
costs in real time. It is **not** a dataset; the MJPC runtime *is* the
controller, executed step-by-step in a GUI.

### 6.3 Other plausible host environments

The MuJoCo model is portable to:
- **`dm_control`** — wrap as a `composer.Task`, define your own reward
  and observation specs. Standard for RL research.
- **Gymnasium / SB3** — wrap the MuJoCo bindings in a `gym.Env`; train
  PPO / SAC / etc. on raw thrust commands.
- **Differentiable sims (mujoco-mjx, brax)** — port the model to MJX for
  gradient-based trajectory optimisation or policy gradients.
- **IsaacLab** — convert the URDF/USD analog and train at scale on GPU.
- **Imitation-learning frameworks (diffusion policy, ACT, FM-PCC, D3IL)**
  — requires generating expert demonstrations first; the MJPC planner
  itself can serve as the demonstrator if you record its rollouts.

---

## 7. Bottom-Line Summary

- The Skydio X2 in MuJoCo MPC is a **rigid-body quadrotor** with **13-D
  state**, **4-D thrust action**, and the standard cross-arm
  propeller-torque geometry.
- The MJPC "task" layers on **11 waypoints + 8 gates + a 4-block
  quadratic residual + a 0.5 m-trigger waypoint-advance rule**.
- The residual is consumed by a **predictive-sampling planner** (32
  rollouts, 50-step horizon, spline controls). The planner is what does
  the work; the model is inert.
- The model is a faithful approximation of a **real Skydio X2 drone**,
  minus aerodynamics, motor lag, sensor noise, and battery effects.
- Anywhere MuJoCo runs, this model runs — **MJPC is one specific host
  among many** (Menagerie, dm_control, Gymnasium, MJX, IsaacLab, …).
  The math in §2 is the same in all of them; only what wraps it changes.
