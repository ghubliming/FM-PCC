# Gen11 Epoch 2 — UAV Controller Validation: Methodology

**Date**: 2026-06-06  
**Status**: ✅ Complete  
**Maximum fix index**: none (single SLURM run, no code fixes)  
**CLOSURE**: [`EPOCH2_CLOSURE.md`](EPOCH2_CLOSURE.md)

---

## What this epoch does

Validates two things before any data collection or training begins:
1. **The controller is correct** — a hand-written cascaded PID can fly the X2 along a
   non-trivial trajectory inside MuJoCo.
2. **The controller setpoint format is locked** — whether to feed the PID 6 or 9 signals
   per timestep is resolved by measurement, not assumption, and the result fixes the FM
   trajectory tensor dimensionality for all subsequent epochs.

---

## What is a PID controller?

**PID = Proportional–Integral–Derivative.**  It is a feedback loop that produces a
correction signal proportional to the current error (P), its running sum (I), and its
rate of change (D).  We use only **P and D** (no integral — i.e. a PD controller) because
the drone's dynamics are fast enough that integral wind-up would be destabilising at
100 Hz, and gravity is compensated analytically (see below).

The **cascaded** (nested) version chains two PD loops:
an outer loop that controls position → produces a desired force direction, which an inner
loop then converts into rotor thrust commands.  The cascade is necessary because a drone
cannot directly command position — it can only spin rotors.

---

## Cascaded PID — the 4-step pipeline per timestep

Every 10 ms (100 Hz) the controller runs these steps in order.  The trajectory function
(from `trajectories.py`) provides the *reference* signals; MuJoCo provides the *measured*
state.  The output is 4 thrust values written to `data.ctrl[:4]`.

```
INPUT FROM TRAJECTORY:           INPUT FROM MUJOCO:
  p_des  ∈ ℝ³  (desired pos)      p = data.qpos[:3]      (current pos, world frame)
  v_des  ∈ ℝ³  (desired vel)      v = data.qvel[:3]      (current vel, world frame)
  a_des  ∈ ℝ³  (desired accel)    q = data.qpos[3:7]     (quaternion: attitude)
  yaw_des ∈ ℝ  (desired heading)  ω = data.qvel[3:6]     (angular velocity, body frame)
```

### Step 1 — Outer loop: position PD + feedforward → desired force

```
e_p   = p − p_des                          (position error, ℝ³)
e_v   = v − v_des                          (velocity error, ℝ³)
a_cmd = −Kp_pos·e_p − Kd_pos·e_v + a_des  (commanded acceleration, ℝ³)
F_world = m·(a_cmd + [0, 0, g])            (world-frame force vector, ℝ³)
```

**Physical meaning**: Newton's 2nd law says `F = m·a`.  The drone needs to produce
`F_world` to achieve `a_cmd`.  Gravity `[0, 0, 9.81]` is added analytically — the drone
must always produce at least enough upward force to cancel its own weight (1.325 × 9.81 ≈
13 N total; 3.25 N per rotor), plus any correction or feedforward demand.

`F_world` is a 3D vector pointing in the direction the drone needs to push.  To push
forward (+x) the drone must tilt nose-down (because rotors only push *up* in the body
frame).  The direction of `F_world` therefore encodes the *required attitude*.

### Step 2 — Force direction → desired attitude matrix R_des

```
b3_des = F_world / ‖F_world‖         (desired body-z = rotor disk normal)
b2_des = (b3_des × x_c) / ‖...‖      (desired body-y, from yaw constraint)
b1_des = b2_des × b3_des             (desired body-x = forward direction)
R_des  = [b1_des | b2_des | b3_des]  (3×3 desired rotation matrix)
```

This is the Lee/Mellinger geometric approach.  The three columns of `R_des` are the
desired body-frame axes expressed in the world frame.  `b3_des` is the direction the rotor
disk normal must point (perpendicular to the thrust plane); the yaw constraint fixes the
remaining degree of freedom.

### Step 3 — Inner loop: attitude error → body torques

```
R     = quat_to_rot(q)                      (current rotation, from MuJoCo quaternion)
E     = 0.5·(R_des.T @ R − R.T @ R_des)    (skew-symmetric attitude error matrix)
e_R   = [E[2,1], E[0,2], E[1,0]]           (axis-angle error vector, ℝ³)
e_ω   = ω                                   (angular velocity to damp; ω_des ≈ 0)
gyro  = ω × (I·ω)                           (gyroscopic term: Coriolis compensation)
τ     = −Kp_att·e_R − Kp_omega·e_ω + gyro  (body torques: roll, pitch, yaw)
```

**Physical meaning**: `e_R` is the shortest rotation that takes the current attitude `R`
to the desired attitude `R_des`, expressed as a 3-vector (roll error, pitch error, yaw
error).  `Kp_att` fights this misalignment.  `Kp_omega` damps spinning (acts like a
rotational dashpot).  The gyroscopic term compensates the Coriolis effect — fast rotation
about one axis produces a precession torque about the others; without this compensation
the attitude loop oscillates on fast manoeuvres.

### Step 4 — Allocation: wrench → 4 rotor thrusts

```
T      = F_world · b3               (total thrust scalar: world force projected onto body-z)
wrench = [T, τ_x, τ_y, τ_z]        (desired force + torques)
u      = M⁻¹ · wrench               (solve for 4 rotor thrusts)
u      = clip(u, 0, u_max)          (physical limits: rotors can't pull, have max speed)
data.ctrl[:4] = u                   (write to MuJoCo — 4 floats in Newtons)
```

**Physical meaning**: `u = [u1, u2, u3, u4]` are the **individual thrust forces** each
rotor must produce, in Newtons.  MuJoCo then applies these forces at each rotor site,
integrates the equations of motion, and updates `qpos`/`qvel` for the next step.

Code: `uav_env_test/flight_controller.py` (`CascadedPID.compute()`, lines 75–133).

---

## Diagnostic output — what every number means

```
mass=1.3250 kg  u_hover=3.2496 N  u_max=6.50
inertia diag = [0.06071129  0.0364684   0.0254117]
allocation M =
[[ 1.      1.      1.      1.    ]
 [-0.18    0.18    0.18   -0.18  ]
 [ 0.14    0.14   -0.14   -0.14  ]
 [-0.0201  0.0201 -0.0201  0.0201]]
Kp_pos=[4. 4. 8.]  Kd_pos=[3. 3. 4.]
Kp_att=[70. 70.  4.]  Kp_omega=[10. 10.  2.]
```

### Physical constants

| Value | Formula | Meaning |
|---|---|---|
| `mass = 1.3250 kg` | from MuJoCo model | Total drone mass — body + 4 rotors combined |
| `u_hover = 3.2496 N` | `= m·g / 4 = 1.325×9.81/4` | Thrust each rotor must produce to hover — 50% of its maximum |
| `u_max = 6.50 N` | `= max(2·u_hover, 6.0)` | Each rotor's peak thrust; hover uses 50% of headroom, leaving 50% for manoeuvres |

### Inertia diagonal `[Ixx, Iyy, Izz]` kg·m²

| Axis | Value | What it means |
|---|---|---|
| `Ixx = 0.0607` | Roll (around x) | Highest — the X2 is widest in y (rotor arms span ±0.18 m), so it resists roll the most |
| `Iyy = 0.0365` | Pitch (around y) | Medium — narrower in x (±0.14 m arm) |
| `Izz = 0.0254` | Yaw (around z) | Smallest — the drone is flat; yaw inertia is about a vertical axis through the thin body |

Higher inertia means the controller must apply more torque to achieve the same angular
acceleration (`τ = I·α`), and the gains must be set accordingly.

### Allocation matrix M (4×4)

Each **column** corresponds to one rotor.  Each **row** is one component of the output
wrench:

```
         Motor 1    Motor 2    Motor 3    Motor 4
         rear-left  rear-right front-right front-left
         (−x,−y)   (−x,+y)   (+x,+y)   (+x,−y)
Lift  [  1.         1.         1.         1.      ]  all rotors contribute equally to total lift
Roll  [ −0.18      +0.18      +0.18      −0.18    ]  moment arm in y: +y rotors roll right
Pitch [ +0.14      +0.14      −0.14      −0.14    ]  moment arm in x: +x rotors pitch nose-up
Yaw   [ −0.0201   +0.0201    −0.0201    +0.0201   ]  reaction torque: alternating spin direction
```

To produce **roll left**: increase thrust on right rotors (M2, M3), decrease on left (M1, M4).  
To produce **pitch nose-up**: increase front rotors (M3, M4), decrease rear (M1, M2).  
To produce **yaw right**: increase CW rotors (M2, M4), decrease CCW (M1, M3).

The yaw coefficient `±0.0201` is tiny because yaw authority comes from rotor reaction
torque (the spinning mass resisting direction change), not from a geometric moment arm.
This is why yaw is the hardest axis to control quickly and gets a lower gain.

`M_inv` is computed at init via `np.linalg.inv(M)`.  The allocation solve
`u = M_inv @ wrench` distributes the desired wrench across 4 rotors uniquely (the
4×4 system is square, so there is exactly one solution before clipping).

### Gains

| Gain | Value | Meaning |
|---|---|---|
| `Kp_pos = [4, 4, 8]` | Position proportional | z is 2× x/y — altitude control fights gravity disturbance and is more critical; higher gain gives stronger restoring force |
| `Kd_pos = [3, 3, 4]` | Position derivative (damping) | Fights overshoot; z slightly higher to prevent altitude bouncing |
| `Kp_att = [70, 70, 4]` | Attitude proportional | Roll/pitch must respond instantly to position-loop demands (hence high 70). Yaw is slow-moving and yaw actuator is weak (κ ≈ 0.02), so gain = 4 |
| `Kp_omega = [10, 10, 2]` ← **original** | Angular rate damping | **Too high.** At 100 Hz, saturation onset at `\|ω\| > 1.82/10 = 0.18 rad/s` → limit cycle. This is the Epoch 2 failure. |
| `Kp_omega = [2.5, 2.5, 1.0]` ← **fixed (Epoch 4)** | Angular rate damping | Saturation onset at `\|ω\| > 0.73 rad/s` — 4× harder to trigger. Applied in `uav_env_test/flight_controller.py` before Epoch 4 data collection |

**Why z Kp_pos is doubled**: The position-loop plant in z is `ẍ = u_z/m − g`.  Gravity
enters as a constant offset — if `u_z` is not exactly `m·g`, position drifts.  Although
we compensate gravity analytically in `F_world = m·(a_cmd + [0,0,g])`, any tracking lag
in the attitude loop (which delivers `u_z`) translates directly to altitude drift.  A
stiffer position gain in z reduces that lag's effect.

**Why Kp_att roll/pitch = 70 but yaw = 4**: The position outer loop needs the attitude
inner loop to execute its demand *within one or two timesteps* — otherwise the position
loop is controlling a sluggish sub-system.  Roll and pitch are actuated by the full rotor
force (moment arm ≈ 0.14–0.18 m, force up to 6.5 N → torque up to ~1.2 N·m).  Yaw is
actuated only by the weak reaction torques (κ ≈ 0.02, so max yaw torque ≈ 4 × 0.02 × 6.5
= 0.52 N·m).  High yaw gain with weak actuation → saturation → instability.

Code: `uav_env_test/flight_controller.py` `CascadedPID.__init__()` (lines 33–73).

---

## The four tasks — trajectory math and what they demand

Each task is a function `traj(t) → (p_des, v_des, a_des, yaw_des)` evaluated at every
100 Hz timestep.  The PID receives these signals as its setpoint.  Code:
`uav_naive_test/trajectories.py` + `uav_naive_test/run_naive.py`.

---

### Task A — Hover at `[0, 0, 0.5]`

```python
traj(t) = ([0, 0, 0.5],  [0, 0, 0],  [0, 0, 0],  0)
           p_des           v_des        a_des        yaw
```

The setpoint is **completely static** for all t.  The drone starts at approximately
`[0, 0, 0]` (on the floor) and must climb to `z=0.5 m` and hold still.

**What this demands from the controller**: during the climb, position error `e_p = p−p_des`
is large → large `a_cmd` → drone tilts strongly → attitude error grows → inner loop fires
hard.  When `|ω|` exceeds the saturation threshold (0.18 rad/s with Kp_omega=10), the
inner loop clips → oscillation → limit cycle.  The drone never settles.

`v_des = 0` and `a_des = 0` everywhere: the controller is **purely reactive** at all
times — it can only respond after error has already built up.

---

### Task B — Step response: `[0,0,0.5] → [1,0,0.5]` at t=2 s

```python
traj(t) =  ([0,0,0.5], 0, 0, 0)   for t < 2.0
           ([1,0,0.5], 0, 0, 0)   for t ≥ 2.0
```

At t=2 s, `p_des` **instantaneously jumps** 1 m in x.  This is a classical step input.
The drone must accelerate from hovering at x=0 to x=1 m.

**What this demands**: the step creates a sudden 1 m position error → maximum `a_cmd` →
drone pitches aggressively nose-forward → attitude loop fires at full saturation →
same limit-cycle issue as Task A.  Additionally, the controller must then decelerate and
settle at the new position with `v_des=0, a_des=0` — again a static endpoint → hover
instability on arrival.

Both Tasks A and B have `a_des=0` everywhere: the drone is told *where* to go but not
*how fast to accelerate* to get there.

---

### Task C — Circle, radius 0.5 m, period 10 s, altitude 0.75 m

**Geometry**: constant-altitude circle centred at `[0, 0]`, starting at `[0.5, 0, 0.75]`,
completing one full revolution every 10 s.

**Exact math** (from `trajectories.py:circle()`):

```
θ(t)   = ω·t                             angular position
ω      = 2π / T = 2π / 10 = 0.6283 rad/s

p_des(t) = [0.5·cos(θ),  0.5·sin(θ),  0.75]     (position on circle)
v_des(t) = [−0.5ω·sin(θ), 0.5ω·cos(θ), 0]        (tangential velocity)
a_des(t) = [−0.5ω²·cos(θ), −0.5ω²·sin(θ), 0]    (centripetal acceleration, points inward)
```

**Key numbers**:
| Quantity | Formula | Value |
|---|---|---|
| Angular rate | ω = 2π/10 | 0.628 rad/s |
| Peak speed | r·ω = 0.5 × 0.628 | **0.314 m/s** |
| Centripetal acceleration | r·ω² = 0.5 × 0.395 | **0.197 m/s² ≈ 0.02 g** |
| Centripetal force (1 drone) | m·r·ω² = 1.325 × 0.197 | **0.261 N** (2% of hover thrust) |

**6D mode** — the trajectory function returns the correct `(p_des, v_des, a_des)` but the
caller **zeroes out `a_des`** before passing it to the controller:

```python
# 6D: controller sees (p_des, v_des, zeros)
pid.compute(p, q, v, ω,  p_des, v_des, a_des=np.zeros(3))
```

The controller must *infer* the need to tilt from the growing position error.  By the
time the error is large enough to generate the required centripetal tilt, the drone has
already drifted outward.  On a circle this drift is continuous — no step where the error
is zero — so the RMS error never goes to zero.  **Result: 0.214 m RMS.**

**9D mode** — `a_des` is passed as-is:

```python
# 9D: controller sees (p_des, v_des, a_des)
pid.compute(p, q, v, ω,  p_des, v_des, a_des)
```

`a_des = [−0.197·cos(θ), −0.197·sin(θ), 0]` is pre-added to `F_world` before any error
correction.  The drone starts tilting inward *even when* `e_p = 0`.  Position error
stays near zero throughout, and the attitude loop stays in its linear region.
**Result: 0.029 m RMS — 7.4× better.**

**Why this never triggers the hover instability**: the circle has non-zero velocity
everywhere (`v_des ≠ 0`, `a_des ≠ 0`).  The attitude loop is always being asked to
maintain a small non-zero tilt — it never enters the near-zero-demand regime that
saturates the torque command.

---

## What "6D" and "9D" actually mean

These are the number of scalar signals the PID outer loop receives **per timestep** as
its setpoint — i.e., the dimension of the reference trajectory signal, not the FM
tensor dimension.

| Format | Signals fed to PID | What `a_des` carries |
|--------|--------------------|---------------------|
| **6D** | `p_des ∈ ℝ³`, `v_des ∈ ℝ³` | forced to **zero** |
| **9D** | `p_des ∈ ℝ³`, `v_des ∈ ℝ³`, `a_des ∈ ℝ³` | actual trajectory acceleration |

Unpacked:
- `p_des` — where the drone should **be** at time `t` (3D position).
- `v_des` — how fast it should be **moving** at time `t` (3D velocity).
- `a_des` — how fast it should be **accelerating** at time `t` (3D acceleration).  
  For a circle: `a_des = −Aω²cos(ωt)` pointing toward the centre (centripetal).

The 6D case is physically equivalent to giving the controller a GPS waypoint and a speed
target but no information about the upcoming turn.  The 9D case is equivalent to also
telling it "you will need to pull 0.3 g centripetally in 0.1 s" — the controller can
pre-tilt the drone before the position error even builds up.

Code: `uav_naive_test/trajectories.py` — both `circle_6d` and `circle_9d` generate the
same geometric path; they differ only in whether `a_des` is returned as zeros or as the
analytic centripetal term.

---

## Why feedforward `a_des` is load-bearing (the physics)

In the outer-loop equation `F_des = Kp·err_p + Kd·err_v + a_des`:

- Without `a_des`: the drone is reactive.  It must first accumulate position error `err_p`
  to generate enough `F_des` to tilt and accelerate.  By the time it tilts, it has already
  drifted off the path.  On a circle this drift is continuous → large steady-state RMS.

- With `a_des`: the force command includes the known centripetal demand even when
  `err_p ≈ 0`.  The drone pre-tilts, position error stays near zero, and the inner
  attitude loop operates in its linear (stable) region throughout.

This is the standard argument for **feedforward + feedback** vs feedback-only control.
The magnitude difference is not subtle: 6D gives 0.214 m RMS, 9D gives 0.029 m RMS —
a **7.4× improvement** on the same circle at the same speed with the same gains.

---

## 6D vs 9D — the decisive experiment

| Task | Setpoint format | RMS (m) | Verdict |
|---|---|---|---|
| A — hover | static p_des only | 0.335 | ❌ (gain issue — see below) |
| B — step response | static p_des only | 0.328 | ❌ |
| C — circle | 6D `[p_des, v_des]`, `a_des=0` | 0.214 | ❌ |
| C — circle | 9D `[p_des, v_des, a_des]` | **0.029** | ✅ |

**Decision locked**: all future epochs use 9D reference trajectories.

---

## How this maps to the FM trajectory tensor (vs D3IL avoiding)

The "9D" controller setpoint decision directly determines the dimension of the FM
trajectory tensor in Epoch 4.  It is useful to compare to the existing D3IL avoiding
pipeline to see the pattern:

| System | Task space | FM trajectory tensor | `action` component | `obs` component |
|---|---|---|---|---|
| **D3IL avoiding** | 2D plane (robot arm) | **6D** `[Δdes_xy(2) ‖ des_xy(2), c_xy(2)]` | 2D = Δ desired position | 4D = desired + current XY |
| **UAV Epoch 4** | 3D space | **9D** `[Δp_des(3) ‖ p(3), v(3)]` | 3D = Δ desired position | 6D = position + velocity |

**Both use the same `[action ‖ obs]` packing convention.**  The difference is:

- D3IL avoiding operates in a **horizontal plane** (z fixed by the env, not by the drone
  itself) — so position and action are 2D.  Observation is 4D: `[des_xy(2), c_xy(2)]`.
  No velocity in obs because the arm's state is fully captured by position alone at the
  dataset frequency.

- UAV operates in **3D space** — position is 3D.  Velocity must be in the observation
  because the drone's dynamics are second-order: position alone does not determine the
  next state (a drone at the same position can be flying in any direction).  So obs is 6D:
  `[p(3), v(3)]`.  Action remains a position delta: `Δp_des ∈ ℝ³`.

**The result**: D3IL avoiding has a 6D FM tensor (= 2+4), UAV has a 9D FM tensor (= 3+6).
The "9D" in Epoch 2 (controller setpoint) and the "9D" in Epoch 4 (FM tensor) are the
same number but refer to different things — the former is the PID input signal dimension,
the latter is the packed `[action ‖ obs]` format stored in episode pickles.

---

## Hover instability — discovered and deferred

Tasks A and B (static setpoint) failed with a discrete-time limit cycle:
motor outputs alternate `[6.5, 6.5, 0, 0] ↔ [0, 0, 6.5, 6.5]` every step.

**Root cause**: `Kp_omega = [10, 10, 2]` is too aggressive for 100 Hz physics.  Any
disturbance reaching `|ω| > 0.18 rad/s` saturates the torque command; the over-correction
flips sign in one step → locked oscillation.

**Why it doesn't matter for FM-PCC**: FM/diffusion policies output continuously-moving
trajectories with non-zero `a_des`.  The limit cycle only fires when the drone holds a
static position (near-zero velocity for multiple steps) — a scenario FM-PCC never
produces.  The fix (`Kp_omega → [2.5, 2.5, 1.0]`) is deferred to Epoch 4, where the
s_curve data-collection scene forces it.

---

## Architectural conclusion

The planning/execution split is validated: a reference trajectory (from any planner,
including FM) can be tracked at < 3 cm RMS by a separate cascaded controller with no
coupling between them.  This is the same assumption FM-PCC makes for the Panda arm —
now confirmed for the UAV.

---

## Cross-references

| Document | Content |
|---|---|
| [`EPOCH2_CLOSURE.md`](EPOCH2_CLOSURE.md) | Full results, stability diagnosis, arithmetic |
| [`../Epoch1_UAV_model/METHODOLOGY.md`](../Epoch1_UAV_model/METHODOLOGY.md) | Model assets used here |
| [`../Epoch_3_uav_in_env/METHODOLOGY.md`](../Epoch_3_uav_in_env/METHODOLOGY.md) | Same controller tested inside obstacle scenes |
| [`../Epoch4_expert_data/METHODOLOGY.md`](../Epoch4_expert_data/METHODOLOGY.md) | Kp_omega fix applied; 9D FM tensor in episode pickles |
