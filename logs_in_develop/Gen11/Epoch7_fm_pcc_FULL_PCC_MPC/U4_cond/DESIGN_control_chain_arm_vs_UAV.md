# Control Chain: Arm (IK) vs UAV (PID) vs MJPC vs UAV-Flow

**Date:** 2026-06-28  
**Path:** `logs_in_develop/Gen11/Epoch7_fm_pcc_FULL_PCC_MPC/U4_cond/`  
**Companion doc:** [`DESIGN_data_flow_FM_to_MuJoCo.md`](DESIGN_data_flow_FM_to_MuJoCo.md) (12D tensor, V1/V2/V3 velocity)

---

## §1 — The core question

> Is the avoiding/DPCC arm task replanning every step (A→B→C receding-horizon), or does it plan the full path A→Z then IK once?

**Answer:** Receding-horizon MPC — FM replans EVERY step. IK also runs inside EVERY step.

---

## §2 — Avoiding/DPCC arm: exact replanning pattern

### §2.1 The outer loop (FM is called every timestep)

`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` L292-328:

```python
for _ in range(args.max_episode_length):          # one FM call per loop iteration
    ...
    start = time.time()
    action, samples = policy(conditions={0: obs}, batch_size=args.batch_size,
                             horizon=args.horizon, ...)   # ← FM + QP EVERY step
    ...
    if 'avoiding' in exp:
        next_pos_des = action + obs[:2]           # Δ(x,y) → absolute (x,y)_des
        obs, rew, terminated, info = env.step(
            np.concatenate((next_pos_des, fixed_z, [0, 1, 0, 0]), axis=0)
        )                                         # 7D Cartesian pose sent to env
```

Policy class (`flow_matcher_v3/sampling/policies.py` L92):
```python
action = actions[which_trajectory, 0]            # only actions[0] executed (first of H)
```

So the FM generates an **H-step plan** (`horizon=8` by default) at every control step, but **only the first action `actions[0]` is executed**. The rest of the horizon is discarded. At the next step, FM replans from the new observation. This is **receding-horizon MPC**.

### §2.2 What `env.step` receives

The step input is a 7D Cartesian pose:
```
[x_des, y_des, z_fixed, qx=0, qy=1, qz=0, qw=0]
```

- `(x_des, y_des)` = FM output (position delta applied to current position)
- `z_fixed` = constant (arm stays at fixed height in the avoiding task)
- Quaternion = fixed orientation (arm always points same way)

**FM never outputs joint angles. It outputs Cartesian position deltas.**

### §2.3 What IK does inside env.step (every step)

`d3il/environments/d3il/d3il_sim/controllers/IKControllers.py` — `CartPosImpedenceController.getControl()`:

The IK problem: given desired Cartesian EE position `p_des ∈ ℝ³`, find joint velocities `q̇_d ∈ ℝ⁷` that move the EE toward `p_des`.

**Math (Jacobian pseudo-inverse, L58-73):**
```
error:   Δx = p_des - p_curr                  [EE position error in ℝ³]
task acc: ẍ_des = Kp · Δx                     [PD in Cartesian space]

Jacobian: J ∈ ℝ³ˣ⁷  (linear part of full 6D Jacobian)
Weighted: J_W = J · W    (W = diag([1,1,1,1,1,1,1]) = identity, L63)

Regularized pseudo-inverse solve:
  (J W J^T + λI) q̇_d = ẍ_des - J q̇_null     [L72]
  q̇_d = W J^T · q̇_d + q̇_null               [map back to joint space, L73]

null-space: q̇_null = Kp_null (q_rest - q)    [keeps config near rest posture]
```

Joint velocity `q̇_d` is then integrated → desired joint position → PD joint controller → joint torques → MuJoCo physics `mj_step`.

### §2.4 Full avoiding arm chain (per step)

```
FM (H=8 plan, Cartesian)
  ↓ action[0] = Δ(x,y)  (first step only; rest discarded)
  ↓ next_pos_des = obs[:2] + action       [eval L323]
  ↓
env.step([x_des, y_des, z_fixed, quat])   [7D Cartesian pose]
  ↓
IK: Jacobian pseudo-inverse              [d3il IKControllers.py]
  J q̇_d ≈ Kp (p_des - p_curr)
  ↓
q̇_d → integrate → q_d                   [joint target]
  ↓
JointPDController → torques              [Robots.py L72]
  ↓
MuJoCo mj_step (7-DOF arm dynamics)
  ↓
new obs (Cartesian EE position) → back to FM
```

**Key property:** FM lives entirely in **Cartesian EE space** (2D for avoiding). IK is a **transparent layer** — it converts FM's Cartesian output to joint control invisibly. FM doesn't know joints exist. This is why no velocity is needed in the FM tensor: the IK+PD controller handles velocity implicitly from the position error.

---

## §2.5 — "Per-waypoint replan + IK" vs "all-waypoints-then-IK": which is it?

> Theoretically, is the avoiding task: (A) replan one waypoint → IK → replan next → IK …, or (B) plan the whole path A→Z first, then IK the entire path once?

### §2.5.1 What the code actually does — definitively (A)

Trace the exact call stack for ONE FM control step:

**Step 1 — FM emits exactly ONE next-waypoint** (`eval_flow_matching_v3_ode_selectable.py` L314, L323):
```python
action, samples = policy(...)             # H=8 plan generated, but…
next_pos_des = action + obs[:2]           # …only action[0] used → ONE (x,y) setpoint
obs, rew, terminated, info = env.step(np.concatenate((next_pos_des, fixed_z, quat)))
```

**Step 2 — env.step sets ONE Cartesian setpoint, then IK tracks it over 35 substeps** (`gym_env_wrapper.py` L83-93):
```python
self.controller.setSetPoint(action)                       # ONE waypoint
self.controller.executeControllerTimeSteps(self.robot, self.n_substeps, block=False)  # IK
for i in range(self.n_substeps):          # n_substeps = 35  (avoiding.py L55)
    self.scene.next_step()                # 35 physics steps tracking that ONE waypoint
```

**Step 3 — back to the top.** FM replans the next single waypoint from the new observation.

So per FM step: **1 waypoint → IK servo over 35 physics substeps → replan**. The full A→Z path is **never assembled into one trajectory**. There is no "IK the whole path" anywhere in the code. It is unambiguously **(A): per-waypoint replan, IK every substep**.

```
FM step k:    obs_k → FM → action[0] → waypoint_k → IK × 35 substeps → obs_{k+1}
FM step k+1:  obs_{k+1} → FM → action[0] → waypoint_{k+1} → IK × 35 substeps → …
              └─ replans from scratch each time; horizon[1:] discarded ─┘
```

> Note the layering: FM *internally* predicts an H=8 horizon (a short A→…→H lookahead), but only `horizon[0]` is committed. So even the "mini-plan" is consumed receding-horizon, never IK'd as a block.

### §2.5.2 The theoretical alternative (B), and why it is NOT used

(B) "plan full path A→Z, then IK once" = **open-loop trajectory playback**:
```
FM → full path [w_0, w_1, …, w_Z]  (one shot)
  → IK each w_i → joint trajectory [q_0, …, q_Z]
  → play back q_0…q_Z blindly, no replanning
```

This is a valid and common pattern in classical robotics (offline trajectory optimization → spline → joint-space playback). But it is **deliberately not** what an FM/diffusion policy with DPCC does, because:

1. **No feedback against model error.** Open-loop playback assumes the world matches the plan. Any disturbance, contact, or model mismatch accumulates with no correction. Receding-horizon re-grounds on the *real* observation every step.
2. **The projector (DPCC QP) must see the live state.** Obstacle/dynamics constraints are enforced per-step against the current position. A one-shot path can't react to a constraint that becomes active mid-execution.
3. **The policy is reactive by construction.** FM is conditioned on `obs` (current state). Calling it once throws away its core capability — closing the loop on what actually happened.

### §2.5.3 What is most reasonable IRL

| Pattern | When it's the right call IRL |
|---|---|
| **(A) Per-step replan + IK** (this code) | Reactive tasks, obstacles, contact, model uncertainty, learned policies. **The correct default for FM/diffusion + DPCC.** This is MPC. |
| **(B) Full path → IK once → playback** | Highly repeatable, structured, low-uncertainty motions (pick-and-place on a jig, CNC, welding) where the environment is known exactly and speed/smoothness matter more than reactivity. |
| **(C) Hybrid: plan full path, but re-solve IK per step tracking it** | Long-horizon tasks where a global plan aids consistency but you still want closed-loop tracking (e.g. a global path planner feeding a local MPC tracker). Common in autonomous driving / mobile robots. |

**Verdict for the avoiding task specifically:** (A) is not just what the code does — it's the *correct* design. The whole point of avoiding is reacting to obstacle layouts that vary per episode, with a learned multimodal policy whose value comes from re-conditioning on state. Open-loop (B) would defeat both the policy and the safety projector.

**Where (C) becomes interesting for us:** the UAV task could benefit from a global FM path + local MJPC/PID tracker split (see §6) — that's the planner/tracker hierarchy. But even there, the *tracker* still runs closed-loop per step; only the global waypoint sequence is planned ahead. Nobody runs pure (B) with a learned policy.

### §2.5.4 It's a philosophy/design axis, not a fixed property of the task

The crucial point: **(A) and (B) are both technically valid — the choice is a design decision driven by what you optimize for.** The same FM output (a trajectory) can be consumed either way. Nothing about the arm, the IK solver, or the FM forces per-substep replanning; it's chosen.

| | (A) IK every substep — **current** | (B) IK the whole trajectory |
|---|---|---|
| **Optimizes for** | reactivity / real-time correction | smoothness / offline optimality / speed |
| **Consumes FM** | one waypoint at a time, replan each step | the full H-horizon (or full A→Z) as one block |
| **Feedback** | closed-loop every substep | open-loop after the one-shot solve |
| **Handles disturbance/model error** | yes (re-grounds on real obs) | no (drifts; needs re-plan trigger) |
| **Compute pattern** | cheap per step, paid every step | one big solve up front, then cheap playback |
| **Why our code picks it** | eval is **real-time / receding-horizon** by design — the whole pipeline (FM conditioning + DPCC live constraints) assumes per-step state | would require batching the horizon through IK and trusting it open-loop |

So the honest statement is: **the current design IKs every substep *because we run real-time, receding-horizon* — but the very same FM trajectory could be IK'd as a whole block if the goal were offline smooth playback instead.** The codebase commits to (A); (B) is a legitimate alternative you'd reach for under different requirements (known environment, no disturbances, smoothness/throughput priority).

This mirrors the UAV discussion in §6: PID-per-substep vs MJPC-over-horizon is the *same axis* — how much of the trajectory you commit to before re-grounding on reality. Real-time pushes you toward per-step; offline optimality pushes you toward whole-trajectory.

---

## §3 — UAV chain (our FM-PCC Gen11)

From `eval_fm_uav.py` L340-390 and `flight_controller.py`:

```
FM (H=8 plan, 9D obs = [p_des|p|v])
  ↓ action[0] = Δp_des  (3D, first step only)
  ↓ p_des += action; v_des = action / dt_fm       [eval L374-375]
  ↓
CascadedPID.compute(p, q, v, ω, p_des, v_des)    [flight_controller.py L79]
  ├── Outer loop: position PD + feedforward
  │     e_p = p - p_des
  │     e_v = v - v_des
  │     a_cmd = -Kp_pos·e_p - Kd_pos·e_v + a_des   [L91]
  │     F_world = m·(a_cmd + g)                     [L94]
  │
  ├── Attitude: F_world direction → R_des
  │     b3_des = F_world / |F_world|               [L101]
  │     R_des from (b3_des, yaw_des)
  │
  ├── Inner loop: SO(3) attitude PD (Lee 2010)
  │     E = ½(R_des^T R - R^T R_des)              [L120]
  │     e_R = vee(E); e_ω = ω_body
  │     τ = -Kp_att·e_R - Kp_ω·e_ω + ω×(I·ω)    [L125]
  │
  └── Allocation: M u = [T; τ_x; τ_y; τ_z]
        u = M⁻¹ [T, τ]                             [L135]
        ↓ 4 motor thrusts
        ↓
MuJoCo mj_step (rigid-body quadrotor dynamics)
  ↓
new obs (p, v from qpos/qvel) → back to FM
```

This runs at `decim` physics steps per FM query (L314: `decim = round(1/(dt·33))`), so FM runs at 33 Hz, PID runs at physics rate (typically ~500 Hz).

**Why velocity MUST be in the FM tensor:**
- PID needs `v_des` for the derivative term. `v_des = action/dt_fm` is derived from the FM action.
- PID also needs real `v` (from `data.qvel[:3]`) for the damping term.
- Without `v_des`: PID drives to zero velocity at each waypoint → sluggish/oscillatory.
- Without real `v` feedback: open-loop, unstable (double-integrator plant).

**Why this differs from the arm:**
- Arm: IK converts position error → joint torques implicitly. Position-only input is sufficient because the IK+PD layer handles all velocity/dynamics internally.
- UAV: No IK equivalent. Motors produce thrust → force → acceleration → velocity → position (two integrations). Must explicitly close the velocity loop, which requires feeding real velocity into the PID.

---

## §4 — MJPC quadrotor

**Source:** `mujoco_mpc/mjpc/tasks/quadrotor/quadrotor.cc` + `task.xml`

MJPC is a **sampling-based Model Predictive Controller** that directly optimizes motor thrusts over a prediction horizon.

### §4.1 Cost function (residuals)

`quadrotor.cc` L37-57 — four residual groups:
```cpp
// Residual (0): position error
mju_sub(residuals + 0, position_sensor, mocap_pos, 3);   // p - p_goal ∈ ℝ³

// Residual (1): linear velocity (penalize motion)
mju_copy(residuals + 3, linear_velocity, 3);              // v ∈ ℝ³

// Residual (2): angular velocity (penalize spin)
mju_copy(residuals + 6, angular_velocity, 3);             // ω ∈ ℝ³

// Residual (3): control deviation from hover thrust
thrust_hover = total_mass * g / n_motors;
residuals[9 + i] = ctrl[i] - thrust_hover;               // u_i - u_hover ∈ ℝ
```

The MJPC cost is `J = Σ w_i ||r_i||²`. MJPC minimizes this over motor thrust sequences.

### §4.2 How MJPC solves the problem (task.xml L13-25)

```xml
<numeric name="agent_horizon"   data="0.5"/>   <!-- 0.5 s prediction horizon -->
<numeric name="agent_timestep"  data="0.01"/>  <!-- 10ms = 100 Hz control -->
<numeric name="sampling_trajectories" data="32"/>    <!-- 32 candidate sequences -->
<numeric name="sampling_sample_width" data="0.01"/>  <!-- noise σ on controls -->
```

At each step:
1. Sample 32 random perturbations of the current control sequence
2. Roll each forward for 0.5 s (50 steps) using **full MuJoCo nonlinear physics**
3. Compute the cost `J = Σ ||r||²` for each rollout
4. Update control via weighted combination of candidates
5. Execute first control action; repeat at 100 Hz

**The key difference:** MJPC embeds the full rigid-body dynamics (position → velocity → acceleration → thrust → position) **inside the optimizer**. The planner sees the plant as a black box and directly optimizes motor thrusts. No explicit PID, no explicit velocity controller.

### §4.3 MJPC goal interface

The goal is set via `mocap_pos` (a MuJoCo marker you can write to). `TransitionLocked` in `quadrotor.cc` L60-88 auto-advances through waypoints when within 0.5 m. So:
- MJPC input: `goal_position ∈ ℝ³` (a 3D setpoint)
- MJPC output: `[u1, u2, u3, u4]` (4 motor thrusts, Newtons)

No velocity needed from the planner. MJPC reasons about velocity internally through its physics rollouts.

---

## §5 — UAV-Flow (the external UAV FM repo at `/workspaces/UAV-Flow`)

**Source:** `UAV-Flow/UAV-Flow-Eval/batch_run_act_all.py` + `OpenVLA-UAV/vla-scripts/deploy.py`

UAV-Flow is **not a physics-based controller**. It is:
- **Model**: OpenVLA (Vision-Language-Action) — a large VLM fine-tuned for drone navigation
- **Action space**: `[Δx, Δy, Δz, Δyaw]` relative position+heading delta (4D)
- **Simulation backend**: Unreal Engine via UnrealCV API

### §5.1 The control loop (`batch_run_act_all.py` L290-328)

```python
response = send_prediction_request(image, proprio, instruction, server_url)
action_poses = response.get('action')               # list of (Δx, Δy, Δz, Δyaw)

for action_pose in action_poses:
    relative_x, relative_y, relative_z = float(action_pose[0:3])
    absolute_pos = [global_x + initial_x, global_y + initial_y, relative_z + initial_z, yaw]

    # DIRECT TELEPORTATION — no physics, no PID, no motor control:
    env.unwrapped.unrealcv.set_obj_location(player, absolute_pos[:3])
    env.unwrapped.unrealcv.set_rotation(player, absolute_pos[3] - 180)
```

**The drone is TELEPORTED to each position**. There is no physics simulation — the drone instantly appears at the commanded location. This sidesteps the entire motor/dynamics control problem.

### §5.2 Control chain comparison

```
UAV-Flow:
  OpenVLA → (Δx, Δy, Δz, Δyaw)
    → compute absolute position
    → set_obj_location()           ← TELEPORT (no physics)
    → set_rotation()               ← TELEPORT orientation
  Result: drone jumps to commanded position

Our FM-PCC Gen11:
  FM → Δp_des
    → p_des += Δp_des; v_des = Δp_des/dt
    → CascadedPID(p, q, v, ω, p_des, v_des)
    → 4 motor thrusts
    → mj_step (full rigid-body physics)
  Result: drone physically flies to commanded position
```

UAV-Flow solves the control problem by **eliminating it**: use an engine that can teleport objects. Valid for visual navigation research, but not a dynamics-aware controller.

---

## §6 — The "IK-style UAV" idea: can we control the UAV with position-only (no velocity in FM tensor)?

> **➡ This idea is now being implemented as its own Epoch.** See
> [`../../Epoch8_UAV_Mjpc_thrust_control/PLAN_MJPC_Thrust_Control.md`](../../Epoch8_UAV_Mjpc_thrust_control/PLAN_MJPC_Thrust_Control.md)
> — strict-DPCC 9D position planner (`[action|p_des|p]`, velocity dropped) + MJPC optimal-control
> tracker, added **beside** the E7 PID/12D path (both kept). Headline finding: the existing expert
> dataset is **sliced, not recollected** (reuses the `cond_mode` column-slice precedent).

### §6.1 What the user asks

> Can we use the same principle as the arm IK — feed position into an optimal controller (like MJPC) that handles the remaining dynamics, so the FM only needs to output position, not velocity?

**Answer: Yes, architecturally. This is exactly what MJPC does.** The question is whether it's practical for our pipeline.

### §6.2 Proposed architecture: FM → p_des → MJPC → thrusts

```
FM (33 Hz, position planner):
  input:  obs = [p(3)] or [p_des(3), p(3)]   ← velocity dropped from tensor
  output: Δp_des(3) per step

  ↓ p_des += Δp_des

MJPC (100-500 Hz, physics-aware tracker):
  input:  p_des as mocap_pos (goal position)
  action: sample 32 thrust-sequences, roll forward via mj_step
  output: [u1, u2, u3, u4] motor thrusts

  ↓ mj_step (full physics)
  ↓ new p, v → FM obs (only p needed!)
```

### §6.3 Why this would work

The MJPC optimizer sees the **full plant dynamics** in its rollouts:
```
x[t+1] = f_mujoco(x[t], u[t])     (nonlinear rigid-body + rotors)
```
It doesn't need `v_des` from the FM because it infers the required velocity profile internally: to reach `p_des` while minimizing velocity residual and staying near hover thrust.

This is exactly analogous to the arm:
- **Arm**: IK takes `p_des` → computes `J^† (p_des - p)` → joint torques. Velocity is internal.
- **UAV with MJPC**: MJPC takes `p_des` → optimizes over thrust sequences → motor commands. Velocity is internal.

The FM tensor could be simplified to **6D**: `[p_des(3) | p(3)]` — no velocity needed.

### §6.4 Why we currently use PID (and what we'd lose/gain by switching)

| | Current: FM → PID | Alternative: FM → MJPC |
|---|---|---|
| FM tensor | 9D `[p_des, p, v]` | 6D `[p_des, p]` or even 3D `[p]` |
| Velocity in FM | yes (V2=v_des, V1=v feedback) | no — MJPC handles it internally |
| Compute @ control rate | PID: O(1), ~μs | MJPC: 32×50 rollouts, ~5-50 ms |
| Physics fidelity | PID approximates: linear position PD | MJPC exact: full nonlinear physics |
| Feasibility at 33 Hz | yes | depends on MJPC speed |
| Stability guarantee | PD poles (tunable, fast) | cost function (harder to tune) |
| DPCC compatibility | ✅ p_des/action directly constrainable | ✅ same interface (p_des is the goal) |

**The PID was chosen because:**
1. MJPC runs at 100-500 Hz with 32 rollouts per step. Each rollout is ~50 mj_steps (0.5 s horizon, 0.01 s timestep). That's 1600 `mj_step` calls per MJPC update — feasible at ~50 Hz but the overhead is orders of magnitude more than PID.
2. PID at ~500 Hz with FM at 33 Hz = 15 PID steps per FM step. MJPC at 50 Hz with FM at 33 Hz = 1.5 MJPC steps per FM step — marginally enough.
3. PID stability is well understood. MJPC stability depends on cost tuning and horizon.
4. Our pipeline records `v` anyway (for PID damping and obs normalization). Dropping it from the FM tensor is a training change, not just an inference change — would require full retrain.

**What MJPC would buy:**
- Velocity feedforward automatically emerges from physics rollouts — no need to hand-tune `Kd_pos`.
- Handles attitude dynamics more explicitly (MJPC already penalizes `ω` residual).
- Can incorporate obstacle avoidance in the cost function (if residuals include obstacle proximity).
- FM tensor simplification: drop velocity from obs/action, train simpler model.

### §6.5 Concrete math for MJPC as "UAV IK"

The IK analogy made precise:

**Robot arm IK:**
```
Given: p_des ∈ ℝ³ (EE target)
Find:  q̇_d ∈ ℝ⁷ (joint velocities)
Via:   q̇_d = W J^T (J W J^T + λI)^{-1} Kp(p_des - p_curr)
Cost:  min ||q̇_d||_W s.t. J q̇_d ≈ ẍ_des
```

**MJPC "IK" for UAV:**
```
Given: p_des ∈ ℝ³ (position target, from FM)
Find:  u ∈ ℝ⁴ (motor thrust sequence, length N=50)
Via:   min_{u_{0..N}} Σ_t [ w_p||p_t - p_des||² + w_v||v_t||² + w_ω||ω_t||² + w_u||u_t - u_hover||² ]
       s.t. x_{t+1} = f_mujoco(x_t, u_t)
Solve: sampling MPC (32 candidates, weighted average update)
```

Both solve "given a goal Cartesian position, what low-level control achieves it?" The difference is the plant model: arm uses linearized kinematics (Jacobian), MJPC uses nonlinear rigid-body dynamics.

### §6.6 Would velocity disappear from the 12D tensor?

If using FM → MJPC, the training transition would change:

**Current 12D (from `DESIGN_data_flow_FM_to_MuJoCo.md` §1):**
```
[action(0:3) | p_des(3:6) | p(6:9) | v(9:12)]
  Δp_des        commanded     real      real vel
```

**Position-only 9D (proposed):**
```
[action(0:3) | p_des(3:6) | p(6:9)]
  Δp_des        commanded     real
  (no velocity column)
```

Or even 6D if `p_des` is dropped (FM only receives real position):
```
[action(0:3) | p(3:6)]
  Δp_des        real pos
```

The `deriv` constraint (`p_des[t+1] = p_des[t] + dt·action[t]`) in DPCC would still apply — it only touches the `action` and `p_des` columns. Compatible.

**What you'd lose by dropping velocity from training data:**
- V3 (FM-predicted velocity, dims 9-11): was never used in control — no loss.
- V1 (real `v`, used by PID damping): if switching to MJPC, not needed from FM.
- V2 (`v_des = action/dt_fm`, used by PID): if switching to MJPC, not needed from FM.

**The catch:** Existing expert data (`uav_expert_data_collect`) already records 12D transitions. Switching to 9D or 6D requires either re-recording expert data OR slicing the existing 12D dataset and dropping the velocity columns before training.

---

## §6.7 — Stop-and-go vs continuous flight: the role of velocity feedforward

> Claim under test: "the UAV currently does strict stop-and-go; if we use the old UAV velocity we can fly continuously through all waypoints — but only if the frequency is high enough."

### §6.7.1 First, a correction to the premise

The **current E7 system is NOT strictly stop-and-go.** It already carries a velocity feedforward:

`eval_fm_uav.py` L375 + L383:
```python
v_des = action / dt_fm                    # velocity feedforward = displacement / FM timestep
u = pid.compute(p, q, v, om, p_des, v_des)
```

And the expert paths are already blended (no zero-velocity stops):
`uav_expert_data_collect/generator.py` L159-165 — *"U9: pillar_path is now blended (no zero-velocity stops)"*, *"hovers removed (smooth blended_path)."*

So E7 is **continuous-flight capable today**, because the PID is told at every waypoint *"be at p_des AND moving at v_des≠0,"* which lets the drone blend through without braking.

### §6.7.2 The actual control-theory fact (this is the part that's correct)

The distinction the claim points at is real and important:

| Controller reference | Behaviour at a waypoint | Result |
|---|---|---|
| **Position only** (`p → p_des`, implicit `v → 0`) | error zero ⇒ hover ⇒ brake | **stop-and-go** |
| **Position + velocity feedforward** (`p → p_des` AND `v → v_des≠0`) | nonzero velocity target ⇒ carry momentum | **continuous flight** |

Why position-only stops: with only a position setpoint, when the drone reaches `p_des` the position error is zero, the controller has no remaining command, and (with any velocity damping) it settles to `v=0`. It must re-accelerate for the next waypoint → stop-and-go.

Why velocity feedforward flows through: `v_des = Δp_des/dt_fm` is the *speed the path wants at this instant*. Handing the next waypoint a nonzero entry velocity means the drone is never asked to be stationary at an interior waypoint — it threads them.

### §6.7.3 The "only if frequency is high enough" qualifier — also correct

Velocity feedforward is **necessary but not sufficient**. Continuous flight also needs the replan/control frequency high enough that:

1. **The velocity reference stays fresh.** `v_des` is piecewise-constant between FM steps (zero-order hold). If FM runs too slowly, `v_des` is stale by the time the drone acts on it — the drone overshoots or lurches between waypoints. At 33 Hz the hold interval is ~30 ms, short enough that consecutive `v_des` form a near-continuous reference.
2. **Waypoint spacing ≤ what the drone can blend.** Far-apart waypoints (low freq) force large heading changes per step that the attitude loop can't track smoothly → effective stop-and-go even with feedforward.
3. **The feedforward and the path agree.** `v_des` must point along the path tangent. A smooth blended expert path (E7's U9 paths) gives consistent tangents; a jagged path gives `v_des` that whipsaws.

So: **velocity feedforward + high frequency + smooth path ⇒ continuous flight.** Drop any one and you degrade toward stop-and-go.

### §6.7.4 Consequence for the E8 MJPC (position-only) design

This is the crucial caveat for the FM→MJPC direction (§6.2–§6.6): **dropping velocity from the FM tensor risks reintroducing stop-and-go**, because:

- The FM no longer emits a velocity target — it only emits `p_des`.
- MJPC's stock cost penalizes `||v||²` (E7 §4.1, Residual 1), which actively **brakes at the goal**. A pure position goal + velocity penalty = the drone wants to arrive *and stop*.

Two ways E8 keeps continuous flight despite dropping velocity from the *learned tensor*:

- **(a) Receding goal.** If `p_des` is always pushed ahead of the drone (the FM keeps emitting forward deltas at 33 Hz), MJPC chases a target it never catches → it never reaches the braking condition → continuous flight emerges from the moving setpoint. This is the cleanest answer and needs high FM frequency — *exactly the "freq high enough" condition*.
- **(b) Reduce/zero the velocity penalty** in the UAV-tracker task so MJPC doesn't brake, and let the position residual pull it forward. Risks overshoot; tune `w_v`.

**Net:** velocity can leave the *learned FM tensor* without forcing stop-and-go — but only if the receding-goal frequency is high enough (a) and/or the tracker cost is tuned not to brake (b). The "old velocity" (explicit `v_des` feedforward) is the *simplest* guarantee of continuous flight; MJPC must recover the same effect through a moving goal or a relaxed velocity penalty. See [`../../Epoch8_UAV_Mjpc_thrust_control/PLAN_MJPC_Thrust_Control.md`](../../Epoch8_UAV_Mjpc_thrust_control/PLAN_MJPC_Thrust_Control.md) §4.4.

---

## §6.8 — Is EVERY DPCC-pattern task stop-and-go? (arm avoiding vs SafeFlowMPC vs MJPC)

> Hypothesis under test: "the avoiding-d3il arm is the same stop-and-go problem; ALL d3il/DPCC-pattern designs are stop-and-go unless you do one-shot full-trajectory IK."

**Verdict: the hypothesis is half right and half wrong.** The d3il/DPCC pattern is indeed the *weakest* guarantee of continuity (no velocity anywhere), but it is NOT inherently stop-and-go, and "one-shot IK" is NOT the only escape. **SafeFlowMPC is the decisive counterexample: it is receding-horizon (replans constantly) AND continuous — because it puts velocity inside the trajectory representation.** So the real axis is not "replan vs one-shot" — it's *where velocity lives*.

### §6.8.1 Is the avoiding-d3il arm stop-and-go?

Check the arm's actual controller. `IKControllers.py` `CartPosImpedenceController`:
```python
def reset(self):
    self.desired_c_vel = np.zeros((3,))      # ← velocity reference is ZERO
def getControl(self, robot):
    xd_d = self.desired_c_pos - robot.current_c_pos   # position error only
    target_c_acc = self.pgain * xd_d                   # P on position; NO velocity FF
```

So the arm is **position-only, no velocity feedforward** — structurally the *same* category as the E8 position-only MJPC idea, NOT the same as the E7 UAV PID (which has `v_des`). 

Why it still looks smooth: the **receding-goal mechanism** (E7 §6.7.4 option a). The setpoint is recomputed every FM step as `next_pos_des = action + obs[:2]` = *current position + small delta* (`eval...py` L323). The target is perpetually ~Δ ahead of where the arm actually is, so the arm chases a goal it never catches → continuous motion. **It is continuous by frequency + receding goal, not by velocity.** Drop the frequency and the P-controller settles to each setpoint within its 35 substeps before the next arrives → it degrades to micro stop-and-go. So: the arm is **conditionally continuous**, riding entirely on the moving goal.

### §6.8.2 What SafeFlowMPC does differently (the counterexample)

SafeFlowMPC (`/workspaces/SafeFlowMPC`) is a flow-matching planner + Acados safety filter for a 7-DOF arm — same *family* as FM-PCC, but it is **continuous by construction**:

- **The FM plans the whole horizon at once.** `x_current` is the entire `n_horizon=16` trajectory, not a single next-step (`SafeFlowMPC.py` L199-242). The flow field refines the whole trajectory each step.
- **The trajectory lives in JERK space.** `jerk_to_position` (L254-281) integrates jerk → acceleration → velocity → position:
  ```python
  q[k+1]   = q[k] + dq[k]·dt + ½ddq[k]·dt² + (jerk terms)
  dq[k+1]  = dq[k] + ddq[k]·dt + (jerk terms)      # ← velocity at EVERY knot
  ddq[k+1] = ddq[k] + (jerk terms)
  ```
  Velocity (and acceleration, and jerk) is **intrinsic to the planned state at every interior knot.** The plan is a jerk-limited smooth spline.
- **Receding-horizon shift, still replanning.** `update_current_solution()` (L230) shifts the trajectory forward each step and re-runs flow + safety filter — it replans constantly, exactly like DPCC.
- **It only brakes at the FINAL goal.** Success = `dist_goal < 0.03 AND ||dq|| < 0.05` (L442) — zero velocity *only at the terminal*. Interior knots carry nonzero velocity by design → no interior stops.

So SafeFlowMPC is **receding-horizon AND continuous**, disproving "stop-and-go unless one-shot." The trick: velocity is part of what the FM plans, not something a downstream tracker has to invent or feed-forward.

### §6.8.3 The real taxonomy — *where does velocity live?*

| # | Pattern | Where velocity lives | Continuous? | Examples |
|---|---|---|---|---|
| 1 | **Velocity intrinsic to the planned trajectory** | inside the FM output (jerk/vel/acc/pos spline) | **Yes, by construction** | **SafeFlowMPC** |
| 2 | **Velocity as explicit tracker feedforward** | `v_des` fed to PID alongside `p_des` | Yes, if freq high + smooth path | **UAV FM-PCC E7** (PID, 12D) |
| 3 | **Position-only + receding goal** | nowhere — relies on goal always being ahead | Conditionally (freq-dependent; weakest) | **arm avoiding-d3il**, **E8 MJPC position-only** |
| 4 | **Position-only + arrival-triggered waypoints** | nowhere; advance only after reaching | **No — inherently stop-and-go** | **MJPC stock quadrotor** (`TransitionLocked`, advance at 0.5 m, `quadrotor.cc` L76) |
| 5 | **One-shot full-trajectory IK** | inside the offline spline | Yes, but open-loop (no reactivity) | classical offline traj-opt → playback |

Reading this table against the hypothesis:
- "ALL DPCC-pattern is stop-and-go" → **false.** DPCC (category 3) is *conditionally* continuous via receding goal; and SafeFlowMPC (category 1) is unconditionally continuous while still replanning.
- "unless one-shot IK" → **false.** One-shot (category 5) is one way to be continuous, but category 1 (velocity-in-the-plan) is continuous *and* reactive — strictly better than one-shot for safety/reactivity.
- The hypothesis's *kernel of truth*: the bare DPCC position-only pattern (category 3) has **no velocity anywhere**, so it is the closest to stop-and-go and the most frequency-fragile. The arm avoiding-d3il only escapes via the receding goal.

### §6.8.4 Consequence for our designs

- **E7 UAV (category 2)** is already on solid ground: explicit `v_des` feedforward → continuous.
- **avoiding-d3il arm (category 3)** is continuous only because of its receding goal + high substep rate. It is *not* a velocity-aware design; it's the same fragility class as position-only MJPC.
- **E8 position-only MJPC (category 3 → risk of 4)** must actively avoid sliding into category 4: keep the goal receding at high FM frequency and/or down-weight the MJPC velocity penalty (§6.7.4), else MJPC's `||v||²` cost turns it into arrival-triggered stop-and-go.
- **The clean publishable alternative is category 1 (SafeFlowMPC-style):** make the FM plan velocity (or jerk) directly, so continuity is structural rather than frequency-dependent. For the UAV this would mean the FM tensor carries velocity *as part of the planned trajectory the tracker executes* (not merely as inert V3 dims) — i.e. the tracker consumes the planned velocity, the way SafeFlowMPC's robot controller consumes the planned `dq`. This is the strongest design and worth noting as a future direction beyond E8.

---

## §6.9 — How does SafeFlowMPC guarantee control frequency? (and your "stale velocity → stuck at 0" worry)

> Worry under test: "if the FM computation takes too long, the planned velocity is no longer real-time — does the robot get stuck and the velocity drop to 0?"

**For SafeFlowMPC: no, it does not get stuck — and this is exactly the payoff of putting velocity *inside the buffered trajectory* (category 1).** Three mechanisms in the code guarantee it:

### §6.9.1 The robot executes a *buffered whole-horizon trajectory*, not a single instantaneous command

`step()` sends the **entire** jerk+position horizon to the robot each cycle (`SafeFlowMPC.py` L417):
```python
self.robot_controller.send_joint_jerk_trajectory(jerk_traj, q_traj)   # whole n_horizon=16 trajectory
```
The trajectory is **time-indexed** — knot k has the right velocity for time k, knot k+1 for time k+1, etc. (the `jerk_to_position` integration, L254-281, fills `dq` at every knot). So while the planner is busy computing the *next* plan, the robot keeps executing the *remaining knots* of the last plan — each with its own correct, time-appropriate velocity. **A slow FM does not freeze the command; the robot flies along the buffer.** Velocity does not collapse to 0, because the buffer still has live, forward-moving knots.

This is the structural difference from categories 2–3: there, the controller holds a *single* setpoint (`p_des`/`v_des`) between FM steps. If the FM stalls there, `v_des` goes stale (wrong direction) or the position goal stops receding → the tracker settles → velocity → 0 → stuck. SafeFlowMPC never holds a single point; it holds a whole time-consistent trajectory.

### §6.9.2 Hard per-step deadline + safe fallback

There is an explicit real-time budget (L78): `self.time_limit = 0.8 * self.config.dt_sim` (80% of the control period). If a flow/safety-filter step blows it (L343-353):
```python
if use_safety_filter and (time.perf_counter() - t_start > self.time_limit) and limit_time:
    print("Time limit exceeded ... falling back to last safe trajectory")
    self.x_current = self.last_safe_trajectory      # reuse last feasible plan
    break
```
So the planner **never overruns the control period** — it aborts and reuses `last_safe_trajectory` (L351-352), which is itself a full velocity-carrying horizon. There is always something feasible to execute on time.

### §6.9.3 Async robot clock + time sync

The robot controller runs on its own real-time clock; the planner re-aligns to it rather than blocking it (L457-463):
```python
t_sleep = robot_controller.t_current - iiwa.time.to_sec() - dt_sim - 0.015
time.sleep(max(0, t_sleep))
```
The robot keeps playing the buffered trajectory at the hardware rate; the planner just makes sure its next plan lands before the buffer drains.

### §6.9.4 Worst case is a *safe decel*, not a *stuck-with-wrong-velocity*

If a new plan never arrives, the robot eventually reaches the end of the buffered horizon. Every planned trajectory is feasible and ends in a safe state (the terminal condition the planner solves for; success is `dist<0.03 AND ||dq||<0.05`, L442). So the absolute worst case is the robot finishing a safe trajectory and decelerating cleanly — **not** lurching with a stale velocity. Your "stuck at 0" failure is precisely what categories 2–3 risk and category 1 avoids.

### §6.9.5 Direct answer + the lesson for E8

- **Is the velocity stale if the FM is slow?** Only by at most the buffer depth, and it is *time-indexed* so it stays correct for the knots being consumed — not a single frozen number. A late plan is covered by the buffer; a very late plan is covered by the safe fallback.
- **Does it get stuck / velocity → 0?** No. It flies the buffer; worst case it ends safely. The collapse-to-0 only happens in single-setpoint trackers (UAV PID `v_des`, position-only goal).
- **Lesson for the E8 UAV→MJPC plan:** MJPC's stock interface is single-goal (`mocap_pos`), which is category-3 fragile — a slow FM means the goal stops receding and the `||v||²` cost brakes to 0 (your exact worry, made real). To get SafeFlowMPC's guarantee, E8 would need to **feed MJPC a short time-indexed reference trajectory** (the FM's H-step plan, velocity included) rather than one point — i.e. move E8 from category 3 toward category 1. Noted as the robustness upgrade in the E8 plan.

---

## §7 — Summary table: four control chains

| | Avoiding/DPCC arm | UAV FM-PCC (Gen11) | MJPC quadrotor | UAV-Flow |
|---|---|---|---|---|
| **Planner** | FM (2D Cartesian Δ) | FM (3D Δp_des) | — (MJPC IS the planner+controller) | OpenVLA (4D Δpose) |
| **What planner outputs** | Δ(x,y) | Δp_des | N/A | Δ(x,y,z,yaw) |
| **Low-level controller** | IK (Jacobian pseudo-inv) | CascadedPID (Lee 2010) | Sampling MPC (32 rollouts) | None (teleport) |
| **Controller input** | p_des (Cartesian) | p_des + v_des + real p + real v | p_des (goal position) | absolute pos |
| **Controller output** | joint velocities → torques | 4 motor thrusts | 4 motor thrusts | set_obj_location |
| **Physics** | MuJoCo 7-DOF arm | MuJoCo quadrotor | MuJoCo quadrotor | Unreal Engine (kinematic) |
| **Velocity in FM tensor** | No | Yes (V1, V2, V3) | No (not applicable) | No |
| **Replanning frequency** | Every step (receding horizon) | Every `decim` physics steps (33 Hz) | N/A (continuous) | Every step |
| **Velocity handled by** | IK PD error term implicitly | PID Kd term explicitly | MJPC physics rollout | Not applicable |

---

## §8 — Key takeaways

1. **Both arm and UAV replan every step** (receding-horizon MPC). IK ≠ "plan once A→Z". IK runs on each step's position target.

2. **The arm IK is a velocity-free interface** because IK + PD absorbs velocity internally through the Jacobian. FM only needs to know where to go next, not how fast.

3. **Our UAV PID requires velocity** because the cascaded PID explicitly needs:
   - `v_des` (derivative reference) to avoid driving to zero velocity at each waypoint
   - real `v` (from MuJoCo) for closed-loop stability (double-integrator plant needs velocity feedback)

4. **MJPC is the IK analog for UAVs**: feed goal position → MJPC handles the full dynamics internally. Velocity vanishes from the FM tensor. Cost: ~50× more compute than PID per control step.

5. **UAV-Flow is not a dynamics controller** — it teleports the drone. The "control problem" is eliminated by the simulator's cheat mode.

6. **The proposed FM → MJPC architecture is feasible but requires**:
   - Re-training FM with a position-only observation space (6D or 9D instead of 12D)
   - Wrapping MJPC as the low-level tracker (exposing `set_goal_position → motor_thrusts` API)
   - Verifying MJPC tracks FM waypoints at 33 Hz with acceptable lag
   - This is a future research direction, not a patch

---

## §9 — Code reference index

| Claim | File | Line |
|---|---|---|
| FM called every step in avoiding | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | L292 (`for _ in range(max_episode_length)`) |
| Only `action[0]` executed | `flow_matcher_v3/sampling/policies.py` | L92 |
| 7D Cartesian pose sent to env.step | `eval_flow_matching_v3_ode_selectable.py` | L324 |
| ONE setpoint + IK over n_substeps (per-waypoint, not whole-path) | `d3il/d3il_sim/gyms/gym_env_wrapper.py` | L83-93 |
| n_substeps = 35 (physics steps per FM waypoint) | `d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py` | L55 |
| IK Jacobian pseudo-inverse | `d3il/d3il_sim/controllers/IKControllers.py` | L58-73 |
| UAV FM at 33 Hz, PID at physics rate | `FM_v3_uav_test/eval_fm_uav.py` | L314 (`decim`), L378 (`for _ in range(decim)`) |
| PID outer loop math | `uav_env_test/flight_controller.py` | L88-91 |
| PID allocation M u = wrench | `uav_env_test/flight_controller.py` | L134-135 |
| MJPC residuals (pos+vel+ω+ctrl) | `mujoco_mpc/mjpc/tasks/quadrotor/quadrotor.cc` | L37-57 |
| MJPC horizon 0.5 s, 32 candidates | `mujoco_mpc/mjpc/tasks/quadrotor/task.xml` | L15-17, L22 |
| UAV-Flow teleport | `UAV-Flow/UAV-Flow-Eval/batch_run_act_all.py` | L327-328 |
| v_des = action/dt_fm | `FM_v3_uav_test/eval_fm_uav.py` | L375 |
| p_des update (free-running Euler) | `FM_v3_uav_test/eval_fm_uav.py` | L374 |
| Arm IK has ZERO velocity feedforward (position-only) | `d3il/d3il_sim/controllers/IKControllers.py` | L44-45, L58-59 |
| Arm receding goal = current pos + Δ | `eval_flow_matching_v3_ode_selectable.py` | L323 |
| SafeFlowMPC plans whole horizon (n_horizon=16) | `SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py` | L199-242 |
| SafeFlowMPC jerk→acc→vel→pos (velocity intrinsic) | `SafeFlowMPC/safe_flow_mpc/SafeFlowMPC/SafeFlowMPC.py` | L254-281 |
| SafeFlowMPC receding-horizon shift + replan | `SafeFlowMPC/.../SafeFlowMPC.py` | L230, L325-417 |
| SafeFlowMPC brakes only at terminal goal | `SafeFlowMPC/.../SafeFlowMPC.py` | L442 |
| MJPC stock quadrotor arrival-triggered advance (stop-and-go) | `mujoco_mpc/mjpc/tasks/quadrotor/quadrotor.cc` | L76 |
| SafeFlowMPC hard time budget (0.8·dt_sim) | `SafeFlowMPC/.../SafeFlowMPC.py` | L78 |
| SafeFlowMPC time-limit → fallback to last_safe_trajectory | `SafeFlowMPC/.../SafeFlowMPC.py` | L343-353 |
| SafeFlowMPC sends whole jerk+pos trajectory to robot | `SafeFlowMPC/.../SafeFlowMPC.py` | L417 |
| SafeFlowMPC async robot-clock time sync | `SafeFlowMPC/.../SafeFlowMPC.py` | L457-463 |
