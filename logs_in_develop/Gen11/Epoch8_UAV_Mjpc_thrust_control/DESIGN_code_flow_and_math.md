# E8 Code Flow & Math — FM→MJPC Thrust Control

**Scope:** UAV Epoch 8 (`cond_mode='pos_only'`, `controller='mjpc'`).
**Companion:** [`PLAN_MJPC_Thrust_Control.md`](PLAN_MJPC_Thrust_Control.md) · [`CHANGELOG_E8_Implementation.md`](CHANGELOG_E8_Implementation.md)

---

## 1 Big Picture

```
pkl (T,9)         FM-ODE          MJPC
[p_des|p|v]  →   samples H=8   →  optimal   →  u[4]  →  MuJoCo
               position traj     thrusts       physics
```

The FM replans at ~33 Hz; only the first waypoint drives the tracker each substep. MJPC recovers the velocity profile from MuJoCo physics — velocity is NOT in the FM tensor (that's the whole point of E8).

---

## 2 Dataset (train time)

**File:** `flow_matcher_v3_uav/datasets/d4rl.py:sequence_dataset`

```
raw pkl ep['obs'] : (T, 9)   [p_des(3) | p(3) | v(3)]
raw pkl ep['actions'] : (T-1,3)  Δp_des

cond_mode='pos_only' slice:
  obs    → obs[:, 0:6]      # (T,6)   [p_des | p],  v dropped
  action → ep['actions']    # (T-1,3) Δp_des, unchanged
```

No data regeneration — pure column slice. Resulting **transition** fed to SequenceDataset:

```
τ  =  [ Δp_des(3) | p_des(3) | p(3) ]   dim = 9
       ← action  →← ────── obs ──────→
```

---

## 3 FM Training Math

Flow Matching ODE on transition sequences of length H=8:

```
x₀ ~ N(0, I)            noise
x₁ ~ data (τ sequence)  target

Conditional flow:  x_t = (1-t)·x₀ + t·x₁,   t ∈ [0,1]

Velocity target:   v* = x₁ - x₀

Loss:  L = E[ || v_θ(x_t, t) - v* ||² ]
```

Time sampling (`t_schedule='beta'`, default): `t = 1 - Beta(α=1.5, β=1.0)` skewing toward t≈1 (data-side).
Alternatively `t_schedule='logit_normal'` (U7): `t = σ(randn·p_std + p_mean)`, p_mean=-0.4, p_std=1.0.

At inference: ODE solved from t=0→1 (noise→data) via fixed-step Euler in `FlowMatchingODE.forward()`.

---

## 4 Eval Loop Code Flow

**File:** `FM_v3_uav_test/eval_fm_uav.py:rollout_one`

```
OUTER LOOP  (FM replan at ~33 Hz)
│
├─ Build obs:  obs = [p_des | p]   (6D, from env state)
├─ FM.forward(obs, batch_size)     →  traj (H×9)  sampled trajectories
├─ Projector selects best traj     →  traj[best]  (H×9)
├─ Extract action[0]:              →  Δp_des       (3D)
│
├─ p_des += Δp_des                 (Euler step on the goal)
│
└─ INNER LOOP  (physics substeps, typically 1:1 at 33 Hz)
   │
   ├─ tracker.compute(p, q, v, om, p_des, v_des)  →  u[4]  thrusts
   └─ mujoco.step(u)               →  new (p, q, v, om)
```

**`v_des` is still passed** (from `Δp_des / dt_fm`) but `MJPCTracker.compute` ignores it — accepted for API parity with `CascadedPID`.

---

## 5 MJPC Tracker Math

**File:** `FM_v3_uav_test/mjpc_tracker.py:MJPCTracker.compute`

```python
qpos = [p(3) | q(4)]          # 7D free-joint position
qvel = [v(3) | ω(3)]          # 6D velocity from MuJoCo state
agent.set_state(qpos, qvel, mocap_pos=p_des)   # goal = p_des

for _ in range(planner_steps):   # typically 10 (cartpole.py default)
    agent.planner_step()         # sampling MPC iteration

u = agent.get_action()[:4]      # 4 motor thrusts
```

MJPC internally minimizes over a short horizon H_mjpc (≈0.3 s, tunable):

```
J  =  Σ_{t=0}^{H}  [ w_p‖p(t) - p_des‖²  +  w_v‖v(t)‖²  +  w_u‖u(t)‖² ]

subject to  MuJoCo dynamics  p(t+1), v(t+1) = f(p(t), v(t), u(t))
```

Optimizer: **sampling MPC** — draws N random control sequences, rolls them out, keeps the lowest-cost one (or MPPI-style weighted average). `n_trajectories=16` controls the fan.

Only `u[0]` (the first action) is ever applied — classic receding-horizon.

---

## 6 Contrast with E7 PID

| | E7 CascadedPID | E8 MJPCTracker |
|---|---|---|
| **FM obs** | `[p_des\|p\|v]` 9D | `[p_des\|p]` 6D |
| **Transition dim** | 12 | 9 |
| **Velocity in FM** | Yes | No |
| **Tracker** | Cascade: pos→vel→att→ω→u | Joint opt: (pos,vel,u) together |
| **Control law** | PD analytically | Sampling MPC numerically |
| **v_des used** | Yes (feedforward) | No (MuJoCo internal) |
| **Retrain** | — | Required (shape mismatch) |
| **Checkpoint suffix** | *(none)* | `_cmpos_only_ctrlmjpc` |

---

## 7 Path Discrimination

`_uav_exp_name(args)` in `config/uav.py`:

```
(p_des, pid)      →  H8_Dmodels.diffusion.FlowMatchingODE
(pos_only, pid)   →  H8_D…_cmpos_only
(pos_only, mjpc)  →  H8_D…_cmpos_only_ctrlmjpc   ← E8
```

Same function used by BOTH train and plan blocks → paths always agree, existing E7 checkpoints unaffected.
