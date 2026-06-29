# Dataset Origin — PID Role, MJPC Impact, Recollect Decision

**Context:** E8 uses `cond_mode='pos_only'` which slices the existing PID-collected dataset.
The question is whether collecting new data with MJPC would be necessary or beneficial.

---

## 1 How Expert Data Is Collected (Epoch 4)

### 1.1 The Geo Env (MuJoCo Geometric Scenes)

The "Geo Env" is a set of 4 MuJoCo scenes with static geometric obstacles:

| Scene | Obstacles | Homotopy classes |
|---|---|---|
| `empty` | none | N/A |
| `corridor` | 2 flat walls at y=±0.5 | L, C, R |
| `s_curve` | 4 wall segments (two chicanes) | default |
| `pillars` | 6 cylinders (3 pairs) | (L,L,L) (L,R,L) (R,L,R) (R,R,R) |

Each episode is one randomised trial: random start position, random altitude `z ∈ [0.90, 1.30]`, random duration within scene bounds. Homotopy label (which side of each obstacle the drone passes) is chosen deterministically and cycled for balance.

### 1.2 Two-Layer Pipeline: Geometry First, Physics Second

**Layer 1 — Geometric reference trajectory** (`uav_expert_data_collect/trajectories.py`)

A pure-math function `traj_fn(t) → (p_des, v_des, a_des, yaw_des)`. No MuJoCo, no physics. This is a `blended_path` (U9) — cubic fillets through waypoints that guarantee the drone stays in the correct homotopy channel with ≥8 cm clearance. The waypoint geometry is hard-coded per scene (e.g. `_Y_L = -1.11`, `_Y_R = +1.11` for pillars).

**Layer 2 — Physics execution** (`uav_expert_data_collect/generator.py:run_trial`)

The `CascadedPID` tracks `p_des(t)` in the MuJoCo scene at 100 Hz:
```python
p_des, v_des, a_des, yaw_des = traj_fn(t)        # pure geometry
u = pid.compute(p, q, v, om, p_des, v_des, ...)  # physics-aware
data.ctrl[:4] = u
mujoco.mj_step(model, data)
# record: steps.append({'p': p, 'v': v, 'p_des': p_des, 'q': q})
```

Episodes are **rejected** if contact fraction too high or drone crashes (floor < 0.5 m).

### 1.3 What Gets Recorded

`dataset_writer.py:rollout_to_episode` converts the raw rollout to the pkl schema:

```
obs     (T, 9)   = [p_des(3) | p(3) | v(3)]
actions (T-1, 3) = Δp_des = np.diff(targets, axis=0)   ← LINE 71
targets (T, 3)   = p_des sequence (+ constant noise offset per episode)
```

**Critical:** `actions = Δp_des` = the GEOMETRY layer differenced. Not the PID output. Not `Δp`.

---

## 2 What Role Did PID Play?

### 2.1 The PID is NOT an Inverse Solver — It is a Feedback Cascade

A common misreading: "feed position from Geo constraints → PID inversely computes velocity."
**This is wrong.** The CascadedPID (`uav_env_test/flight_controller.py`) computes **motor thrusts** from measured errors using 4 sequential closed-loop layers (Mellinger / Lee 2010 structure):

```
INPUT:  p_des(t), v_des(t), a_des(t)  ← from traj_fn (pure geometry)
        p(t), v(t), q(t), ω(t)        ← from MuJoCo state (real physics)

Layer 1 — Position PD + feedforward
  e_p   = p - p_des
  e_v   = v - v_des
  a_cmd = -Kp_pos·e_p - Kd_pos·e_v + a_des
  F_world = mass · (a_cmd + g)         ← desired world-frame thrust force

Layer 2 — Desired attitude (SO(3))
  b3_des = F_world / |F_world|         ← rotor disk must point this way
  R_des  = [b1_des | b2_des | b3_des]  ← full desired rotation matrix

Layer 3 — Attitude PD on SO(3) error (Lee 2010)
  e_R  = vee(R_des^T R - R^T R_des) / 2
  τ    = -Kp_att·e_R - Kp_ω·ω + gyro_compensation

Layer 4 — Thrust allocation
  [T; τ_x; τ_y; τ_z] → M_inv → u[4]  ← 4 motor thrusts

OUTPUT: u[4] → data.ctrl → mujoco.mj_step() → new p, v, q, ω
```

**Velocity is NOT computed by the PID.** `v(t)` in the dataset comes from MuJoCo's physics integrator stepping forward under the motor thrusts the PID commanded. The PID uses `v` as a *measured input* to its derivative term, never as an output.

### 2.2 What PID Contributes vs. What It Does Not

| What PID contributes | What PID does NOT contribute |
|---|---|
| `u[4]` → MuJoCo step → `p(t), v(t)` in dataset obs | `p_des(t)` — purely from `traj_fn`, geometry only |
| Episode quality: contact fraction, floor rejection | `actions = Δp_des` — diff of `p_des`, pure geometry |
| Tracking error: `p(t) = p_des(t) + ε_pid` | The homotopy routing or obstacle-avoidance path shape |
| The specific oscillation / overshoot signature in `v(t)` | The waypoint geometry baked into `targets` |

**PID's sole role:** provide the physics bridge so the MuJoCo simulation produces realistic `p(t)` and `v(t)` traces along the geometric reference path. The FM's *targets* (`Δp_des` actions) are 100% controller-independent.

---

## 3 What MJPC Replaces — and When

**MJPC only operates at EVAL time. It never touched data collection.**

At eval time, MJPC replaces the CascadedPID as the inner-loop tracker:

```
TRAINING (data collect, Epoch 4):
  traj_fn(t) → p_des  ──→  CascadedPID  ──→  u[4]  ──→  MuJoCo  ──→  p, v  (in dataset)

EVAL E7 (PID tracker, same as collection):
  FM generates p_des  ──→  CascadedPID  ──→  u[4]  ──→  MuJoCo  ──→  p, v  (real flight)

EVAL E8 (MJPC tracker, new):
  FM generates p_des  ──→  MJPCTracker  ──→  u[4]  ──→  MuJoCo  ──→  p, v  (real flight)
```

Both PID and MJPC take `p_des` as input and output `u[4]` motor thrusts. They differ only in HOW they compute those thrusts:

| | CascadedPID | MJPCTracker |
|---|---|---|
| Input | `p_des, v_des, p, v, q, ω` | `p_des, p, v, q, ω` (no `v_des` needed) |
| Method | Analytic feedback cascade (4 layers) | Sampling MPC: N rollouts over horizon H, minimize `J = w_p‖p-p_des‖² + w_v‖v‖² + w_u‖u‖²` |
| Velocity profile | Inferred from PD derivative term | Optimized jointly with position over H steps |
| Computation | O(1), microseconds | O(N·H·physics), milliseconds |
| Tuning | Kp, Kd gains per layer | N trajectories, horizon, cost weights |

The FM never sees motor thrusts at any point. The FM generates `p_des` — a POSITION GOAL — and the tracker (PID or MJPC) is entirely responsible for getting the drone there.

## 4 If We Redesigned From the Geo Env With MJPC

To collect new data using MJPC instead of PID, the changes in `generator.py:run_trial` would be:

```python
# replace:
pid = _make_pid(model, gain_variant)
...
u = pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)

# with:
tracker = MJPCTracker(model, task_id=mjpc_task_id, ...)
...
u = tracker.compute(p, q, v, om, p_des)
```

Everything else stays: `traj_fn`, waypoints, homotopy routing, rejection thresholds, `dataset_writer`. The resulting pkl schema would be **identical**: same `obs` shape, same `actions` definition. Only `p(t)` and `v(t)` inside `obs` would change (reflecting MJPC tracking instead of PID tracking).

---

## 5 Comparison: Recollect vs Reuse

| Factor | Reuse PID data | Recollect with MJPC |
|---|---|---|
| `actions = Δp_des` | Identical — pure geometry | Identical — same traj_fn |
| `p_des` in obs | Identical — same traj_fn | Identical — same traj_fn |
| `p` in obs | PID tracking error (obs mismatch at E8 eval) | MJPC tracking error (matched to E8 eval) |
| `v` in obs | From PID dynamics | From MJPC dynamics |
| FM learns obstacle avoidance | Yes — from `Δp_des` shape, controller-irrelevant | Yes — identical |
| Train/eval distribution match (`p`) | Small shift (good PID → `p ≈ p_des`) | Exact match |
| Train/eval distribution match (`v`) | Shifted | Exact — but E8 drops `v` anyway |
| Cost | Zero (slice existing pkl) | Full re-collection run (~hours on cluster) |
| Risk | Low — `pos_only` drops `v`, only `p` mismatch remains | Zero distribution shift, but operational cost |

### The E8-specific wildcard: `cond_mode='pos_only'` drops velocity

With E7's `cond_mode='p_des'`, the obs is `[p_des | p | v]` — both `p` and `v` carry PID-specific dynamics. With E8's `cond_mode='pos_only'`, obs becomes `[p_des | p]` — velocity is **already removed**. The only remaining PID-specific signal in the training distribution is `p` (actual position), which reflects tracking error.

For a well-tuned controller, tracking error is small: `p ≈ p_des`. So:

```
training obs:  [p_des | p_pid]   where  p_pid = p_des + ε_pid,   ε_pid small
eval obs:      [p_des | p_mjpc]  where  p_mjpc = p_des + ε_mjpc,  ε_mjpc small
```

The FM sees `[p_des | p_des + small_error]` in both cases. If MJPC tracks at least as well as PID (which it should — MPC is the optimizer; PID is the approximation), the distribution shift is neutral to favorable.

Additionally: `anchor_to_p=True` (E7 U4 fix) already rebinds `p_des ← p_real + action` every step, directly correcting drift from `p` mismatch. This mechanism works for MJPC too.

---

## 6 Decision: No Recollection Needed

**Reason:** The FM is learning a GEOMETRIC PATH PLANNER — it generates `Δp_des` sequences that avoid obstacles within the correct homotopy. These are derived entirely from `traj_fn` (geometry), not from the PID. The PID only contributes `p(t)` to the obs, and this contribution:

1. Is already small (`p ≈ p_des` for a well-tuned cascade)
2. Becomes even smaller for MJPC (better tracker → tighter tracking)
3. Is partially corrected by `anchor_to_p`
4. Is further reduced in E8 because `pos_only` already dropped the more controller-sensitive `v` column

**MJPC "can 100% do it" reasoning:** MJPC and PID are both tracking the SAME geometric reference `p_des(t)`. If the PID successfully reaches each waypoint without collision (which is what the rejection filter verifies), MJPC — a strictly more capable optimizer over the same physics model — will also reach each waypoint. The FM never saw motor thrusts or PID intermediate states; it saw positions. Any controller that produces `p ≈ p_des` is interchangeable at the FM's observation level.

**What to monitor at eval (not a blocking issue):**
- If MJPC has higher tracking error than PID (e.g. due to sampling MPC noise), the `[p_des | p]` distribution seen at inference will be slightly further from training than expected. This would show as increased FM replanning uncertainty, not hard failure. Tune `mjpc_trajectories` and `mjpc_planner_steps` to bring tracking error down.
- Use `anchor_to_p=True` in the eval config to further neutralize any residual drift.

---

## 7 Summary

```
traj_fn(t)  →  p_des(t)  [PURE GEOMETRY — controller-independent]
                   ↓
              p_des diff  →  Δp_des = actions  [PURE GEOMETRY]
                   ↓
         PID / MJPC tracks p_des  →  p(t), v(t)  [CONTROLLER-SPECIFIC]

FM learns:  Δp_des sequences  (geometry)    ← no controller bias
FM conditions on: [p_des | p]               ← p has tiny PID bias → negligible for E8
E8 drops: v                                 ← removes the more controller-sensitive signal
```

**Verdict: reuse the PID dataset. No recollection.**
The dataset is controller-agnostic at the action level (what the FM generates) and nearly controller-agnostic at the observation level after the `pos_only` velocity drop.
