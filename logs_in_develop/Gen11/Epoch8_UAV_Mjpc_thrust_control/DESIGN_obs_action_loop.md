# UAV FM — Observation / Action Loop Explained

**Confusion addressed:** where does `p_des` in obs come from? what does the NN actually output? where does `v_des` come from?

---

## 1 What the FM Network Outputs

The FM network outputs **one thing only:**

```
action = Δp_des   (3D — x, y, z position delta)
```

It never outputs velocity, attitude, or motor thrusts. Everything else is derived.

---

## 2 Where `p_des` in the Observation Comes From

`p_des` is **NOT output by the NN**. It is a running accumulation of all past FM actions.

```
Episode start:
  p_des ← init_pos          (drone's starting position)

Each FM step k:
  obs   = [p_des | p | v]   ← p_des from the PREVIOUS step fed back in
  Δp_des = FM(obs)           ← NN output
  p_des  = p_des + Δp_des    ← accumulated (free-running Euler)
```

After N steps:
```
p_des_N = init_pos + Δp_des_0 + Δp_des_1 + ... + Δp_des_{N-1}
```

`p_des` in obs is the **commanded goal position** — where the planner has told the drone to be. `p` (from MuJoCo) is where it actually is.

---

## 3 Where `v_des` Comes From — and Why

### 3.1 Why the PID needs `v_des` at all

The CascadedPID position loop:
```python
e_p   = p - p_des
e_v   = v - v_des
a_cmd = -Kp_pos * e_p - Kd_pos * e_v + a_des
```

The `Kd_pos * e_v` term is the velocity damping. Without `v_des` (i.e. `v_des = 0`):
```
e_v = v_real - 0 = v_real
→ Kd term = -Kd_pos * v_real   ← always fights velocity → brakes to zero every step
```

With `v_des ≠ 0`:
```
e_v = v_real - v_des
→ if drone is already at v_des: e_v = 0 → no braking → continuous flight
```

**`v_des` tells the PID "you are supposed to be moving — don't brake."** Without it, the PID treats any nonzero velocity as an error and kills it. That is exactly `pid_stopgo` (U2) — `v_des=0` is the strict stop-and-go mode.

### 3.2 Why NOT from the expert dataset

The expert data records `v` (the drone's actual velocity during collection). But at eval time the FM is generating a **different** sequence of `p_des` waypoints — not replaying the expert trajectory. The expert's velocity at step `k` corresponded to the expert's smooth `traj_fn(t)` at that moment. The FM's waypoints are different, so the expert's `v` has no meaning here.

Also: during data collection, the expert controller received `v_des` from `traj_fn(t)` — an analytically differentiable smooth curve. At eval there is no such curve. The FM only produces discrete `Δp_des` steps.

### 3.3 Why NOT from the physics control frequency (dt_physics = 0.01 s)

The physics runs at 100 Hz (dt = 0.01 s), but `p_des` only changes at the FM rate (33 Hz, dt_fm ≈ 0.030 s). The PID holds `p_des` constant for 3 physics substeps (`decim = 3`).

The intended velocity is "cover `Δp_des` in the time between FM goal updates":
```
v_des = Δp_des / dt_fm        (correct — FM update period)
      ≠ Δp_des / dt_physics   (wrong — 3× too fast, would overshoot wildly)
```

Using `dt_physics` would tell the PID to reach `p_des` in 0.01 s instead of 0.030 s — a commanded speed 3× higher than the FM intended. Using `dt_fm` matches the actual goal-update cadence.

### 3.4 Implemented Formula — `pid_const_v` (U3)

```python
norm  = ||action||
v_des = (action / norm) * v_des_magnitude    if norm > 1e-6
      = zeros(3)                              otherwise (hovering)
```

`v_des_magnitude` is a config key (default `0.4` m/s, matches expert dataset average speed). This:

- **Ignores `dt_fm`** — no timing sensitivity, no jitter propagation
- **Overrides the default control-rate formula** — direction from FM action; magnitude from config, not clock arithmetic
- **Stable across runs** — same speed regardless of inference latency

**All controller options:**

| `controller=` | `v_des` | Notes |
|---|---|---|
| `pid` | `action / dt_fm` | E7 legacy; timing-sensitive |
| `pid_stopgo` | `zeros(3)` | U2; strict brake-to-zero |
| `pid_const_v` | `unit(action) * v_des_magnitude` | **U3 — implemented** |
| `mjpc` | N/A (internal) | E8; cluster-only |

### 3.5 The Expert vs Eval `v_des` Mismatch

During **data collection** (Epoch 4), the PID received `v_des` from the analytic trajectory function:

```python
p_des, v_des, a_des, yaw_des = traj_fn(t)   # smooth, analytically differentiable curve
u = pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)
```

`v_des` here is the exact time-derivative of the smooth geometric path — stable, noise-free, consistent with `p_des` by construction.

At **eval time**, there is no `traj_fn`. The FM produces discrete `Δp_des` steps, so `v_des` is approximated:

```python
v_des = action / dt_fm    # finite difference — noisier, jitter-sensitive
```

| | Data collection | Eval |
|---|---|---|
| `v_des` source | `traj_fn(t)` analytic derivative | `Δp_des / dt_fm` finite difference |
| Stability | Smooth, exact | Step-to-step variation, dt jitter |
| `a_des` fed to PID | Yes (from `traj_fn`) | No (defaults to 0) |

**Does this matter?** The FM never sees `v_des` — it only sees `v` (real velocity from MuJoCo). What matters is whether the PID, driven by the approximate `v_des`, produces a `v` at eval that matches the `v` distribution in training. If yes, the FM's obs is consistent and it plans well. If the approximation is too rough, the PID behaves differently → `v` at eval drifts from the training distribution → FM sees unfamiliar obs.

This is a secondary source of train/eval mismatch on top of the `p` tracking-error mismatch discussed in [`DESIGN_dataset_pid_vs_mjpc.md`](DESIGN_dataset_pid_vs_mjpc.md).

---

## 4 Full Step-by-Step Loop

```
INIT:
  p_des = init_pos
  p, v  = MuJoCo initial state

FM STEP k  (runs at 33 Hz):
  ┌─────────────────────────────────────────────────────┐
  │  1. Build obs:                                       │
  │       obs = [p_des | p | v]   (9D raw)              │
  │            ^current goal ^real pos ^real vel         │
  │                                                      │
  │  2. FM inference:                                    │
  │       Δp_des = FM(obs)        (3D — only NN output) │
  │                                                      │
  │  3. Update goal:                                     │
  │       p_des = p_des + Δp_des  (free-running Euler)  │
  │                    OR                                │
  │       p_des = p    + Δp_des   (anchor_to_p — safer) │
  │                                                      │
  │  4. Derive v_des (pid_const_v — U3):                 │
  │       v_des = unit(Δp_des) * v_des_magnitude        │
  │              (ignores dt_fm; magnitude from config)  │
  └─────────────────────────────────────────────────────┘

PHYSICS SUBSTEPS  (runs at 100 Hz, 3 steps per FM step):
  ┌─────────────────────────────────────────────────────┐
  │  for each substep:                                   │
  │    p, q, v, ω = MuJoCo state                        │
  │    u[4] = PID.compute(p, q, v, ω, p_des, v_des)    │
  │    MuJoCo.step(u)                                    │
  └─────────────────────────────────────────────────────┘

  → new p, v fed into next FM step's obs
```

---

## 5 The Free-Running Drift Problem

With `p_des = p_des + Δp_des` (default), `p_des` accumulates errors:

```
step 0:  p_des = init + Δp_0                   (fine)
step 1:  p_des = init + Δp_0 + Δp_1            (fine)
...
step 50: p_des = init + Σ Δp                   (may be far from real p)
```

If the PID has any tracking error, `p` lags behind `p_des`. Over many steps, the gap `p_des - p` grows. The FM then sees an obs where `p_des` is far ahead of `p` — a distribution it may never have seen in training (training had good PID tracking, so `p ≈ p_des`). This causes the FM to generate worse actions → the problem compounds.

```
Free-running:
  p_des ──────────────────────────→  (runs ahead)
  p     ───────────────────→         (lags, tracking error)
        gap grows over time ↑
```

---

## 6 `anchor_to_p` Fix (U4 fix_5)

```python
p_des = p + Δp_des    # rebind to real position every step
```

Instead of accumulating, `p_des` is always grounded to the drone's actual position plus the FM's next step. The gap `p_des - p` stays bounded to `|Δp_des|` (one FM action magnitude):

```
anchor_to_p:
  p_des = p + Δp    (always close to p)
  p     ───────────→
        gap ≤ |Δp| always ✓
```

---

## 7 Dimension Summary

| Mode | obs fed to FM | NN output | transition dim |
|---|---|---|---|
| E7 default (`p_des`) | `[p_des\|p\|v]` 9D | `Δp_des` 3D | **12D** |
| E8 pos_only | `[p_des\|p]` 6D | `Δp_des` 3D | **9D** |

`v` in E7 obs = **input conditioning** so the FM knows current momentum when planning the next `Δp_des`. It is not the source of `v_des`.

---

## 9 The Zero-Latency Assumption — Fundamental Design Flaw

**Short answer: yes, you are correct.**

The entire obs/action loop works *only if FM inference is instantaneous*. The design implicitly assumes Δt_infer = 0. In reality it is not.

### What the assumption looks like

```
ideal (Δt_infer = 0):
  t=0:  obs ← [p_des | p(0) | v(0)]
        action = FM(obs)           ← takes 0 ms
        p_des updated, v_des set
        PID uses p(0), v(0), v_des — all fresh

real (Δt_infer > 0, e.g. 10 ms):
  t=0:  obs ← [p_des | p(0) | v(0)]   ← snapshot
  t=10ms: action = FM(obs)             ← computed from 10ms-old obs
        p_des updated, v_des set
        PID uses p(0), v(0) from obs — STALE
        real drone is now at p(10ms), v(10ms) — never seen by FM
```

FM gave an action based on where the drone *was*, not where it *is*. The PID then chases a goal computed from stale state.

### Three compounding problems

**1. Obs staleness.** `v` and `p` sampled at t=0 are fed to the PID at t=Δt_infer. The drone has moved during FM inference. The obs the FM saw no longer reflects reality when its action is applied.

**2. `v_des` frozen for the entire substep window.** Whether using `action/dt_fm` (pid) or `unit(action)*v_des_magnitude` (pid_const_v), `v_des` is held constant for the full 30ms FM step (3 physics substeps). The drone's actual velocity changes continuously during those 30ms. The PID velocity error `e_v = v_real - v_des` accumulates drift within the window.

**3. Expert collection had neither problem.** During expert data collection:
```python
p_des, v_des, a_des = traj_fn(t)   # evaluated at exact current time t — always fresh
u = pid.compute(p, q, v, ω, p_des, v_des, a_des)   # per-physics-step, 100 Hz
```
- `v_des` updated at 100 Hz from an analytic function — never stale, never frozen
- No inference latency — trajectory is a closed-form formula
- `a_des` feedforward cancels inertia exactly

At eval, `v_des` updates at 33 Hz (FM rate) and is stale by Δt_infer on top of that. This is a **third dimension of train/eval mismatch**, on top of the `p` tracking error mismatch (§3.5) and the v_des formula mismatch (§3.5).

### When does the design hold?

Only when Δt_infer << dt_physics = 10ms. For a small UAV-FM network on GPU this might be 1–3ms — marginal. On CPU or with larger models it breaks down clearly.

### What a correct design would require

| Fix | What it means |
|---|---|
| Predict-ahead obs | At t=0, predict p(t+Δt_infer), v(t+Δt_infer) using physics model; feed predicted state to FM |
| Async FM thread | PID runs at 100Hz independently; FM computes in background; PID uses latest available action |
| Online Δt_infer measurement | Measure actual FM wall-clock time each step; use it in v_des formula (if using pid default) |
| Retrain with latency | Inject artificial latency during data collection so FM trains on stale obs |

None of these are implemented. The current design is **zero-latency–only**. It works to the extent that GPU inference is fast and the drone dynamics are slow enough that 10ms staleness doesn't destabilize the PID.

---

## 10 Why SafeFlow MPC Can Track `v_des` and `a_des` — The Wasted H-Step Plan

The FM does NOT output only one action. It outputs an **H-step trajectory plan** — all H future Δp_des at once:

```python
action, traj = policy({0: obs}, batch_size=batch_size, horizon=horizon)
# action          = traj.actions[which][0]    ← only step 0, what we execute
# traj.actions[which]  shape: (H, 3)          ← ALL H future Δp_des, already computed
```

**We currently throw away `traj.actions[which][1:]` entirely.**

From consecutive plan steps you can reconstruct exactly what the expert collection had:

```python
acts = traj.actions[which]           # (H, 3)
v_des      = acts[0] / dt_fm         # velocity for this step
v_des_next = acts[1] / dt_fm         # velocity for NEXT step (free — already computed by FM)
a_des      = (v_des_next - v_des) / dt_fm   # acceleration feedforward
```

| Source | v_des | a_des | Latency-safe? |
|---|---|---|---|
| Expert (`traj_fn`) | analytic derivative at each 100Hz PID step | analytic 2nd derivative | Yes (zero latency, continuous) |
| `pid` (E7) | `acts[0] / dt_fm` — single step finite diff | None (a_des=0) | No — dt_fm jitter |
| `pid_const_v` (U3) | `unit(acts[0]) * 0.4` — constant magnitude | None | No — magnitude is wrong |
| **H-step plan** (not yet implemented) | `acts[0] / dt_fm` — from FM's own plan | `(v[1]-v[0])/dt_fm` — from FM's own plan | Partial — lookahead covers latency window |

The H-step plan also **partially mitigates the zero-latency flaw** (§9): while the PID executes step k for 30ms (3 substeps), the plan already knows step k+1's Δp_des. The latency Δt_infer is spent executing step k while step k+1 is already planned — so the next obs snapshot is already pre-answered.

This is what SafeFlow MPC exploits: the FM's receding-horizon plan is not just for collision avoidance — it gives a smooth sequence of `v_des` and `a_des` feedforward **for free**, matching the expert collection structure.

**Current status: unimplemented.** This would require reading `traj.actions[which][1]` in `rollout_one` and passing `a_des` to the PID. The PID already accepts `a_des` (see `tracker.compute(p, q, v, om, p_des, v_des, a_des=a_des)`).

---

## 8 One-Line Summary Per Variable

| Variable | Source | Notes |
|---|---|---|
| `action = Δp_des` | FM network output | Only NN output |
| `p_des` in obs | Accumulated past actions | `init_pos + Σ Δp_des` (or anchored) |
| `p` in obs | MuJoCo state | Real drone position |
| `v` in obs | MuJoCo state | Real drone velocity (E7 only) |
| `v_des` to PID | `unit(Δp_des) * v_des_magnitude` (`pid_const_v`) | Config-driven magnitude; ignores dt_fm |
| `u[4]` motor thrusts | PID or MJPC | Never touches FM |
