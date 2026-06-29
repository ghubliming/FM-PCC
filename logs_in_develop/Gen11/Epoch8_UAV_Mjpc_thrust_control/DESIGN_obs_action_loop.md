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

## 8 One-Line Summary Per Variable

| Variable | Source | Notes |
|---|---|---|
| `action = Δp_des` | FM network output | Only NN output |
| `p_des` in obs | Accumulated past actions | `init_pos + Σ Δp_des` (or anchored) |
| `p` in obs | MuJoCo state | Real drone position |
| `v` in obs | MuJoCo state | Real drone velocity (E7 only) |
| `v_des` to PID | `unit(Δp_des) * v_des_magnitude` (`pid_const_v`) | Config-driven magnitude; ignores dt_fm |
| `u[4]` motor thrusts | PID or MJPC | Never touches FM |
