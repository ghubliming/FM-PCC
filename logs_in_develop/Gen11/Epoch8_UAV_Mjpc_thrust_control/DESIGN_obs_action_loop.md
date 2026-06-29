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

## 6 `anchor_to_p` Fix (U4 fix_5) — Is It Correct? Inside or Outside H8?

```python
p_des = p + Δp_des    # rebind to real position every step
```

### Where does the drift happen — inside H8 or outside H8?

**Outside H8 — between FM re-queries.** This is the only place drift accumulates.

Within one FM step (inside the 30ms execution window): `p_des` is FROZEN for all 3 physics substeps. There is no drift within one step — p_des doesn't move.

Inside the FM's H=8 internal plan: the FM plans `p_des[1] = p_des[0] + action[0]`, `p_des[2] = p_des[1] + action[1]`, etc. This is free-running accumulation WITHIN the plan — it is correct by design (the dynamics constraint enforces it). No anchor needed here; there is no "real p" inside a prediction.

The drift in §5 happens ACROSS many FM steps — each re-query, free-running `p_des += action` lets p_des run ahead of p_real. After 50 steps, the gap `p_des - p_real` is large.

**anchor_to_p is applied at the FM step boundary — outside H8. It is correctly placed.**

### Does it introduce a new distribution shift?

Yes — a bounded, mild one. In the training data (expert collection):

```
Training:  p_des[k+1] = p_des[k] + action[k]      ← pure free-running (traj_fn)
Eval free: p_des[k+1] = p_des[k] + action[k]      ← same formula — but gap GROWS
Eval anch: p_des[k+1] = p_real[k] + action[k]     ← different! uses real p
```

`p_des[k]` and `p_real[k]` are different when PID has tracking error. So anchor_to_p gives a different `p_des[k+1]` than training expected.

BUT: if the PID tracks well, `p_real[k] ≈ p_des[k]` → `p_des_anch ≈ p_des_free` → no shift.

The two failures compared:

| Mode | `p_des - p_real` over time | Severity |
|---|---|---|
| Free-running | Grows unboundedly as PID error accumulates | Catastrophic — FM obs leaves training distribution |
| anchor_to_p | Always = `action[k]` (one FM step magnitude) | Bounded — small and close to training |

In training, `p_des - p_real = ε_track` (small, expert PID tracked well). With anchor_to_p: `p_des - p_real = action - (p_real_new - p_real_old)`. If drone tracks its goal well, `p_real_new - p_real_old ≈ action` → `p_des - p_real ≈ 0`. Close to training distribution.

**anchor_to_p is correct.** The bounded distribution shift it introduces is far smaller than the unbounded drift it prevents.

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

**Short answer: yes, you are correct — and it is worse than it first appears.**

### The core contradiction

The FM obs includes `v` (real velocity). Including `v` in the obs is only meaningful if `v` is **current at the moment the FM acts on it**. But FM inference takes Δt_infer > 0. So:

```
t=0:    obs ← [p_des | p(0) | v(0)]   ← snapshot
t=Δt:   action = FM(obs)               ← FM finally decides, based on v(0)
        but real drone is now at v(Δt) — different, already moved on
        the v(0) the FM used has already passed. it is no longer real time.
```

**FM made a planning decision based on a velocity that no longer exists.** The action is applied to a drone in a different dynamic state than what the FM observed.

### Why velocity is worse than position

Position and velocity are both stale by Δt_infer, but they have very different sensitivity:

| State | Changes in Δt_infer = 10ms | Relative to FM step magnitude | Severity |
|---|---|---|---|
| `p` | v × Δt ≈ 0.4 × 0.01 = **4 mm** | ~3% of one Δp_des step (~0.12m) | Low |
| `v` | a × Δt ≈ 2.0 × 0.01 = **20 mm/s** | ~5% of v_des (0.4 m/s) | Moderate |

Under PID control, the drone applies accelerations of 1–5 m/s² continuously. In 10ms, velocity can shift by 10–50 mm/s — a non-trivial fraction of the nominal flight speed. **Velocity is an inherently fast-moving signal.** Feeding a 10ms-stale velocity into the FM for planning is not "slightly stale" — for a dynamic system, velocity has already meaningfully changed.

Position is slower (velocity × time, second-order), so positional staleness is more tolerable.

### Three compounding problems

**1. `v` in obs is non-real-time by construction.** If the FM obs includes `v`, zero latency is a hard requirement — not a performance preference. Any Δt_infer > 0 means the FM receives a velocity that has already passed. The FM's next action is computed for a drone that no longer has that velocity.

**2. `v_des` frozen for the entire substep window.** `v_des` is set once per FM step (30ms, 3 physics substeps). The PID velocity error `e_v = v_real - v_des` accumulates throughout those 30ms as v_real evolves. Whether using `action/dt_fm` or `pid_const_v`, v_des is a fixed feedforward for a moving target.

**3. Expert collection had none of these problems.** During expert data collection:
```python
p_des, v_des, a_des = traj_fn(t)   # evaluated at exact current time t — always fresh
u = pid.compute(p, q, v, ω, p_des, v_des, a_des)   # per-physics-step, 100 Hz
```
- No FM inference — trajectory is a closed-form formula evaluated at the exact moment of use
- `v_des` updated every 10ms (100 Hz), never frozen for 30ms
- `a_des` feedforward cancels inertia exactly

At eval, `v` in obs is Δt_infer stale; `v_des` updates at 33 Hz; `a_des = 0`. Three simultaneous regressions.

### `pos_only` (9D) partially addresses this

By dropping `v` from the FM obs (`cond_mode='pos_only'`), the FM no longer depends on an instantaneous signal it can never truly have:

| `cond_mode` | FM obs | Latency sensitivity |
|---|---|---|
| `p_des` (12D) | `[p_des \| p \| v]` | High — v is fast-moving, stale in 10ms |
| `pos_only` (9D) | `[p_des \| p]` | Lower — p is slow-moving, 4mm drift in 10ms |

`pos_only` does not fix the problem — p is still stale — but position changes slowly enough that the staleness is tolerable. Velocity is not.

### When does the 12D design hold at all?

Only when Δt_infer << 10ms (one physics dt). On GPU with a small network, inference may be 1–3ms. At that point:
- v drift = 2 m/s² × 0.002s = 4 mm/s — acceptable
- p drift = 0.4 m/s × 0.002s = 0.8 mm — negligible

So the 12D design is GPU-speed-dependent and degrades gracefully as inference slows. The 9D (`pos_only`) design is more robust to latency — the FM simply does not ask for the one signal that cannot be provided in real time.

### What a correct design would require

| Fix | What it means |
|---|---|
| Predict-ahead obs | At t=0, predict p(t+Δt_infer), v(t+Δt_infer) using a physics rollout; feed predicted state |
| Async FM + PID | PID runs at 100Hz independently on latest action; FM computes in background |
| Retrain with injected latency | During data collection, feed Δt-stale obs so FM learns to plan under staleness |
| Drop `v` from obs | `pos_only` — FM never depends on the real-time signal it cannot have |

None of the first three are implemented. The last one (`pos_only`) is implemented as E8.

### Is the Velocity Fed Back at the Next FM Step Actually Real-Time?

This is the sharpest version of the question. The answer depends on **simulation vs real robot**, and on **whether we re-query every step or every H steps**.

**In MuJoCo simulation — v IS exact at every FM step boundary.**

Trace the exact loop from `rollout_one`:

```python
for k in range(n_fm):
    # A. Sample state — AFTER previous iteration's mj_step calls completed
    v = data.qvel[:3].copy()          # real v AFTER step k-1's physics
    obs = [p_des, p, v]

    # B. FM inference — NO mj_step called here
    action, traj = policy(obs, ...)   # wall-clock time passes, sim time FROZEN
    action = action[:3]               # only action[0] used — NOT H steps

    # C. Update goal, freeze v_des
    p_des = p + action
    v_des = ...                       # frozen for next 3 substeps

    # D. Physics substeps — sim time advances HERE
    for _ in range(decim):            # 3 iterations
        p = data.qpos[:3].copy()      # real p each substep
        v = data.qvel[:3].copy()      # real v each substep (for PID, not FM)
        u = pid.compute(p, q, v, om, p_des, v_des)
        mujoco.mj_step(model, data)   # only now does sim time advance

# Next iteration k+1: v sampled at A is AFTER all 3 mj_step calls above — exact.
```

`mj_step` is the only thing that advances simulation time. FM inference (step B) is pure Python compute — zero simulation time passes. So:

- `v` at FM step k+1 = `data.qvel` after step k's 3 `mj_step` calls = **exactly the real velocity in simulation, no staleness**
- Simulation does not "fly" between our calls; it only advances when we tell it to

**The §9 latency problem is a real-robot concern, not a simulation concern.**

In simulation, v is always perfectly synchronized at FM step boundaries. The within-step error is different: `v_des` is frozen for 3 substeps while `v_real` evolves under PID forces — but the PID itself reads real `p` and `v` every substep (step D above, lines 415-416).

**On a real robot — v IS stale.**

The drone flies in real clock time. During FM inference (Δt_infer ≈ 10ms), the real drone has moved. When you sample v for step k+1, it is v at the wall-clock moment of sampling — but the FM for step k+1 still takes another Δt_infer to compute, during which v moves on again.

**"Execute H=8 then re-query" vs "re-query every step" — this is the key.**

We execute only `action[0]` per FM call — **1 step = 30ms** — then immediately re-query from the real state. We do NOT execute all H=8 steps before re-querying.

```
Receding horizon (what we do):
  step k:   FM → plan[0..7], execute plan[0] → 30ms → re-query with real v
  step k+1: FM → plan[0..7], execute plan[0] → 30ms → re-query with real v
  ...
  v at each query = real velocity after previous step's physics ✓

Open-loop H=8 execution (what we do NOT do):
  block k:  FM → plan[0..7], execute ALL plan[0..7] → 240ms → re-query
            plan[1..7] were computed from v(T), but actual drone has v(T+30ms)...v(T+210ms)
            plan is increasingly wrong by step 7
  v at next query = 240ms of drift, no correction in between ✗
```

H=8 is the FM's **planning foresight** (how far ahead it considers when picking action[0]), not the **execution window**. The FM looks 8 steps ahead to plan better, but hands us only step 0 to execute. Then we re-plan.

**Summary:**

| Context | v at next FM query | Real-time? |
|---|---|---|
| MuJoCo simulation | after k's `mj_step` calls | Exact — sim time was frozen during inference |
| Real robot | v when sampling for step k+1 | Slightly stale by Δt_infer (§9 problem) |
| Open-loop H=8 then re-query | v after 240ms of flight | Stale by 240ms — plan diverges badly |
| Receding horizon (our design) | v after 30ms of flight | Exact in sim; minor staleness on real robot |

---

### How SafeFlow MPC Addresses This — And Its Own Limitation

**pos_only (no v in FM obs) solves the real-time staleness problem but introduces a new one: velocity continuity is no longer guaranteed.**

```
12D (v in obs):
  FM sees v_real → plans Δp_des that continues smoothly from current momentum
  → guaranteed velocity continuity at FM step boundaries
  → REQUIRES real-time v → structurally impossible (§9 problem)

pos_only (no v in obs):
  FM does NOT see v_real → plans Δp_des from position alone
  → no guarantee that acts[0]/dt_fm matches v_real from previous step
  → velocity discontinuity possible at each FM step boundary
  → avoids the staleness problem
  → effectively a "soft stop-and-go" at step boundaries
```

**So pos_only SafeFlow IS a form of stop-and-go.** Not literal v=0 braking (v_des ≠ 0), but the FM at each step ignores current momentum. It plans from position, hands a new v_des to the controller, and the controller has to fight any mismatch between v_des and v_real. This is a velocity jerk at every FM step boundary.

With v in obs (12D): FM conditions on momentum → smooth continuation → true continuous flight.
Without v in obs (pos_only): FM blind to momentum → potential velocity discontinuity every 30ms.

**Why pos_only is still used:**

The tradeoff is real but the alternative is worse:

| | 12D with v in obs | pos_only without v |
|---|---|---|
| Real-time v dependency | Structurally required, impossible on real robot | None |
| Velocity continuity | Guaranteed (FM conditions on it) | Not guaranteed |
| Flight character | Truly continuous | Soft stop-and-go at FM boundaries |
| Sim only? | Works in sim (v is exact); breaks on real robot | Works in both |

**MJPC partially compensates.** MJPC receives `p_des` and internally solves for optimal `u[4]` over a physics horizon, planning its own velocity profile to reach `p_des` smoothly. It doesn't need v_des from FM at all. So MJPC can bridge the velocity discontinuity at step boundaries better than CascadedPID (which has no such internal look-ahead).

**v_des and a_des from the H-step plan (§10) partially compensate for PID.** From consecutive plan steps:
```python
v_des = acts[0] / dt_fm
a_des = (acts[1] - acts[0]) / dt_fm²
```
These are smooth within one H-step plan (the FM produced a coherent plan). At the boundary between receding-horizon calls, consecutive plans' acts[0] values should be close (because the FM re-plans from a nearly-continued state). So the velocity jerk at boundaries is small in practice, not catastrophic.

**The honest summary:**

The real-time v problem (§9) and the stop-and-go problem are **in direct tension**:
- Feed v into FM obs → solves stop-and-go, creates real-time staleness requirement
- Drop v from FM obs → solves staleness, creates soft stop-and-go

There is no free lunch. MJPC as inner loop is the closest to escaping this tension — it handles velocity internally and only needs positional goals from FM.

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

## 11 Receding Horizon as Self-Correction — Why §9 Drift Doesn't Compound

### Is SafeFlow MPC stop-and-go?

**No.** SafeFlow MPC (and our implementation) is true receding-horizon MPC — the drone flies continuously. There is no "stop and peek." There IS a latency window (Δt_infer, §9) where the drone keeps flying on the previous p_des and v_des while the next FM call is computing, but this is not a deliberate stop — it's just the inference gap.

### Clarifying H=8 — Plan Horizon ≠ Execution Horizon

This is the critical point:

```
H = 8  →  FM plans 8 future Δp_des steps at once
           but we EXECUTE only step 0 (30 ms, 3 physics substeps)
           then immediately re-query FM from the new real state

Execution horizon = 1 FM step = 30 ms   (always, regardless of H)
Planning horizon  = 8 FM steps = 240 ms (how far FM sees ahead when choosing step 0)
```

We do NOT execute all 8 planned steps before re-planning. The FM re-queries every 30ms with fresh (p, v). H=8 is foresight for better planning, not a long open-loop execution window.

### The Drift Window is Bounded to 30ms

The zero-latency drift (§9) only accumulates **within one 30ms FM step**:

```
FM step k:
  t=0 ms:   obs snapshot [p_des | p(0) | v(0)]
  t=Δt ms:  action = FM(obs)          ← stale by Δt_infer
             PID executes on stale p_des, frozen v_des for 30ms
             tracking error may accumulate
  t=30 ms:  FM step k DONE

FM step k+1:
  t=30 ms:  NEW obs snapshot [p_des | p(30ms) | v(30ms)]   ← REAL current state
             FM re-plans from actual position and velocity
             tracking error from step k is NOT carried forward into FM's obs
```

**The receding horizon is the self-correction mechanism.** Even if the PID drifted during step k, FM step k+1 sees the real (p, v) and adjusts its action accordingly. The error doesn't compound across steps — it's reset at every 30ms re-query.

### anchor_to_p Closes the Loop

Without `anchor_to_p`: `p_des = p_des + Δp_des` — p_des can drift ahead of real p across many steps even if FM re-queries correctly (see §5).

With `anchor_to_p=True`: `p_des = p + Δp_des` — p_des is also reset from real position at every step. Combined with receding-horizon re-querying of (p, v), both the goal and the obs are grounded every 30ms.

```
Every 30ms:
  obs = [p_des | p_real | v_real]   ← grounded (anchor_to_p + fresh query)
  action = FM(obs)
  p_des = p_real + action            ← immediately re-anchored
```

### Summary: Three Timescales

| Timescale | What happens | Self-correcting? |
|---|---|---|
| Within physics substep (10ms) | PID tracks p_des at 100Hz | Yes — PID feedback |
| Within FM step (30ms) | v_des frozen, obs stale | No — open-loop drift (§9) |
| Across FM steps (30ms boundary) | FM re-queries real (p, v), anchor_to_p resets p_des | **Yes — receding horizon** |

The §9 zero-latency flaw is real but its blast radius is one 30ms window. The receding horizon ensures errors do not accumulate across the episode. This is why the design works in practice even without the §9 fixes — the FM continuously re-plans from reality.

---

## 12 Fundamental: Waypoint-Based Planning is Inherently Stop-and-Go

**Any controller whose goal is to REACH position B will brake to zero at B.**

This is first principles. A position controller minimises position error `e_p = p - p_des`. The only way to have `e_p → 0` stably is to also have velocity `v → 0` at B — otherwise the drone overshoots. This is true regardless of whether the inner loop is PID, MJPC, or IK.

```
Move A → B:
  controller minimises |p - B|
  to settle at B: v must → 0
  otherwise: drone overshoots B, oscillates

→ any pure waypoint tracker is inherently stop-and-go
```

**This applies to ALL of: SafeFlow, D3IL, DPCC, our UAV FM — anything that outputs position waypoints.**

### Three escapes from stop-and-go

**Escape 1: FM update rate faster than settling time.**

If the FM updates p_des → C before the drone reaches B, the drone is always chasing a moving target — it never settles at any waypoint. This is why the avoiding robot arm appears continuous:

```
step k:   p_des = B  (drone chases B)
step k+1: p_des = C  (drone chases C before reaching B)
step k+2: p_des = D  ...
→ drone follows a smoothed path through B, C, D — never stops
```

This only works if the controller bandwidth is LOW relative to the FM update rate — i.e. the drone moves slowly toward each waypoint so the next one arrives first. Robot arm at high frequency: works. UAV at 33Hz with fast PID: may actually reach B → stops → next FM step.

**Escape 2: Explicit v_des feedforward.**

Tell the controller "at waypoint B, your desired velocity is v_des ≠ 0." The position error term still pushes toward B, but the velocity error term now says "don't brake — move at v_des." The drone passes through B at v_des instead of stopping.

```
e_v = v_real - v_des    (with v_des = acts[0]/dt_fm, non-zero)
→ PID does not brake at B
→ drone passes through B continuously
```

This is what `v_des` feedforward does (§3). It is a PATCH on the stop-and-go problem — not a solution. The drone still doesn't know about C when passing B; it just doesn't stop at B because it was told not to.

**Escape 3: Controller horizon extends past current waypoint.**

If the inner-loop controller (MJPC) has a horizon that covers multiple FM steps, it can plan a smooth trajectory THROUGH B toward future waypoints — naturally generating non-zero velocity at B.

```
MJPC horizon = 0.3s ≈ 10 FM steps
→ MJPC "sees" not just B but also C, D, E...
→ plans a smooth path that passes through B without stopping
→ velocity at B is naturally non-zero
```

This is the principled escape. MJPC is the right tool here — but only if it has enough horizon to see past the current waypoint.

### Why the robot arm appears continuous (but isn't really)

The avoiding task robot arm uses escape 1 implicitly: the arm's position control bandwidth is high — it tracks commanded positions nearly instantly. So the FM update arrives before the arm fully settles at B. The arm ends up smoothly following the sequence of position commands.

The "dynamics" (velocity) is irrelevant because the arm moves so fast it's always mid-transit when the next command arrives.

### Why the UAV is different

The UAV settles slowly — PID takes ~100ms+ to converge to p_des. At 33Hz (FM step = 30ms), the next p_des arrives long before the drone reaches the current one. So technically escape 1 applies — but unlike the robot arm, the drone has real momentum and inertia. Without v_des feedforward, PID actively BRAKES velocity (§3.1) even while pursuing a waypoint, making flight slower and jerkier.

### The Closed Loop: Non-Zero Velocity at Waypoints Requires Zero Latency

This is the hardest insight. Stop-and-go and the zero-latency problem (§9) are **two sides of the same coin**:

```
Stop-and-go (v=0 at waypoints):
  → no latency problem — controller doesn't need to know velocity in real time
  → drone settles at B, waits for next p_des
  → jerky flight, but correct

Continuous flight (v≠0 at waypoints):
  → v_des must be computed NOW and applied to the drone's state NOW
  → but FM has inference latency Δt_infer
  → by the time v_des is applied, drone has already moved
  → v_des is stale → tracking error → drift
  → the longer the inter-waypoint window (30ms for UAV), the worse the error
```

**If you want non-zero velocity at waypoints, you MUST have zero-latency state feedback.**

Because: v_des computed from `acts[0]/dt_fm` at time T is the correct velocity only AT time T. During the 30ms execution window, the drone moves, the path curves, the required velocity direction and magnitude change. A single frozen v_des cannot capture this. You need continuous real-time velocity updates — which requires zero latency.

**Why the robot arm "escapes" this problem:**

The arm doesn't escape it in theory — it escapes it in practice because:
- The arm's bandwidth is so high that each inter-waypoint motion takes ~1ms
- In 1ms at low speed, position error < 0.5mm, velocity drift negligible
- The 0-latency error is too small to matter

The UAV has a 30ms inter-waypoint window at 33Hz. In 30ms at 0.4 m/s, the drone travels 12mm. Velocity drift over 30ms with wrong v_des is significant. The problem is real, not negligible.

**The fundamental dilemma — no free lunch:**

| Choice | Velocity at waypoints | Latency requirement | Tracking quality |
|---|---|---|---|
| Stop-and-go (v_des=0) | Zero — drone stops | None | Correct but jerky |
| v_des feedforward (patch) | Non-zero but stale | Requires near-zero latency | Drifts if latency > few ms |
| MJPC long horizon | Non-zero, internally planned | Position-only from FM (latency-tolerant) | Smooth, principled |
| Full trajectory (not waypoints) | Continuous, computed offline | Requires exact model of drone | Best — but no adaptive replanning |

**The only true escape: MJPC with long horizon or full offline trajectory generation.**

MJPC internally generates smooth velocity profiles over its planning horizon — it doesn't need v_des from FM at all. It solves the inter-waypoint velocity problem internally using its physics model. The FM only provides p_des (latency-tolerant position goal). MJPC handles the rest.

Everything else is a patch with bounded validity.

### Summary

| Escape | Mechanism | Works for UAV? |
|---|---|---|
| Fast FM update rate (escape 1) | Drone always chasing — never settles | Partially — PID still brakes, drift still occurs |
| v_des feedforward (escape 2) | PID told not to brake at waypoints | Patchy — stale by Δt_infer, drifts in 30ms window |
| Long MJPC horizon (escape 3) | Controller plans smooth velocity internally | Yes — principled; only needs p_des from FM |

**Without any of these: stop-and-go is the correct default.** `pid_stopgo` (U2) is honest — it accepts the physics rather than patching around it with stale velocity feedforward.

---

## 13 First Principles — What Does the Inner-Loop Controller Actually Need from FM?


**From first principles: any controller only needs ONE thing from the FM — the goal position `p_des`.**

Every other piece of information (p, v, q, ω) the controller reads from its own sensors. FM is the high-level planner; the controller is the executor. Their interface is purely positional.

### State feedback vs feedforward intent

| Information | Source | FM needs to provide? |
|---|---|---|
| `p` real position | Robot's own sensors (encoder, MuJoCo) | Never |
| `v` real velocity | Robot's own sensors | Never |
| `q` attitude | IMU | Never |
| `ω` angular rate | IMU | Never |
| `p_des` goal | FM plan | **Yes — this is the only FM output** |
| `v_des` desired velocity | FM plan (optional feedforward) | Only for PID — not for MJPC/IK |
| `a_des` desired acceleration | FM plan (optional feedforward) | Only for PID — not for MJPC/IK |

### Does each controller type need anything besides `p_des`?

**IK (robot arm avoiding task):** No. IK is purely geometric — given `p_des`, compute `q_des`, track with joint PD. The joint PD reads `dq` from its own encoders. FM only provides `p_des`. Velocity is implicit in consecutive position commands.

**MJPC (UAV E8):** No. MJPC receives `p_des` and internally plans an optimal trajectory to reach it — computing its own `v`, `a`, and `u[4]` profile using a physics model. FM's only contract with MJPC is positional. Velocity is handled entirely inside MJPC.

**CascadedPID (UAV E7):** Technically no — but it behaves as stop-and-go (§3.1). Without `v_des`:
```
e_v = v_real - 0 = v_real  →  always brakes to zero
```
`v_des` feedforward is needed for continuous flight — but this is a **PID limitation**, not a fundamental requirement. PID is a reactive controller with no internal trajectory planning. It needs to be told the intent ("you are supposed to be moving").

### The velocity problem is a PID problem, not an FM problem

```
IK   → p_des only → works perfectly     (position tracking is fast, velocity implicit)
MJPC → p_des only → works perfectly     (internal physics model handles velocity)
PID  → p_des only → stop-and-go         (needs v_des feedforward to not brake)
PID  → p_des + v_des → continuous       (feedforward tells PID not to fight velocity)
```

If the inner loop is capable enough (MJPC or high-bandwidth position tracking), FM only outputs `p_des` and the controller handles everything else. The fact that our CascadedPID needs `v_des` from FM is because PID has no internal foresight — it reacts to errors without knowing the plan.

### Why the avoiding task works with position-only FM

The avoiding robot arm/car has a high-bandwidth position controller — it reaches commanded positions in one timestep. Velocity self-regulates from consecutive position commands (`v_implicit = Δp/dt`). FM never needs to know or command velocity because the inner loop handles it implicitly.

### Why UAV is harder

The UAV has real inertia — it cannot reach `p_des` in one timestep. The CascadedPID needs 30ms (3 substeps) to move toward `p_des`, and during those 30ms it will brake unless told `v_des`. MJPC solves this by planning internally — but introduces other complexity (requires a MuJoCo task definition, gRPC communication, cluster-only).

**The cleanest design:** FM outputs `p_des` only. Inner loop handles everything else. PID is a workaround that requires v_des feedforward. MJPC is the principled solution.

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
