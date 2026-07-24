# MuJoCo Sim vs IRL — Velocity Staleness, Frozen Time, and v_des

**Questions answered:** Is v in the 12D obs v_des or v_real? Does MuJoCo freeze simulation time
during FM inference? Does the §9 zero-latency problem (DESIGN_obs_action_loop.md) actually apply
in simulation? Is v_des override from expert dataset meaningful?

---

## 1 What Is `v` in the 12D Obs Tensor?

`v` is **real velocity from MuJoCo physics** — not v_des.

```python
# rollout_one(), eval_fm_uav.py line 381:
v = data.qvel[:3].copy()                            # ← real velocity from physics engine
obs = np.concatenate([p_des, p, v]).astype(np.float32)  # [p_des(3D) | p(3D) | v(3D)]
```

`v_des` is computed LATER from the FM action (lines 425–431), purely to feed the PID:

```python
# after FM returns:
if controller == 'pid_stopgo':
    v_des = np.zeros(3)
elif controller == 'pid_const_v':
    v_des = unit(action) * v_des_magnitude          # ← derived from action
else:
    v_des = action / dt_fm                          # ← derived from action
```

**v_des never enters the FM obs.** The FM only sees `[p_des | p_real | v_real]`.

The 12D label refers to transition_dim = action(3) + obs(9) = 12. `v` in obs = `v_real`.

---

## 2 Does MuJoCo Freeze While FM Computes? — Yes, Always

The simulation loop in `rollout_one()` is:

```python
for k in range(n_fm):
    # A. Sample real state — AFTER previous iteration's mj_step calls
    v = data.qvel[:3].copy()
    obs = np.concatenate([p_des, p, v])

    # B. FM inference — no mj_step called here
    action, traj = policy({0: obs}, ...)   # wall-clock time passes, sim time FROZEN

    # C. Derive v_des, update p_des ...

    # D. Physics substeps — simulation time advances ONLY here
    for _ in range(decim):               # decim = 3 (100Hz / 33Hz)
        v = data.qvel[:3].copy()         # real v each substep (for PID, not FM)
        u = tracker.compute(p, q, v, om, p_des, v_des)
        mujoco.mj_step(model, data)      # ← simulation time advances ONLY here
```

`mujoco.mj_step()` is the **only** call that advances simulation time. During step B
(`policy(obs, ...)`), wall-clock time passes (GPU inference, ODE integration) but
**zero simulation time passes**. MuJoCo is deterministic and passive — it only steps
when you call `mj_step`.

When FM returns at step B, `data.qvel` is still exactly the state at the end of
step k−1's physics. The `v` sampled at A is therefore **perfectly current** — no staleness.

---

## 3 Does the §9 Zero-Latency Problem Apply in Simulation?

**No. The §9 problem is a real-robot concern only.**

| Context | v at FM obs | Staleness |
|---|---|---|
| MuJoCo simulation | `data.qvel` after k−1's `mj_step` calls | **Zero** — sim frozen during inference |
| Real robot (IRL) | sensor reading at the moment of sampling | **Stale by Δt_infer ≈ 10ms** |

In MuJoCo: FM inference time is irrelevant — simulation time does not pass. The `v`
fed into obs is the exact velocity at the FM step boundary, regardless of how long
inference takes (1ms or 1s — makes no difference to the simulation state).

On a real drone: the drone keeps flying during inference. If inference takes 10ms,
the drone has moved: `v(t + 10ms) ≠ v(t)`. The FM makes a decision based on a
velocity that has already passed. This is the §9 structural problem.

**Consequence for evaluation:** MuJoCo eval results are optimistic for real-robot
deployment. The sim drone has perfect velocity information at every FM step. A real
drone would have 10ms-stale velocity, introducing the drift described in §9.

If GPU inference is fast (1–3ms), the real-robot staleness is small:
```
v drift = a × Δt_infer ≈ 2 m/s² × 0.002s = 4 mm/s → acceptable
```

If inference is slow (50ms+), the staleness becomes significant:
```
v drift = 2 m/s² × 0.05s = 100 mm/s → 25% of nominal flight speed
```

The 12D (v in obs) design is therefore GPU-speed-dependent for real-robot use.

---

## 4 Is `v_des` Override From the Expert Dataset Meaningful?

### What "override to expert dataset v_des" means

At eval, v_des for `pid_const_v` is:
```python
v_des_magnitude = mean(|action|) × DATASET_HZ    # calibrated from dataset
v_des = unit(action) * v_des_magnitude            # direction from FM, magnitude from dataset mean
```

The user's idea: instead of this, use the expert's actual `v_real` values from the dataset
as v_des — matching training conditions more closely.

### Why it makes sense (the appeal)

During expert data collection:
```python
p_des, v_des, a_des = traj_fn(t)   # analytic smooth trajectory
u = pid.compute(p, q, v, om, p_des, v_des, a_des)
```
Expert's `v_des` matched `p_des` exactly — both came from the same analytic curve.
Expert's `v_real ≈ v_des` because the PID tracked well.

So `v_real` in the dataset ≈ the `v_des` the PID received during collection.
At eval, if we could match `v_des` to what the dataset's PID saw, the PID would
reproduce similar behavior → `v_real` at eval would stay in-distribution for the FM obs.

### Why it is problematic

**Problem 1: FM generates different waypoints.**
Expert `v_real[k]` was the velocity at expert waypoint k on the smooth analytic curve.
At eval, FM's `p_des[k]` is a different waypoint — the FM is generating its own trajectory,
not replaying the expert. Expert `v_real[k]` has no spatial relationship to FM's `p_des[k]`.
Using it as v_des tells the PID to chase a velocity that belongs to a different path.

**Problem 2: Only valid when v_real ≈ v_des in distribution.**
The user notes this correctly: "only when real v is same/similar to v_des is making sense."
Expert v_des varied along the smooth curve — it was higher on straight segments, lower near
turns. Applying those velocities to FM's different waypoints introduces random mismatch.

**Problem 3: `pid_const_v` already uses the dataset mean.**
`v_des_magnitude = mean(|action|) × DATASET_HZ` is already the mean of `action/dt_fm`
over the entire dataset — effectively the average v_des the expert's PID received.
The direction comes from the FM's current action (correct intent direction).
This is already a dataset-calibrated approximation without the per-step mismatch problem.

### Comparison: stop-and-go vs expert v_des override

| Option | v_des | Valid when | Failure mode |
|---|---|---|---|
| `pid_stopgo` | 0 | Always — drone just stops and goes | Jerky, slow; correct but non-smooth |
| `pid_const_v` | `unit(action) × dataset_mean` | FM direction is correct; magnitude is right on average | Magnitude wrong for fast/slow segments |
| Expert v_des per-step | Expert dataset v_real[k] | FM replays exact expert waypoints | Random mismatch if FM diverges from expert |
| `pid` (legacy E7) | `action / dt_fm` | Inference latency is negligible | Noisy when dt_fm jitters |

**Verdict:** Expert v_des override is not better than `pid_const_v` for the general case.
`pid_const_v` already extracts the global mean from the dataset without the per-step
path-mismatch problem. The override would only be meaningful if FM is perfectly replicating
expert waypoints (it is not — it generates its own trajectory).

**Compared to stop-and-go:**
The user asks if expert v_des override makes more sense than `pid_stopgo`.
Yes — any non-zero v_des tells the PID not to brake, which is better than stopping at
every waypoint. But `pid_const_v` with dataset-calibrated magnitude already achieves this
without the path-mismatch issue. The comparison is `pid_const_v` vs `pid_stopgo`, and
`pid_const_v` is the better design (U3).

---

## 5 Summary

| Question | Answer |
|---|---|
| `v` in 12D obs tensor = v_des? | **No.** `v` = `data.qvel[:3]` (real MuJoCo velocity). v_des is derived from the FM action after inference and never enters obs. |
| MuJoCo frozen during FM inference? | **Yes.** `mj_step` is the only time-advance. FM inference is pure compute — zero sim time passes. v is perfectly synchronized at every FM step boundary. |
| §9 zero-latency problem in simulation? | **Does not apply.** Simulation freezes. §9 is a real-robot problem only. MuJoCo eval is therefore optimistic for IRL deployment. |
| Expert v_des override — meaningful? | **Marginal.** Appealing but only valid if FM replicates expert waypoints. `pid_const_v` already uses the dataset mean implicitly and is more robust. |
| IRL: does drone drift during FM inference? | **Yes.** Real drone keeps flying during Δt_infer. If inference is fast (1–3ms), drift is ~4mm/s — tolerable. If slow (50ms+), staleness is significant. |
