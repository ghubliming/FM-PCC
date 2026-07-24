# Epoch 8 — UAV MJPC Thrust Control (position-only FM → MJPC optimal control)

**Date:** 2026-06-28
**Status:** PLAN (no code yet)
**Design basis:** [`../Epoch7_fm_pcc_FULL_PCC_MPC/U4_cond/DESIGN_control_chain_arm_vs_UAV.md`](../Epoch7_fm_pcc_FULL_PCC_MPC/U4_cond/DESIGN_control_chain_arm_vs_UAV.md) §6 (the "IK-style UAV" idea) and §2.5.4 (real-time vs whole-trajectory axis)
**Companion:** [`../Epoch7_fm_pcc_FULL_PCC_MPC/U4_cond/DESIGN_data_flow_FM_to_MuJoCo.md`](../Epoch7_fm_pcc_FULL_PCC_MPC/U4_cond/DESIGN_data_flow_FM_to_MuJoCo.md) (12D tensor, V1/V2/V3 velocity)

---

## §0 — TL;DR

Add an **optional** second control backend beside the current E7 cascaded-PID:

- **Planner (new, optional):** FM trained on a **strict-DPCC 9D transition** `[action(3) | p_des(3) | p(3)]` — velocity dropped. Pure position planner.
- **Tracker (new, optional):** MJPC sampling-MPC optimal-control solver replaces the PID. FM feeds it a **position goal** (`p_des`); MJPC outputs the 4 motor thrusts, reading the drone's velocity from MuJoCo state directly (not from the FM tensor).

**Nothing is removed.** The current 12D `[action|p_des|p|v]` + cascaded-PID path stays the default. The new mode is selected by config flags.

**Dataset answer (the headline question):** **No recollection, no redesign. Pure column slice of the existing `.pkl`.** The expert data already stores `[p_des|p|v]`; the new mode slices `[p_des|p]` (drops `v`, cols 6:9) — exactly mirroring the existing `cond_mode='real_p'` precedent. See §3.

---

## §1 — Motivation (why this Epoch exists)

From the E7 design doc §6: the cascaded-PID requires velocity in the FM tensor (V1 real `v` for damping, V2 `v_des=action/dt` for the derivative reference). The arm's IK avoids this because the IK+PD layer absorbs velocity internally — FM only outputs Cartesian position.

**MJPC is the IK analog for the UAV** (E7 §6.5): feed it a goal position, it optimizes motor thrusts over a short physics-rollout horizon, inferring the required velocity profile internally. This lets the FM tensor drop velocity and become a strict-DPCC position planner — closer to the canonical DPCC `[action | state]` layout, and arguably cleaner for publication (the planner/tracker split of E7 §2.5.4 made literal).

This Epoch is the **experimental arm** that tests whether FM→MJPC can match or beat FM→PID, without sacrificing the working E7 pipeline.

---

## §2 — Architecture (both backends side-by-side)

```
                              ┌─────────────────────────────────────────┐
   obs ──► FM ──► action[0] ─►│  DPCC projector (QP, obstacle+deriv)     │─► p_des (safe)
                              └─────────────────────────────────────────┘
                                                  │
                  ┌───────────────────────────────┴───────────────────────────────┐
                  │                                                                 │
        E7 DEFAULT (unchanged)                                       E8 NEW (optional)
        ┌───────────────────────┐                                  ┌────────────────────────────┐
        │ CascadedPID.compute(  │                                  │ MJPC tracker:              │
        │   p,q,v,ω, p_des,v_des)│                                  │  set_state(qpos,qvel,      │
        │  needs v_des & real v  │                                  │            mocap_pos=p_des)│
        │  → 4 motor thrusts     │                                  │  planner_step()            │
        └───────────────────────┘                                  │  get_action() → 4 thrusts  │
                  │                                                  └────────────────────────────┘
                  │                                                                 │
                  └───────────────────────────────┬───────────────────────────────┘
                                                   ▼
                                         data.ctrl[:4] = u  →  mujoco.mj_step
```

**Key:** velocity never leaves MuJoCo in the E8 path. MJPC reads `qvel` via `set_state`; the FM tensor doesn't carry it. The PID path is byte-for-byte the E7 behaviour.

### §2.1 Transition-dim comparison

| Mode | obs | action | transition | velocity in tensor? | controller |
|---|---|---|---|---|---|
| **E7 default** (`cond_mode='p_des'`) | `[p_des|p|v]` 9D | Δp_des 3D | **12D** | yes (V1/V2/V3) | cascaded PID |
| `cond_mode='real_p'` (existing) | `[p|v]` 6D | Δp 3D | 9D | yes (just v) | cascaded PID |
| **E8 new** (`cond_mode='pos_only'`) | `[p_des|p]` 6D | Δp_des 3D | **9D** | **no** | **MJPC** |

> The E8 "9D" the user asked for = `action(3) + obs6D[p_des,p]`. It is 9D not 6D *because UAV is xyz (3D)* where the arm-avoiding analog is xy (2D). The strict-DPCC `[action|state]` shape is preserved; only velocity is removed from `state`.

---

## §3 — Dataset: SLICE, do not recollect (the headline answer)

### §3.1 What the expert `.pkl` already stores

`uav_expert_data_collect/dataset_writer.py` L10-12 + L53-54:
```python
obs     : (T, 9)   [p_des(3), p(3), v(3)]      # everything we need is already here
actions : (T-1, 3) [Δp_des]                    # position-delta convention
targets : (T, 3)   absolute p_des              # debug only
```

### §3.2 The slicing precedent already in the loader

`flow_matcher_v3_uav/datasets/d4rl.py` L88-95 — there is **already** a `cond_mode` switch that column-slices the stored 9D obs at load time:
```python
obs = np.asarray(ep['obs'], dtype=np.float32)   # (T, 9) [p_des|p|v]
if cond_mode == 'real_p':
    p_real = obs[:, 3:6]                          # measured position
    actions = np.diff(p_real, axis=0)            # Δp
    obs = obs[:, 3:9]                             # (T, 6) [p|v]
else:  # 'p_des'
    actions = ep['actions']                       # Δp_des
```

The new mode is one more branch — a strict subset of the 12D, **no regeneration**:
```python
elif cond_mode == 'pos_only':
    # E8: drop velocity. obs = [p_des | p] (6D); action = Δp_des (already stored).
    actions = np.asarray(ep['actions'], dtype=np.float32)  # (T-1,3) Δp_des — unchanged
    obs = obs[:, 0:6]                                       # (T,6) [p_des|p], drop v (cols 6:9)
```

### §3.3 Why slicing is valid (not a shortcut)

- The recorded `(p_des, p, action)` are **controller-agnostic at the trajectory level**. The expert flew the path with PID, but the *waypoints* `p_des` and the *realized positions* `p` are facts about the trajectory, not the controller. MJPC will track the same `p_des` sequence.
- Velocity is not lost from the world — it still lives in MuJoCo `qvel`, which MJPC reads via `set_state`. It is only removed from the **learned tensor**.
- `observation_dim`/`action_dim` are auto-derived from data shape (`sequence.py` L52-53), so the 6D obs flows through training/eval with no hard-coded dim to chase.

**Verdict: zero dataset regeneration. No new expert collection. No `.pkl` schema change.** A single new `cond_mode` branch slices `[p_des|p]` from the existing files.

> ⚠ Caveat to verify on cluster: the normalizer (`SafeLimitsNormalizer`/`LimitsNormalizer`) is re-fit on the sliced 6D obs. The E7 doc flagged the `eps=1` constant-dimension widening issue for `action_z`; that lives in the *action* normalizer and is unchanged here. The obs slice only removes columns, so obs normalization is unaffected for the surviving columns.

---

## §3.5 — Do we need to retrain? (depends on which variant)

The dataset is **sliced, not recollected** (§3) — but slicing the data is separate from whether the FM **model** must be retrained. There are two variants of this Epoch, with different answers:

| Variant | FM tensor | Controller | Retrain FM? | Why |
|---|---|---|---|---|
| **V1 — controller-only swap** (de-risk first) | **keep 12D** `[action|p_des|p|v]` | MJPC | **NO** | Same FM checkpoint as E7. MJPC just consumes `p_des`; the velocity columns the FM still outputs (V3, already inert in control per E7 §7.3) are ignored. Pure eval-side change. |
| **V2 — strict 9D** (the clean version) | **9D** `[action|p_des|p]` | MJPC | **YES** | Input obs is 6D (was 9D) and the transition is 9D (was 12D). The network's input/output dims change, so a 12D checkpoint **cannot** be loaded — the model must be retrained on the sliced data. |

### §3.5.1 Order — DECISION: skip V1, go straight to V2

> **🟢 DECISION (user, 2026-06-28): SKIP V1, do V2 directly. This is OK / approved.**
> V1 is kept documented below for reference (do not delete), but we will **not** run it. The
> 12D-FM + MJPC controller-only swap is judged a waste of time — the end state is V2 anyway, and
> the 12D and 9D tensors are strictly different shapes (a guaranteed retrain at the end), so V1's
> "no retrain" saving doesn't carry forward. Go straight to the strict-9D retrain.

1. **V1 (no retrain) — DESCRIBED FOR REFERENCE, NOT EXECUTED:** point MJPC at the existing E7 12D checkpoint, feed it `p_des`, ignore the FM's velocity output. Would isolate the MJPC tracker risk (task build, real-time budget — §4.3/§4.4) without a retrain. **Skipped per decision above.**
2. **V2 (retrain) — THE PATH WE TAKE:** retrain the FM on the **sliced 9D** data (`cond_mode='pos_only'`) to get the clean strict-DPCC position planner. From-scratch FM train on the same (sliced) dataset — no new data collection, just a new training run. The MJPC tracker risks (§4.3/§4.4) are absorbed directly in V2.

### §3.5.2 Why V2 can't reuse the V1 checkpoint

The FM backbone's first layer width = `transition_dim` and its conditioning width = `observation_dim`. Dropping velocity changes both (12→9, 9→6). PyTorch `state_dict` shapes won't match → load fails. This is a genuine retrain, not a fine-tune. (Cost: one training run on the cluster; dataset prep is free — it's the §3.2 slice.)

### §3.5.3 Stop-and-go caveat ties in here

Per E7 design doc §6.7.4: V2 drops the explicit `v_des` feedforward, which risks stop-and-go unless the receding goal frequency is high enough or the MJPC velocity penalty is relaxed (§4.4). V1 keeps the 12D FM, so the `v_des` signal still *exists* in the FM output even if MJPC doesn't use it — another reason V1 is the safer first step. If continuous flight proves hard under V2's position-only goal, that's a signal to keep velocity (stay closer to V1) rather than force the strict-9D design.

---

## §4 — MJPC tracker integration (eval side)

### §4.1 The MJPC API we use

`mujoco_mpc/python/mujoco_mpc/agent.py`:
- `set_state(qpos, qvel, ..., mocap_pos=p_des, mocap_quat=...)` L183 — set drone state + **goal position** (`mocap_pos`).
- `planner_step()` L274 — run the sampling optimizer (32 rollouts × 50-step horizon, see E7 §4.2).
- `get_action()` L226 — return the latest 4 motor thrusts.
- `get_total_cost()` / `get_cost_term_values()` — diagnostics for the real-time logger.

### §4.2 Eval loop (E8 path), mirroring `eval_fm_uav.py` L340-390

```python
# build once per scene:
mjpc_agent = agent_lib.Agent(task_id='UAV Tracker', model=mjpc_model)   # see §4.3

for k in range(n_fm):                       # 33 Hz FM planner
    p = data.qpos[:3].copy()
    obs = np.concatenate([p_des, p]).astype(np.float32)   # 6D — NO velocity
    action, traj = policy({0: obs}, batch_size=..., horizon=horizon)
    action = action[:3]                       # Δp_des
    p_des = p_des + action                     # (or anchor_to_p: p = p + action)

    for _ in range(decim):                     # physics-rate MJPC tracker
        mjpc_agent.set_state(qpos=data.qpos, qvel=data.qvel, mocap_pos=p_des)
        mjpc_agent.planner_step()              # optimize thrusts (reads qvel internally)
        u = mjpc_agent.get_action()            # 4 motor thrusts
        data.ctrl[:4] = u
        mujoco.mj_step(model, data)
```

**Velocity flows MuJoCo → MJPC via `set_state(qvel=...)`, never through the FM.** This is the whole point.

### §4.3 The MJPC task — "UAV Tracker" variant (implementation detail / risk)

The stock `mujoco_mpc/mjpc/tasks/quadrotor/` task has built-in gate waypoints and `TransitionLocked` auto-advance (E7 §4.3). We do **not** want its waypoint logic — FM+DPCC owns the path. Two options:

- **Option A (preferred): minimal tracking task.** A stripped task XML using the same Skydio-X2 body but with a single position-tracking residual `||p − mocap_pos||²` (+ small `||v||²`, `||ω||²`, `||u − u_hover||²` regularizers, copied from `quadrotor.cc` L37-57). We overwrite `mocap_pos = p_des` every FM step. No gate geometry, no auto-advance.
- **Option B: reuse stock task, pin mode.** Use `set_task_parameters({'task_transition': 'Stage1'})` to freeze auto-advance and override `mocap_pos` each step. Faster to prototype but carries the stock cost weights.

**Obstacles:** MJPC stays a *pure tracker* — it only tracks the safe `p_des` the DPCC projector already produced. Obstacles remain handled upstream by the FM+DPCC QP, exactly as in E7. (Stretch goal: fold obstacle proximity into the MJPC cost for a second safety layer — out of scope for the first cut.)

### §4.4 Real-time budget risk

MJPC is ~50× heavier than PID per control step (E7 §6.4: 32 rollouts × 50 mj_steps = 1600 mj_step calls per planner update). At `decim ≈ 15` physics steps per FM query, calling `planner_step()` every physics step may blow the 33 Hz budget. Mitigations to test:
- Call `planner_step()` once per FM step (33 Hz) and only `get_action()` per physics step (the planner internally feedbacks the spline).
- Lower `sampling_trajectories` (32 → 8/16) and `agent_horizon` (0.5 s → 0.25 s).
- Record `total_ms`/`planner_ms` via the existing real-time recorder (`REALTIME_RECORDING`) to quantify feasibility — this is the primary success metric for "can it close the loop in budget".

---

## §5 — Config changes (all additive, all optional)

`config/uav.py` — new opt-in keys (defaults preserve E7 exactly):

```python
# E8 (optional) — strict-DPCC position-only planner + MJPC tracker.
'cond_mode':        'p_des',     # DEFAULT unchanged. Set 'pos_only' for E8 9D transition.
'controller':       'pid',       # DEFAULT 'pid' (E7). Set 'mjpc' for the optimal-control tracker.
'mjpc_task_id':     'UAV Tracker',
'mjpc_trajectories': 16,         # sampling fan (tune for budget)
'mjpc_horizon':      0.3,        # s (tune for budget)
'mjpc_planner_per': 'fm_step',   # 'fm_step' (33 Hz replan) | 'physics_step' (heavy)
```

- `cond_mode='pos_only'` triggers the §3.2 slice + 9D transition (training + eval must match, encoded in checkpoint path like the existing `cond` key handling).
- `controller='mjpc'` swaps the tracker in `eval_fm_uav.py` only. `'pid'` is untouched.
- Checkpoint path gets a `_cm{cond_mode}` (or reuse existing key) so `pos_only` checkpoints never collide with `p_des` ones.

---

## §6 — Files touched (planned; nothing dropped)

| File | Change | Additive? |
|---|---|---|
| `flow_matcher_v3_uav/datasets/d4rl.py` | add `cond_mode=='pos_only'` branch (1 slice) | ✅ new branch |
| `flow_matcher_v3_uav/datasets/sequence.py` | pass-through (already forwards `cond_mode`) | ✅ none/minor |
| `config/uav.py` | add E8 opt-in keys; defaults = E7 | ✅ additive |
| `FM_v3_uav_test/eval_fm_uav.py` | `if controller=='mjpc'` branch wrapping the tracker loop | ✅ new branch, PID untouched |
| `FM_v3_uav_test/train_fm_uav.py` | none (dims auto-derive from sliced data) | ✅ none |
| `mjpc/tasks/uav_tracker/` (new) | minimal position-tracking MJPC task (Option A) | ✅ new task |
| NEW `Epoch_8.../CHANGELOG.md` | written after implementation | ✅ new |

---

## §7 — Open questions to resolve before coding

1. **MJPC task build:** does the local `mujoco_mpc` have a Python-loadable Skydio-X2 model we can point a tracking task at, or must we compile a new task into the C++ agent server? (Affects Option A vs B in §4.3.)
2. **Replan cadence:** `planner_step()` at 33 Hz vs physics rate — measure both for budget (§4.4).
3. **Normalizer refit:** confirm 6D obs normalization is sane on the cluster (no constant-dim widening surprises — §3.3 caveat).
4. **Checkpoint path key:** reuse the existing `cond`/`cond_mode` path token or add `_cm{cond_mode}` — must not collide with E7 `p_des` checkpoints.

---

## §8 — Success criteria

**V1 (controller-only swap, NO retrain) — do first:**
- [ ] Eval with `controller='mjpc'` on the **existing E7 12D checkpoint** flies corridor/pillars, tracking FM `p_des` (velocity output ignored).
- [ ] Real-time recorder shows MJPC `total_ms` and whether it fits the 33 Hz budget (the gating risk — §4.4).
- [ ] E7 PID path verified unchanged (regression guard).

**V2 (strict 9D, retrain) — only after V1 validates MJPC:**
- [ ] FM retrains on `cond_mode='pos_only'` (9D transition) from the **sliced** existing dataset — no recollection.
- [ ] Continuous flight (not stop-and-go) confirmed under position-only goal (§3.5.3 / E7 §6.7.4) — via receding-goal frequency and/or relaxed MJPC velocity penalty.
- [ ] A/B: FM(12D)+PID (E7) vs FM(9D)+MJPC (E8) — success rate + tracking error.

---

## §9 — Why this is defensible for publication

This makes the E7 §2.5.4 planner/tracker axis literal and gives a clean ablation:
- **FM(12D) + PID** (E7): velocity in the learned tensor; linear-PD tracker.
- **FM(9D) + MJPC** (E8): strict-DPCC position planner; nonlinear optimal-control tracker that infers velocity from physics.

Same FM family, same DPCC projector, same MuJoCo plant — only the planner tensor and the tracker change. That isolates the contribution of "should the learned model carry velocity, or should the tracker recover it?" — the exact question E7 §6 raised.

---

*This plan implements the FM→MJPC direction proposed in [`DESIGN_control_chain_arm_vs_UAV.md`](../Epoch7_fm_pcc_FULL_PCC_MPC/U4_cond/DESIGN_control_chain_arm_vs_UAV.md) §6.2–§6.6.*
