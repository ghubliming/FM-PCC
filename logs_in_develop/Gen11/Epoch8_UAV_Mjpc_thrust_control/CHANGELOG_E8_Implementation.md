# CHANGELOG — E8 MJPC Thrust Control (implementation)

**Date:** 2026-06-28
**Plan:** [`PLAN_MJPC_Thrust_Control.md`](PLAN_MJPC_Thrust_Control.md)
**Decision:** V1 skipped — V2 (strict-9D + MJPC) implemented directly (PLAN §3.5.1).

---

## What landed

The optional FM→MJPC path: a strict-DPCC **9D** position planner (`[action|p_des|p]`, velocity
dropped) + an **MJPC** optimal-control thrust tracker, added **beside** the E7 cascaded-PID / 12D
path. Defaults preserve E7 byte-for-byte. Selected entirely by config (`cond_mode`, `controller`).

All new code reuses existing patterns — the MJPC driving loop mirrors the repo's own
`mujoco_mpc/python/mujoco_mpc/demos/agent/cartpole.py`; the dataset slice mirrors the existing
`cond_mode='real_p'` branch; the tracker exposes the PID's `.compute()` signature so the eval loop
is controller-agnostic (no fork).

---

## Files touched

| File | Change | Lines |
|---|---|---|
| `flow_matcher_v3_uav/datasets/d4rl.py` | **NEW branch** `cond_mode=='pos_only'` → `obs=[p_des|p]` (slice cols 0:6, drop velocity); action `Δp_des` unchanged. Pure slice of the 9D pkl — **no data regeneration**. | +1 branch + docstring |
| `config/uav.py` | **NEW** `_uav_exp_name(args)` conditional exp_name; `cond_mode`+`controller` keys in BOTH train & plan blocks; both blocks switched `exp_name → _uav_exp_name`. MJPC knobs documented. | +helper, +keys |
| `FM_v3_uav_test/mjpc_tracker.py` | **NEW FILE** — `MJPCTracker` class. PID-compatible `.compute(p,q,v,om,p_des,…)→4 thrusts`. Mirrors cartpole.py's `set_state → N×planner_step → get_action`. Lazy mujoco_mpc import (cluster-only). | new (~110) |
| `FM_v3_uav_test/eval_fm_uav.py` | `load_pcc_config` reads E8 keys; `rollout_one` builds pid|mjpc tracker, obs width by `cond_mode`, calls `tracker.compute`, closes mjpc; `_run_variant` reads controller/cond_mode, segregates output via `_ctrl{controller}` eval tag, plumbs through. | ~6 edits |

No files removed. No existing behaviour changed when `cond_mode='p_des'` + `controller='pid'`.

---

## Key design points

### Path discrimination (backward-compatible)
`_uav_exp_name` appends a suffix **only when non-default**, so existing E7 checkpoints
(`H{horizon}_D{diffusion}`) are NOT orphaned:

| cond_mode | controller | savepath suffix |
|---|---|---|
| `p_des` | `pid` | *(none)* — identical to E7 |
| `pos_only` | `mjpc` | `_cmpos_only_ctrlmjpc` |
| `pos_only` | `pid` | `_cmpos_only` |

Verified standalone: default → `H8_Dmodels.diffusion.FlowMatchingODE` (unchanged); E8 →
`…_cmpos_only_ctrlmjpc`. **The controller type IS in the config/checkpoint path**, as requested.
Eval (`build_experiment`) loads from the TRAIN block's `exp_name`→`savepath`, so train/eval paths
always agree.

### Velocity leaves the FM tensor, not the world
`cond_mode='pos_only'` → FM obs is 6D `[p_des|p]` (transition 9D). The drone's real velocity still
enters MJPC via `set_state(qvel=…)` — MJPC recovers the velocity profile from physics internally.

### Controller-agnostic inner loop
`MJPCTracker.compute()` matches `CascadedPID.compute()` exactly, so the physics loop calls
`tracker.compute(p,q,v,om,p_des,v_des)` regardless. `v_des`/`a_des`/`yaw_des` accepted for API
parity but unused by MJPC.

### Reuse over rewrite (per request)
- MJPC loop = cartpole.py pattern (`set_state → 10× planner_step → get_action`), not invented.
- Dataset slice = `real_p` branch pattern.
- Output segregation = existing `eval_tag` mechanism (like `_anchorP`).

---

## Verification (Docker dev env — no torch/mujoco/mujoco_mpc)

- `py_compile` passes: `d4rl.py`, `config/uav.py`, `mjpc_tracker.py`, `eval_fm_uav.py`.
- `_uav_exp_name` logic validated standalone: backward-compat default + E8 suffixes correct.
- `pos_only` slice validated standalone: `(T,9)→(T,6)`, velocity dropped, actions unchanged.
- **NOT run end-to-end** — MJPC needs the compiled `mujoco_mpc` agent_server + GPU/mujoco
  (cluster-only). The `MJPCTracker` raises a clear error if `mujoco_mpc` is absent.

---

## How to run (user)

1. **Config:** in `config/uav.py`, set BOTH blocks (`flow_matching_v3_uav` + `plan_flow_matching_v3_uav`):
   ```python
   'cond_mode':  'pos_only',
   'controller': 'mjpc',
   ```
   (optionally tune `mjpc_trajectories`, `mjpc_horizon`, `mjpc_planner_steps`, `mjpc_task_id` in the plan block).
2. **Retrain** (required — 9D ≠ 12D shape): the trainer slices the existing dataset, no recollection.
3. **Eval** with the same config → results land in `…/plans/…/<variant>_ctrlmjpc/`.

### Open items carried from the PLAN (not code bugs)
- **MJPC task** (`mjpc_task_id`, default `'Quadrotor'`): the stock task has gate waypoints +
  auto-advance. For a pure tracker, point this at a minimal position-tracking task (PLAN §4.3).
- **Real-time budget** (PLAN §4.4): MJPC is heavy; tune `mjpc_planner_steps`/`mjpc_trajectories`.
- **Stop-and-go risk** (E7 design §6.7.4): keep the goal receding / down-weight the MJPC velocity
  penalty so position-only tracking doesn't brake at each waypoint.
