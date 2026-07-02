# U6 — Rebuild MJPC Tracker: gRPC C++ Binary → MJX Python Predictive Sampling

**Date:** 2026-07-01  
**Scope:** `FM_v3_uav_test/mjpc_tracker.py`, `FM_v3_uav_test/eval_fm_uav.py`,
`config/uav.py`, `third_party/mujoco_mpc/mujoco_mpc/mjx/`  
**Principle:** translate/reuse — zero invented MPC math; DeepMind's `predictive_sampling.py` used verbatim.

---

## Problem (C++ gRPC approach, Fix_5 → ABORT)

Three-layer deployment failure on the Slurm cluster:

| Step | Failure |
|---|---|
| Import | `ModuleNotFoundError: mujoco_mpc` (job 22209) |
| Binary | `RuntimeError: agent_server not found` (job 22265) |
| Runtime | `si_addr=0x4` SIGSEGV in the C++ gRPC server (commit `4282d73` — ABORT) |

Six build jobs, `patchelf` RPATH rewriting, bundled conda runtime libs — SIGSEGV persisted.
Root cause: task XML / GL resource mismatch inside the C++ process on the eval node.
Undebuggable without shell access to the exact node.

**Decision:** the gRPC server architecture is not viable on this cluster. Switch to pure Python.

---

## Solution — MJX Predictive Sampling

DeepMind's own Python implementation of the predictive-sampling planner already exists in the repo:

```
/workspaces/mujoco_mpc/python/mujoco_mpc/mjx/predictive_sampling.py
```

It uses **MJX** (MuJoCo's JAX backend) — `jax.vmap` runs all N candidate rollouts in parallel on GPU. No gRPC, no subprocess, no binary.

The only dependency change: `pip install "jax[cuda12]" mujoco-mjx` on the cluster.

---

## Files Changed

### NEW — `third_party/mujoco_mpc/mujoco_mpc/mjx/__init__.py`
Empty. Makes `mujoco_mpc.mjx` importable from the bundled package.

### NEW — `third_party/mujoco_mpc/mujoco_mpc/mjx/predictive_sampling.py`
Copied verbatim from `/workspaces/mujoco_mpc/python/mujoco_mpc/mjx/predictive_sampling.py`.

**Only two changes** (principle: translate/reuse, not invent):
1. Removed `from brax import base as brax_base` — brax only used in `mpc_rollout`
2. Removed `mpc_rollout` function — it's for offline evaluation, not inference; uses brax

All inference symbols preserved unchanged: `Planner`, `_rollout`, `get_actions`,
`improve_policy`, `resample`, `set_state`.

### REWRITTEN — `FM_v3_uav_test/mjpc_tracker.py`
`MJPCTracker` class completely rewritten. External API **unchanged** —
`.compute(p, q, v, om, p_des, ...)` signature identical, so `rollout_one` in
`eval_fm_uav.py` requires zero changes.

Key logic:

```python
# UAV position-tracking cost — passed as cost_fn to Planner
def uav_pos_cost(mx_model, mx_data, p_des):
    pos = mx_data.qpos[:3]   # free-joint qpos[:3] = xyz
    return jnp.sum((pos - p_des) ** 2), ()

planner = ps.Planner(model=mjx.put_model(model), cost=uav_pos_cost,
                     noise_scale=0.3, horizon=horizon_steps,
                     nspline=horizon_steps, nsample=n_trajectories,
                     interp='zero', instruction_fn=dummy)
self._improve_jit = jax.jit(ps.improve_policy)   # JIT once, warm on first call
```

In `compute()`:
```python
mx_data = mjx.make_data(mx_model).replace(qpos=qpos, qvel=qvel)
self._policy, _ = self._improve_jit(planner, mx_data, p_des_j, self._policy, rng)
action = np.array(self._policy[0])
self._policy = ps.resample(planner, self._policy, 1)
return action[:4]
```

**Deprecated / removed from `__init__`:**
- `task_id` param → ignored (absorbed via `**_`)
- `planner_steps` param → gone (one `improve_policy` call = equivalent of N gRPC `planner_step()` calls)

### MODIFIED — `FM_v3_uav_test/eval_fm_uav.py`

`load_pcc_config()`: replaced 4 `mjpc_*` keys with 2 `mjx_*` keys:
```python
# before:
cfg['mjpc_task_id']      = str(getattr(plan_args, 'mjpc_task_id', 'Quadrotor'))
cfg['mjpc_trajectories'] = int(getattr(plan_args, 'mjpc_trajectories', 16))
cfg['mjpc_horizon']      = float(getattr(plan_args, 'mjpc_horizon', 0.3))
cfg['mjpc_planner_steps']= int(getattr(plan_args, 'mjpc_planner_steps', 10))

# after:
cfg['mjx_n_samples'] = int(getattr(plan_args, 'mjx_n_samples', 16))
cfg['mjx_horizon']   = float(getattr(plan_args, 'mjx_horizon', 0.3))
```

`_run_variant()`: updated `mjpc_kwargs` dict:
```python
# before:
mjpc_kwargs = {'task_id': ..., 'n_trajectories': ..., 'horizon': ..., 'planner_steps': ...}

# after:
mjpc_kwargs = {'n_trajectories': config.get('mjx_n_samples', 16),
               'horizon':        config.get('mjx_horizon', 0.3)}
```

### MODIFIED — `config/uav.py`

Added MJX knobs to `plan_flow_matching_v3_uav` block:
```python
'mjx_n_samples': 16,   # parallel candidate trajectories
'mjx_horizon':   0.3,  # planning window in seconds
```
Updated controller comment to reference MJX + install command.

### DELETED — `Slurm_Codes/sbatch/uav_fm/build_mjpc_agent_server.sh`
No binary to build. Six jobs of cmake/patchelf work replaced by one pip install.

---

## What Was NOT Changed

| Component | Reason |
|---|---|
| `rollout_one()` in `eval_fm_uav.py` | `tracker.compute()` API identical |
| `CascadedPID` | Unaffected |
| `third_party/mujoco_mpc/mujoco_mpc/agent.py` | Kept (proto stubs harmless; removal is low priority) |
| Training code, model weights, dataset | Unaffected |
| `controller='mjpc'` config key | Kept — now backed by MJX instead of gRPC |

---

## Cluster Install (One-Time)

```bash
conda activate FMPCC
pip install "jax[cuda12]" mujoco-mjx
# brax NOT required
```

Verify:
```bash
python3 -c "from mujoco import mjx; import jax; print('MJX OK, JAX', jax.__version__)"
```

---

## Runtime Notes

- **JIT warmup**: first `compute()` call compiles the JAX graph (~5–10 s). All subsequent calls: ~1–5 ms on GPU.
- **Tuning**: `mjx_n_samples=16`, `mjx_horizon=0.3` are starting points. Profile against the 33 Hz budget (~30 ms). Drop `mjx_n_samples` to 8 if over budget.
- **float64**: MJX defaults to float32; `jax[cuda12]` enables float64 via `jax.config.update("jax_enable_x64", True)` if needed (add to top of `mjpc_tracker.py`).

---

## Post-U6 Fixes

### JAX 0.5+ compatibility shim (job 22958)

**Error** (pillars eval, node i6-gpu-1):
```
AttributeError: module 'jax.extend.backend' has no attribute 'backends'
  in mujoco/mjx/_src/io.py → has_cuda_gpu_device() → backend.backends()
```

**Root cause**: `pip install "jax[cuda12]" mujoco-mjx` pulled JAX 0.5+ which removed
`jax.extend.backend.backends()`. mujoco-mjx's `has_cuda_gpu_device()` calls it to check
for CUDA, crashing at `mjx.put_model()` before any JAX computation starts.

**Fix** (`FM_v3_uav_test/mjpc_tracker.py`, inside `__init__` before `from mujoco import mjx`):
```python
import jax.extend.backend as _jax_eb
if not hasattr(_jax_eb, 'backends'):
    _jax_local = jax
    _jax_eb.backends = lambda: {d.platform for d in _jax_local.devices()}
```

`jax.devices()` is a stable long-term API; `.platform` returns `'cuda'`/`'cpu'`.
The set comprehension replicates the dict-key check `'cuda' in backends()` correctly.
Shim is a no-op on JAX 0.4.x where `backends()` already exists.

### MJX zero-policy init: drone free-falls on startup

**Error** (pillars eval, mjpc controller):
Drone falls from step 0 at ~8.4 m/s² — almost free-fall — despite correct PID gravity
compensation. `pid_stopgo` variant hovering correctly; only `mjpc` falls.

**Root cause**: `self._policy = jnp.zeros((horizon_steps, nu))` initializes all motor
thrusts to zero. `improve_policy` explores `policy ± noise_scale` (0.3 N/motor per call).
Hover requires ≈ 2.45 N/motor. Starting from zero, it takes ceil(2.45/0.3) ≈ 8 FM steps
to climb to hover — the drone is in free-fall the entire ramp-up phase.

**Fix** (`FM_v3_uav_test/mjpc_tracker.py`, after `Planner(...)` construction):
```python
g_mag = float(np.linalg.norm(model.opt.gravity)) or 9.81
body_id = model.body('x2').id
mass = float(model.body_subtreemass[body_id])
u_hover_init = mass * g_mag / float(nu) if nu > 0 else 0.0
ctrl_ceil = float(model.actuator_ctrlrange[:nu, 1].min()) if nu > 0 else 13.0
self._policy = jnp.full((horizon_steps, nu), float(min(u_hover_init, ctrl_ceil)))
```

Policy now starts at hover thrust so the planner's first rollout already stabilizes the
drone. `noise_scale=0.3` then explores ±12% of hover — correct operating region from step 0.

---

### MJX CYLINDER-BOX collision not implemented (job 22962)

**Error** (pillars eval, same node):
```
NotImplementedError: (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented.
  in mujoco/mjx/_src/io.py → put_model → _put_model_jax
```

**Root cause**: MJX's JAX backend only implements a subset of collision pair types.
The pillars scene UAV XML has cylindrical pillar geoms; MJX can't upload that model.

**Fix** (`FM_v3_uav_test/mjpc_tracker.py`, before `mjx.put_model`):
```python
_orig_contype     = model.geom_contype.copy()
_orig_conaffinity = model.geom_conaffinity.copy()
model.geom_contype[:]     = 0
model.geom_conaffinity[:] = 0
try:
    mx_model = mjx.put_model(model)
finally:
    model.geom_contype[:]     = _orig_contype
    model.geom_conaffinity[:] = _orig_conaffinity
```

Collisions are disabled only for the MJX model snapshot. The MJX planner needs UAV
dynamics (gravity + rotor thrust) only — obstacle avoidance is handled upstream by
the FM policy's DPCC projection step. The MuJoCo `model` object is restored in the
`finally` block so the actual rollout simulation remains unaffected.
