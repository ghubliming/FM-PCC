# MEMO — Drop C++ `agent_server`, Use MJX Python Solver

**Date:** 2026-07-01  
**Context:** Gen11 E8 UAV MJPC thrust controller  
**Status:** PLAN (not implemented)

---

## Correction from Earlier Plan

The Python MPC solver **already exists in the repo** — no C++ translation required.

`/workspaces/mujoco_mpc/python/mujoco_mpc/mjx/predictive_sampling.py`

This is DeepMind's own Python implementation of the predictive-sampling planner used inside MJPC.
It uses **MJX** (MuJoCo's JAX backend): vectorised rollouts via `jax.vmap` over GPU.
No gRPC. No subprocess. No binary.

**But** the three required packages are not currently installed:

```
jax:  NOT installed
brax: NOT installed
mujoco.mjx (mujoco-mjx): NOT installed
```

Verified by:
```bash
python3 -c "import jax"      # ModuleNotFoundError
python3 -c "import brax"     # ModuleNotFoundError
python3 -c "from mujoco import mjx"  # ModuleNotFoundError
```

The plain `mujoco` pip package (already installed) is the **physics engine only** — it has no MPC solver built in.

---

## Why the C++ `agent_server` Approach Failed

| Layer | Failure | Evidence |
|---|---|---|
| Python import | `ModuleNotFoundError: mujoco_mpc` | job 22209 |
| Binary absent | `RuntimeError: agent_server not found` | job 22265 |
| Binary crashes | `si_addr=0x4` (SIGSEGV) | commit `4282d73` — ABORT |

Six build jobs, RPATH patching with `patchelf`, bundling all conda runtime libs — none resolved the SIGSEGV. The crash is inside the C++ gRPC server (likely task XML / GL resource mismatch on the eval node). Undebuggable without shell access to the exact node.

---

## The Python Path: MJX Predictive Sampling

### What the existing solver provides

`predictive_sampling.py` exposes:

| Symbol | Role |
|---|---|
| `Planner` (dataclass) | Holds model, cost_fn, noise_scale, horizon, nsample |
| `improve_policy(p, data, instruction, policy, rng)` | One MPC step: samples N noisy action sequences, vmap-rolls them through MJX physics, returns best |
| `resample(p, policy, steps_per_plan)` | Shift policy window for receding-horizon |
| `mpc_rollout(...)` | Full closed-loop rollout (for offline evaluation) |

`improve_policy` is the hot path:
- Samples `nsample` perturbed action sequences
- Runs `jax.vmap(_rollout, ...)` — **parallel GPU rollouts**, no Python loop
- Returns `argmin(costs)` — best trajectory's first action

### What we need to provide

A **cost function** `cost_fn(model, data, instruction) -> (scalar, aux)` that scores a state.
For position tracking:

```python
def uav_position_cost(model, data, p_des):
    pos = data.qpos[:3]
    return jnp.sum((pos - p_des) ** 2), ()
```

And `instruction_fn(model, data) -> (instruction, userdata)` that returns `p_des` from the
live rollout data (we pass it in via `mjx.Data.userdata` or just close over it).

---

## Two-Step Install Plan

### Step 1 — Install on the cluster (one-time, inside FMPCC env)

```bash
pip install "jax[cuda12]" mujoco-mjx brax
```

`mujoco-mjx` ships `mujoco.mjx`; it does NOT conflict with the existing `mujoco` install.
JAX GPU backend matches the A5000s on the eval nodes (CUDA 12, driver 530).

### Step 2 — Copy solver into repo

```bash
cp -r /workspaces/mujoco_mpc/python/mujoco_mpc/mjx \
       /workspaces/FM-PCC/third_party/mujoco_mpc/mujoco_mpc/
```

No code change to the solver itself — just make it importable from our bundled package.

---

## What to Change / Delete / Add

### DELETE

| Path | Reason |
|---|---|
| `Slurm_Codes/sbatch/uav_fm/build_mjpc_agent_server.sh` | No binary to build anymore |
| `third_party/mujoco_mpc/mujoco_mpc/mjpc/agent_server` (binary, if present) | Not used |
| `third_party/mujoco_mpc/mujoco_mpc/proto/` | gRPC stubs, no longer called |
| `third_party/mujoco_mpc/mujoco_mpc/agent.py` | gRPC client, replaced by MJX planner |

### ADD

| Path | What |
|---|---|
| `third_party/mujoco_mpc/mujoco_mpc/mjx/` | Copy from `/workspaces/mujoco_mpc/python/mujoco_mpc/mjx/` |
| `FM_v3_uav_test/mjpc_tracker.py` (rewrite) | `MJPCTracker` wraps `predictive_sampling.Planner` + `improve_policy`. Same `.compute(p, q, v, om, p_des)` signature. Defines the UAV position-tracking cost fn. |

### CHANGE

| File | Change |
|---|---|
| `FM_v3_uav_test/mjpc_tracker.py` | Full rewrite (see below) |
| `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` | Remove `third_party/mujoco_mpc` from PYTHONPATH (keep `third_party/mujoco_mpc` path if mjx is bundled there) |
| `config/uav.py` plan block | `mjpc_planner_steps` → gone (no longer meaningful); keep `mjpc_trajectories`, `mjpc_horizon` |

### `mjpc_tracker.py` rewrite (sketch)

```python
import numpy as np
import mujoco
from mujoco import mjx
import jax, jax.numpy as jnp
from mujoco_mpc.mjx import predictive_sampling as ps

class MJPCTracker:
    def __init__(self, model, scene='', task_id='Quadrotor',
                 n_trajectories=16, horizon=0.3, planner_steps=10):
        # horizon_steps: convert seconds → MuJoCo steps
        dt = model.opt.timestep
        horizon_steps = max(1, int(horizon / dt))

        def cost_fn(mx_model, mx_data, p_des):
            pos = mx_data.qpos[:3]
            return jnp.sum((pos - p_des) ** 2), ()

        def instruction_fn(mx_model, mx_data):
            return mx_data.userdata[:3], mx_data.userdata  # p_des stored in userdata

        mx_model = mjx.put_model(model)
        self.mx_model = mx_model
        self.planner = ps.Planner(
            model=mx_model, cost=cost_fn,
            noise_scale=0.5, horizon=horizon_steps,
            nspline=horizon_steps, nsample=n_trajectories,
            interp='zero', instruction_fn=instruction_fn,
        )
        self.policy = jnp.zeros((horizon_steps, model.nu))
        self.rng = jax.random.PRNGKey(0)
        self._improve = jax.jit(ps.improve_policy)  # JIT-compile once

    def compute(self, p, q, v, om, p_des, **_):
        # Build MJX data from current state
        mx_data = mjx.make_data(self.mx_model)
        qpos = jnp.array([*p, *q], float)
        qvel = jnp.array([*v, *om], float)
        p_des_j = jnp.array(p_des, float)
        mx_data = mx_data.replace(qpos=qpos, qvel=qvel,
                                   userdata=jnp.pad(p_des_j, (0, mx_data.userdata.shape[0]-3)))
        self.rng, rng = jax.random.split(self.rng)
        self.policy, _ = self._improve(self.planner, mx_data, p_des_j, self.policy, rng)
        action = np.array(self.policy[0])
        self.policy = ps.resample(self.planner, self.policy, 1)
        return action[:4]

    def close(self): pass
```

---

## Risk / Notes

- **First `_improve` call is slow** (JIT compile, ~5–10 s). Subsequent calls: GPU-parallel, ~1–5 ms.
- **userdata size**: MuJoCo model must have `userdata` of at least 3 floats in the XML, or pass `p_des` as a JAX closure instead.
- **brax dependency**: `mpc_rollout` imports `brax.base.State` but `improve_policy` (our hot path) does not touch brax. If brax install fails, we can strip the import from `predictive_sampling.py` (the `mpc_rollout` function is for offline evaluation, not needed at inference).
- **CUDA 12 vs 11**: cluster nodes have driver 530 → CUDA 12. `jax[cuda12]` is the correct variant.

---

## Migration Checklist

- [ ] `pip install "jax[cuda12]" mujoco-mjx` on cluster (brax optional — only needed for `mpc_rollout`)
- [ ] Copy `mjx/` folder into `third_party/mujoco_mpc/mujoco_mpc/mjx/`
- [ ] Rewrite `FM_v3_uav_test/mjpc_tracker.py` using MJX planner sketch above
- [ ] Add `nspline` / `horizon_steps` params to `config/uav.py` + `uav_projection.yaml`
- [ ] Delete gRPC files and build sbatch
- [ ] Test: first call takes ~10 s JIT warmup, then profile `compute()` for 33 Hz budget
