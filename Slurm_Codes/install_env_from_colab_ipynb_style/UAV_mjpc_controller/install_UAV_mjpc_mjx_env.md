# UAV `mjpc` Controller — Isolated `FMPCC_mjx` Conda Env Setup

This guide is for anyone installing FM-PCC **from scratch** who also wants to run the
UAV `controller='mjpc'` eval (MJX predictive-sampling tracker, Gen11 Epoch8 U6). Do
this **after** you've already completed the main `remote_setup_guide.md` and have a
working `FMPCC` conda env.

> [!CAUTION]
> **Do not `pip install jax[cuda12] mujoco-mjx` into your main `FMPCC` env.**
> `mujoco-mjx` requires `mujoco>=3.x`. The rest of the repo (d3il / avoiding-d3il,
> everything **except** the UAV `mjpc` controller) is built and validated against
> `mujoco==2.3.7` (see `requirements.txt`). Installing `mujoco-mjx` directly into
> `FMPCC` silently upgrades `mujoco` (and drags `numpy` up to `2.x` with it), which
> breaks d3il's XML scene parsing (`Schema violation: unrecognized attribute:
> 'collision'`) and segfaults `pinocchio` (compiled against NumPy 1.x's ABI). This
> already happened once — see
> `logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/U6_rebuild_mujoco_MPC/fix_1/`
> for the recovery writeup. Use a **separate cloned env** instead, as below.

---

## 1. Prerequisite

You have a working `FMPCC` conda env (per `remote_setup_guide.md`) and can already run
non-`mjpc` UAV eval jobs (`controller='pid'`, `'pid_stopgo'`, `'pid_const_v'`) or the
avoiding-d3il pipeline successfully.

Do **not** proceed until this is confirmed working:
```bash
conda activate FMPCC
python -c "import mujoco; print(mujoco.__version__)"   # expect 2.3.7
python -c "import numpy; print(numpy.__version__)"     # expect 1.26.4
```

---

## 2. Clone the Env (Fast — No Reinstall)

`conda create --clone` hardlinks the existing packages instead of re-downloading or
rebuilding anything, so this step is fast even though `FMPCC` has a lot of packages:

```bash
conda create -n FMPCC_mjx --clone FMPCC
conda activate FMPCC_mjx
```

`FMPCC` itself is untouched by anything you do inside `FMPCC_mjx` from this point on.

---

## 3. Install JAX + MJX (Only Into the Clone)

```bash
pip install "jax[cuda12]" mujoco-mjx
```

This upgrades `mujoco` to `3.x` inside `FMPCC_mjx` (`numpy` may or may not be touched
depending on what pip resolves — it stayed at `1.26.4` in our validated run, since
`jax`'s `numpy>=1.22` bound was already satisfied). Either way this only happens inside
`FMPCC_mjx`, not `FMPCC`.

You will likely see a pip resolver warning like:
```
ERROR: pip's dependency resolver does not currently take into account all the packages
that are installed. ... gymnasium-robotics 1.2.4 requires mujoco<3.0,>=2.3.3, but you
have mujoco 3.10.0 which is incompatible.
```
**This is harmless — ignore it.** `gymnasium_robotics` is leftover baggage from cloning
`FMPCC`; nothing in the UAV/`mjpc` code path (`FM_v3_uav_test/`, `flow_matcher_v3_uav/`)
imports it, so the version conflict never actually triggers.

---

## 4. Verify

> [!CAUTION]
> **Do not run the `import mujoco` check on the login node.** MuJoCo 3.x's
> `mujoco/__init__.py` unconditionally sets up a renderer at import time. With
> `MUJOCO_GL` unset (as it is on an interactive login shell — only the sbatch script
> exports it) it falls back to software OSMesa rendering, which needs a native
> `libOSMesa` binding that login nodes typically don't have. You'll get:
> ```
> AttributeError: 'NoneType' object has no attribute 'glGetError'
> ```
> This looks like a broken install but isn't — it's a rendering-backend probe failing
> on a machine with no GPU/EGL context, nothing to do with whether `mujoco-mjx`
> installed correctly. Verify inside a real GPU allocation instead:

```bash
srun --partition=gpu-1-student --gres=gpu:1 --pty bash
conda activate FMPCC_mjx
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python -c "import mujoco; print('mujoco', mujoco.__version__)"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python -c "from mujoco import mjx; print('mjx OK')"
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 python -c "import jax; print('jax', jax.__version__, jax.devices())"
exit   # release the interactive allocation when done
```

`jax.devices()` should list a `cuda` device. If this all passes here, the real
`mjpc` eval job (which sets these same env vars itself — see `eval_fm_uav.sh`) will
work too; that job is ultimately the real test.

---

## 5. Known Compatibility Shims (Already Patched in the Repo)

If your `jax`/`mujoco-mjx` pip resolution pulls slightly different versions than what
was validated, you may hit these — both are already fixed in
`FM_v3_uav_test/mjpc_tracker.py`, so you shouldn't need to do anything, but they're
documented here in case a future `pip install` drifts again:

- **JAX 0.5+**: `AttributeError: module 'jax.extend.backend' has no attribute
  'backends'` — `mujoco-mjx`'s CUDA-detection call was removed upstream; shimmed via
  `jax.devices()`.
- **MJX CYLINDER-BOX collisions**: `NotImplementedError:
  (mjtGeom.mjGEOM_CYLINDER, mjtGeom.mjGEOM_BOX) collisions not implemented` — MJX only
  implements a subset of collision pair types. The pillars scene's cylindrical
  obstacles trigger this; collisions are disabled on the MJX model snapshot only (the
  real MuJoCo rollout still has full collision) since the MJX planner only needs UAV
  dynamics, not obstacle collision — avoidance is handled by the FM policy's
  projection step.

See `logs_in_develop/Gen11/Epoch8_UAV_Mjpc_thrust_control/U6_rebuild_mujoco_MPC/CHANGELOG_U6_mjx_tracker.md`
for the full history and reasoning.

---

## 6. Running an `mjpc` Eval Job — Fully Automatic Env Selection

You do **not** need to remember which env to activate. `config/uav.py`'s
`plan_flow_matching_v3_uav['controller']` value is what actually selects the tracker,
and `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` reads that value directly and activates
the matching conda env **before running anything**:

```python
# config/uav.py — plan_flow_matching_v3_uav block
'controller': 'mjpc',   # <- set this
```

```bash
sbatch Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh pillars 6
```

The job log will show which env it picked:
```
[ env-select ] config/uav.py controller='mjpc' -> conda env 'FMPCC_mjx'
```

Switch `controller` back to `'pid_stopgo'` (or any non-`mjpc` value) and the next job
submission automatically goes back to the plain `FMPCC` env — no other change needed.

---

## 7. Sanity Checklist

| Check | Env | Expected |
|---|---|---|
| `import mujoco; mujoco.__version__` | `FMPCC` | `2.3.7` |
| `import mujoco; mujoco.__version__` | `FMPCC_mjx` | `3.x` |
| `import numpy; numpy.__version__` | `FMPCC` | `1.26.4` |
| `from mujoco import mjx` | `FMPCC` | `ImportError` (expected — not installed here) |
| `from mujoco import mjx` | `FMPCC_mjx` | succeeds |
| avoiding-d3il eval job | `FMPCC` | runs normally |
| UAV `controller='pid_stopgo'` eval | `FMPCC` (auto-picked) | runs normally |
| UAV `controller='mjpc'` eval | `FMPCC_mjx` (auto-picked) | runs, ~5–10s JIT warmup on first `compute()` call |

If any row disagrees with "Expected", something drifted — re-read the CAUTION box at
the top before installing anything else into either env.
