# Changelog — HardFlow replication: SLURM entries + FMPCC-clone bridge

**Date:** 2026-07-15
**Author:** Claude (coding), per user request
**Spec:** `EVALUATION_hardflow_readiness_and_iMF_swap.md` §Part 3 (this folder)
**Scope:** cluster orchestration only. **No HardFlow source touched. No FMPCC pipeline code touched.**

---

## Hardcoded env block (update — mjx-style)

Following the `sbatch/uav_fm/eval_fm_uav.sh` (`FMPCC_mjx` clone) precedent, `_hardflow_common.sh` now **hardcodes** the env block instead of `${VAR:-default}` overrides — because the job may be submitted from an already-activated FMPCC shell, where inherited env vars would make dynamic defaults unreliable:
- `CONDA_ENV_NAME="hardflow_clone"` (hardcoded literal; the reconciled clone, never live FMPCC).
- `FMPCC_ROOT`/`REPO`/`CONDA_DIR`/`HARDFLOW_REPO`/`HARDFLOW_LOG_COLLECT` hardcoded (only `$HOME` stays dynamic).
- **`PYTHONPATH` reset FRESH** to `"$HARDFLOW_REPO"` (was appended) — prevents an inherited FMPCC `d3il` path from shadowing HardFlow's bundled `d3il`.
- Dropped the `CONDA_ENV_NAME==FMPCC` fail-closed guard (moot now the name is a hardcoded literal). The env sanity-check (`import tyro, gym`) stays.

## Vendored HardFlow into FM-PCC (update)

- **Copied** the HardFlow `d3il`/avoiding working tree into **`FM-PCC/HardFlow/`** (~62 MB, 426 files) — `hardflow/`, `run/`, `run_scripts/`, and the `d3il/` sim + 96 avoiding demos. **Excluded `.git`** (upstream history is ~469 MB) and all pycache/egg/pyc. HardFlow is now versioned *with* FM-PCC and reaches the cluster via `git pull` (no separate clone).
- **Edited `FM-PCC/HardFlow/.gitignore`**: removed the upstream `d3il/`, `datasets/`, `dataset/`, `experiment/` ignore lines so FM-PCC **tracks** the sim + data (upstream ignored them). Kept the rest (`*.pth`, `**/logs/`, `__pycache__`, `**_generated/`, …) so checkpoints/logs still never get committed. Verified: `git add --dry-run` → 426 files, 393 under `d3il/`, 96 demos, **zero** `.pth`/`logs`/`pyc`.
- **Changed the bridge default** `HARDFLOW_REPO` from `$FMPCC_ROOT/HardFlow` (sibling) to **`$REPO/HardFlow`** (inside FM-PCC), matching the vendored location.

## What was added

Four new files under `Slurm_Codes/sbatch/hardflow/` (new dir, sibling of `sbatch/iMF/`):

| File | Type | Purpose |
|---|---|---|
| `_hardflow_common.sh` | sourced bridge | activates the FMPCC **clone** env, sets PYTHONPATH, GPU/EGL guard, symlinks HardFlow `logs/` into `FM-PCC/logs/hardflow`, sanity-checks the env, `cd`s into the HardFlow repo |
| `train_hardflow.sh` | sbatch | trains the FM backbone via HardFlow's own `run_scripts/train.sh` |
| `eval_hardflow.sh` | sbatch | fits dynamics (guard) then evals each `METHODS` entry via HardFlow's own `run_scripts/eval_<method>.sh` |
| `hardflow_pipeline.sh` | sbatch | one-job chain: (train if no ckpt) → fit_dynamics → eval |

All four are `chmod +x` and pass `bash -n` syntax check. **None were executed** (cluster-only; no Python/GPU here).

---

## Design decisions realized in code

1. **Runs against a CLONE of FMPCC, never the live env.**
   `CONDA_ENV_NAME` is **hardcoded** to `hardflow_clone` (see "Hardcoded env block" update above — the earlier `==FMPCC` fail-closed guard was dropped as moot once the name is a literal). The env is assumed already built + reconciled by hand (user builds it manually) — the sbatch **never creates or mutates** any conda env.

2. **Env is decoupled — one variable.**
   Everything except `conda activate "$CONDA_ENV_NAME"` is env-agnostic. Build the env however you like; point `CONDA_ENV_NAME` at it.

3. **No auto-install at job time.**
   The bridge only *sanity-checks* (`python -c "import tyro, gym"`), prints the gym version, warns if gym is not `0.20.x`, and aborts if tyro/gym are missing. It does **not** pip-install (avoids races across concurrent jobs and masking env problems).

4. **Outputs land in FMPCC logs via a symlink, not `--log_folder`.**
   `HardFlow/logs` → `FM-PCC/logs/hardflow` (gitignored). Chosen because `eval.py` uses `cfg.log_folder` for most paths **but hardcodes `"logs"` for the dynamics model** (`eval.py:517`); with `cwd = HardFlow repo`, the symlink catches both. A `--log_folder` flag would silently miss the dynamics path.

5. **PYTHONPATH puts HardFlow first; no editable install.**
   So `import hardflow` and `import d3il` resolve to HardFlow's **own bundled** `d3il` (which registers `avoiding-v0`), not the clone's d3il. `pip install -e .` is deliberately avoided (it would persist HardFlow's `d3il` into the env and could shadow other jobs).

6. **GPU/EGL isolation guard preserved** (FMPCC standing rule): `MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=${CUDA_VISIBLE_DEVICES%%,*}`, aborts on EGL≠CUDA leak.

7. **`fit_dynamics` guard** in eval/pipeline: runs `run/fit_dynamics.py` once if `linear_model.npz` is absent, so `--dynamics_constraint` methods (e.g. `hardflow_new`) never silently run without the model.

8. **Science stays in HardFlow's scripts.** The sbatch bodies only orchestrate; they call `run_scripts/{train,eval_<method>}.sh` verbatim so the paper hyper-parameters (H16, `ode_t_steps=10`, `random_repeat=50`, `controller=rh`, `constraint=novel`, …) stay baked in and un-duplicated.

---

## Knobs (env vars; overridable at submit time)

| Var | Default | Meaning |
|---|---|---|
| `CONDA_ENV_NAME` | `hardflow_clone` | clone env to activate (must not be `FMPCC`) |
| `HARDFLOW_REPO` | `$HOME/FMPCC/HardFlow` | where HardFlow was git-pulled |
| `HARDFLOW_LOG_COLLECT` | `$REPO/logs/hardflow` | output collection dir inside FM-PCC |
| `METHODS` | `hardflow_new original` | eval methods (add `oc_flow gradient_guidance` freely; `hardflow`/`projection*` need l4casadi) |
| `SKIP_TRAIN` | `0` | pipeline: `1` to use a downloaded `.pth` |

---

## How to run (user does this on the cluster)

```
# 0. one-time (manual): build + reconcile the clone (see spec §3.05)
conda create --name hardflow_clone --clone FMPCC
conda activate hardflow_clone && pip install tyro "gym==0.20.0"

# 1. train (or download the .pth and skip)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_hardflow.sh
# 2. eval (smoke test first)
METHODS="original" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_hardflow.sh
# or the whole chain
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/hardflow_pipeline.sh
```

Results: `FM-PCC/logs/hardflow/avoiding-v0/eval/<exp>/trajectories.csv`; aggregate with HardFlow's `notebooks/collect_results.ipynb` off-cluster.

---

## NOT done (out of scope / user's side)

- Building/reconciling the conda clone (user does manually).
- Obtaining the FM checkpoint (train job or Google-Drive download).
- l4casadi CUDA build (only needed for `hardflow`/`projection*` methods).
- Any first-run validation on the cluster (no execution possible here).
- The iMF backbone swap (Part 2 of the spec — separate future work).

## Watch-items on first run

- gym actually downgraded to `0.20.0` in the clone and the avoiding rollout steps (highest risk).
- numpy left at `1.26` unless something demands `2.0` (spec §3.05.3).
- `import d3il` resolves to HardFlow's copy (print `d3il.__file__` if unsure).
- `HardFlow/logs` was not a pre-existing real dir (bridge aborts if so — move it).
