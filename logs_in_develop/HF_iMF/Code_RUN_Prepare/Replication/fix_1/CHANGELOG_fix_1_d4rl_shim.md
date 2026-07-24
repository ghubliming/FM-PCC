# Fix 1 — `d4rl` shim (first cluster run of the HardFlow bridge)

**Date:** 2026-07-16
**Trigger:** first real submission `eval_hardflow` (job 23486, node i6-gpu-1) — the bridge worked end-to-end (env activate, gym 0.20, GPU guard, log symlink, cwd), then `fit_dynamics.py` crashed on a missing dependency.
**Rule kept:** no edits to HardFlow source; fix is bridge-level only.

---

## Symptom

```
[ HF-EVAL ] dynamics model missing -> python run/fit_dynamics.py
  run/fit_dynamics.py:15  from hardflow.datasets.sequence import SequenceDataset
   hardflow/datasets/preprocessing.py:3  from .d4rl import load_environment
    hardflow/datasets/d4rl.py:5  import d4rl
ModuleNotFoundError: No module named 'd4rl'
```

## Root cause

- `hardflow_clone` is cloned from **FMPCC**, and **FMPCC has no `d4rl` package**. HardFlow's own `environment.yml` installs d4rl from git, but we build the env by cloning FMPCC (+ gym 0.20 + tyro), so HardFlow deps that FMPCC lacks surface at runtime. `d4rl` is the first.
- **It is a missing PACKAGE, not a missing file.** All HardFlow `.py` files loaded fine (they resolve from the repo via PYTHONPATH). The crash is a third-party `import d4rl`.
- `D4RL_SUPPRESS_IMPORT_ERROR=1` does not help: that flag is read *inside* d4rl's `__init__` to swallow its sub-import errors — it requires d4rl to be installed. Here the package is entirely absent, so the import fails first.

## Why a shim (not installing real d4rl)

`hardflow/datasets/d4rl.py` does `import d4rl` at the top but **never calls any `d4rl.<symbol>`** on the avoiding path:
- `load_environment()` → `gym.make("avoiding-v0")` (env registered by **d3il**, not d4rl)
- `sequence_dataset()` → `env.get_dataset()` (d3il `.pkl` data)
- grep for `d4rl.` across HardFlow → **zero** functional uses.

So the import is dead code inherited from diffuser. Installing real d4rl (mujoco_py / dm_control / old-gym build pain) is heavy and pointless. An empty shim satisfies the import with zero risk.

## Change

**New file:** `Slurm_Codes/sbatch/hardflow/shims/d4rl.py` — an empty module (docstring only) that exists solely to satisfy `import d4rl`.

**Edited:** `Slurm_Codes/sbatch/hardflow/_hardflow_common.sh` §5 PYTHONPATH:
```diff
- export PYTHONPATH="$HARDFLOW_REPO"
+ export PYTHONPATH="$HARDFLOW_REPO:$REPO/Slurm_Codes/sbatch/hardflow/shims"
```
HardFlow repo stays FIRST (so `hardflow`/`d3il` resolve to HardFlow's bundled copies); `shims/` is appended so it only supplies the otherwise-missing top-level `d4rl` and shadows nothing real.

**No HardFlow source touched.** Shim + PYTHONPATH only.

## Verified (locally, import-level)

```
PYTHONPATH=.../shims python -c "import d4rl; print(d4rl.__file__)"  ->  .../shims/d4rl.py
bash -n _hardflow_common.sh  ->  OK
```

## Next

- Re-`git pull` on the cluster (brings the shim + PYTHONPATH change) and resubmit `eval_hardflow` (`METHODS="original"` smoke test).
- **Expect possibly more clone-vs-HardFlow gaps** beyond d4rl — same pattern (a package FMPCC lacks). If another `ModuleNotFoundError` appears, decide per case: real install vs shim (shim only when the import is provably unused, as here).
- If a run ever accesses a real `d4rl.<attr>`, the shim raises `AttributeError` at that line — the signal to install the real package instead.

## Guardrail note

The escalation ladder for these: **(1) is the missing thing actually used?** grep first. **(2) if unused → shim** (this fix). **(3) if used → install the same thing FMPCC/DPCC uses**, copied, not invented.
