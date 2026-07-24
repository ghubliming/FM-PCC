# Gen13 fix-2 — missing train+eval pipeline, and noisy console logs

**Date:** 2026-07-18
**Triggers:** (1) user noticed no combined iMF train→eval job existed, unlike the FM path's `hardflow_pipeline.sh`; (2) user flagged that eval console logs are unreadable (a known issue reproduced in an earlier real replication run, job 23565).

---

## Part A — missing iMF pipeline (now added)

### Gap
The FM (replication) track has `Slurm_Codes/sbatch/hardflow/hardflow_pipeline.sh` — one job doing train(if-needed) → fit_dynamics → eval. For iMF, coding_1 only produced the two halves separately: `train_imf_hardflow.sh` and `eval_imf_hardflow.sh`. **No combined pipeline existed.** Confirmed by listing the directory — user's suspicion was correct.

### Fix
Added **`Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh`**, mirroring `hardflow_pipeline.sh`'s structure exactly, plus the two iMF-specific steps:

```
0. gates (G0 + G1)         — abort BEFORE any GPU spend if they fail
1. train                   — skipped if checkpoint exists, or SKIP_TRAIN=1
2. fit_dynamics             — skipped if already present
3. eval E1-E4 matrix        — loops IMF_METHODS x IMF_KS (default: {original,hardflow_new} x {1,2})
```

One command now runs the whole Gen13 experiment:
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
```
Knobs: `N_TRAIN_STEPS`, `IMF_DATA_PROPORTION`, `IMF_P_STD` (train); `IMF_METHODS`, `IMF_KS`, `IMF_CP` (eval); `SKIP_TRAIN=1` (reuse an existing checkpoint). The separate `train_imf_hardflow.sh` / `eval_imf_hardflow.sh` remain useful for finer control (e.g. inspect the training curve before committing to eval).

New file only — additive, `bash -n` clean.

---

## Part B — console log noise (fixed for the iMF path)

### Root cause
`run/eval.py`'s `run_env()` calls `pbar.set_postfix(...)` **every single timestep** (up to 100 steps × 50 episodes). Under SLURM, stdout is redirected to a log **file**, not a real terminal — tqdm's carriage-return trick never collapses in place, so every update gets dumped as raw text. This produced the multi-thousand-character unreadable log lines seen in the real replication run (job 23565, `eval_hardflow` — visible in that log's `original` method output).

### Constraint
`run/eval.py` is **pre-existing** HardFlow code (predates Gen13, vendored during replication). Gen13's rule §0.1 forbids editing any existing HardFlow file — so the fix could not touch `run_env` in place.

### Fix (iMF path only)
Forked `run_env` → **`_run_env_quiet`**, added directly inside **`run/eval_imf.py`** (a Gen13-owned file — editing it is unrestricted). Functionally **identical** to the original (same env stepping, violation counting, image saving, return values — CSV numbers stay apples-to-apples with the FM baselines). Only the progress-reporting changed:
- **stdout is a real terminal** (`sys.stdout.isatty()` True): live tqdm bar, same as before.
- **stdout is redirected** (the normal SLURM case): **no tqdm** — one compact line printed at episode end, e.g.
  ```
  [ eval_imf ] episode 7: terminated  steps=23  violations=1  reward=0.000
  ```
  instead of up to 100 raw postfix dumps per episode.

`check_violation` (pure, no printing) is still imported unchanged from `run.eval`; only `run_env` needed forking. Call site in `evaluate_imf()` updated to use `_run_env_quiet`. Syntax-checked (`py_compile`); `git status` confirms only `run/eval_imf.py` (Gen13's own file) touched — **`run/eval.py` remains byte-identical**.

### Scope — what's NOT fixed
**The FM baseline path (`run/eval.py`, used by `eval_hardflow.sh`/`hardflow_pipeline.sh`) still has the noisy behavior.** It was intentionally left alone because it is pre-existing code protected by the Gen13 no-edit rule. This is a cosmetic-only issue (does not affect CSV/success/violation numbers), so it doesn't block anything — but if you want the FM replication logs cleaned up too, that requires an explicit one-time exception to the no-edit rule (a small, low-risk change: same `isatty()` guard applied to `run/eval.py`'s `run_env`). Flagging for a decision rather than doing it silently.

## Status
Both gaps closed for the iMF track. `git status HardFlow/` shows only Gen13-owned files touched (`run/eval_imf.py`); new sbatch file added. Ready to submit `imf_pipeline_hardflow.sh` for a full, quiet-logged train→eval run.
