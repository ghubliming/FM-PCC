# Gen13 U12.2 — CHANGELOG: family-first folder layout (imf/ mf/ af/)

Follow-up to U12 (which fixed the `H16_imf_*`-named, PNG-flooded output). U12 still put every
Mix-ML run flat under `flow/` and `eval/` — `mf` and `af` runs sat side by side with no grouping.
This pass makes **the model family the first folder level**, for both training checkpoints and eval
results, additively.

## Before (U12) → After (U12.2)

```
# training checkpoints
logs/avoiding-v0/flow/H16_ml_mf_100k/model_ema_*.pth        ->  logs/avoiding-v0/flow/mf/H16_ml_mf_100k/model_ema_*.pth
logs/avoiding-v0/flow/H16_ml_af_100k/model_ema_*.pth        ->  logs/avoiding-v0/flow/af/H16_ml_af_100k/model_ema_*.pth

# eval results
logs/avoiding-v0/eval/H16_ml_mf_100k/hfproj_K2_n200/        ->  logs/avoiding-v0/eval/mf/H16_ml_mf_100k/hfproj_K2_n200/
logs/avoiding-v0/eval/H16_ml_af_100k/raw_K1_n200/           ->  logs/avoiding-v0/eval/af/H16_ml_af_100k/raw_K1_n200/
```

So `ls logs/avoiding-v0/flow/` now shows exactly three folders — `imf/ mf/ af/` — each holding that
family's runs, instead of a flat pile of `H16_ml_*` names you have to read to tell apart.

## How — one variable, zero new logic

Both `run/train_ml.py` and `run/eval_imf.py` build their output dir as
`os.path.join(log_folder, env, "flow"|"eval", cfg.exp_name)`, and `save_config()`
(`run/utils.py`) does `os.makedirs(exp_dir, exist_ok=True)`. **`exp_name` is the only free
component** — same trick U12 used to nest `raw_K1_n200` under a run; this time the *default*
`exp_name` itself gets a `<ml_type>/` prefix, so the family folder falls out for free with
**no change to any frozen path-building code**.

## Files changed (all in the additive Mix-ML path; frozen iMF scripts untouched)

- **`HardFlow/run_scripts/train_ml.sh`** — default `exp_name` computation changed from
  `H16_ml_${ml_type}_${steps_tag}` to `${ml_type}/H16_ml_${ml_type}_${steps_tag}`. An explicit
  `ML_EXP_NAME` override is still used **verbatim** (no auto-prefix — caller owns the full path).
- **`Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh`** — the orchestrator's own `EXP_NAME`
  (forwarded to both train and eval as `ML_EXP_NAME`) gets the same `${ML_TYPE}/` prefix, so train
  and eval automatically land in the same family folder without either side re-deriving anything.
  Header/usage comments and the final "Results land in…" line updated.
- **`Slurm_Codes/sbatch/hardflow/train_ml_hardflow.sh`** — its independent `_exp` derivation (used
  only for the end-of-job `ls` listing) synced to match.
- **`HardFlow/run_scripts/eval_raw_ml.sh`, `eval_hfproj_ml.sh`** — no logic change; the
  family-first nesting arrives automatically because `exp_name = flow_exp_name/<method>_K<k>...`
  and `flow_exp_name` (= `ML_EXP_NAME`) already carries the `<ml_type>/` prefix. Comment + cosmetic
  default (`imf/H16_ml_imf_100k`) updated to match.
- **`Slurm_Codes/sbatch/hardflow/eval_ml_hardflow.sh`** — header comment + cosmetic default updated
  to the family-first form; the closing `find` summary already handles nested paths unchanged.

## Backward compatibility with the already-run U11/U12 checkpoints

Your two existing checkpoints (`H16_ml_mf_100k`, `H16_ml_af_100k`) live at the **old flat path** —
they were trained before this change. They are **not moved or broken**: pass `ML_EXP_NAME` explicitly
(exactly as before) and both train-side loading and eval-side output resolve to the old flat location,
since an explicit override is never auto-prefixed:
```bash
# still works, unchanged — loads the OLD flat checkpoint, writes eval un-nested:
ML_EXP_NAME=H16_ml_af_100k ML_CP=4 ML_METHODS="hfproj" ML_KS="2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_ml_hardflow.sh
```
**Optional migration** (cluster-side, your call, not run by me): to bring the two existing runs into
the new tree, `mv` them into their family folder before evaluating with the new default name:
```bash
mkdir -p logs/avoiding-v0/flow/mf logs/avoiding-v0/flow/af
mv logs/avoiding-v0/flow/H16_ml_mf_100k logs/avoiding-v0/flow/mf/
mv logs/avoiding-v0/flow/H16_ml_af_100k logs/avoiding-v0/flow/af/
# then the default (no ML_EXP_NAME override) resolves correctly:
ML_TYPE=mf N_TRAIN_STEPS=100000 SKIP_TRAIN=1 ML_CP=4 ML_METHODS="hfproj" ML_KS="2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh
```
Any already-collected eval `.npz`/`.csv` results (the completed U11/U12 dirs) can be moved the same
way into `eval/mf/…` / `eval/af/…`, or just left where they are — nothing reads the old flat eval
dirs, so leaving them is harmless.

## What was deliberately NOT touched
- Frozen iMF baseline: `train_imf.py`, `train_imf.sh`, `train_imf_hardflow.sh`,
  `eval_original_imf.sh`, `eval_hardflow_new_imf.sh`, `eval_imf_hardflow.sh`,
  `imf_pipeline_hardflow.sh` — `git status --porcelain` confirms zero diff. `H16_imf_*` runs keep
  their existing flat layout exactly as before.
- The HardFlow / MLbone math, the U12 PNG-flood guard (`HF_EVAL_SAVE_PNG`), the U12 `raw`/`hfproj`
  naming — all unchanged, just relocated one level deeper.
- No files were moved on disk by this change (this is a code-only, code-container session — no
  cluster filesystem access here). The migration commands above are for you to run if wanted.

## Validation (local, syntax-only)
- `bash -n` passes on all 6 touched scripts.
- Simulated the actual variable-resolution logic (not just read the code) for three cases:
  1. **Fresh run, no override** → `flow/mf/H16_ml_mf_100k/model_ema_4.pth` +
     `eval/mf/H16_ml_mf_100k/{raw,hfproj}_K{1,2}_n200/` — family-first confirmed.
  2. **Re-eval an old flat checkpoint** (`ML_EXP_NAME=H16_ml_af_100k`) → still loads
     `flow/H16_ml_af_100k/model_ema_4.pth` and writes `eval/H16_ml_af_100k/hfproj_K2_n200/` —
     backward compatibility confirmed.
  3. **Clobber guard on the new nested path** → correctly detects an existing
     `flow/mf/H16_ml_mf_100k/model_ema_4.pth` and aborts as designed.
- `git status --porcelain` on the frozen-iMF file list: empty (untouched).

## How to run (unchanged entrypoint)
```bash
ML_TYPE=mf N_TRAIN_STEPS=100000 ML_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh
ML_TYPE=af N_TRAIN_STEPS=100000 ML_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh
# both land under logs/avoiding-v0/{flow,eval}/<mf|af>/H16_ml_<type>_100k/...
```
