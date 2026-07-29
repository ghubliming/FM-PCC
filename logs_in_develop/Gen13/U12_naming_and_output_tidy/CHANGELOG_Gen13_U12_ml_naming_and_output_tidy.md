# Gen13 U12 — CHANGELOG: kill the "imf" naming for Mix-ML runs + tidy the output tree

**Why:** Gen13 is no longer "HardFlow + iMF". U11 reassembled it into **HF_Mix_ML** (imf | mf | af),
but the eval leg still *reused the frozen iMF eval scripts*, so an AlphaFlow run's results landed as:

```
logs/avoiding-v0/eval/H16_imf_hardflow_new_K2_from_H16_ml_af_100k_n200/128_real.png
                       ^^^ says "imf" for an AF run   ^^^^^^^^^^^^^^^^^ redundant _from_   ^^^ PNG flood
```

— the name lies about the objective, it's flat (every method×K×run in one folder), and the
**unconditional per-episode `*_real.png`** (thousands at n=200) filled the cluster disk and crashed
AF `hardflow_new K2` (`OSError: Errno 28`). U12 fixes all three, **additively**.

## The result — objective-named, one folder per run

```
logs/avoiding-v0/eval/
  H16_ml_mf_100k/            <- one folder per TRAINING run
    raw_K1_n200/  raw_K2_n200/          <- unguided
    hfproj_K1_n200/  hfproj_K2_n200/    <- HardFlow-projected  (+ trajectories.csv)
  H16_ml_af_100k/
    raw_K1_n200/ ... hfproj_K2_n200/
```

- `raw` = unguided (was `original`), `hfproj` = HardFlow-projected (was `hardflow_new`) — named after
  what they are, not "imf".
- Grouped **one parent folder per run**, so `mf` vs `af` vs a re-train never intermix.
- Enabled by a one-liner: the eval output dir is `…/eval/<exp_name>/` and `save_config` does
  `os.makedirs(exist_ok=True)`, so a **slashed** `exp_name` (`H16_ml_mf_100k/hfproj_K2_n200`) creates
  the nested tree for free — **no change to the frozen `eval_imf.py` path logic.**

## Files

### NEW — `HardFlow/run_scripts/eval_raw_ml.sh`, `eval_hfproj_ml.sh`
Clean siblings of the frozen `eval_original_imf.sh` / `eval_hardflow_new_imf.sh`. **Identical python
call and identical `--guidance_method original_imf` / `hardflow_new_imf`** (the code contract into
`eval_imf.py` is untouched — only the *output name* changed). They read `ML_EXP_NAME` (fallback
`IMF_EXP_NAME`), `ML_CP`/`ML_K` (fallback `IMF_CP`/`IMF_K`), and build the tidy nested `exp_name =
<run>/{raw,hfproj}_K<k>[_n<n>]`.

### NEW — `Slurm_Codes/sbatch/hardflow/eval_ml_hardflow.sh`
Clean sibling of `eval_imf_hardflow.sh`. Job name **`hf_ml_eval`** (was `hf_imf_eval`). Loops
`ML_METHODS` (default `"raw hfproj"`) × `ML_KS` calling `eval_${method}_ml.sh`; prints a tidy
per-run `find` tree at the end. **Exports `HF_EVAL_SAVE_PNG=0` by default** → no PNG flood.

### EDIT — `HardFlow/run/eval_imf.py`  (the only shared-file change; default-preserving)
The unconditional `save_single_trajectory_image(... f"{run_id}_real.png")` is now gated:
```python
if os.environ.get("HF_EVAL_SAVE_PNG", "1") != "0":
    ...save the *_real.png...
```
**Default `"1"` ⇒ every existing run, including all frozen iMF baselines, is byte-identical.** Only the
Mix-ML eval (which exports `HF_EVAL_SAVE_PNG=0`) skips the render. This is the same default-off pattern
already in the file (`imf_plot_fan`). Nothing else in `eval_imf.py` changed.

### EDIT — `Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh`  (U11 file, mine)
Eval leg repointed `eval_imf_hardflow.sh` → **`eval_ml_hardflow.sh`**. `EVAL_EXPORT` now forwards
`ML_EXP_NAME`/`ML_CP` (+ `ML_METHODS`/`ML_KS`/`HF_EVAL_SAVE_PNG`); legacy `IMF_METHODS`/`IMF_KS` still
accepted as fallbacks. Comments + "Results land in …" updated to the nested tree.

## What was deliberately NOT touched
- **Frozen iMF path byte-identical:** `eval_original_imf.sh`, `eval_hardflow_new_imf.sh`,
  `eval_imf_hardflow.sh`, `train_imf.py`, `train_imf.sh` — `git status` confirms unchanged.
  The iMF baseline still runs exactly as before with its old names.
- The training exp-name stays `H16_ml_<type>_<steps>k` — already readable, and the `ml_` prefix must
  stay to avoid colliding with the frozen `H16_imf_*` checkpoints.
- The HardFlow / MLbone **math** — untouched (U11).

## Validation (local, syntax-only — no pipeline run here)
- `bash -n` passes on all 3 new scripts + the edited pipeline.
- `python3 -m py_compile HardFlow/run/eval_imf.py` passes.
- Naming simulated: AF run → `…/eval/H16_ml_af_100k/{raw,hfproj}_K{1,2}_n200/` (see below).
- `git status` confirms only the 2 intended edits + 3 new files; every frozen iMF file untouched.

## How to run (unchanged entrypoint; cleaner output)
```bash
# full pipeline (train + tidy eval), one seed
ML_TYPE=mf N_TRAIN_STEPS=100000 ML_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh
ML_TYPE=af N_TRAIN_STEPS=100000 ML_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/ml_pipeline_hardflow.sh

# re-eval an existing checkpoint only (e.g. redo the disk-crashed AF K2 at n=200):
ML_EXP_NAME=H16_ml_af_100k ML_CP=4 ML_METHODS="hfproj" ML_KS="2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_ml_hardflow.sh

# want the renders back for a debug run:  add  HF_EVAL_SAVE_PNG=1
```

## Follow-ups (not in U12)
- Re-run AF `hfproj K2` to n=200 (U11 truncated at 129 — the very disk crash this closes).
- Re-eval cp `3` (step ~75k) for MF/AF (U11 §2: the u-head diverged after; cp `4` is past the minimum).
