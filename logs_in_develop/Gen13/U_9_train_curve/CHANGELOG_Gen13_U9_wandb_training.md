# Gen13 U9 — W&B logging for BOTH trainings (iMF + FM), copied from FMPCC Gen3v4

**Date:** 2026-07-20 · **Two fixes in one changelog**, as requested.
**Track back to the code:** `grep -rn "U9" HardFlow/run/train_imf.py HardFlow/run/train_fm.py Slurm_Codes/sbatch/hardflow/_hardflow_common.sh`
**Context:** `WHERE_ARE_THE_TRAINING_CURVES.md` (same folder) established: no W&B anywhere, iMF logged to CSV only, and **FM was never trainable** in `hardflow_clone`.

---

## 1. What was copied from FMPCC Gen3v4 (and why it transfers cleanly)

Source: `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py` + `Slurm_Codes/sbatch/iMF/train_imf.sh`.

| Gen3v4 element | Reused? | Note |
|---|---|---|
| `sanitize_wandb_env()` | ✅ **copied verbatim** (logic) | clears malformed `WANDB_SERVICE` tokens that make `wandb.init()` hang on this cluster — Gen3v4 hit exactly this |
| `wandb.init(project/entity/group/name/config, reinit=True)` in **try/except** | ✅ copied | logging must never kill a multi-hour training job |
| sbatch key-file login: `$HOME/FMPCC/.wandb_api_key` → `WANDB_API_KEY` + `WANDB_MODE=online` | ✅ copied | identical convention, identical path |
| `--use-wandb` / `--wandb-project` CLI flags | ✅ adapted to tyro (`--use_wandb`, `--wandb_project`) | HardFlow uses tyro, not argparse |
| `log_wandb_from_losses()` — re-reading `losses.pkl` each epoch | ❌ **not needed** | Gen3v4 needed it because its trainer was opaque; our loops hold `infos` directly, so we `wandb.log(...)` inline — simpler and live |
| `upload_wandb_artifact()` (checkpoint upload) | ❌ skipped | 14 MB × 5 per run; the cluster keeps them. Easy to add later if wanted |

**Environment:** `wandb==0.17.5` is already in the FMPCC requirements, and `hardflow_clone` is a clone of FMPCC ⇒ **wandb is already installed. No pip install needed.**

---

## 2. FIX 1 — iMF training (Gen13) now logs to W&B

| File | Change |
|---|---|
| `imf/imf_config.py` | `use_wandb: bool = False`, `wandb_project = "FMPCC-HF-iMF"`, `wandb_entity`, `wandb_group` |
| `run/train_imf.py` | `sanitize_wandb_env()` + `init_wandb()`; logs **all six** metrics (`loss`, `raw_mse_u`, `raw_mse_v`, `a0_mse`, `fm_frac`, `h_mean`) every `log_freq`; `wandb_run.finish()` at the end |
| `run_scripts/train_imf.sh` | `USE_WANDB` (default **1**) → passes `--use_wandb`; `WANDB_PROJECT` knob |

Logging is **layered and independently fail-safe**: CSV always; TensorBoard if installed; W&B if a key exists. A W&B failure prints a warning, sets `wandb_run = None`, and training continues.

## 3. FIX 2 — FM training is now RUNNABLE (and logged)

This is the more consequential fix. `run/train.py` is pre-existing and frozen, and it has **two** problems:

1. `from torch.utils.tensorboard import SummaryWriter` at **module level, un-guarded** ⇒ in `hardflow_clone` (no tensorboard) FM training **crashes after ~4 s**. This is what killed pipeline job 23559 and forced the replication onto the downloaded checkpoint.
2. It logs **one scalar** to TensorBoard only — no CSV, no W&B.

**Fix: `run/train_fm.py`, an additive sibling.** Training math is **identical** — same `TemporalUnet`, same `FlowMatcher('cfm')`, same optimiser / `CosineAnnealingLR(T_max=save_freq*2)` / EMA 0.995 / checkpoint cadence (the block is marked `---- identical to run/train.py ----`). Only logging differs, so a checkpoint from it faithfully reproduces the original.

| File | Type | Change |
|---|---|---|
| `run/train_fm.py` | 🆕 | try-import tensorboard, `metrics.csv`, W&B, tty-gated tqdm, `FmTrainingConfig` |
| `run_scripts/train_fm_wandb.sh` | 🆕 | same params as `train.sh` (H16, 1e6, batch 32, lr 2e-4, ema .995, save_freq 50000) |
| `Slurm_Codes/sbatch/hardflow/train_fm_hardflow.sh` | 🆕 | the job |

**`run/train.py` and `run_scripts/train.sh` remain untouched** (verified).

## 4. Shared — W&B login in the bridge

`_hardflow_common.sh` §6b now reads `$FMPCC_ROOT/.wandb_api_key` and exports `WANDB_API_KEY` + `WANDB_MODE=online`, exactly as `sbatch/iMF/train_imf.sh` does. If the key is absent it prints a clear message and everything falls back to CSV — **W&B never blocks a job**. Because it lives in the shared bridge, *every* hardflow sbatch inherits it.

## 5. ⭐ COMMANDS

**Prerequisite (once):** ensure `$HOME/FMPCC/.wandb_api_key` exists on the cluster (same file Gen3v4 uses — likely already there). Verify in the log for `W&B key found -> WANDB_MODE=online`.

```bash
cd /u/home/llim/FMPCC/FM-PCC && git pull

# (a) train iMF longer, now with live W&B curves
N_TRAIN_STEPS=300000 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh

# (b) train FM ourselves — finally possible, and gives a comparable curve
N_TRAIN_STEPS=100000 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_fm_hardflow.sh
```
Knobs: `USE_WANDB=0` (disable) · `WANDB_PROJECT=…` · `FM_EXP_NAME=…` · `N_TRAIN_STEPS=…`

⚠️ **Watch the 24 h cap.** iMF ran 100 k in 4 h 12 m (~3.95 it/s) ⇒ **300 k ≈ 12.5 h** (fits), **1 M ≈ 42 h** (does **not** fit — would need checkpoint/resume). FM's speed is unmeasured; start at 100 k.

## 6. Why this matters for the Gen13 verdict

`WHERE_ARE_THE_TRAINING_CURVES.md` §4b flagged a genuine confound: **the FM baseline had 1 M training steps, our iMF had 100 k — a 10× asymmetry.** Fix 2 removes the blocker that made this untestable. Two experiments are now possible:

1. **FM at 100 k** — matched-budget training. If FM@100k still beats iMF@100k at matched K, the fix_7.3 refutation is confirmed *without* the training-budget confound. **This is the cheaper and more decisive of the two.**
2. **iMF at 300 k+** — tests whether more training closes the gap. §4's curve (plateau from ~25 k, −5.4 % over the last 50 k) predicts it will not, but it is now directly testable with live curves.

## 6b. 🛑 OVERWRITE SAFETY — a real data-loss bug, caught before running

**The question "will the new train overwrite the old files?" was correct, and the answer was YES for iMF.**

`run_scripts/train_imf.sh` had `exp_name="H16_imf_100k"` **hardcoded**. Running `N_TRAIN_STEPS=300000` would therefore have written into the *same* directory, and `run/utils.py:save_config` overwrites silently (it only prints *"old configs, checkpoints … will be overwritten"*). That would have destroyed:

- `model_ema_{0..4}.pth` — **the checkpoint behind every Gen13 result** (u_5 n=200, fix_7, fix_7.3)
- `metrics.csv` — the training curve analysed in §4 of the companion MD

Two independent protections added:

**(1) exp_name now encodes the step budget** — different budgets can never collide:

| `N_TRAIN_STEPS` | directory |
|---|---|
| 100000 | `H16_imf_100k` (unchanged — backward compatible) |
| 300000 | `H16_imf_300k` (**new dir**, existing artifacts untouched) |

**(2) refuse-to-clobber guard** — if the target dir already holds a *finished* run (final `model_ema_<cp>.pth`), the script **aborts** with instructions unless `FORCE_OVERWRITE=1`.

Verified by simulation against a mock existing checkpoint:

| case | result |
|---|---|
| re-run at 100k (the dangerous one) | **ABORTED ✅** |
| new budget 300k | proceeds → `H16_imf_300k` |
| `FORCE_OVERWRITE=1` | proceeds (explicit opt-in) |
| `IMF_EXP_NAME=H16_imf_rerun` | proceeds → new dir |

**FM side:** was already safe (`H16_1e6steps_wandb` ≠ the downloaded `H16_1e6steps`), but now additionally (a) budget-tagged as `H16_fm_<N>k`, (b) **hard-refuses** to write to `H16_1e6steps` — the authors' downloaded checkpoint that backs the entire replication — and (c) carries the same finished-run guard.

`--time` note: the sbatch `ls` lines were also hardcoded to the old name and now derive the same tag.

## 6c. 🛑 EVAL PROVENANCE — a *silent wrong-result* bug, also caught before running

Follow-up question: *"after training, what's the next command — and will eval overwrite things / can we tell which training an eval came from?"* Checking this exposed a bug **worse than overwriting**:

**`flow_exp_name="H16_imf_100k"` was HARDCODED in all three iMF eval scripts.** After training `H16_imf_300k`, running eval would have:
1. **loaded the OLD 100 k checkpoint** — silently, no error,
2. **written to the OLD result dir** (`H16_imf_hardflow_new_K5`), overwriting the u_5/fix_7.3 results,
3. produced numbers that *look* like they evaluate the new training but do not.

A wrong-results bug is worse than a data-loss bug: data loss is obvious, this is invisible.

**Fixed — two changes, both backward compatible:**

| | before | after |
|---|---|---|
| checkpoint loaded | hardcoded `H16_imf_100k` | **`${IMF_EXP_NAME:-H16_imf_100k}`** — same var name as training, so train and eval stay in sync |
| output dir | `H16_imf_hardflow_new_K5` | same **unless** a non-default checkpoint is used, then **`…_from_<training>`** |

Verified:

| invocation | loads | writes |
|---|---|---|
| *(no env — legacy)* | `H16_imf_100k` | `H16_imf_hardflow_new_K5` ← **unchanged** |
| legacy, `RANDOM_REPEAT=200` | `H16_imf_100k` | `H16_imf_hardflow_new_K5_n200` ← **unchanged** (u_5's dir) |
| `IMF_EXP_NAME=H16_imf_300k` | **`H16_imf_300k`** | `H16_imf_hardflow_new_K5_from_H16_imf_300k` |
| `IMF_EXP_NAME=H16_imf_300k`, `RANDOM_REPEAT=200` | `H16_imf_300k` | `…_K5_from_H16_imf_300k_n200` |

Every existing result directory keeps its exact name, so all prior MDs stay valid. `eval_smoothness_diag.sh` got the same treatment for **both** backbones (`IMF_EXP_NAME` / `FM_EXP_NAME` + `FM_CP`).

### The post-training command sequence

```bash
# 1. train (new dir, guarded)
N_TRAIN_STEPS=300000 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh
#    -> logs/hardflow/avoiding-v0/flow/H16_imf_300k/{model_ema_0..12.pth, metrics.csv}
#    NOTE cp index = N_TRAIN_STEPS/25000  (300k -> 12, NOT 4)

# 2. eval it — MUST pass IMF_EXP_NAME and the matching IMF_CP
IMF_EXP_NAME=H16_imf_300k IMF_CP=12 IMF_K=5 RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_imf_hardflow.sh

# 3. matched-budget battery against the new checkpoint
IMF_EXP_NAME=H16_imf_300k IMF_CP=12 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_matched_nfe_hardflow.sh
```

⚠️ **`IMF_CP` must be set too.** The checkpoint index is `N_TRAIN_STEPS / save_freq(25000)` — a 300 k run's final checkpoint is `model_ema_12.pth`, not `model_ema_4.pth`. Leaving `IMF_CP=4` would evaluate a 100 k-equivalent *intermediate* checkpoint of the new run — another silent-wrong-result trap.

## 7. Verification (container)

`py_compile` clean on `train_imf.py` / `train_fm.py`; `bash -n` clean on all three shell scripts; frozen files (`run/train.py`, `run_scripts/train.sh`, `run/eval.py`) confirmed untouched. Not executed here (no torch/wandb in this container) — W&B behaviour is exercised on the cluster, and every W&B path is wrapped so a failure degrades to CSV rather than aborting.

**Note:** `use_wandb` defaults to `False` in the config but the run scripts pass `--use_wandb` unless `USE_WANDB=0`. So sbatch runs get W&B by default; direct `python run/train_imf.py` calls do not, unless asked.
