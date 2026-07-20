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

## 7. Verification (container)

`py_compile` clean on `train_imf.py` / `train_fm.py`; `bash -n` clean on all three shell scripts; frozen files (`run/train.py`, `run_scripts/train.sh`, `run/eval.py`) confirmed untouched. Not executed here (no torch/wandb in this container) — W&B behaviour is exercised on the cluster, and every W&B path is wrapped so a failure degrades to CSV rather than aborting.

**Note:** `use_wandb` defaults to `False` in the config but the run scripts pass `--use_wandb` unless `USE_WANDB=0`. So sbatch runs get W&B by default; direct `python run/train_imf.py` calls do not, unless asked.
