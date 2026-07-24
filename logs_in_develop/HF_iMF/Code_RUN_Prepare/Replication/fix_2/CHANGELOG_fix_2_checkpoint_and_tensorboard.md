# Fix 2 — pretrained checkpoint + `tensorboard` (getting eval to actually run)

**Date:** 2026-07-18
**Trigger:** pipeline run `hardflow_pipeline` (job 23559, node i6-gpu-1). The bridge, d4rl shim (fix_1), and dynamics all worked; the run exposed the last two gaps between "clone runs code" and "eval produces numbers."
**Scope:** no repo/code change. Both items are handled cluster-side (place a file / install one package). The sbatch bridge is already correct.

---

## First, the good news (a fix_1 follow-up resolved)

The `No fitted dynamics found` worry from the previous run is **not a problem**. This run proved it — for the method that needs dynamics:
```
[hardflow_new] Loading fitted dynamics from logs/avoiding-v0/dynamics/linear_model.npz
              Fitted dynamics loaded successfully
```
`original` still prints `No fitted dynamics found` **by design** (it doesn't use a dynamics constraint). So the fit_dynamics → symlink → eval path works end-to-end. Nothing to fix there.

---

## Problem A — training aborts: `tensorboard` missing

```
run/train.py:12  from torch.utils.tensorboard import SummaryWriter
ModuleNotFoundError: No module named 'tensorboard'
```

**Root cause:** same family as d4rl (fix_1) — `hardflow_clone` is cloned from FMPCC, which lacks `tensorboard`; HardFlow needs it (it's in HardFlow's `environment.yml`). **Unlike d4rl, `tensorboard` is actually USED** (training-curve logging via `SummaryWriter`), so it must be **installed**, not shimmed.

**Caveat observed:** `run_scripts/train.sh` swallowed the Python non-zero exit, so the pipeline **continued to the eval steps** after training had actually failed — which is why the log then shows the eval checkpoint error below rather than stopping at the train error. Don't be misled: the *first* failure in a pipeline run is the real one.

## Problem B — eval aborts: pretrained checkpoint does not exist

```
FileNotFoundError: 'logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth'
```

**Root cause — and the key conceptual point:** HardFlow is **train THEN eval**, two separate steps. The eval methods (`hardflow_new`, `original`, `projection`, …) are **inference-time constrained-sampling algorithms with no weights of their own** — they load a **frozen, pretrained flow-matching model** (`model_ema_20.pth`) and steer its sampling. So eval **cannot run without that checkpoint.**

**HardFlow does NOT ship the checkpoint.** Verified: zero `.pth` in the repo. README (lines 73–75) only says it is *"expected at"* `logs/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth` and links a Google-Drive *download*. Nothing (not the repo, not our scripts) fetches it automatically. In job 23559 the checkpoint was missing because training didn't complete (Problem A) **and** nothing had been downloaded.

---

## The fix (chosen: DOWNLOAD the released weights)

The checkpoint can be obtained two ways; we took the fast one.

**Path B — download (chosen):** manually downloaded the released `.pth` (~20 MB) from the README Google-Drive link and placed it at the exact expected path:
```
/u/home/llim/FMPCC/FM-PCC/logs/hardflow/avoiding-v0/flow/H16_1e6steps/model_ema_20.pth
```
(`HardFlow/logs` is symlinked to `FM-PCC/logs/hardflow`, so that is the real location. Filename **must** be `model_ema_20.pth` — eval builds `model_ema_<flow_cp>.pth` with `flow_cp=20`.)

- **`tensorboard` is NOT needed for this path** — eval never imports it. Only training does.
- After placing the file, resubmit eval:
  ```bash
  cd /u/home/llim/FMPCC/FM-PCC
  METHODS="original" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_hardflow.sh
  ```

## The alternative (Path A — train it yourself)

Only if you want your **own** checkpoint instead of the authors'. Longer and needs the extra package:
```bash
conda activate hardflow_clone
pip install tensorboard          # Problem A fix — required ONLY for training
# then:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_hardflow.sh   # ~1e6 steps, long
```
Training writes the same `model_ema_20.pth` to the same path, after which eval runs identically. `tensorboard` produces training-curve logs under the run's log dir.

---

## Summary table

| Item | Needed for | Missing because | Resolution |
|---|---|---|---|
| `model_ema_20.pth` (checkpoint) | **eval** (all methods) | HardFlow doesn't ship weights; not trained/downloaded yet | **download** the released `.pth` → place at the path above (or train) |
| `tensorboard` (package) | **train only** | clone from FMPCC lacks it | `pip install tensorboard` in the clone — **skip if downloading** |

## Escalation ladder reminder (from fix_1)

Missing thing → **(1) is it actually used?** (grep). **(2) unused → shim** (d4rl, fix_1). **(3) used → install** (tensorboard, this fix). Missing **data/weights** (not a package) → **obtain the artifact** (train or download the checkpoint, this fix) — never shimmable.

## Status after this fix

Bridge ✅ · d4rl shim ✅ · fit_dynamics ✅ · dynamics load (hardflow_new) ✅ · checkpoint ✅ (placed). Next: resubmit eval and confirm `trajectories.csv` is written under `logs/hardflow/avoiding-v0/eval/<exp>/`.
