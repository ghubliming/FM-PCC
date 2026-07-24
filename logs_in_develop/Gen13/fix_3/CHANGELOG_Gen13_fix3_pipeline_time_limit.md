# Gen13 fix-3 — pipeline jobs stuck PENDING forever: `--time` exceeded the 24h partition ceiling

**Date:** 2026-07-18
**Trigger:** user submitted `imf_pipeline_hardflow.sh`; `squeue` showed:
```
23577  llim  0:00  2  gpu-1-student  (PartitionTimeLimit)  gres:gpu:1  PENDING  imf_pipeline_hardflow
```
**Question asked:** is the memory (24h cap) wrong, or is some `.sh` not updated/correct? **Answer: the memory was right — two `.sh` files were wrong.**

---

## 1. Diagnosis

`(PartitionTimeLimit)` is SLURM's queue-eligibility check, not a runtime kill — it means the **requested** `--time` exceeds the partition's actual maximum, so the job can **never** be scheduled. It sits `PENDING` indefinitely with no further error.

Grepped every `#SBATCH --time` in the repo (`Slurm_Codes/sbatch/**/*.sh`). Result: **every single `_pipeline.sh` in the repo requests `00:10:00` — except two, which requested `36:00:00`:**

| File | `--time` | Why |
|---|---|---|
| `Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh` (Gen13) | 36:00:00 ❌ | ran train (24h budget) + eval (12h budget) **inline in one job**, so I summed the two sub-budgets into one `--time` |
| `Slurm_Codes/sbatch/hardflow/hardflow_pipeline.sh` (Replication track) | 36:00:00 ❌ | identical bug, same author, same mistake |
| every other `*_pipeline.sh` (`sbatch/iMF/imf_pipeline.sh`, `sbatch/dpcc_pipeline.sh`, `sbatch/fm_visual_aligning/...`, etc.) | 00:10:00 ✅ | these are **orchestrators**: trivial jobs that just call `sbatch` twice with `--dependency=afterok`, never run the actual work inline |

So the root cause wasn't a wrong number — it was the wrong **architecture**. My two pipeline scripts didn't follow the repo's established pattern; they invented a different one (single inline job) that happens to be mathematically incompatible with a 24h ceiling whenever the summed phases exceed it.

## 2. Fix — converted both to the established orchestrator pattern

Rewrote `hardflow_pipeline.sh` and `imf_pipeline_hardflow.sh` to mirror `Slurm_Codes/sbatch/iMF/imf_pipeline.sh` exactly:

- The pipeline script itself: `--time=00:10:00`, 1 CPU, 2G, no GPU — it does nothing but check for an existing checkpoint (`SKIP_TRAIN`/file-exists logic preserved) and call `sbatch --parsable` for the train job, then `sbatch --parsable --dependency=afterok:$TRAIN_ID` for the eval job (or eval standalone if training is skipped).
- Each **chained** job (`train_hardflow.sh`/`train_imf_hardflow.sh`, `eval_hardflow.sh`/`eval_imf_hardflow.sh`) keeps its own already-correct ≤24h budget (24h, 12h respectively) — nothing about those files needed to change.
- Eval knobs (`METHODS`, `IMF_METHODS`, `IMF_KS`, `IMF_CP`) are forwarded to the chained eval job via `sbatch --export`.
- Log grouping preserved: sub-jobs share the pipeline's own `$SUBMIT_DATE`/`$SUBMIT_TIME` for unified log folders, same convention as the existing iMF orchestrator.

Both pass `bash -n`. Repo-wide grep confirms **no `#SBATCH --time` anywhere now exceeds 24h**.

## 3. What the user needs to do

- **Cancel the stuck job:** `scancel 23577` (it will never run on its own).
- **Re-pull** this fix, then resubmit:
  ```bash
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
  ```
  This now returns almost immediately (the orchestrator itself finishes in seconds) and prints the train/eval job IDs it chained; monitor both with `squeue -u $USER`.

## 4. Memory updated

`slurm-sbatch-is-real-entrypoint.md` amended with this concrete incident: exceeding 24h means **never scheduled**, not "killed mid-run" (stronger than previously stated) — and a new explicit rule: never inline multiple long-running phases (train + eval) into one job's `--time`; always use the dependency-chain orchestrator pattern for any script that would otherwise need to sum phase budgets.

## 5. Status

Both pipeline scripts fixed and verified (syntax + repo-wide time-cap grep). No other `--time` value in the repo violates the 24h ceiling. The usage guide changelog (`Gen13/init/CHANGELOG_Gen13_coding1_imf_package_and_usage.md`) already documents `imf_pipeline_hardflow.sh` as "one command" — that remains true; only its internal mechanics changed (now chains two real jobs instead of running inline).
