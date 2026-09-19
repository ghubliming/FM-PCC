# SLRUM DATE 2026-09-17

This runbook closes the two missing rows in `tab:avoiding-dpcc-protocol` (ledger R8). The scripts under
`Slurm_Codes/temp_bash/` are intentionally gitignored and must be copied to the remote manually.

**Ownership boundary:** the orchestrator only submits and validates remote Slurm training/evaluation jobs. It
does not download, copy, export, or delete results. Copying the template to the remote and downloading completed
results are manual human tasks.

## Run map

| lacking-run topic | dated orchestrator | real Slurm entrypoint | child job / state |
| :-- | :-- | :-- | :-- |
| [FM at K=1,2; five seeds × two episodes](PENDING_20260917_dpcc_protocol_rows_table61.md#what-is-needed) | [`pipeline_20260917_r8_dpccproto_all.sh`](../../../../Slurm_Codes/temp_bash/pipeline_20260917_r8_dpccproto_all.sh) | [`eval_fmv3_ode_job.sh`](../../../../Slurm_Codes/sbatch/eval_fmv3_ode_job.sh) | job **25878** · RUNNING at first check |
| [CI-MeanFM U-Net α_end=0.2; train seeds 7–10](PENDING_20260917_dpcc_protocol_rows_table61.md#what-is-needed) | same driver | [`train_alphaflow.sh`](../../../../Slurm_Codes/sbatch/AlphaFlow/train_alphaflow.sh) | job **25879** · PENDING `(Resources)` at first check |
| [CI-MeanFM U-Net α_end=0.2; evaluate K=1,2 over seeds 6–10](PENDING_20260917_dpcc_protocol_rows_table61.md#what-is-needed) | same driver | [`eval_alphaflow.sh`](../../../../Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh) | job **25880** · PENDING `(Dependency)` · `afterok:25879` |

The CI-MeanFM training also advances general-ledger row R6, but this wave evaluates only the R8 protocol.

## Submission attempts

| attempt | command / outcome | jobs created |
| :-- | :-- | :-- |
| 2026-09-17 attempt 0 | invoked the copied orchestrator with `bash`; stopped in the login shell before submission. The original seed-6 preflight incorrectly looked for a nonexistent `state_latest.pt` alias; corrected from `analysis_results_checkpoint/16_09_logs_tree.txt` to the exact U-Net α_end=0.2 seed-6 `state_80000.pt` path | none |
| 2026-09-17 attempt 1 | invoked the corrected orchestrator with `bash` again; the Slurm-only guard stopped immediately and instructed use of `submit.sh` | none |
| 2026-09-17 attempt 2 | corrected login-shell driver completed submission; seed-6 `state_80000.pt` preflight passed and the dependency graph was created | **25878, 25879, 25880** |

Initial scheduler snapshot: job 25878 was running; job 25879 was waiting for GPU resources; job 25880 was
correctly waiting on `afterok:25879`. The `/u/home` filesystem reported 7.9 GB available (7.0 TB filesystem,
rounded to 100% used). Training therefore remains disk-sensitive; job 25879's disk preflight and checkpoint
writes must be checked in its log.

## One-command submission

Run the dated aggregate driver directly, following the established `temp_bash` convention:

```bash
bash Slurm_Codes/temp_bash/pipeline_20260917_r8_dpccproto_all.sh
```

The login-shell driver submits FM evaluation and CI-MeanFM training as the two independent initial jobs, then
submits CI-MeanFM evaluation with `afterok:<training-job-id>`. At most two jobs from this wave run concurrently.
Each child retains its own `24:00:00` allocation; the phases are never combined into one over-limit allocation.
The driver prints all three IDs directly in the terminal. Copy them into the run map. Its internal `sbatch`
calls use the same date-organized log naming as `submit.sh`, following the repository's established dependency
pipeline pattern. All logs land under `Slurm_Codes/logs/<SLURM DATE>/`.

## Identity checks

| job | log must show |
| :-- | :-- |
| FM eval | `NFE budgets to evaluate: 1 2`; results suffix `_msgdpccproto`; seeds 6–10; `n_trials=2` |
| CI-MeanFM train | `AF_BONE='unet'`; `AF_ALPHA_END='0.2'`; seeds `7 8 9 10`; completed numbered checkpoints for all four seeds |
| CI-MeanFM eval | `AF_SEEDS` = 6–10; `AF_EPOCH='latest'`; `AF_NTRIALS='2'`; K=1,2; checkpoint tree contains `bbunet` and `ae0.2` |

If training fails or times out, Slurm does not release the dependent evaluation. Record the failed training
job and do not bypass the dependency; a resume wave must preserve the same train-before-eval ordering.

## Completion record and owner

| check | owner | result |
| :-- | :-- | :-- |
| FM K1/K2 Slurm job completed and remote folders verified | Slurm job + human verification | ✅ 2026-09-17 23:30 UTC — job 25878, `NFE budgets to evaluate: 1 2`, seeds 6–10, ten `…_msgdpccproto/<seed>` savepaths, `Evaluation completed successfully` |
| CI-MeanFM seeds 7–10 training completed and checkpoints verified | Slurm job + human verification | ✅ 2026-09-18 11:08 UTC — job 25879, `AF_BONE='unet'`, `AF_ALPHA_END='0.2'`, seeds 7/8/9/10, `Training complete for all seeds` |
| CI-MeanFM K1/K2 Slurm job completed and remote folders verified | Slurm job + human verification | ✅ 2026-09-18 11:47 UTC — job 25880, `AF_SEEDS='6 7 8 9 10'`, `AF_EPOCH='latest'`, `AF_NTRIALS='2'`, K=1,2, `bbunet`/`ae0.2` in the path, ten savepaths. Its `[hardflow][BLOCKED]` lines are the expected K=2 degeneracy guard on the unrelated HardFlow variants, not a failure |
| Required result folders downloaded from the cluster | **human only** | `XXX` |
| `avoiding_rules_by_protocol.py` re-run locally | human/local follow-up | `XXX` |
| Table 6.1 rows filled; R8 closed | human/local follow-up | `XXX` |

## Closure, 2026-09-19

All three jobs finished. **R8 and R19 are data-complete** at DPCC's protocol: flow matching and CI-MeanFM
(U-Net, $\alpha_{\mathrm{end}}=0.2$) now both exist at $\nfe=1,2$ over seeds 6–10 at `n_trials: 2`, tagged
`_msgdpccproto`. R6 (CI-MeanFM beyond seed 6 on obstacle avoidance) has its trainings; whether it closes
depends on which evaluations the extended protocol still needs.

Because the queue is now empty of this wave, `config/projection_eval.yaml` is free to be switched to
`n_trials: 20` for phase B of the 2026-09-18 wave (group I, ledger R18) — see
[`SLURM_RUNBOOK_20260918_pending_runs.md`](SLURM_RUNBOOK_20260918_pending_runs.md).
