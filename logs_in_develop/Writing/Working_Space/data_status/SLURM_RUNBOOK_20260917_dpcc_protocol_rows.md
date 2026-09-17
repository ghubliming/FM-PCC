# SLRUM DATE XXX

Replace `XXX` with the actual cluster submission date. This runbook closes the two missing rows in
`tab:avoiding-dpcc-protocol` (ledger R8). The scripts under `Slurm_Codes/temp_bash/` are intentionally
gitignored and must be copied to the remote manually.

**Ownership boundary:** the scripts only submit and validate remote Slurm training/evaluation jobs. They do
not download, copy, export, or delete results. Copying the templates to the remote and downloading completed
results are manual human tasks.

## Run map

| lacking-run topic | temporary submission file | real Slurm entrypoint | Slurm job / state |
| :-- | :-- | :-- | :-- |
| [FM at K=1,2; five seeds × two episodes](PENDING_20260917_dpcc_protocol_rows_table61.md#what-is-needed) | [`submit_20260917_r8_fm_dpccproto.sh`](../../../../Slurm_Codes/temp_bash/submit_20260917_r8_fm_dpccproto.sh) | [`eval_fmv3_ode_job.sh`](../../../../Slurm_Codes/sbatch/eval_fmv3_ode_job.sh) | job `XXX` · ⏳ |
| [CI-MeanFM U-Net α_end=0.2; train seeds 7–10](PENDING_20260917_dpcc_protocol_rows_table61.md#what-is-needed) | [`submit_20260917_r8_cimeanfm_train.sh`](../../../../Slurm_Codes/temp_bash/submit_20260917_r8_cimeanfm_train.sh) | [`train_alphaflow.sh`](../../../../Slurm_Codes/sbatch/AlphaFlow/train_alphaflow.sh) | job `XXX` · ⏳ |
| [CI-MeanFM U-Net α_end=0.2; evaluate K=1,2 over seeds 6–10](PENDING_20260917_dpcc_protocol_rows_table61.md#what-is-needed) | [`submit_20260917_r8_cimeanfm_eval.sh`](../../../../Slurm_Codes/temp_bash/submit_20260917_r8_cimeanfm_eval.sh) | [`eval_alphaflow.sh`](../../../../Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh) | job `XXX` · blocked until training ✅ |

The CI-MeanFM training also advances general-ledger row R6, but this wave evaluates only the R8 protocol.

## Submission order

The FM evaluation is independent and may run immediately. Submit the CI-MeanFM training separately. Only
after its job finishes successfully, run the CI-MeanFM evaluation template; that template refuses to submit
unless `state_latest.pt` exists for all five seeds.

```bash
bash Slurm_Codes/temp_bash/submit_20260917_r8_fm_dpccproto.sh
bash Slurm_Codes/temp_bash/submit_20260917_r8_cimeanfm_train.sh
# After the training job is COMPLETED:
bash Slurm_Codes/temp_bash/submit_20260917_r8_cimeanfm_eval.sh
```

Copy each reported job ID into the run map. All three submissions go through `Slurm_Codes/submit.sh`; their
logs therefore land under `Slurm_Codes/logs/<SLURM DATE>/` with the job ID in the filename.

## Identity checks

| job | log must show |
| :-- | :-- |
| FM eval | `NFE budgets to evaluate: 1 2`; results suffix `_msgdpccproto`; seeds 6–10; `n_trials=2` |
| CI-MeanFM train | `AF_BONE='unet'`; `AF_ALPHA_END='0.2'`; seeds `7 8 9 10`; four completed `state_latest.pt` files |
| CI-MeanFM eval | `AF_SEEDS` = 6–10; `AF_EPOCH='latest'`; `AF_NTRIALS='2'`; K=1,2; checkpoint tree contains `bbunet` and `ae0.2` |

If training fails or times out, do not submit evaluation. Re-run the training template with `AUTO_RESUME=1`,
then record the additional training job ID in the same run-map cell.

## Completion record and owner

| check | owner | result |
| :-- | :-- | :-- |
| FM K1/K2 Slurm job completed and remote folders verified | Slurm job + human verification | `XXX` |
| CI-MeanFM seeds 7–10 training completed and checkpoints verified | Slurm job + human verification | `XXX` |
| CI-MeanFM K1/K2 Slurm job completed and remote folders verified | Slurm job + human verification | `XXX` |
| Required result folders downloaded from the cluster | **human only** | `XXX` |
| `avoiding_rules_by_protocol.py` re-run locally | human/local follow-up | `XXX` |
| Table 6.1 rows filled; R8 closed | human/local follow-up | `XXX` |
