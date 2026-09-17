# REQUEST — D3IL-avoiding at the protocol of DPCC for the flow-based models (U-Net)

**2026-09-17 · needed for** `tab:avoiding-dpcc-protocol` (v3 §6.1.1), the primary comparison of Chapter 6.

## Protocol
DPCC's released evaluation (`aux_repo/dpcc/config/projection_eval.yaml`): **`n_trials: 2`** for each of **5
training seeds** → 10 episodes per geometry; the paper's "five training seeds and ten test seeds". Ours: seeds
6–10, three geometries, tightened `dpcc-{r,c,t}` rules, 4 candidate plans.

## What exists (committed 15-09 batch)
| model | budget | at 5 seeds × 2 episodes? |
| :-- | :-- | :-- |
| diffusion (DPCC) | K20 | ✅ `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` |
| flow matching (U-Net) | K20 | ✅ `H8_K20_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE` |
| flow matching (U-Net) | K1, K2 | ❌ |
| MeanFlow (**U-Net** `bbunet`) | K1, K2 | ✅ **found 2026-09-17** — present all along under the same `Folder_Name` as the DiT rows; separable only by `Full_Path` (`…_bbunet_…`). Filled into `tab:avoiding-dpcc-protocol` in v3.28 |
| α-Flow (**U-Net**, α_end 0.2) | K1, K2 | ❌ — the checkpoint exists for **seed 6 only**; 5-seed 2-episode α-Flow rows are **SiT/DiT** |

## Runs (all evaluation is cheap: 2 episodes)
1. **Flow matching K1, K2** — config default already 5 seeds × 2:
   ```bash
   FMV3_FLOW_STEPS="1 2" FMPCC_RUN_MSG=dpccproto ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/eval_fmv3_ode_job.sh
   ```
2. ~~**MeanFlow U-Net K1, K2**~~ — **not needed**: the runs exist in the 15-09 batch (see the table above).
3. **α-Flow U-Net, α_end 0.2** — train seeds 7–10 first, then evaluate all five. The YAML currently
   names seed 6 only, so the five-seed evaluation must use the `AF_SEEDS` runtime override. Evaluate
   `latest`, not `best`: the latter is the known mid-homotopy checkpoint-selection artefact.
   ```bash
   TRAIN_SEEDS="7 8 9 10" AF_BONE=unet AF_ALPHA_END=0.2 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/train_alphaflow.sh
   # Submit only after training completes successfully:
   AF_BONE=unet AF_ALPHA_END=0.2 AF_EPOCH=latest AF_SEEDS="6 7 8 9 10" AF_NTRIALS=2 AF_FLOW_STEPS="1 2" FMPCC_RUN_MSG=dpccproto ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh
   ```
   ⚠️ Cluster disk was ≈7 GB free on 2026-09-16 (`Gen15/Campaign_20260916_uav_5seed_training`); four α-Flow
   U-Net trainings write ≈ 430 MiB each. Check before submitting.

Separated, guarded one-off drivers and the job-ID ledger are in
[`Slurm_Codes/temp_bash/`](../../../Slurm_Codes/temp_bash/) and
[`SLURM_RUNBOOK_20260917_dpcc_protocol_rows.md`](../../../logs_in_develop/Writing/Working_Space/data_status/SLURM_RUNBOOK_20260917_dpcc_protocol_rows.md).

## Then, locally
`python3 Data_Analysis/DA_in_Paper/analysis/avoiding_rules_by_protocol.py` after adding the new folders to its
`DPCC protocol` block; fill the *pending* rows of `tab:avoiding-dpcc-protocol` and the `\hole` below it.
