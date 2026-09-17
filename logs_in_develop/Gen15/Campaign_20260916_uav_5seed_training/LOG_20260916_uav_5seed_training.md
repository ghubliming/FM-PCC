# LOG — UAV 5-seed training (seeds 7–10)

*Gen15 · campaign `Campaign_20260916_uav_5seed_training` · opened 2026-09-16.
Closes thesis item **R1** in `Writing/Working_Space/data_status/PENDING_20260916_missing_data_and_analyses.md`
(every quadrotor claim is single-seed). Training only; evaluation is a follow-up campaign step.*

> ## ❌ IMPOSSIBLE — not run (2026-09-16)
> **Blocked by cluster disk, not GPU time.** After `tools/clean_weights/clean_weights.py --apply`
> (run log `logs/_clean_weights_runlogs/clean_weights_20260916_173211.log`): 100.3 GB used,
> 1.5 GB free before, 5.7 GB freed → **≈ 7.2 GB free**.
> - Training writes 6 periodic `state_<N>.pt` + `state_best.pt` per seed: ≈ 210 MiB (fm, diffusion) /
>   ≈ 430 MiB (mf, af) → **≈ 15 GB for the 48 trainings** (≈ 4.4 GB even with best+latest pruning after every job).
> - Evaluating seeds 7–10 adds tens of GB of `plans/` (`logs/UAV_MIX` is already 13.8 GiB, mostly plans).
> - With `afterany` chaining, a full disk would silently fail every remaining job (cf. VA job 24838, `No space left on device`).
>
> **Decision: UAV stays at single seed (seed 6), stated as a limitation — same as Visual Aligning**
> (`Gen14/Campaign_20260916_va_5seed_assessment/`). Thesis item R1 closes as *not feasible*.
> `Slurm_Codes/sbatch/uav_mix/TEMP_train_5seed_missing.sh` was **never submitted**; the plan below is kept for the record only.

## 1. Plan

**Grid:** 4 engines × 3 scenes × seeds 6–10. Seed 6 is already trained for every cell
(16-09 cluster tree `Data_Analysis/analysis_results_checkpoint/16_09_logs_tree.txt`), so
**48 trainings** are missing: seeds 7, 8, 9, 10.

| engine | checkpoint folder (`logs/UAV_MIX/uav-<scene>/mix_uav_<engine>/…`) | env | ~h/seed (seed 6) |
| :-- | :-- | :-- | --: |
| fm | `H8_Dmodels.diffusion.FlowMatchingODE_9D` | — | 2.7 |
| mf | `H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet` | — | 7.5 |
| af | `H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet` | `UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2` | 3 |
| diffusion | `H8_Dmodels.ddpm_diffusion.GaussianDiffusion_9D_K20` | — (K20 only) | 2.5 |

**Scenes:** `corridor`, `pillars`, `s_curve`. `corridor_v2` (u17cv2) is an evaluation geometry; training uses `corridor`.

**Submission:** `Slurm_Codes/sbatch/uav_mix/TEMP_train_5seed_missing.sh` (thin login-node submitter, run with `bash` from repo root).
- One seed per job (`train_mix_uav.sh <engine> <scene> <seed>`), `--time=24:00:00`.
- **Max 2 jobs running at once:** all jobs are split into 2 lanes; each lane is one `--dependency=afterany` chain,
  so everything is scheduled at submit time. Order is seed-first (all seed-7 jobs, then 8, 9, 10), each job
  placed on the lane with the smaller expected load. `afterany`: a failed/timed-out job does not block its lane.
- A seed is skipped if its `<seed>/state_best.pt` exists (a crashed partial run also has one — check before submitting).

```bash
bash Slurm_Codes/sbatch/uav_mix/TEMP_train_5seed_missing.sh           # dry run
bash Slurm_Codes/sbatch/uav_mix/TEMP_train_5seed_missing.sh --submit
```

**Expected wall clock:** 48 jobs ≈ 188 GPU-h → ≈ 94 h per lane, **≈ 4 days** end to end (plus queue).

## 2. Slurm job index — not used (never submitted)

Submit date: _____ · git rev: _____ · Slurm logs: `Slurm_Codes/logs/<date>/`

Record `jobID (lane)` per cell, from the `--submit` output.

| engine | scene | seed 7 | seed 8 | seed 9 | seed 10 | status / notes |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| fm | corridor | | | | | |
| fm | pillars | | | | | |
| fm | s_curve | | | | | |
| mf | corridor | | | | | |
| mf | pillars | | | | | |
| mf | s_curve | | | | | |
| af | corridor | | | | | |
| af | pillars | | | | | |
| af | s_curve | | | | | |
| diffusion | corridor | | | | | |
| diffusion | pillars | | | | | |
| diffusion | s_curve | | | | | |

## 3. Issues / reruns

Campaign cancelled before submission — cluster disk (see banner).

## 4. Next

None. Reopen only if ≥ ~15 GB (training) plus evaluation space can be freed on the cluster.

**Thesis (v3.17, 2026-09-16):** stated once in §5.5 (seed 6 within the available storage and compute); the across-seed variation is shown on D3IL-avoiding instead (`tab:seed-spread`); the single-seed caveats in Chapter 6 were removed. Ledger item closed ❌ in `Writing/Working_Space/data_status/PENDING_20260916_missing_data_and_analyses.md`.
