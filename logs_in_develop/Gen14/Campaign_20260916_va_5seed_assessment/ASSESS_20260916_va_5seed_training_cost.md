# ASSESS — Is a 5-seed replication of Visual Aligning (Gen14) workable?

*Gen14 · `Campaign_20260916_va_5seed_assessment` · 2026-09-16 · assessment only, nothing submitted.
Relates to thesis items **R4** (alignment seeds beyond 6) in
`Writing/Working_Space/data_status/PENDING_20260916_missing_data_and_analyses.md`.
Sibling UAV campaign: `Gen15/Campaign_20260916_uav_5seed_training/`.*

## 1. Sources

- Slurm training logs `Slurm_Codes/logs/2026-0[89]-*/*train_mix_visual_aligning_*.log` — time = sum of the
  per-epoch tqdm durations (100 epochs × 1000 steps, bs 64).
- Cross-check: `dataset_config.pkl` → `losses.pkl` timestamps in the cluster tree
  `Data_Analysis/analysis_results_checkpoint/16_09_logs_tree.txt`. Both agree within ~0.1 h
  (the logs print no `JOB END`, so the epoch sum is the wall clock minus a few minutes of setup/image loading).
- Paper models: `Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md` §flagship + v3 `tab:va-models`.

## 2. Measured training time per seed (seed 6, one GPU, i6-gpu-1)

**Paper arms** (the four rows of `tab:va-models`):

| engine | checkpoint | job | train h / seed |
| :-- | :-- | :-- | --: |
| mf (flagship) | `VisualMeanFlow_…_filmv1_Emf_tslogit_normal` | 24124 | **9.4** |
| af (consistency training, floor 0.2) | `VisualAlphaFlow_…_filmv1_…_afschsigmoid_AFAFend0p2` | 25372 | 4.0 |
| diffusion K20 (DPCC baseline) | `H8_K20_…VisualGaussianDiffusion_aw10_…_filmv1_Ediffusion` | 24407 | 4.0 |
| fm | `VisualFlowMatching_…_filmv1_Efm` | 24345 | 3.2 |
| **sum** | | | **20.6** |

Other Gen14 runs, for reference (not paper arms): mf `filmv2` 11.6 h (24454) · mf DiT 12.9 h (24874) ·
mf `mf_dit` adaLN 9.4 h (25047) · af `filmv2` 5.7 h (24457) · af `filmv1` sigmoid 5.2 h (24156) ·
af const 0.05 4.5 h (25241) · af end 0.05 4.1 h (25376) · diffusion K100 4.0 h (24340).
Failures: 24222 (cancelled at start), 24838 (disk full → time limit), 25043 (`vis_lr_scale` guard).

## 3. Cost of seeds 7–10

| scope | GPU-h | wall clock at max 2 parallel |
| :-- | --: | --: |
| VA, 4 paper arms × 4 seeds | **82** | **≈ 41 h (1.7 d)** |
| UAV campaign (already planned, Gen15) | 188 | ≈ 94 h (3.9 d) |
| **UAV + full VA, same 2-slot budget** | **270** | **≈ 135 h (5.6 d)** |

**Evaluation is the larger cost, not training.** VA rollouts run a median of **400 control steps**
(11,298 rollouts in `batch_va2_20260915_100754`) at 190–300 ms/step for K=20 unprojected
(`tab:va-models`) → **≈ 1.3–2 min per episode, ≈ 2–3.5 GPU-h per (arm, variant) at 10 contexts × 1 episode**,
more for the projected variants and for 3 episodes per context. Re-evaluating every paper
configuration for 4 extra seeds is roughly another training-sized bill *per arm*, on top of §3.

## 4. Verdict

- **Training alone is workable** (≈ 1.7 days at 2 parallel), but it only makes sense if the seeds are then
  evaluated, and together with the UAV campaign it pushes the queue to **≈ 6 days of training plus evaluation
  on both tasks**.
- **The VA result does not need it as badly as UAV does.** The load-bearing VA claim (MeanFlow vs the
  diffusion baseline) is paired over 10 contexts and decisive at seed 6 (−0.3744 m, 0/10, p = 0.0020;
  MeanFlow vs flow matching 9/0, p = 0.0039). The null results (fm vs diffusion, af vs diffusion, p = 1.0)
  would not become claims with more seeds either. UAV, by contrast, has single-seed claims in every scene.
- **Decision (author, 2026-09-16): no VA training.** VA stays at the current single seed (seed 6),
  stated as a limitation (`\guard` in §6.2 stays). No partial replication either. UAV gets the 2-slot queue.

## 5. Status

| item | status |
| :-- | :-- |
| VA 5-seed training | ❌ not done — author decision 2026-09-16; single seed (6) is final |

**Thesis (v3.17, 2026-09-16):** stated once in §5.5 (seed 6 within the available storage and compute); the across-seed variation is shown on D3IL-avoiding instead (`tab:seed-spread`); the single-seed caveats in Chapter 6 were removed. Ledger item closed ❌ in `Writing/Working_Space/data_status/PENDING_20260916_missing_data_and_analyses.md`.
