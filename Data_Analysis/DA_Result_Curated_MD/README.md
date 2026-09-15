# Curated analysis archive

This directory preserves curated reports, snapshots, proposals, and the historical key-headlines
notebook. It is an analysis archive and navigation aid. It is not the main thesis evidence entry.

For the paper/thesis, start at `Data_Analysis/DA_in_Paper/analysis/INDEX.md`. That index identifies
the analysis of record for every reported result; `Data_Analysis/DA_in_Paper/` also owns every paper
figure and its plotting code. For the retained 2026-09-15 CSV export, start at
`Data_Analysis/analysis_results_checkpoint/15-09/LEDGER_20260915.md`.

## Status map — reviewed 2026-09-15

| item | status | use |
| :-- | :-- | :-- |
| `NOTEBOOK_20260829_key_headlines.md` | working notebook, updated through 2026-09-15 | historical headline navigation; do not use as the main thesis entry |
| `Report_20260819_MF_UNet/` | retained | avoiding MeanFlow report; its n=2 protocol and later n=20 checks must travel with its claims |
| `Report_20260903_AF_UNet/` | retained | avoiding consistency-target report, seed 6; its figures and source analysis remain useful |
| `DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md` | retained | cross-family avoiding analysis; backbone and coverage caveats remain binding |
| `DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md` | retained methodology | trial-count sensitivity and baseline protocol |
| `DA_20260827_mpc_candidate_fan_avoiding.md` | retained | candidate-fan study under its stated protocol |
| `Proposal_20260905_HF_minK_mf_af_unet/` | mixed proposal/result folder | the proposal's “nothing submitted” header is historical; use its dated DA for the completed result |
| `ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md` | partially superseded | use for source-level mechanism only; its empirical open question was answered later |
| `SNAPSHOT_20260826_visual_avoiding_env_status.md` | historical snapshot | dated whole-environment context, not a current thesis result |
| `DA_20260913_uav_pillars_diffusion_baseline_reference.md` | retained | concise pointer to the full corrected-pillars analysis; one seed and unmatched K |
| `Report_20260909_MJPC_vs_PID_s_curve/` | partially invalid | raw-plan and DPCC controller observations remain; HardFlow rows are invalid due to the pre-fix `x_active` bug |
| every `outdated_*` item | outdated archive | keep for provenance; read its banner and replacement pointer before using anything |

## Current replacements for stale UAV summaries

- Corrected pillars and diffusion reference:
  `logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260912_pillars_diffusion_baseline_reference.md`.
- Corridor paper evaluation:
  `logs_in_develop/Gen15/U16/DA_20260914_corridor_v2_paper_full.md`.
- The September 15 checkpoint contains the corresponding UAV batch, including all 624
  corridor-v2 rollout rows.

There is no single current “whole UAV ranking”: scenes answer different questions, and the main
corrected scene studies are one seed. Report each scene, geometry, success definition, and protocol.

## Maintenance rules

- Preserve outdated reports; add or update an `OUTDATED`/`PARTIALLY INVALID` banner and name the
  replacement or limitation.
- Small corrections belong in the affected document. A changed result needs a new dated analysis,
  then a row update in `DA_in_Paper/analysis/INDEX.md`.
- Keep data provenance in the analysis that generated the number. The September 15 checkpoint is
  the source for new work; it does not silently replace an older analysis's recorded input batch.
- Do not build thesis claims from global candidate rankings. Filter by task, protocol, geometry,
  backbone, K, constraint method, seed coverage, and quality flags first.
