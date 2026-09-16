# INDEX — where the thesis data and analyses live

The thesis draws its numbers and figures from two folders **outside** `Writing/`. This index links them,
together with the notes kept here. Paths are relative to this file.

## This folder
| file | what |
| :-- | :-- |
| [`NOTE_20260914_official_DA_in_Paper.md`](NOTE_20260914_official_DA_in_Paper.md) | `DA_in_Paper` is the official data and figure store — the rules |
| [`PENDING_20260916_missing_data_and_analyses.md`](PENDING_20260916_missing_data_and_analyses.md) | what the thesis still lacks: runs and analyses, with where each gap shows in the draft |
| [`outdated_DATASTATUS_20260910_v3_entry_readiness.md`](outdated_DATASTATUS_20260910_v3_entry_readiness.md) | historical readiness ledger; superseded |

Naming: `NOTE_<date>_<topic>.md` states how things are; `PENDING_<date>_<topic>.md` lists what is missing.
A newer PENDING supersedes the older one — rename the old one with the prefix `outdated_`.

## `Data_Analysis/DA_in_Paper/` — official, live (numbers and figures in the thesis)
| path | what |
| :-- | :-- |
| [`README.md`](../../../../Data_Analysis/DA_in_Paper/README.md) | rules: data, figures, numbers |
| [`analysis/INDEX.md`](../../../../Data_Analysis/DA_in_Paper/analysis/INDEX.md) | **every thesis result → its analysis of record**, with protocol |
| [`analysis/va_results.py`](../../../../Data_Analysis/DA_in_Paper/analysis/va_results.py) | recomputes every §6.2 alignment number from the 15-09 batch |
| [`analysis/uav_results.py`](../../../../Data_Analysis/DA_in_Paper/analysis/uav_results.py) | recomputes every §6.3 quadrotor number from the 15-09 batch |
| [`plotting/`](../../../../Data_Analysis/DA_in_Paper/plotting/) | all figure code; data paths only in `sources.py` |
| [`plotting/NOTEBOOK_20260915_figure_data_sources.md`](../../../../Data_Analysis/DA_in_Paper/plotting/NOTEBOOK_20260915_figure_data_sources.md) | per-figure provenance audit |
| [`plotting/REQUEST_20260916_cluster_fm_plan_panels.md`](../../../../Data_Analysis/DA_in_Paper/plotting/REQUEST_20260916_cluster_fm_plan_panels.md) | cluster run for the missing plan-matrix panels |
| [`figures/MANIFEST.md`](../../../../Data_Analysis/DA_in_Paper/figures/MANIFEST.md) | every figure in the store: generated, vendored, planned, excluded |
| `data/` | prepared inputs (scene extract, cut dashboards, MuJoCo renders) |

**Data source of record:** [`Data_Analysis/analysis_results_checkpoint/15-09/`](../../../../Data_Analysis/analysis_results_checkpoint/15-09/) — one committed batch per environment. `temp/` is local and uncommitted.

## `Data_Analysis/DA_Result_Curated_MD/` — curated reports (cited through the index above)
| path | what |
| :-- | :-- |
| [`README.md`](../../../../Data_Analysis/DA_Result_Curated_MD/README.md) | folder guide |
| [`Report_20260819_MF_UNet/`](../../../../Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/README.md) | MeanFlow U-Net on obstacle avoidance; raw-plan panels |
| [`Report_20260903_AF_UNet/`](../../../../Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) | consistency training U-Net; four-model aggregate; unprojected goal rate |
| [`Report_20260909_MJPC_vs_PID_s_curve/`](../../../../Data_Analysis/DA_Result_Curated_MD/Report_20260909_MJPC_vs_PID_s_curve/README.md) | s-curve controller comparison — ⚠️ endpoint-projection rows invalid |
| [`DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md`](../../../../Data_Analysis/DA_Result_Curated_MD/DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md) | baseline at 10 vs 100 episodes |
| [`DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md`](../../../../Data_Analysis/DA_Result_Curated_MD/DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md) | flow matching vs baseline, matched budget |
| [`DA_20260913_uav_pillars_diffusion_baseline_reference.md`](../../../../Data_Analysis/DA_Result_Curated_MD/DA_20260913_uav_pillars_diffusion_baseline_reference.md) | pillars diffusion reference (pointer) |
| [`ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md`](../../../../Data_Analysis/DA_Result_Curated_MD/ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md) | consistency training vs MeanFlow on alignment |

`outdated_*` files there are kept for history and are not cited.

## Analyses of record outside both folders (linked from `analysis/INDEX.md`)
- Alignment: `logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md`
- Quadrotor: `logs_in_develop/Gen15/U16/DA_20260914_corridor_v2_paper_full.md`,
  `logs_in_develop/Gen15/Campaign_20260907_five_missions/CLOSURE_20260910_uav_engine_ladder_final.md`
- Endpoint projection at low budget: `logs_in_develop/aggregated_hardflow_lowK/DA_20260824_does_HF_pay_when_it_actually_runs.md`
