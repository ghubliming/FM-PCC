# Figure manifest — generated, do not edit

Built by `plots/make_figs.py` on **2026-09-12**.
Regenerate with `python3 plots/make_figs.py`; to point at new data, edit
`plots/sources.py` and nothing else.

| figure | corpus | protocol |
| :-- | :-- | :-- |
| `fig_avoiding_pareto_aggregate.svg` | `temp/2508/batch_avoiding_combined_20260825_143212` | 5 seeds (6-10) x 20 trials = 100 episodes per cell |
| `fig_avoiding_pareto_top-left-hard.svg` | `temp/2508/batch_avoiding_combined_20260825_143212` | 5 seeds (6-10) x 20 trials = 100 episodes per cell |
| `fig_avoiding_pareto_top-right-hard.svg` | `temp/2508/batch_avoiding_combined_20260825_143212` | 5 seeds (6-10) x 20 trials = 100 episodes per cell |
| `fig_avoiding_pareto_both-hard.svg` | `temp/2508/batch_avoiding_combined_20260825_143212` | 5 seeds (6-10) x 20 trials = 100 episodes per cell |
| `fig_avoiding_k_ladder.svg` | `temp/2508/batch_avoiding_combined_20260825_143212` | 5 seeds (6-10) x 20 trials = 100 episodes per cell |
| `fig_avoiding_projector_cost.svg` | `temp/0609/I/batch_avoiding_combined_20260906_125724` | 4 seeds (7-10) x 3 geometries x 2 trials = 24 rollouts per row |

## Vendored — copied in, not generated

These were produced by the evaluation's own diagnostics and cannot be rebuilt
from a CSV. Each was checked against `/workspaces/aux_repo/` and is ours.

| figure | source | provenance |
| :-- | :-- | :-- |
| `fig_raw_plans_meanflow_K1` | `Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/fig6a_plans_mfunet_K1_seed6_both-hard.png` | Report_20260819_MF_UNet fig 6a; per-episode MPC diagnostics, unprojected arm, seed 6, both-hard, K=1. Ours; verified absent from aux_repo. |
| `fig_raw_plans_diffusion_K1` | `Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/fig6b_plans_dpcc_K1_seed6_both-hard.png` | Report_20260819_MF_UNet fig 6b; same protocol, diffusion baseline at K=1. Ours; verified absent from aux_repo. |
| `fig_raw_plans_meanflow_K2` | `Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/fig6c_plans_mfunet_K2_seed6_both-hard.png` | Report_20260819_MF_UNet fig 6c; same protocol, K=2. Ours; verified absent from aux_repo. |
| `fig_raw_goal_reached_K1` | `Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/fig7_raw_diffuser_K1.svg` | Report_20260903_AF_UNet fig 7; goal reached on the unprojected arm at K=1, top-right-hard, seed 6, 20 trials. Ours; verified absent from aux_repo. |

## Corpus registry

Mirrors §10 of `data_status/DATASTATUS_20260910_v3_entry_readiness.md`.
A `missing` row is a download away; a `blocked` grade is a *run* away.

| key | on disk | grade | protocol |
| :-- | :-- | :-- | :-- |
| `avoiding_t2` | yes | ready | 5 seeds (6-10) x 20 trials = 100 episodes per cell |
| `avoiding_af_unet` | yes | partial | seed 6, 20 trials |
| `avoiding_minK` | yes | partial | 4 seeds (7-10) x 3 geometries x 2 trials = 24 rollouts per row |
| `avoiding_t1` | yes | partial | 5 seeds x 2 trials = 10 episodes per cell |
| `visual_aligning` | yes | partial | seed 6, 10 enumerated contexts x 1 trajectory (paired) |
| `uav` | yes | blocked | seed 6, n = 10 rollouts |

## Format

SVG. `\includegraphics` needs PDF, so run `tools/svg2pdf.sh` before a LaTeX build;
the container that writes the thesis has no SVG converter, which is why the figures
are committed as SVG and converted where the build happens.
