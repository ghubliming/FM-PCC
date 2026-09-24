---
name: ntrial20-campaign-census
description: D3IL-avoiding twenty-episode (_msg20trials) campaign — only MeanFM(U-Net)/FM at K1,K2(,K5) are complete; every K20 cell is truncated; the 20-trial "AlphaFlow" cells are SiT alpha_end 0 (not CI-MeanFM); rebuilt section + DA in Working_Space/ntrial20_appendix_20260924
metadata:
  type: project
---

Census of 2026-09-24, from `temp/23-09/batch_avoiding_combined_20260923_110010(TenpK2D)` and the cluster tree
`analysis_results_checkpoint/23-09-23-21-Berlin-Time-logs_tree.txt`. A complete flow campaign has
5 seeds x 3 geometries x 20 episodes, which is 4706 files in the tree.

- **Complete, and the same `state_best.pt` as the chapter's two-episode rows:** MeanFM (`bbunet`) K1/K2/K5 and FM K1/K2/K5.
  MeanFM K10 has complete per-step cells, but the run was cut on seed 10 / both-hard.
- **Truncated:** Diffusion K20 (both-hard has seed 6 only, 562 files), FM K20, MeanFM K20.
- **Wrong model:** `AlphaFlowODE_msg20trials` is `bbsit` + `ae0.0`. CI-MeanFM (`bbunet` + `ae0.2`) exists at twenty
  episodes for seed 6 only (`_msgafon02_s6`).

The author (2026-09-24) says K20 at twenty episodes is infeasible in this setup. The appendix shows only the correct
low-K runs against the two-episode main text, with statistics. The result is the new section `app:avoiding-twenty` plus
its DA in `logs_in_develop/Writing/Working_Space/ntrial20_appendix_20260924/`. `09_appendix.tex` was not touched; v3
must `\input` the section and re-point `app:avoiding-twenty-episode`.

Open watch item: MeanFM's chapter 29/30 under `t` comes from one failure at seed 7 / top-right-hard. The twenty-episode
run of the same checkpoint and seeding reads 20/20 there. It repeats across K1/K2/K5/K10 and is unexplained.

**Why:** the dead block quoted K20 rows built on 5+5+1 seeds and CI-MeanFM rows from one seed. The author flagged the
campaign as suspicious twice (v3.55, 2026-09-24).

**How to apply:** quote twenty-episode numbers only from the kept cells and the DA above, and never from a K20
`_msg20trials` cell. Related: [[thesis-draft-ownership]], [[da-requires-csv-never-from-logs]], [[thesis-prose-style]].
