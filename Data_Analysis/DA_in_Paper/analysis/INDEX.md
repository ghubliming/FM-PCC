# Analysis index — the evidence behind every thesis number

**Live since 2026-09-14.** One row per result the thesis states, with the analysis it is taken from.
The analyses stay where they were written; this index links them. When a thesis number changes, the
analysis changes first and this row is updated with it.

**This is the main analysis entry for the paper/thesis.** The curated headlines notebook is a
historical working notebook and is not used as an analysis of record.

Thesis sections refer to `Working_Space/v3` (Ch 5–6). Links are relative to this file and were checked
to exist when the index was written.

| thesis | result | cited in the draft as | analysis | protocol |
| :-- | :-- | :-- | :-- | :-- |
| §5.3.1 `tab:target`, §6.1.1 `tab:state-headline` | DPCC baseline reproduced; MeanFlow vs baseline | DA_20260827 §10.1 | [`DA_20260827_mpc1_full_seeds_state_avoiding.md`](../../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md) | 5 seeds × 20 episodes |
| §5.5.1 `tab:tiers` | baseline at 10 vs 100 episodes per geometry | DA_20260819 (n=2 vs n=20) | [`DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md`](../../DA_Result_Curated_MD/DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md) | 5 seeds × 2 / × 20 |
| §6.1.1 `tab:state-fm`, matched-budget control | flow matching vs baseline; the 1.4× × 15× split | DA_20260819 §1, §1b | [`DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md`](../../DA_Result_Curated_MD/DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md) | 5 seeds × 20 episodes |
| §6.1.2 `tab:state-lowk` | cheapest full-success configuration per model | Report_20260903 §1 | [`README.md`](../../DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) | seed 6, 20 episodes |
| §6.1.3 `fig:raw-plans` | plans without projection, 6.0 vs 28.0 violations | Report_20260819_MF_UNet §7 | [`README.md`](../../DA_Result_Curated_MD/Report_20260819_MF_UNet/README.md) | seed 6, both-hard |
| §6.1.3 | goal reached without projection, K=1 | Report_20260903_AF_UNet §7 | [`README.md`](../../DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) | seed 6, 20 episodes |
| §6.1.4, §6.4.2 | endpoint vs per-step projection at low budget | DA_20260906 §3 | [`DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md`](../../DA_Result_Curated_MD/Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md) | 4 seeds × 3 geometries × 2 episodes |
| §5.1.1 `tab:avoiding-geometries`, §5.2.1 | demonstrations satisfying each obstacle-avoidance geometry (0, 1, 2 of 96) | extracted for the figures | [`avoiding_scene.json`](../data/avoiding_scene.json) — built by [`plotting/extract/avoiding_scene.py`](../plotting/extract/avoiding_scene.py) from `config/projection_eval.yaml`, `scripts/eval.py` and the D3IL dataset | 96 demonstrations |
| §6.1.5 | genuine steps confirmed at run time | job 25444 evaluation log | [`18_51_26_eval_meanflow_hardflow_25444.log`](../../../temp/0609/I/2026-09-05/18_51_26_eval_meanflow_hardflow_25444.log) | — |
| §6.1.6 | candidate plans and selection rules | DA_20260827 §2, §6, §10.5 | [`DA_20260827_mpc1_full_seeds_state_avoiding.md`](../../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md) | 5 seeds × 2 episodes (fan study) |
| §6.2.1–§6.2.3 | alignment: models, step budget, projection | Gen14 CLOSURE_20260907 | [`CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md`](../../../logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md) | seed 6, 10 paired contexts |
| §6.2.1 | consistency training vs MeanFlow at K=20 | DA_20260907 Gate 1 | [`DA_20260907_Gen14_Gate1_AF_vs_MF_K20_flagship_KILL.md`](../../../logs_in_develop/Gen14/DA_20260907_Gen14_Gate1_AF_vs_MF_K20_flagship_KILL.md) | seed 6, 10 paired contexts |
| §6.2.3 | endpoint projection at K=10 and K=20 | Gen14 CLOSURE_20260907 R.2, §5 | [`CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md`](../../../logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md) | seed 6, 10 paired contexts |
| §6.3.1 | corridor with slide | U16 DA_20260914 corridor_v2 | [`DA_20260914_corridor_v2_paper_full.md`](../../../logs_in_develop/Gen15/U16/DA_20260914_corridor_v2_paper_full.md) | seed 6, 12 flights |
| §6.3.2 | pillars: model means | Gen15 CLOSURE_20260910 §2 | [`CLOSURE_20260910_uav_engine_ladder_final.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/CLOSURE_20260910_uav_engine_ladder_final.md) | seed 6, 10 flights |
| §6.3.2 | pillars: endpoint vs per-step projection | T3 | [`DA_20260910_T3_hardflow_vs_dpcc_pillars_K5.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260910_T3_hardflow_vs_dpcc_pillars_K5.md) | seed 6, 10 flights |
| §6.3.2 | pillars: diffusion reference | DA_20260912 rev. 2 | [`DA_20260912_pillars_diffusion_baseline_reference.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260912_pillars_diffusion_baseline_reference.md) | seed 6, 10 flights |
| §6.3.3 | s-curve and the tracking controller | Gen15 CLOSURE_20260910 §3, §6; T5 | [`DA_20260909_T5_mjpc_vs_pid_s_curve.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md) | seed 6 |

## Corpora

The batch directories behind these analyses are registered, with their protocols, in
[`../plotting/sources.py`](../plotting/sources.py) (`CORPORA`). `temp/` is a local drop folder and is not
version-controlled.
