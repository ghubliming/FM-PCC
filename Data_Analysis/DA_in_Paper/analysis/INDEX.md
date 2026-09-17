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
| §6.1.1 `tab:avoiding-dpcc-protocol`, `tab:state-headline`; §5.3.1 `tab:target` | every selection rule per model, at DPCC's protocol (5 seeds × 2 episodes) and extended (5 × 20); baseline reproduction | (new, 2026-09-17) | [`avoiding_rules_by_protocol.py`](avoiding_rules_by_protocol.py) — `analysis_results_checkpoint/15-09/batch_avoiding_combined_20260915_100757`; missing rows: [`REQUEST_20260917_avoiding_dpcc_protocol.md`](../plotting/REQUEST_20260917_avoiding_dpcc_protocol.md) | 5 seeds × 2 / × 20 |
| §5.5.1 `tab:seed-spread` | variation across training seeds 6–10: MeanFlow, flow matching, diffusion | (new, 2026-09-16) | [`avoiding_seed_spread.py`](avoiding_seed_spread.py) — per-seed values from `analysis_results_checkpoint/15-09/batch_avoiding_combined_20260915_100757` | 5 seeds × 20 episodes, top-left-hard + top-right-hard |
| §5.5.1 `tab:tiers` | baseline at 10 vs 100 episodes per geometry | DA_20260819 (n=2 vs n=20) | [`DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md`](../../DA_Result_Curated_MD/DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md) | 5 seeds × 2 / × 20 |
| §6.1.1 `tab:state-fm`, matched-budget control | flow matching vs baseline; the 1.4× × 15× split | DA_20260819 §1, §1b | [`DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md`](../../DA_Result_Curated_MD/DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md) | 5 seeds × 20 episodes |
| §6.1.2 `tab:state-lowk` | cheapest full-success configuration per model | Report_20260903 §1 | [`README.md`](../../DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) | seed 6, 20 episodes |
| §6.1.3 `fig:raw-plans` | plans without projection, 6.0 vs 28.0 violations | Report_20260819_MF_UNet §7 | [`README.md`](../../DA_Result_Curated_MD/Report_20260819_MF_UNet/README.md) | seed 6, both-hard |
| §6.1.3 | goal reached without projection, K=1 | Report_20260903_AF_UNet §7 | [`README.md`](../../DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) | seed 6, 20 episodes |
| §6.1.4, §6.4.2 | endpoint vs per-step projection at low budget | DA_20260906 §3 | [`DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md`](../../DA_Result_Curated_MD/Proposal_20260905_HF_minK_mf_af_unet/DA_20260906_hf_minK_mfunet_A1_K2_K3_K5.md) | 4 seeds × 3 geometries × 2 episodes |
| §5.1.1 `tab:avoiding-geometries`, §5.2.1 | demonstrations satisfying each obstacle-avoidance geometry (0, 1, 2 of 96) | extracted for the figures | [`avoiding_scene.json`](../data/avoiding_scene.json) — built by [`plotting/extract/avoiding_scene.py`](../plotting/extract/avoiding_scene.py) from `config/projection_eval.yaml`, `scripts/eval.py` and the D3IL dataset | 96 demonstrations |
| §6.1.5 | genuine steps confirmed at run time | job 25444 evaluation log | [`18_51_26_eval_meanflow_hardflow_25444.log`](../../../temp/0609/I/2026-09-05/18_51_26_eval_meanflow_hardflow_25444.log) | — |
| §6.1.6 | candidate plans and selection rules | DA_20260827 §2, §6, §10.5 | [`DA_20260827_mpc1_full_seeds_state_avoiding.md`](../../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md) | 5 seeds × 2 episodes (fan study) |
| §6.2 `tab:va-models`, §6.2.1–6.2.3 | alignment: MeanFlow vs flow matching vs diffusion vs consistency training, unprojected, paired over 10 contexts; budget ladder | CLOSURE_20260907 §1, §3, R.1–R.4 | [`va_results.py`](va_results.py) — recomputes every quoted number from `analysis_results_checkpoint/15-09/batch_va2_20260915_100754` (same 14,102 rollouts as [`CLOSURE_20260907`](../../../logs_in_develop/Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md)) | seed 6, 10 **training** contexts |
| §6.2.2 `tab:va-projection` (v3.26) | alignment: endpoint vs per-step projection, MeanFlow K10/K20 tightened, all rules — violation-free contexts, violating steps, moved, median distance, ms/step; diffusion per-step untightened | CLOSURE_20260907 §2, R.2 | [`va_results.py`](va_results.py) block *projection methods* | seed 6, 10 training contexts |
| §6.3 `tab:uav-corridor` (`sec:res:uav:corridor`, `sec:res:uav:projection`) | corridor v2: passed / collision-free / S&C at K1/K3/K5 + diffusion; endpoint vs per-step | DA_20260914 corridor v2 §3–§7 | [`uav_results.py`](uav_results.py) (tag `u17cv2`) + [`DA_20260914_corridor_v2_paper_full.md`](../../../logs_in_develop/Gen15/U16/DA_20260914_corridor_v2_paper_full.md) | seed 6, 12 flights (4 per route) |
| §6.3 `tab:uav-pillars`, `tab:uav-pillars-best` (`sec:res:uav:pillars`) | pillars: S&C (**goal passed + collision-free**, v3.28) per projection configuration for all four models, full constraint set; best projected configuration per model; 7-configuration budget means; physical safety by method | CLOSURE_20260910 §2, DA_20260912 | [`uav_results.py`](uav_results.py) (tag `u7hg`) + [`CLOSURE_20260910`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/CLOSURE_20260910_uav_engine_ladder_final.md), [`DA_20260912`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260912_pillars_diffusion_baseline_reference.md) | seed 6, 10 flights |
| §6.3 `tab:uav-scurve`, `tab:uav-controller` (`sec:res:uav:scurve`, `sec:res:uav:controller`) | s-curve unprojected listing; caveat: tracking controller (MuJoCo MPC vs brake-to-rest) | CLOSURE_20260910 §3, DA_20260909_T5 | [`uav_results.py`](uav_results.py) + [`DA_20260909_T5`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md) (endpoint-projection rows invalid, pre-fix) | seed 6, 10 flights; 3 with MPC |

## UAV success criterion (changed 2026-09-17, v3.28)

The thesis scores an aerial flight by whether it **passes the goal** — crosses the finish line of its scene —
which is the success criterion of D3IL-avoiding transferred to the air, reported together with collision-free
flight as S&C. The strict *within 0.30 m of the route's goal point* criterion (`n_success`,
`n_success_and_constraints`) is **no longer reported**: on scenes whose constraints push a route off its
nominal end point it penalises flights that solved the task. `uav_results.py` prints the crossing criterion
everywhere; the columns remain in the CSV.

**Consequence, recorded here because it changes a headline:** on UAV-pillars the ordering of the two leading
models swaps. At the strict criterion MeanFM led (mean 0.618 against FM 0.327); at the goal-passed criterion
FM leads (mean 0.718, ceiling 1.00, 8 of 11 configurations ≥ 0.8) against MeanFM (0.627, ceiling 0.90, 5 of
11), while MeanFM finishes in fewer control steps (442 against 562 at each model's best projected
configuration). Older UAV analyses in `logs_in_develop/Gen15/` and `DA_Result_Curated_MD/` report the strict
criterion and are not rewritten; read their numbers against this note.

## Corpora

The batch directories behind these analyses are registered, with their protocols, in
[`../plotting/sources.py`](../plotting/sources.py) (`CORPORA`). `temp/` is a local drop folder and is not
version-controlled.
