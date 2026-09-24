# DA — 2026-09-24 · D3IL-avoiding at twenty episodes per seed: what is complete, what is dropped, and how it compares with the protocol of DPCC

**Request (author, 2026-09-24):** the twenty-episode (`n_trials 20`) runs cannot be completed at K20 in our
setup, and the K20 diffusion run is badly incomplete. Keep only the twenty-episode data that is really
correct, put it at the end of the appendix as the feasible twenty-episode tests, compare it with the
two-episode results of the main text, and add a statistical analysis. Do not touch `v3/chapters/09_appendix.tex`.

**Deliverables (this folder):**
- [`app_ntrial20_feasible.tex`](app_ntrial20_feasible.tex) — the new appendix section, standalone (§9 says where it goes).
- [`ntrial20_da.py`](ntrial20_da.py) — census, tables and statistics. Stdlib only.
- [`ntrial20_cells.csv.gz`](ntrial20_cells.csv.gz) — the 12 759 rows it uses. The script re-runs from this file when the batch is absent, with identical output (checked).
- [`ntrial20_results.json`](ntrial20_results.json) — every number below, machine-readable.

**Inputs:**
- Result data: `temp/23-09/batch_avoiding_combined_20260923_110010(TenpK2D)/candidates_multidimensional_raw.csv` (long format; the batch the author named).
- File tree: `Data_Analysis/analysis_results_checkpoint/23-09-23-21-Berlin-Time-logs_tree.txt` (cluster `logs/`, captured 2026-09-23 23:21).
- Main-text numbers: `v3/chapters/06_results.tex` (`tab:avoiding-dpcc-protocol`, `tab:avoiding-raw-models`, `sec:res:ablations`) and `v3/chapters/09_appendix.tex` (the dead twenty-episode block and `tab:app:avoiding-dpcc-full`).

**Aggregation:** the chapter's convention (`avoiding_unprojected_and_budget20.py`). The point estimate is the mean per
geometry over the seeds, then the mean over the geometries. The spread is the sample SD over the per-seed geometry
means. On complete cells the point estimate is also the pooled rate, so counts are exact: S&C count = Σ rate × episodes.

**DA Target (standing rule):** DPCC K20 / `aw10` / GaussianDiffusion, `dpcc-c-tightened`. Of record at two episodes:
**30/30, 70.1 steps, 553.4 ms/step** (candidate 17). At twenty episodes it exists on only two geometries (§6).

---

## 0 · Verdict

| | campaign (20 episodes/seed) | verdict | why |
| :-- | :-- | :-- | :-- |
| ✅ | **MeanFM (U-Net) K1, K2** | **thesis section** | 5 seeds × 3 geometries × 20 episodes, all 13 variants finished. The checkpoints are those of the chapter's two-episode rows. |
| ✅ | **FM K1, K2** | **thesis section** | same |
| 🟡 | MeanFM K5 | DA only | complete; its two-episode twin is an appendix row, not a chapter row |
| 🟡 | FM K5 | DA only | complete; no two-episode twin on five seeds (seed 6 only) |
| 🟡 | MeanFM K10 | DA only | the reported cells are complete, but the run was cut on seed 10 / both-hard before 5 endpoint-projection variants finished |
| ❌ | **Diffusion K20** | dropped | both-hard: seed 6 only under `r`/`c`, **none** under `t` and unprojected. Tree: **562 files** against 4706 for a complete run |
| ❌ | FM K20 | dropped | both-hard 3/5 seeds (`r`,`c`), 2/5 (`t`, unprojected) |
| ❌ | MeanFM K20 | dropped | top-left 2/5, both-hard 0/5. It also has no U-Net two-episode twin (the two-episode K20 MeanFlow row is `bbmf_dit`) |
| ❌ | "AlphaFlow" K1, K2 `_msg20trials` | dropped | complete, but it is the **wrong model**: `bbsit` with α annealed to **0** (`ae0.0`), not the thesis's U-Net CI-MeanFM with α_end = 0.2 |
| ❌ | CI-MeanFM K1–K20 `_msgafon02_s6` | dropped | right model, but **one training seed** (seed 6) |

**Result.**
- The two protocols agree on every kept cell.
  - **16/16** two-episode S&C counts lie inside the central 95 % range of what a two-episode evaluation returns when its episodes are drawn from the twenty.
  - The rule effects of `sec:res:ablations` reproduce to within 1 step (the MeanFM stall under `c`: 72.0/98.0 against 72.4/97.2).
  - Control steps differ by at most 2.4 per configuration.
- What twenty episodes add is resolution.
  - MeanFM's `t` beats `r`/`c` on S&C by 15–20 of 300 episodes; at two episodes the rules were within 1–2 of 30.
  - FM's three rules, all 30/30 at two episodes, separate to 299/299/297 (K1) and 299/300/295 (K2).

**Watch items for the author (§5):**
1. MeanFM's single two-episode failure under `t` (seed 7, top-right-hard, both K1 and K2) is **not reproduced** by the twenty-episode run of the same checkpoint and seeding. That run has 20/20 there.
2. CI-MeanFM at twenty episodes exists only on seed 6. There its K1 `t` reads **57/60**, while the chapter prints 30/30 on five seeds.

## 1 · Census — every twenty-episode campaign in the batch

File counts come from the tree. A complete flow campaign has **4706 files**: 5 seeds × 910, where 900 = 3 geometries × 300,
plus 156 `all_seeds` plots. Cells are counted from the CSV. The episode count is checked by requiring every success rate to be
k/20; every twenty-episode campaign also has rates off the 1/2 grid, so none of them is a mislabeled two-episode run.

| id | cand. | folder (under `logs/avoiding-d3il/plans/`) | tree | CSV cells (of 15): unproj / r / c / t | verdict |
| :-- | --: | :-- | :-- | :-- | :-- |
| MF1_20 | 168 | `flow_matching_v3_meanflow/…objmeanflow_bbunet…dp0.5/H8_K1_…MeanFlowODE_msg20trials` | 4706 | 15/15/15/15 | ✅ section |
| MF2_20 | 172 | same, `H8_K2_…` | 4706 | 15/15/15/15 | ✅ section |
| MF5_20 | 180 | same, `H8_K5_…` | 4706 | 15/15/15/15 | 🟡 DA only |
| MF10_20 | 165 | same, `H8_K10_…` | 4552 (seed 10: 798/900) | 15/15/15/15 | 🟡 DA only |
| MF20_20 | 169 | same, `H8_K20_…` | 2212 | 8/7/7/7 | ❌ |
| FM1_20 | 188 | `flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/H8_K1_…_msg20trials` | 4706 | 15/15/15/15 | ✅ section |
| FM2_20 | 195 | same, `H8_K2_…` | 4706 | 15/15/15/15 | ✅ section |
| FM5_20 | 198 | same, `H8_K5_…` | 4706 | 15/15/15/15 | 🟡 DA only |
| FM20_20 | 193 | same, `H8_K20_…` | 3877 (seed 8: 727; seeds 9–10: 600) | 12/13/13/12 | ❌ |
| — | 184 | a second `H8_K1_…FlowMatchingODE_msg20trials` (seed 6, 6 files, 2026-08-18) | 6 | no metric rows | ❌ stray |
| AF1_20 | 45 | `flow_matching_v3_alphaflow/…AlphaFlowODE_aw10_bbsit…ae0.0…/H8_K1_…_msg20trials` | 4706 | 15/15/15/15 | ❌ wrong model |
| AF2_20 | 48 | same, `H8_K2_…` | 4706 | 15/15/15/15 | ❌ wrong model |
| CI*_20s6 | 64/67/69/63/66 | `…AlphaFlowODE_aw10_bbunet…ae0.2…/H8_K{1,2,5,10,20}_…_msgafon02_s6` | — | 3/3/3/3 each | ❌ one seed |
| D20_20 | 11 | `diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10/H8_K20_T0.5_…_msg20trials` | **562** | **10/11/11/10** | ❌ |

**The diffusion K20 run in the tree:**
- Seeds 7–10 hold two geometry folders each (80 files). Seed 6 holds three (96 files); its both-hard folder has 16 files instead of 40.
- `all_seeds/` lists only top-left-hard and top-right-hard.
- The CSV agrees: both-hard has seed 6 only, under `r`/`c`.
- The old appendix's baseline row, **0.983 / 69.0 / 563.5**, is exactly this: the per-geometry mean over 5 + 5 + 1 seeds. It gives one seed the same weight as five.

**The two-episode twins** (chapter rows), all 15/15 on every variant, with every rate k/2:

| id | cand. | folder | used for |
| :-- | --: | :-- | :-- |
| MF1_2, MF2_2 | 167, 171 | `…objmeanflow_bbunet…/H8_K{1,2}_Meuler_T0.5_A0.5_B1_…MeanFlowODE` | `tab:avoiding-dpcc-protocol`, `tab:avoiding-raw-models` |
| MF5_2, MF10_2 | 179, 164 | same, K5/K10 | `tab:app:avoiding-dpcc-full` |
| FM1_2, FM2_2 | 189, 196 | `…FlowMatchingODE_a1.5_b1.0_aw10/H8_K{1,2}_…_msgdpccproto` | same chapter tables |
| CI1_2, CI2_2 | 65, 68 | `…ae0.2…/H8_K{1,2}_…_msgdpccproto` | same chapter tables (no twenty-episode twin) |
| D20_2 | 17 | `diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` | the Target |

The script reproduces every chapter cell of these rows to the last printed digit: all 12 per-step MeanFM/FM cells, the 4 unprojected ones, and the K5/K10 appendix rows.

## 2 · Is it the same configuration? Checkpoints, code, seeding

| | checkpoint (`state_best.pt`, tree mtime) | 2-episode eval | 20-episode eval |
| :-- | :-- | :-- | :-- |
| MeanFM (U-Net) | seeds 6–10: 2026-08-06 … 08-10 | 2026-08-11 | 2026-08-13/14 |
| FM | seeds 6–10: 2026-05-06/07 | 2026-09-17 | 2026-08-18/19 |

- **Checkpoint:** every seed folder holds only `state_best.pt` (no numbered states). The plan blocks
  `plan_fm_v3_meanflow` and `plan_fm_v3_ode_selectable` pass `'diffusion_epoch': 'best'` to the loader
  (`epoch=args.diffusion_epoch`). Checked in the config at `eebdeb94` (before 08-11) and `8e8b1ca5` (before 08-18).
  The 2026-08-20 commit `1ce49201` changed only the loaders' default argument from `latest` to `best`, which these calls
  override. Both protocols load the same file.
- **Code, MeanFM (08-11 → 08-13):** `git diff eebdeb94 e58690dc` over config and eval shows one change, the `_msg<tag>`
  results-path machinery. Plan block, fan (`batch_size` 4), projector and sampler are identical.
- **Code, FM (08-18/19 → 09-17):** `git diff 8e8b1ca5 939affbf` over the FM eval path shows three changes:
  - the eval script gained a `--flow-steps` override (K is also in the folder name, so it is the same K);
  - the loader default became `best` (overridden anyway);
  - the fan became env-driven (`FMPCC_MPC_BATCH`, default 4, the same value).

  `diffuser/sampling/` and `diffuser/models/` are unchanged. No functional change in the per-step arms, so the 5–13 %
  lower time per step in the September run (§4.3) is the run environment.
- **Episodes:** episode *i* seeds the sampling noise with `torch.manual_seed(i)`
  (`eval_flow_matching_v3_meanflow.py:626`, `eval_flow_matching_v3_ode_selectable.py:309`). The avoiding reset takes
  no seed (`env.reset()`, `:629`), but `AvoidingEnv._reset_env` beams the robot to the fixed `init_qpos`.
  So the two-episode run is, by design, episodes 0–1 of the twenty-episode run. §5 tests this.

## 3 · The kept cells, twenty against two episodes

Per-step projection, tightened constraints, four candidate plans, temporal U-Net of 4.0 M parameters.
S&C as count and as mean ± seed SD. Steps and ms/step as mean ± seed SD.

| model | K | rule | S&C 20 ep | S&C 2 ep | steps 20 | steps 2 | ms 20 | ms 2 |
| :-- | --: | :-: | :-- | :-- | --: | --: | --: | --: |
| MeanFM | 1 | r | 279/300 (0.930 ± 0.032) | 27/30 (0.900 ± 0.149) | 64.7 ± 1.8 | 64.3 ± 2.6 | 18.4 ± 0.1 | 19.2 ± 1.6 |
| | | c | 283/300 (0.943 ± 0.040) | 28/30 (0.933 ± 0.091) | 72.0 ± 4.1 | 72.4 ± 8.4 | 18.0 ± 0.4 | 18.3 ± 0.5 |
| | | **t ▶** | **298/300 (0.993 ± 0.015)** | **29/30 (0.967 ± 0.075)** | **61.0 ± 1.0** | **58.6 ± 2.2** | **18.1 ± 0.3** | **18.3 ± 0.2** |
| MeanFM | 2 | r | 280/300 (0.933 ± 0.041) | 29/30 (0.967 ± 0.075) | 65.9 ± 1.4 | 65.1 ± 2.6 | 27.3 ± 0.2 | 27.6 ± 0.5 |
| | | c | 278/300 (0.927 ± 0.030) | 28/30 (0.933 ± 0.091) | 98.0 ± 10.4 | 97.2 ± 4.8 | 26.8 ± 0.4 | 27.1 ± 0.2 |
| | | t | 298/300 (0.993 ± 0.009) | 29/30 (0.967 ± 0.075) | 60.4 ± 0.9 | 59.4 ± 2.0 | 27.1 ± 0.3 | 27.7 ± 0.5 |
| FM | 1 | r | 299/300 (0.997 ± 0.007) | 30/30 | 68.3 ± 0.9 | 67.9 ± 1.2 | 19.4 ± 0.6 | 17.4 ± 0.1 |
| | | c | 299/300 (0.997 ± 0.007) | 30/30 | 67.5 ± 1.2 | 67.0 ± 0.8 | 18.9 ± 0.2 | 17.3 ± 0.1 |
| | | t | 297/300 (0.990 ± 0.022) | 30/30 | 67.1 ± 1.6 | 68.2 ± 1.7 | 20.1 ± 1.9 | 17.7 ± 0.6 |
| FM | 2 | r | 299/300 (0.997 ± 0.007) | 30/30 | 68.0 ± 1.6 | 65.7 ± 0.8 | 27.6 ± 0.5 | 26.2 ± 0.9 |
| | | c | 300/300 (1.000 ± 0.000) | 30/30 | 68.4 ± 3.0 | 67.0 ± 3.8 | 27.3 ± 0.3 | 25.5 ± 0.2 |
| | | t | 295/300 (0.983 ± 0.020) | 30/30 | 64.9 ± 1.9 | 63.8 ± 1.9 | 27.9 ± 0.5 | 25.9 ± 0.2 |

Unprojected (`diffuser` arm):

| model | K | success 20 / 2 | S&C 20 / 2 | violating steps 20 / 2 | steps 20 / 2 | ms 20 / 2 |
| :-- | --: | :-- | :-- | :-- | :-- | :-- |
| MeanFM | 1 | 297/300 · 30/30 | 36/300 · 1/30 | 14.5 ± 0.7 · 15.5 ± 2.1 | 61.9 · 63.4 | 9.6 · 10.3 |
| MeanFM | 2 | 299/300 · 30/30 | 36/300 · 1/30 | 15.4 ± 0.9 · 15.5 ± 0.4 | 63.6 · 62.7 | 18.7 · 19.3 |
| FM | 1 | 297/300 · 30/30 | 15/300 · 3/30 | 17.3 ± 1.0 · 16.5 ± 3.4 | 66.5 · 68.6 | 10.3 · 9.0 |
| FM | 2 | 299/300 · 30/30 | 30/300 · 3/30 | 16.4 ± 1.3 · 15.6 ± 2.6 | 63.3 · 61.7 | 19.2 · 17.8 |

**Where the failures sit** (S&C per geometry, twenty episodes of 100 · two episodes of 10):
- MeanFM `t`: top-left 100 · 10 at both K. Top-right 99 · 9 (K1) and 98 · 9 (K2). Both-hard 99 · 10 and 100 · 10.
- MeanFM `r`/`c`: both-hard is the weak geometry, 81–86/100.
- FM: every per-step cell is ≥ 97/100. FM `t` loses its episodes on top-right (K1) and top-left + top-right (K2).

## 4 · Statistics — the two protocols against each other

The thesis section prints counts, seed spreads, the two-episode range and two probabilities, but no test names and no
p-values (thesis-prose rule, 2026-09-17). Everything else stays here.

### 4.1 Success with constraint satisfaction

- *Range*: the central 95 % of the exact distribution of the count, of 30, that a two-episode evaluation returns when its
  two episodes per seed and geometry are drawn without replacement from the twenty (hypergeometric per cell, convolved over
  the 15 cells).
- *Wilson*: the 95 % score interval on the pooled count.
- *Fisher*: two-sided, 2 × 2 on the pooled counts. By design the episodes overlap (§2), so the two samples are not
  independent and this test is conservative; read the *Range* column first.

| cell | 20 ep [Wilson] | 2 ep [Wilson] | range | P(30/30) | 2-ep count inside? | Fisher p |
| :-- | :-- | :-- | :-: | --: | :-: | --: |
| MeanFM K1 r | 279/300 [0.895, 0.954] | 27/30 [0.744, 0.965] | 25–30 | 0.09 | ✅ | 0.47 |
| MeanFM K1 c | 283/300 [0.911, 0.964] | 28/30 [0.787, 0.982] | 26–30 | 0.14 | ✅ | 0.69 |
| **MeanFM K1 t** | 298/300 [0.976, 0.998] | 29/30 [0.833, 0.994] | 29–30 | **0.81** | ✅ | 0.25 |
| MeanFM K2 r | 280/300 [0.899, 0.956] | 29/30 [0.833, 0.994] | 25–30 | 0.11 | ✅ | 0.71 |
| MeanFM K2 c | 278/300 [0.891, 0.951] | 28/30 [0.787, 0.982] | 25–30 | 0.09 | ✅ | 1.00 |
| MeanFM K2 t | 298/300 [0.976, 0.998] | 29/30 [0.833, 0.994] | 29–30 | 0.81 | ✅ | 0.25 |
| FM K1 r | 299/300 [0.981, 0.999] | 30/30 [0.886, 1.000] | 29–30 | 0.90 | ✅ | 1.00 |
| FM K1 c | 299/300 [0.981, 0.999] | 30/30 [0.886, 1.000] | 29–30 | 0.90 | ✅ | 1.00 |
| FM K1 t | 297/300 [0.971, 0.997] | 30/30 [0.886, 1.000] | 29–30 | 0.72 | ✅ | 1.00 |
| FM K2 r | 299/300 [0.981, 0.999] | 30/30 [0.886, 1.000] | 29–30 | 0.90 | ✅ | 1.00 |
| FM K2 c | 300/300 [0.987, 1.000] | 30/30 [0.886, 1.000] | 30 | 1.00 | ✅ | 1.00 |
| **FM K2 t** | 295/300 [0.962, 0.993] | 30/30 [0.886, 1.000] | 28–30 | **0.58** | ✅ | 1.00 |
| MeanFM K1 unproj. | 36/300 [0.088, 0.162] | 1/30 [0.006, 0.167] | 1–7 | 0 | ✅ | 0.23 |
| MeanFM K2 unproj. | 36/300 [0.088, 0.162] | 1/30 [0.006, 0.167] | 1–7 | 0 | ✅ | 0.23 |
| FM K1 unproj. | 15/300 [0.031, 0.081] | 3/30 [0.035, 0.256] | 0–4 | 0 | ✅ | 0.22 |
| FM K2 unproj. | 30/300 [0.071, 0.139] | 3/30 [0.035, 0.256] | 1–6 | 0 | ✅ | 1.00 |

**16/16 inside the range, no Fisher p below 0.2.** Goal reached without projection: 297–299/300 against 30/30, with P(30/30) of 0.72–0.90.

**Resolution, the thing twenty episodes buy.** At two episodes the seed SD of the operating point is 0.075. That is one
lost episode in one seed, i.e. episode noise. At twenty it is 0.015. The rule effect on S&C is resolved only at twenty:

| comparison (20 ep) | counts | Fisher p | at 2 ep |
| :-- | :-- | --: | :-- |
| MeanFM K1 `t` vs `r` | 298 vs 279 /300 | < 0.0001 | 29 vs 27 /30 |
| MeanFM K1 `t` vs `c` | 298 vs 283 /300 | 0.0006 | 29 vs 28 /30 |
| MeanFM K2 `t` vs `c` | 298 vs 278 /300 | < 0.0001 | 29 vs 28 /30 |
| FM K2 `c` vs `t` | 300 vs 295 /300 | 0.06 | 30 vs 30 /30 |

So the chapter's choice of `t` for MeanFM is confirmed on S&C, not only on steps. For FM the rules stay within 5 of 300.

### 4.2 Control steps

"2-ep s.e." is the sampling s.e. of the chapter's step mean under a two-episode protocol, from the twenty-episode per-cell
SDs with finite-population correction; it is approximate, since steps are averaged over successful episodes. "Seed-paired"
is the t-interval (df 4) of the per-seed differences, 2 ep − 20 ep.

| cell | 20 → 2 ep | diff | 2-ep s.e. | z | seed-paired diff [95 %] | seeds shorter at 2 ep |
| :-- | :-- | --: | --: | --: | :-- | :-: |
| MeanFM K1 r / c / t | 64.7→64.3 / 72.0→72.4 / 61.0→58.6 | −0.4 / +0.4 / −2.4 | 1.09 / 1.90 / 1.42 | −0.4 / +0.2 / −1.7 | [−2.8,+2.0] / [−6.0,+6.8] / [−5.0,+0.2] | 3 / 2 / 4 |
| MeanFM K2 r / c / t | 65.9→65.1 / 98.0→97.2 / 60.4→59.4 | −0.8 / −0.8 / −1.0 | 1.21 / 5.20 / 1.23 | −0.7 / −0.2 / −0.8 | [−2.5,+0.8] / [−9.9,+8.3] / [−3.2,+1.2] | 4 / 3 / 3 |
| FM K1 r / c / t | 68.3→67.9 / 67.5→67.0 / 67.1→68.2 | −0.3 / −0.5 / +1.1 | 0.56 / 0.53 / 0.62 | −0.6 / −0.9 / +1.8 | [−1.4,+0.7] / [−2.5,+1.5] / **[+0.2,+2.0]** | 4 / 2 / 0 |
| FM K2 r / c / t | 68.0→65.7 / 68.4→67.0 / 64.9→63.8 | **−2.3** / −1.4 / −1.1 | 0.74 / 0.82 / 0.89 | **−3.1** / −1.7 / −1.2 | **[−4.2,−0.4]** / [−5.3,+2.5] / [−3.9,+1.8] | **5** / 3 / 4 |
| unproj. MF1 / MF2 / FM1 / FM2 | 61.9→63.4 / 63.6→62.7 / 66.5→68.6 / 63.3→61.7 | +1.5 / −0.9 / +2.1 / −1.6 | 0.94 / 1.82 / 1.38 / 0.54 | +1.6 / −0.5 / +1.5 / **−2.9** | [−4.9,+8.0] / [−4.9,+3.1] / [−2.5,+6.6] / **[−2.4,−0.8]** | 2 / 2 / 1 / **5** |

- MeanFM agrees within sampling error on every cell: |z| ≤ 1.7, and every seed-paired interval contains 0.
- FM disagrees on three cells: K2 `r` and K2 unprojected are ~2 steps shorter on all five seeds, and K1 `t` is 1.1 longer.
- Across 16 comparisons a few |z| near 2–3 are expected. The z-scores treat the two episodes as a random pair, but
  episode *i* is seeded by *i* (§2). So the two-episode protocol always evaluates episodes 0 and 1 of every cell: a
  fixed subsample, not a random one.
- A shift shared by all five seeds, like FM K2 `r`, can come from those two noise seeds rather than from sampling
  error. The FM code is functionally unchanged between the two runs (§2).
- Largest difference anywhere: 2.4 steps. None changes an ordering the chapter reports.

### 4.3 Time per control step

| | ratio 2 ep / 20 ep (grand mean) | per seed |
| :-- | :-- | :-- |
| MeanFM, projected | 1.01–1.04 | 0.97–1.14 |
| MeanFM, unprojected | 1.04 (K2), 1.07 (K1) | 1.03–1.13 |
| FM, projected | **0.88–0.95** | 0.75–0.98 |
| FM, unprojected | 0.87 (K1), 0.93 (K2) | 0.85–0.95 |

MeanFM's two evaluations ran two days apart and agree within 4 % (projected). FM's ran a month apart and are 5–13 % faster in September. That is within the ~10–20 % contention band the Pareto rule tolerates. No ms/step claim of the chapter moves.

### 4.4 Rule ordering and the stall under cumulative projection cost

Ranking: S&C first, then fewer steps.

| model K | 20 ep | 2 ep | best rule |
| :-- | :-- | :-- | :-- |
| MeanFM 1 | t > c > r | t > c > r | same |
| MeanFM 2 | t > r > c | t > r > c | same |
| FM 1 | c > r > t | c > r > t | same |
| FM 2 | c > r > t | t > r > c (all 30/30, ordered by steps) | differs, S&C level at 2 ep |

Steps under `c` minus steps under `t`:

| | 20 ep | 2 ep |
| :-- | :-- | :-- |
| MeanFM K1 | +11.0, seed-paired [+6.5, +15.6], 5/5 seeds | +13.8 |
| MeanFM K2 | +37.6, [+24.0, +51.2], 5/5 seeds | +37.8 |
| FM K1 | +0.4 | −1.2 |
| FM K2 | +3.5 | +3.2 |

The stall behind `sec:res:ablations` is a property of the model, not of the two-episode sample.

## 5 · Watch items — not in the thesis section

### 5.1 MeanFM, seed 7, top-right-hard: the chapter's one lost episode is not reproduced

The subset test checks whether the two-episode counts can be episodes 0–1 of the twenty. It needs n2 successes ≤ n20
successes and n2 failures ≤ n20 failures, per seed and geometry.

- It passes on **every FM cell** (K1, K2; per-step and unprojected).
- For MeanFM it fails at exactly one seed–geometry: **seed 7, top-right-hard**. That cell fails
  - under `t` at K1 and K2 (2 ep 1/2, 20 ep 20/20);
  - on goal-reaching under `t` at K5, and under `r` at K5 and K10.
- The chapter's MeanFM `t` count, 29/30 at both K1 and K2, owes its single failure to this cell.
- Raw values, K1 `t`, seed 7 / top-right-hard:
  - 2 ep: `n_success` 0.5, `n_steps` 60.5 ± 5.5 over the two episodes (neither on the step cap).
  - 20 ep: `n_success` 1.0, `n_steps` 69.75 ± 23.6.
- What is ruled out: the two evaluations share the checkpoint (`state_best.pt` of 2026-08-09 for seed 7). They also share
  the code: the only config change between 08-11 and 08-13 is `e58690dc`, the results-path token, checked by `git diff`.
  And they share the seeding (§2).
- The anomaly repeats in **four separate two-episode folders** (K1, K2, K5, K10), always seed 7 / top-right-hard. That
  points to something systematic in the 08-11 evaluations of that cell, not to a one-off floating-point flip.
- Each of those folders holds **6 config snapshots for seeds 7–10** (two evaluation runs), against 3 for seed 6 and 3 for
  every twenty-episode seed. A top-right-hard folder left over from the earlier of the two runs would produce exactly
  this pattern.
- It is not established. What would settle it, as a download and an `ls`, no run:
  - the per-geometry mtimes of `…/H8_K{1,2,5,10}_Meuler_T0.5_A0.5_B1_…MeanFlowODE/7/results/halfspace_*`;
  - the episode 0–1 trajectories of `halfspace_top-right-hard/dpcc-t-tightened.npz` against the `_msg20trials` twin.
- **Effect on the thesis:** if the failure is not reproducible, MeanFM `t` reads 30/30 at the protocol of DPCC. Either value
  lies inside the twenty-episode range (29–30), so no statement of the section changes.
- The ledger line in `data_status/PENDING_20260922_all_lacking_runs.md` §12 ("the first two episodes of the staged
  20-episode artefact are the two the DPCC protocol runs") holds by design, but the data contradict it for this one cell.

### 5.2 CI-MeanFM has no complete twenty-episode evaluation, and its one seed reads lower at K1

- `_msgafon02_s6`, seed 6, is the same checkpoint as the chapter's seed 6: `state_best.pt` is dated 2026-09-02 01:52,
  before both evaluations (09-02 and 09-18).
- At twenty episodes, K1 `t` is **57/60**: top-left 20, top-right 17, both-hard 20. K2 `t` is **60/60**.
- Seed 6 at two episodes is 6/6 for both. Given 57/60, P(6/6) = 0.72, so the two-episode value is unremarkable.
- But the chapter's CI-MeanFM K1 `t` (30/30 on five seeds, the row that "clears the baseline on every axis at once") has no
  twenty-episode support. The one seed that has it shows 3 losses on top-right.
- R25 (CI-MeanFM, 5 seeds × 20 episodes) would settle this. The author ruled it out on cost (ledger §10). Recorded, not requested.

### 5.3 Also noted

The chapter's `tab:avoiding-projectors` already discloses that its CI-MeanFM K10 cell is a twenty-episode run (v3.80). Nothing new.

## 6 · DA Target check

**Of record (two episodes):** DPCC K20 `aw10` `dpcc-c-tightened`, 30/30, 70.1 steps, 553.4 ms. The chapter's verdicts
against it stand; this DA adds nothing at that protocol.

**Matched diagnostic at twenty episodes.** On top-left-hard + top-right-hard the baseline has all five seeds. That gives
5 × 2 × 20 = 200 episodes per row, **the same seeds, geometries and episode count on both sides**.
⚠️ Two geometries only, and both-hard is missing for the baseline. This is a DA diagnostic, not for the thesis: the
author excluded the K20 twenty-episode data.

| row | S&C /200 | steps | ms/step | top-left /100 | top-right /100 | vs Target |
| :-- | :-- | --: | --: | :-- | :-- | :-- |
| Diffusion K20 `r` | 195 | 78.9 ± 8.3 | 573.7 | 100 | 95 | — |
| **Diffusion K20 `c` (Target)** | **195** | **73.8 ± 11.3** | **537.4** | 100 | 95 | — |
| Diffusion K20 `t` | 192 | 82.8 ± 14.7 | 564.3 | 100 | 92 | — |
| **MeanFM K1 `t`** | **199** | **62.1 ± 1.8** | **18.0** | 100 | 99 | **dominates** (S&C ≥, −11.7 steps, 30×) |
| MeanFM K2 `t` | 198 | 61.6 ± 1.0 | 27.3 | 100 | 98 | dominates |
| MeanFM K1 `c` | 198 | 71.8 ± 4.7 | 17.7 | 100 | 98 | dominates |
| FM K1 `c` | 199 | 68.5 ± 1.6 | 18.6 | 100 | 99 | dominates |
| FM K2 `c` | 200 | 68.6 ± 2.2 | 26.9 | 100 | 100 | dominates |
| FM K1 `t` | 197 | 68.2 ± 2.1 | 20.0 | 100 | 97 | dominates |
| FM K2 `t` | 195 | 65.8 ± 2.0 | 27.8 | 98 | 97 | dominates (S&C equal) |

- **Target reached, architecture-matched** (temporal U-Net, 4.0 M on both sides).
- On the two geometries where the twenty-episode baseline is complete, every flow row strictly Pareto-dominates the baseline's best rule.
- The baseline's own weak spot is top-right-hard: 95/100, in 77.6 steps.
- This matches the storyline (MeanFM Pareto-dominates the baseline) at ten times the episodes. It cannot enter the thesis
  while the both-hard geometry is missing.

**Benchmark hierarchy — MeanFM must also beat FM (twenty episodes, all three geometries):**

| | S&C | steps | ms | verdict |
| :-- | :-- | --: | --: | :-- |
| MeanFM K1 `t` vs FM K1 `c` | 298 vs 299 /300 (p = 1.0) | 61.0 vs 67.5; seed-paired −6.5 [−8.4, −4.6], 5/5 seeds | 18.1 vs 18.9 | step win at S&C level within one episode |
| MeanFM K2 `t` vs FM K2 `c` | 298 vs 300 (p = 0.50) | 60.4 vs 68.4; −8.0 [−11.5, −4.5] | 27.1 vs 27.3 | same |
| MeanFM K1 `t` vs FM K1 `t` (same rule) | 298 vs 297 | 61.0 vs 67.1 | 18.1 vs 20.1 | MeanFM dominates |

By the letter of the Pareto rule the first two lines are trade-offs (one or two fewer S&C episodes in 300). The S&C gap
is not resolvable, while the step gap holds on every seed. CI-MeanFM cannot be checked against FM at twenty episodes (§5.2).

## 7 · DA-only cells

| cell | 20 ep | 2 ep (appendix `tab:app:avoiding-dpcc-full`) |
| :-- | :-- | :-- |
| MeanFM K5 `r` | 278/300 · 68.5 ± 1.2 · 224.7 ms | 26/30 · 69.2 ± 4.8 · 216.6 ms |
| MeanFM K5 `c` | 260/300 · 64.0 ± 1.0 · 224.5 ms | 25/30 · 61.5 ± 3.6 · 209.8 ms |
| MeanFM K5 `t` | 294/300 · 60.8 ± 0.7 · 225.0 ms | 28/30 · 60.6 ± 2.4 · 223.8 ms |
| MeanFM K5 unproj. | 32/300 S&C, 299/300 success | 0/30, 30/30 |
| MeanFM K10 `r` | 272/300 · 70.0 ± 1.1 · 396.0 ms | 25/30 · 73.9 ± 2.9 · 391.7 ms |
| MeanFM K10 `c` | 266/300 · 63.2 ± 0.5 · 384.5 ms | 25/30 · 59.4 ± 1.6 · 373.2 ms |
| MeanFM K10 `t` | 292/300 · 61.0 ± 0.8 · 398.2 ms | 28/30 · 63.6 ± 1.9 · 402.3 ms |
| MeanFM K10 unproj. | 32/300 S&C, 297/300 success | 5/30, 30/30 |
| FM K5 `r` / `c` / `t` | 299 / 300 / 300 of 300 · 72.3 / 67.1 / 65.7 steps · 110 ms | — (seed 6 only) |
| FM K5 unproj. | 34/300 S&C · 75.7 steps · 43.9 ms | — |

- All MeanFM K5/K10 two-episode S&C counts lie inside their twenty-episode ranges.
- At K10 all three per-step rules differ by 2.6–3.9 steps, outside two-episode sampling error (|z| 2.2–3.5):
  `r` +3.9, `c` −3.8 (5/5 seeds), `t` +2.6 (0/5).
- MeanFM `t` loses S&C as K grows: 298 → 294 → 292 of 300 from K1/K2 to K5 to K10.
- Adding any of these rows to the section would need its own chapter anchor. The author asked for low K only.

## 8 · Dropped campaigns over the cells they cover — for the record, never for the thesis

| campaign | covers | `r` | `c` | `t` | unproj. S&C |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Diffusion K20 | 11/15 (both-hard: seed 6) | 214/220 · 73.9 · 581 ms | 215/220 · 69.0 · 564 ms | 192/200 · 82.8 · 564 ms | 12/200 |
| FM K20 | 13/15 | 248/260 · 72.8 · 497 ms | 251/260 · 65.6 · 477 ms | 236/240 · 67.3 · 466 ms | 21/240 |
| MeanFM K20 | 7/15 | 126/140 · 73.0 · 1024 ms | 128/140 · 66.4 · 962 ms | 131/140 · 63.5 · 1004 ms | 12/160 |
| α-Flow SiT α_end 0, K1 | 15/15 | 300/300 · 73.4 | 257/300 · 121.6 | 296/300 · 67.7 | 41/300 |
| α-Flow SiT α_end 0, K2 | 15/15 | 295/300 · 72.2 | 48/300 · 184.3 | 291/300 · 67.4 | 53/300 |
| CI-MeanFM seed 6, K1 / K2 | 3/15 | 58/60 · 54/60 | 56/60 · 55/60 | 57/60 · 60/60 | 3/60 · 4/60 |

- The old dead block printed the first three rows as "0.967/0.983/0.960", "0.960/0.970/0.987" and "0.955". Its CI-MeanFM
  K2/K20 pair came from the one-seed cells.
- The α-Flow SiT row at K2 under `c` stalls (48/300, 184 steps). It is not the thesis model and nothing reads it.

## 9 · The new section and how it enters the draft

**`app_ntrial20_feasible.tex`** — `\section{D3IL-avoiding at Twenty Episodes per Seed}\label{app:avoiding-twenty}`.

Contents:
- Two tables:
  - `tab:app:twenty-projected`: 12 rows, S&C counts at 20 and at 2 episodes, the two-episode range, steps and ms at both.
  - `tab:app:twenty-raw`: 4 rows, success, S&C and range, violating steps.
- A `\guard` that names what is complete and what is not.
- Four short paragraphs: agreement, resolution, steps/time, unprojected.
- One `\dataref`.
- All numbers are from §3–§4. Model names follow the naming table: prose uses the mechanism names, tables use MeanFM/FM.

**The no-statistics rule.** The thesis-prose rule of 2026-09-17 bans p-values, test names and "significant". The author
asked for statistical analysis here, so the section carries:
- counts,
- the seed spread,
- the exact two-episode range,
- two probabilities of a 30/30 result.

It carries no test and no p-value. If even the range and the probabilities are unwanted, trimming the "What twenty episodes resolve" paragraph and the Range columns removes them.

**For v3** (not applied; `09_appendix.tex` untouched):
1. `\input` the file as the last section of Appendix B (before `\chapter{Reproducibility}`). To make it literally the last
   page of the appendix, input it after Reproducibility as a `\chapter`.
2. The labels are new, so it coexists with the dead block `app:avoiding-twenty-episode`. When the dead block is dropped,
   re-point `05_setup.tex:903`, `05_setup.tex:1319` and `06_results.tex:755` to `app:avoiding-twenty`.
3. `05_setup.tex:900–903` says the DPCC twenty-episode reproduction "is kept for the record" in the appendix. The new
   section does not carry it (dropped, §1), so that sentence needs rewording or removal at the same time.
4. `data_status/PENDING_20260922_all_lacking_runs.md` §12 carries the episode-identity claim of §5.1. Worth a line there if the npz check is done.

## 10 · Reproduce

```bash
python3 logs_in_develop/Writing/Working_Space/ntrial20_appendix_20260924/ntrial20_da.py            # reads the batch if present
python3 logs_in_develop/Writing/Working_Space/ntrial20_appendix_20260924/ntrial20_da.py <raw.csv>  # or another copy of it
```

- Without the batch it reads `ntrial20_cells.csv.gz`.
- Blocks A–E print the census, the tables, the statistics, the Target check and the dropped cells.
- `ntrial20_results.json` holds every number.
- Stdlib only (csv, gzip, json, math, statistics).
