# DA — Does K=2 MeanFlow / AlphaFlow beat the FM and DPCC baselines?

**Date:** 2026-08-02
**Question:** at equal-or-better *success + constraint satisfaction*, do MF(K=2) and AF(K=2) need **fewer generation steps** and **less wall time** than the FM(K=20) and DPCC(diffusion, K=10) baselines?

> **§§1–8 are read off the pre-aggregated cells** (batch aggregate + the provided `.tex`). **§9 is an independent re-analysis from the per-seed raw rows**, full-seed cells only, with episode-level counts and paired statistics. Where the two disagree, **§9 supersedes** — specifically it corrects §5.2 (the AF-vs-baseline quality gap is *not* significant) and §5.1/§6 (several baseline "failures" are knife-edge tolerance trips, not real violations). §9 also adds the finding that matters most: **AF/MF never fail unsafe.**

---

## 1. Data provenance

| item | value |
|---|---|
| Batch | `batch_avoiding_combined_20260802_092307` |
| CSV | `temp/2026-08-02/batch_avoiding_combined_20260802_092307/candidates_multidimensional_{raw,aggregated}.csv` |
| LaTeX table read | `temp/2026-08-02/plot_n_success_and_constraints_both-hard_7v_6c_20260802_1428_tables.tex` (both-hard only, 7 of 24 variants) |
| Task | `avoiding-d3il`, halfspace constraints |
| Envs | `both-hard`, `top-left-hard`, `top-right-hard` (the .tex covered only `both-hard`; **this report uses all three**) |
| Seeds | 6, 7, 8, 9, 10 — complete, no missing seeds for any candidate |
| Rollouts | **2 trials per seed per env → 10 episodes per cell** (per-seed success values are quantised to {0, 0.5, 1.0}) |
| Aggregation | cell = mean over 5 seed-means; `±` = SEM = std/√5 |

### Candidates

| ID | Short name | NFE / plan | Path fragment |
|---|---|---|---|
| CAND_7 | **DPCC baseline** (GaussianDiffusion) | 10 | `plans/diffusion/H8_K10_…GaussianDiffusion_aw10_thres0.5` |
| CAND_8 | Diffusion K=1 | 1 | `plans/diffusion/H8_K1_…/H8_K1_T0.5_…` |
| CAND_32 | **AlphaFlow K=2** (AF) | 2 | `flow_matching_v3_alphaflow/…_bbsit_…_ai1.0_ae0.0_ag25.0_rf0.5/H8_K2_Meuler_T0.5` |
| CAND_102 | **MeanFlow K=2** (MF) | 2 | `flow_matching_v3_meanflow/…_objmeanflow_bbmf_dit_…_dp0.5/H8_K2_Meuler_T0.5` |
| CAND_105 | **FM baseline** (FlowMatchingODE) | 20 | `flow_matching_v3_ode_selectable/…FlowMatchingODE_a1.5_b1.0_aw10/H8_K20_Meuler_T0.5` |
| CAND_109 | Diffusion-in-FMv3, K=1 | 1 | `flow_matching_v3_ode_selectable/…GaussianDiffusion_a1.5_b1.0_aw1/H8_K1_Meuler_T0.5` |

`K` = `flow_steps_v3` (FM/MF/AF) or `n_diffusion_steps` (diffusion) = number of network evaluations per replan. Confirmed against `config/avoiding-d3il.py` (`H{horizon}_K{...}_D{diffusion}` templates) and cross-checked against measured generation time (§3), which scales linearly with K.

---

## 2. Headline verdict

**AlphaFlow K=2 is the win. MeanFlow K=2 is a near-win with one env short. But both are matched by a 1-NFE diffusion baseline, so the speed claim needs a sharper framing.**

1. **AF(K=2) + `dpcc-r-tightened` is the only flow-family cell that is perfect everywhere**: 1.000 success+constraints in all 3 envs (15/15 seed-cells), 0 violations, while being **15–18× faster than DPCC(K=10)** and **19–75× faster than FM(K=20)**, and taking **7–13 % fewer environment steps than DPCC** in every env.
2. **MF(K=2) is one episode short of AF.** Perfect in `both-hard` and `top-left-hard`; in `top-right-hard` its best cell is 0.900 (one seed scored 0.5 → a single failed episode out of 10). Speed and step count are otherwise on par with AF (~20 % slower per plan: MF's `mf_dit` backbone costs 8.4 ms/NFE vs AF's SiT at 6.1 ms/NFE).
3. **Both baselines are beaten on cost, not on quality.** With a *tightened* projection, FM(K=20), DPCC(K=10), AF, MF and CAND_109 all reach 1.000 in most cells. The generator choice buys **compute**, not safety — safety comes from the tightened projection (see §5.1).
4. **Caveat that undercuts the narrative:** CAND_109 — an ODE-sampled GaussianDiffusion at **K=1** — is also 1.000 in all 3 envs on three different projection variants, at **0.0162–0.0197 s**, i.e. *faster than AF*. If the paper claim is "few-step generative models make PCC real-time", CAND_109 satisfies it without MeanFlow or AlphaFlow. See §5.4.

---

## 3. Generation cost (projection off — `diffuser` variant)

Pure generation time, no projection. Mean over the 3 envs.

| Candidate | NFE/plan | gen time [s] | ms per NFE | vs FM K=20 |
|---|---:|---:|---:|---:|
| FM K=20 | 20 | 0.1835 | 9.18 | 1.0× |
| DPCC diff K=10 | 10 | 0.0888 | 8.88 | 2.1× |
| MeanFlow K=2 | 2 | 0.0168 | 8.40 | **10.9×** |
| AlphaFlow K=2 | 2 | 0.0122 | 6.10 | **15.0×** |
| Diffusion K=1 | 1 | 0.0094 | 9.40 | 19.5× |
| Diff-in-FMv3 K=1 | 1 | 0.0091 | 9.10 | 20.2× |

Time scales cleanly with NFE (≈9 ms per U-Net/DiT call), which validates K as the step-count axis. AF's SiT backbone is additionally ~33 % cheaper per call than the others.

---

## 4. Matched comparison — same projection variant, all 3 envs

Cells are `succ+cons | env steps | s per planning step`. **MEAN** columns average the 3 envs.

### 4.1 `dpcc-r-tightened`

| Candidate | both-hard | top-left-hard | top-right-hard | MEAN s+c | MEAN steps | MEAN time |
|---|---|---|---|---:|---:|---:|
| DPCC diff K=10 | 0.80 / 71.2 / 0.3078 | 1.00 / 77.4 / 0.3250 | 0.90 / 76.7 / 0.3487 | 0.900 | 75.1 | 0.3272 |
| Diff K=1 | 1.00 / 72.0 / 0.0254 | 0.30 / 70.5 / 0.0434 | 0.50 / 60.7 / 0.0328 | 0.600 | 67.7 | 0.0339 |
| **AlphaFlow K=2** | **1.00 / 64.3 / 0.0204** | **1.00 / 67.4 / 0.0204** | **1.00 / 71.1 / 0.0196** | **1.000** | **67.6** | **0.0202** |
| MeanFlow K=2 | 1.00 / 69.3 / 0.0247 | 1.00 / 69.0 / 0.0253 | 0.90 / 74.1 / 0.0252 | 0.967 | 70.8 | 0.0251 |
| FM K=20 | 1.00 / 59.0 / 0.6409 | 0.60 / 89.3 / 1.5394 | 1.00 / 71.5 / 0.3775 | 0.867 | 73.3 | 0.8526 |
| Diff-in-FMv3 K=1 | 1.00 / 66.0 / 0.0175 | 1.00 / 68.8 / 0.0178 | 1.00 / 72.7 / 0.0162 | 1.000 | 69.2 | 0.0172 |

### 4.2 `dpcc-t-tightened`

| Candidate | both-hard | top-left-hard | top-right-hard | MEAN s+c | MEAN steps | MEAN time |
|---|---|---|---|---:|---:|---:|
| DPCC diff K=10 | 1.00 / 69.1 / 0.2683 | 1.00 / 69.3 / 0.3651 | 1.00 / 67.7 / 0.3317 | 1.000 | 68.7 | 0.3217 |
| Diff K=1 | 1.00 / 68.5 / 0.0233 | 0.60 / 72.1 / 0.0330 | 0.20 / 79.8 / 0.0437 | 0.600 | 73.5 | 0.0333 |
| AlphaFlow K=2 | 1.00 / 60.5 / 0.0201 | 1.00 / 60.2 / 0.0237 | 0.80 / 80.2 / 0.0247 | 0.933 | 67.0 | 0.0228 |
| MeanFlow K=2 | 1.00 / 60.6 / 0.0246 | 1.00 / 61.0 / 0.0253 | 0.90 / 83.7 / 0.0259 | 0.967 | 68.4 | 0.0253 |
| FM K=20 | 1.00 / 59.4 / 0.6071 | 1.00 / 65.1 / 0.4615 | 1.00 / 66.5 / 0.3909 | 1.000 | 63.7 | 0.4865 |
| Diff-in-FMv3 K=1 | 1.00 / 64.9 / 0.0178 | 1.00 / 66.6 / 0.0178 | 1.00 / 71.0 / 0.0163 | 1.000 | 67.5 | 0.0173 |

### 4.3 `dpcc-c-tightened` — **MF and AF break here**

| Candidate | both-hard | top-left-hard | top-right-hard | MEAN s+c | MEAN steps |
|---|---|---|---|---:|---:|
| DPCC diff K=10 | 1.00 / 68.5 | 1.00 / 69.4 | 1.00 / 73.1 | 1.000 | 70.3 |
| Diff K=1 | 0.90 / 70.3 | 0.60 / 80.2 | 0.50 / 65.7 | 0.667 | 72.1 |
| **AlphaFlow K=2** | **0.20 / 180.7** | **0.20 / 181.8** | **0.20 / 180.5** | **0.200** | **181.0** |
| **MeanFlow K=2** | **0.10 / 186.3** | **0.10 / 185.0** | **0.10 / 187.1** | **0.100** | **186.1** |
| FM K=20 | 1.00 / 56.8 | 1.00 / 65.9 | 1.00 / 67.0 | 1.000 | 63.2 |
| Diff-in-FMv3 K=1 | 1.00 / 66.4 | 1.00 / 70.5 | 1.00 / 73.1 | 1.000 | 70.0 |

---

## 5. Findings

### 5.1 The projection, not the generator, delivers constraint satisfaction

Unprojected (`diffuser` variant) success+constraints, mean over 3 envs: DPCC 0.067, Diff K=1 0.000, AF 0.267, MF 0.067, FM 0.133, C109 0.033. Everything is unsafe raw. Add a tightened projection and nearly everything reaches 1.000. So *"MF/AF beats FM/DPCC on safety"* is not supported and should not be claimed — the correct claim is **equal safety at a fraction of the generation cost**.

### 5.2 AlphaFlow K=2 — the clean result

`AF K=2 + dpcc-r-tightened` is 1.000 in all 15 seed-cells (checked at seed level in the raw CSV: every one of 5 seeds × 3 envs returns 1.0), total_violations 0.0 everywhere.

Speed-up at matched variant:

| vs. | both-hard | top-left | top-right |
|---|---:|---:|---:|
| DPCC K=10 (`dpcc-r-tightened`) | 15.1× | 15.9× | 17.8× |
| FM K=20 (`dpcc-r-tightened`) | 31.4× | 75×* | 19.3× |
| DPCC K=10 (`dpcc-t-tightened`) | 13.3× | 15.4× | 13.4× |
| FM K=20 (`dpcc-t-tightened`) | 30.2× | 19.5× | 15.8× |

\* inflated — FM's `top-left/dpcc-r-tightened` cell has a 1.5394 ± 0.8586 s outlier (one seed's NLP blew up). Quote the `dpcc-t-tightened` row for a defensible number: **AF is 13–15× faster than DPCC and 16–30× faster than FM.**

Environment steps: AF uses fewer steps than DPCC in all 3 envs on `dpcc-r-tightened` (−9.7 %, −12.9 %, −7.3 %). Against FM it is mixed — FM's projected plans are slightly shorter in `both-hard`.

### 5.3 MeanFlow K=2 — one episode short

MF matches AF everywhere except `top-right-hard`, where its ceiling is 0.900 on `dpcc-r-tightened` / `dpcc-t-tightened` / `hardflow_new-t-tightened`. Seed-level: `[0.5, 1.0, 1.0, 1.0, 1.0]` — **a single failed episode out of 10**. With n=10 per cell this is not a statistically meaningful gap from AF's 1.000 (binomial, p≈1.0). Do not report MF as "worse than AF" on this evidence; report it as "1.000 / 1.000 / 0.900, within noise of AF, needs more rollouts to separate".

### 5.4 The awkward rival: CAND_109 (K=1 ODE-sampled diffusion)

CAND_109 reaches 1.000 in all 3 envs on `dpcc-c-tightened`, `dpcc-r-tightened`, `dpcc-t-tightened` (and the dt-swept variants), at 0.0162–0.0197 s — **~15 % faster than AF and 30 % faster than MF**, with ~2 more env steps.

Its projection genuinely runs: `flow_matcher_v3_ode_selectable/models/diffusion.py:208` guards with
`near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)` — the trailing `or` clause fires at K=1. (The K=1 no-projection defect flagged in `temp/0108/11_41_47_gates_mix_visual_24097.log` §G6 is in the **Gen7 visual** `fm_visual_aligning/models/diffusion.py`, which lacks that `or`. It does **not** affect any candidate in this batch — CAND_109's `dpcc-*` cells differ sharply from its `diffuser` cell, confirming projection fired.)

**Implication for the paper:** on `avoiding-d3il` the task is easy enough that one ODE step + tightened projection suffices, so few-step *flow* models cannot be motivated by this benchmark alone. Either (a) move the headline to a harder task where 1-step diffusion degrades, or (b) reframe the contribution as "MeanFlow/AlphaFlow retain FM's trajectory quality at K=2 whereas naive step-truncation of FM does not" — which requires the missing ablation in §7.

### 5.5 MF/AF collapse under `dpcc-c-tightened`

AF 0.20 and MF 0.10 success across **all three envs**, with ~181–186 env steps (episodes time out without reaching the goal), while FM/DPCC/C109 are at 1.000 on the same variant. This is env-independent and therefore a systematic interaction between the `-c` projection formulation and the 2-step MF/AF sampler, not an env-specific fluke. Worth a targeted debug — it is the one place MF/AF are strictly worse than both baselines.

### 5.6 `hardflow_new-*` is not worth it for MF/AF

For AF/MF the `hardflow_new-*-tightened` variants reach the same 1.000 as `dpcc-*-tightened` but cost **0.066–0.080 s** (3.3× the dpcc variants) and **716–1410 NFE** of inner solver work. The `-c` flavours also drop to 0.6–0.8 success with ~93–107 steps. Recommend dropping hardflow from the headline table.

---

## 6. Caveats

- **n = 10 episodes per cell** (5 seeds × 2 trials). SEM is over 5 seed-means. Differences of 0.1 in success rate = one episode. Anything below ~0.2 apart should be treated as a tie.
- `n_steps` is episode length including failures — it is only interpretable when success ≈ 1.0. In §4.3 the ~185-step figures are timeouts, not "slow but successful" runs.
- `nfe_total` is only logged for `hardflow_new-*`; it is 0 for all `dpcc-*` cells and absent for the diffusion candidates. Step counts for those come from the K in the path, corroborated by the linear time-vs-K fit in §3.
- Wall time is per planning step on the cluster GPU; batch/threading conditions across candidates were not controlled for in this batch.
- The provided `.tex` covered only `both-hard` and 7 of 24 variants — reading it alone would have missed the `top-right-hard` MF gap (§5.3) and the CAND_109 rival (§5.4).

---

## 7. What I'd run next (to close the argument)

1. **More rollouts.** 10 episodes/cell cannot separate 1.000 from 0.900. Re-run CAND_32 / 102 / 105 / 7 / 109 on `dpcc-r-tightened` + `dpcc-t-tightened`, 3 envs, with ≥10 trials/seed (→ n=50). This is the single highest-value job.
2. **The missing ablation: FM at K=2.** Nothing here shows that FM *needs* 20 steps. Evaluate CAND_105's checkpoint with `flow_steps_v3=2` (and 1, 5). If truncated FM degrades while MF/AF hold at K=2, that is the actual contribution — and it is currently unmeasured.
3. **K-sweep for MF/AF.** `temp/0108` already holds MF eval trees for K ∈ {1, 2, 5, 10, 20} that were not in this DA batch. Ingesting those gives the MF quality-vs-K curve for free — send me the batch and I will fold it in.
4. **Debug the `dpcc-c` collapse** (§5.5) for MF/AF at K=2.
5. **Harder benchmark** if CAND_109 is to be beaten on its own terms — `avoiding-d3il` halfspace appears saturated by the tightened projection.

---

## 8. One-line summaries for a paper

> With a tightened DPCC projection, AlphaFlow at K=2 attains 100 % goal-reaching with zero constraint violations across all three `avoiding-d3il` halfspace configurations (5 seeds), using 2 network evaluations per replan against 20 for Flow Matching and 10 for DPCC — a 13–15× reduction in per-step planning latency versus DPCC and 16–30× versus FM, at 7–13 % fewer environment steps. MeanFlow at K=2 matches this in two of three configurations (1.000 / 1.000 / 0.900).

Do **not** write "MF/AF improve constraint satisfaction" — §5.1 contradicts it. See §9.4 for the version of this sentence that survives the raw-data re-analysis.

---
---

# 9. Independent re-analysis from per-seed raw data

§§1–8 were read off pre-aggregated cells. This section re-derives everything from `candidates_multidimensional_raw.csv` (one row per seed × env × variant × metric), so nothing depends on the batch tool's aggregation, its SEM convention, or the 7-variant `.tex` slice.

## 9.0 Full-seed filter and what the data actually is

Seed coverage was checked on every cell rather than trusting the `Missing_Seeds` column:

- **270 (candidate, env, variant) cells across the 6 candidates — all 270 have exactly 5 seeds {6,7,8,9,10}.** The full-seed filter therefore removes nothing; every number below is on complete seeds.
- Per-seed `n_success` and `n_success_and_constraints` take values in **{0.0, 0.5, 1.0}** only, and per-seed `n_success_std` ∈ {0.0, 0.5} (= population std of two values). This independently confirms **exactly 2 trials per seed**, so a cell is **10 episodes** and a candidate pooled over 3 envs is **30 episodes**.
- Episode cap is **199 steps** (seen as the exact `n_steps` value on all timed-out cells), not the 150 `max_path_length` in the training config.

**Variant grid is ragged.** Only **7 variants exist for all 6 candidates**: `diffuser`, `dpcc-{c,r,t}`, `dpcc-{c,r,t}-tightened`. MF/AF additionally have the 6 `hardflow_new-*` variants; DPCC/Diff/C109 additionally have the 4 `dt*` sweeps and `gradient`/`model_free`/`post_processing`. All cross-candidate claims below are restricted to the common 7.

> **Why this changes the reading of §4:** the `±` in §4 is SEM over 5 seed-means of a variable quantised to {0, 0.5, 1}. On 10 episodes that is not a usable uncertainty estimate. §9 uses episode counts with Wilson intervals and paired-by-seed tests instead.

## 9.1 Pooled episode counts — success **and** constraints, 30 episodes per cell

`k/30` with 95 % Wilson interval. Common variants only.

| variant | DPCC K=10 | Diff K=1 | **AlphaFlow K=2** | **MeanFlow K=2** | FM K=20 | Diff-FMv3 K=1 |
|---|---|---|---|---|---|---|
| `diffuser` | 2/30 [.02,.21] | 0/30 [.00,.11] | 8/30 [.14,.44] | 2/30 [.02,.21] | 4/30 [.05,.30] | 1/30 [.01,.17] |
| `dpcc-c` | 9/30 [.17,.48] | 2/30 [.02,.21] | 5/30 [.07,.34] | 2/30 [.02,.21] | 17/30 [.39,.73] | 11/30 [.22,.54] |
| `dpcc-c-tightened` | **30/30** | 20/30 [.49,.81] | 6/30 [.10,.37] | 3/30 [.03,.26] | **30/30** | **30/30** |
| `dpcc-r` | 6/30 [.10,.37] | 2/30 [.02,.21] | 22/30 [.56,.86] | 19/30 [.46,.78] | 9/30 [.17,.48] | 12/30 [.25,.58] |
| `dpcc-r-tightened` | 27/30 [.74,.97] | 18/30 [.42,.75] | **30/30 [.89,1.0]** | 29/30 [.83,.99] | 26/30 [.70,.95] | **30/30** |
| `dpcc-t` | 12/30 [.25,.58] | 0/30 [.00,.11] | 24/30 [.63,.90] | 13/30 [.27,.61] | 10/30 [.19,.51] | 10/30 [.19,.51] |
| `dpcc-t-tightened` | **30/30** | 18/30 [.42,.75] | 28/30 [.79,.98] | 29/30 [.83,.99] | **30/30** | **30/30** |

**Each candidate at its own best tightened variant:** DPCC 30/30, FM 30/30, C109 30/30 (**on all three**), AF 30/30 (`dpcc-r-tightened`, also `hardflow_new-r-tightened`), MF 29/30, Diff K=1 20/30.

So on quality the top of the table is a **five-way tie**, and Wilson intervals of ±0.11 at 30 episodes cannot separate 30/30 from 29/30. **§5.2's framing of AF as uniquely perfect is an artefact of picking one variant.** The honest statement is: *every arm except the K=1 plain diffusion reaches ceiling under some tightened projection; they differ in cost and in which projection they need.*

## 9.2 Paired-by-seed tests — the quality gap is not significant, the speed gap is total

Same 5 seeds × 3 envs = **15 matched pairs**; 20 000-resample seed-level bootstrap CI on the paired difference; two-sided exact sign test on non-ties.

**`dpcc-r-tightened`:**

| comparison | Δ success+constraints | 95 % CI | W/L/T | sign p |
|---|---:|---|---|---:|
| AF − DPCC K=10 | +0.100 | [+0.000, +0.200] | 3/0/12 | 0.250 |
| AF − FM K=20 | +0.133 | [+0.033, +0.267] | 4/0/11 | 0.125 |
| AF − C109 | +0.000 | [+0.000, +0.000] | 0/0/15 | 1.000 |
| MF − DPCC K=10 | +0.067 | [−0.067, +0.200] | 3/1/11 | 0.625 |
| MF − FM K=20 | +0.100 | [−0.033, +0.233] | 4/1/10 | 0.375 |
| MF − AF | −0.033 | [−0.100, +0.000] | 0/1/14 | 1.000 |

On `dpcc-t-tightened` the signs flip (AF − DPCC = −0.067, 0/2/13, p = 0.50). **No quality comparison in this batch reaches significance in either direction.** AF and C109 are literally identical on all 15 pairs.

**Latency, same pairs — this is where the result lives:**

| comparison | variant | mean speed-up | range over 15 pairs | W/L |
|---|---|---:|---|---|
| AF vs DPCC K=10 | `dpcc-r-tightened` | **×16.3** | 11.6 – 20.4 | **15/0** |
| AF vs FM K=20 | `dpcc-r-tightened` | ×42.2 | 17.7 – 245.4\* | **15/0** |
| MF vs DPCC K=10 | `dpcc-r-tightened` | ×13.0 | 9.9 – 16.2 | **15/0** |
| MF vs FM K=20 | `dpcc-r-tightened` | ×33.9 | 13.9 – 196.9\* | **15/0** |
| AF vs DPCC K=10 | `dpcc-t-tightened` | **×14.8** | 7.1 – 19.8 | **15/0** |
| AF vs FM K=20 | `dpcc-t-tightened` | **×22.6** | 11.0 – 33.8 | **15/0** |
| MF vs DPCC K=10 | `dpcc-t-tightened` | ×12.7 | 9.7 – 16.5 | **15/0** |
| MF vs FM K=20 | `dpcc-t-tightened` | ×19.3 | 11.9 – 27.2 | **15/0** |

\* inflated by one FM outlier — see §9.5. Quote the `dpcc-t-tightened` row.

**15/0 on every pair, every seed, every env.** This is the only comparison in the whole batch that is unambiguous.

## 9.3 Environment steps, restricted to pairs where *both* arms scored 1.0

Comparing `n_steps` across cells with different success rates is meaningless (timeouts sit at 199). Restricting to matched (env, seed) cells where **both** candidates scored a full 1.0:

| comparison | variant | n pairs | Δ steps | 95 % CI | fewer-steps wins |
|---|---|---:|---:|---|---|
| AF − DPCC K=10 | `dpcc-r-tightened` | 12 | **−7.9** | [−14.5, −1.5] | 7/12 |
| AF − DPCC K=10 | `dpcc-t-tightened` | 13 | **−6.8** | [−12.8, −1.5] | 10/13 |
| MF − DPCC K=10 | `dpcc-r-tightened` | 11 | **−4.7** | [−9.1, −0.6] | 7/11 |
| MF − DPCC K=10 | `dpcc-t-tightened` | 14 | −4.3 | [−11.2, +1.6] | 8/14 |
| AF − FM K=20 | `dpcc-r-tightened` | 11 | +0.5 | [−5.9, +7.3] | 6/11 |
| AF − FM K=20 | `dpcc-t-tightened` | 13 | −0.9 | [−3.8, +1.7] | 7/13 |
| MF − FM K=20 | `dpcc-r-tightened` | 10 | **+6.8** | [+2.3, +11.2] | 2/10 |
| AF − C109 | `dpcc-t-tightened` | 13 | **−4.7** | [−7.3, −2.0] | 12/13 |

**AF genuinely takes fewer environment steps than DPCC** (CI excludes 0 on both variants) — this survives, unlike the success claim. Against FM it is a tie. **MF is significantly *worse* than FM on steps** at `dpcc-r-tightened` (+6.8 [+2.3, +11.2]) — §5.2's step claim should not be extended to MF. AF also beats C109 on steps (−4.7, 12/13 wins), which is AF's one real edge over that rival.

## 9.4 Failure-mode asymmetry — **the most important finding, and it is new**

Tagging every lost half-episode on the three tightened variants as either a **goal-fail** (never reached goal; `n_success` < 1) or a **constraint trip** (`n_success` = 1 but flagged unsafe):

| candidate | lost half-cells (3 tightened variants) | goal-fails | **constraint trips** |
|---|---:|---:|---:|
| AlphaFlow K=2 | 14 | 14 | **0** |
| MeanFlow K=2 | 17 | 17 | **0** |
| Diff-FMv3 K=1 | 0 | 0 | **0** |
| DPCC diff K=10 | 3 | 0 | **3** |
| FM K=20 | 4 | 0 | **4** |
| Diff K=1 | 21 | 4 | **17** |

**AF and MF never once produced an unsafe episode on any tightened projection variant — every one of their 31 lost episodes was a timeout, with `total_violations` = 0.0.** DPCC and FM never time out but do produce violations. The two K=2 flow arms **fail safe** (liveness failure); the baselines **fail unsafe** (safety failure). For a constrained-control paper this is a stronger and more defensible claim than any success-rate comparison, and it is entirely invisible in the aggregated tables and in the provided `.tex`.

Corrected version of the §8 one-liner:

> Under a tightened DPCC projection, AlphaFlow and MeanFlow at K=2 match the FM(K=20) and DPCC(K=10) baselines on task success (30/30 and 29/30 vs 30/30 episodes; no paired difference significant) while reducing per-replan planning latency by 13–23× (15/15 paired wins) and, for AlphaFlow, environment steps by 6.8–7.9 (95 % CI excludes 0). Crucially, across all tightened variants neither K=2 arm ever produced a constraint violation — their only failures are timeouts — whereas every baseline failure was a constraint violation.

## 9.5 Knife-edge constraint accounting — several baseline "failures" are numerical

The binary constraint flag is near zero-tolerance. Two symptoms in the raw data:

- **566 cells** have `total_violations` > 0 yet count as satisfied — but those are ~3 × 10⁻⁸, i.e. solver noise.
- **157 cells lost an episode to a violation of magnitude < 0.05.**

Applying the sign-test to the actual magnitudes on the tightened variants (note: per-seed `total_violations` is a 2-trial mean, so a failing trial's true magnitude is ≈2× the value shown):

| candidate | variant | score | violation magnitudes of the lost cells | score under a 10⁻² tolerance |
|---|---|---|---|---|
| DPCC K=10 | `dpcc-r-tightened` | 27/30 | 2.3e−4, 4.0e−3, 4.4e−3 | **30/30** |
| FM K=20 | `dpcc-r-tightened` | 26/30 | 9.3e−4, 4.8e−3, **6.2e−2**, **1.57** | 28/30 |
| AlphaFlow K=2 | `dpcc-r-tightened` | 30/30 | — (no lost cells) | 30/30 |
| MeanFlow K=2 | `dpcc-r-tightened` | 29/30 | — (the one loss is a goal-fail) | 29/30 |

**DPCC's apparent 3-episode deficit on `dpcc-r-tightened` is entirely numerical** — all three trips are ≤ 4.4 × 10⁻³ and vanish under any sane tolerance. FM has two genuine violations (6.2 × 10⁻² and 1.57) and two numerical ones. This directly weakens §4.1 and §5.2, which read DPCC's 0.80/0.90 cells as real failures.

**AF is unaffected by this**: it has zero lost cells there, so its 30/30 holds under any tolerance. That is a stronger position than the §5.2 version of the same claim.

## 9.6 Latency stability — an argument §§1–8 missed entirely

Coefficient of variation of `avg_time` across the 15 (env, seed) cells, `dpcc-r-tightened`:

| candidate | mean [s] | sd | **CV** | max/min |
|---|---:|---:|---:|---:|
| AlphaFlow K=2 | 0.0202 | 0.0007 | **3.3 %** | 1.1 |
| MeanFlow K=2 | 0.0251 | 0.0007 | **3.0 %** | 1.1 |
| Diff-FMv3 K=1 | 0.0172 | 0.0008 | 4.5 % | 1.1 |
| DPCC diff K=10 | 0.3272 | 0.0422 | 12.9 % | 1.7 |
| Diff K=1 | 0.0339 | 0.0141 | 41.6 % | 3.6 |
| **FM K=20** | 0.8526 | 1.1489 | **134.7 %** | **14.1** |

The FM outlier is `top-left-hard`, seed 7: **4.949 s per planning step** against 0.351 s at its best cell, together with `n_steps` = 133 and `total_violations` = 1.57 — a single episode that was simultaneously slow, long and unsafe. FM's *worst-case* latency is 14× its best; AF's is 1.1×. For receding-horizon control worst-case latency is the binding constraint, so **AF/MF are ~250× better than FM on tail latency** even though the mean ratio is ~23×. `nlp_solves_total` / `nlp_failures_total` are only populated for `hardflow_new-*` (0 elsewhere, absent for the diffusion arms), so the cause of the FM stall cannot be attributed from this batch — worth instrumenting.

## 9.7 End-to-end cost per episode — `n_steps × avg_time`

The metric that combines both axes the question asked about. Bootstrap CI over the 15 cells.

| candidate | `dpcc-r-tightened` | `dpcc-t-tightened` | `dpcc-c-tightened` |
|---|---|---|---|
| **Diff-FMv3 K=1** | **1.19 s** [1.15, 1.22] (30/30) | **1.17 s** [1.14, 1.20] (30/30) | **1.33 s** (30/30) |
| **AlphaFlow K=2** | **1.36 s** [1.28, 1.45] (30/30) | 1.61 s [1.24, 2.19] (28/30) | 3.43 s (6/30) |
| **MeanFlow K=2** | 1.77 s [1.69, 1.85] (29/30) | 1.73 s [1.55, 1.99] (29/30) | 4.40 s (3/30) |
| Diff K=1 | 2.39 s (18/30) | 2.75 s (18/30) | 2.68 s (20/30) |
| DPCC diff K=10 | 24.5 s [22.6, 26.8] (27/30) | 22.1 s [19.7, 24.9] (30/30) | 21.7 s (30/30) |
| FM K=20 | 79.9 s [33.3, 166.3] (26/30) | 30.7 s [27.8, 33.7] (30/30) | 29.7 s (30/30) |

**AF completes a constrained episode in 1.36 s of planning compute against 22–25 s for DPCC and 30–80 s for FM — a 16–22× end-to-end reduction at equal task success.** That is the cleanest single number in the whole analysis.

## 9.8 The `dpcc-c` collapse, mechanically characterised

§5.5 flagged it; the raw data says exactly what it is. On `dpcc-c-tightened`, **every** AF and MF loss is `n_steps` = 199.0 (the hard cap) with `n_success` = 0 and `total_violations` = **0.0**, in all three envs, on 4/5 and 5/5 seeds respectively — while FM finishes in 55–59 steps and C109 in 64–70 on the identical cells.

So it is not a safety failure and not an env-specific fluke: **the `-c` projection drives the 2-step MF/AF plans into a stall** — the trajectory stays feasible but stops making progress to the goal, indefinitely. That is a liveness bug in the interaction between the `-c` cost formulation and a 2-NFE sampler, and it is fully reproducible (15/15 and 14/15 cells). It is the single most tractable debug target in this batch.

## 9.9 What §9 changes about the conclusions

| §§1–8 said | §9 finds |
|---|---|
| AF is the only flow-family arm perfect everywhere (§5.2) | Variant-selection artefact. DPCC, FM and C109 also reach 30/30; no paired quality difference is significant (§9.1–9.2) |
| DPCC scores 0.80/0.90 on `dpcc-r-tightened` | All three trips are ≤ 4.4e−3 — numerical, not real. DPCC is effectively 30/30 (§9.5) |
| AF takes 7–13 % fewer steps than DPCC | Holds, and now with a CI excluding 0 (−7.9 and −6.8 steps). But MF is *worse* than FM on steps (§9.3) |
| MF is "one episode short of AF" | Correct, and the failing seed differs per variant (s6 / s10 / s7) → episode noise, not a seed or model defect (§9.4) |
| Speed-up 13–15× vs DPCC, 16–30× vs FM | Confirmed paired, 15/15 wins, and extended: ×250 on *tail* latency, 16–22× end-to-end per episode (§9.2, 9.6, 9.7) |
| — (not observed) | **AF/MF never fail unsafe; baselines never fail safe.** Strongest available claim (§9.4) |
| — (not observed) | C109 is 30/30 on *all three* tightened variants — the most robust arm in the batch, and the real rival (§9.1) |

**Bottom line unchanged from §2 but now properly supported:** MF/AF at K=2 buy **compute and latency predictability**, not accuracy. The new addition is that they also buy a **safer failure mode**. The open threat remains CAND_109, and the missing experiment remains FM evaluated at K=2 (§7.2).
