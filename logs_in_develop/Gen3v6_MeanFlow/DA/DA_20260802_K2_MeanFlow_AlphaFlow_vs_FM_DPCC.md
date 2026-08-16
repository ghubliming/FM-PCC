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

---
---

# 10. Addendum — same-K comparison, K-ladders, and MF vs AF

§§1–9 looked at 6 hand-picked candidates. This addendum uses **all 106 candidates in the batch**, grouped by *trained model* so that K-sweeps compare one checkpoint against itself. It answers four questions directly:

1. **Did we beat DPCC/FM at the *same* K?** (the minimum bar)
2. **When DPCC/FM K rises, does MF/AF at K=2 still win?**
3. **Are there high-K MF/AF runs, and what do they say?**
4. **MF vs AF head to head.**

Three of the four answers changed my mind about the paper's framing. **§10.4 in particular invalidates the causal story in §3 and §9.6.**

## 10.0 Inventory and the seed rules used here

Candidates were grouped by their **parent training folder** (the `plans/<family>/<train-config>/` prefix), because `H8_K{n}_...` leaf folders under different parents are different *models*, not different K of the same model. 106 candidates collapse into these usable ladders:

| ladder (one trained model) | K values available | seeds |
|---|---|---|
| `plans/diffusion` GaussDiff aw10 thres0.5 (**DPCC baseline**) | 10, 20 | **5** |
| `flow_matching_v3_ode_selectable` GaussDiff a1.5 b1.0 aw1 | 1, 5, 10, 20 | **5** |
| `flow_matching_v3_ode_selectable` **FlowMatchingODE** aw10 (**FM baseline**) | 5, 20 | 20→**5**, 5→1 |
| `flow_matching_v3_alphaflow` bbsit (**AF, the §§1–9 model**) | 1, 2, 5, 10 | 2→**5**, rest→**4** (seed 6 missing) |
| `flow_matching_v3_meanflow` dp0.5 (**MF, the §§1–9 model**) | 1, 2, 5, 10, 20 | 2→**5**, rest→1 (seed 6 only) |
| `flow_matching_v3_alphaflow(Bf_U3)` bbdit / bbsit / bbunet | 1, 2, 5, 10 (+20 on fix4) | 1 (seed 6) |
| `flow_matching_v3_meanflow(Bf_Fix4 / Bf_Fix5)` | 1, 2, 5, 10, 20 | 1 (seed 6) |

**Seed rule applied throughout §10, per your instruction (full is better, else one seed):**

- **5-seed tables** (30 episodes) where all arms have all seeds.
- **Seeds 7–10 only** (24 episodes) whenever the AF ladder is involved — CAND_30/31/33 are missing seed 6, so seeds 7–10 is the largest *complete, paired* set. CAND_32 is truncated to the same 4 seeds so every comparison is like-for-like.
- **Seed 6 only** (6 episodes) for the MF ladder, the FM K=5 point, and the backbone ablations. These are flagged **[n=1 seed]** everywhere and are *indicative only* — 6 episodes cannot support a claim on its own. They are included because you asked, and because several of them replicate across 3–4 independent training runs, which is what makes them believable (§10.6).

**Critical gap found:** there is **no DPCC and no FlowMatchingODE run at K=2 anywhere in the batch**, at any seed count. The nearest non-flow points are K=1 (`CAND_8`, `CAND_109`). So the literal same-K comparison you asked for **cannot be made at K=2** — §10.1 makes it at K=1 instead, which is a *stricter* bar.

## 10.1 The same-K question — the least result

Since no baseline exists at K=2, the honest same-cost comparison is **at K=1**, where AF, MF, plain diffusion and FMv3-selectable diffusion all have runs. Best tightened variant, seeds 7–10 (24 episodes) where possible:

| arm | K | best tightened variant | s+c | latency [s] | steps | **episode cost [s]** |
|---|---:|---|---|---:|---:|---:|
| **AlphaFlow** | **1** | `dpcc-t-tightened` | **24/24** | 0.0140 | 77.7 | **0.93** |
| GaussDiff-FMv3 (C109) | 1 | `dpcc-t-tightened` | **24/24** | 0.0172 | 69.2 | 1.17 |
| AlphaFlow | 2 | `dpcc-r-tightened` | **24/24** | 0.0202 | 68.4 | 1.38 |
| MeanFlow | 2 | `dpcc-r-tightened` | **24/24** | 0.0251 | 70.8 | 1.79 |
| plain diffusion (CAND_8) | 1 | `dpcc-c-tightened` | 20/30 | 0.0333 | 72.1 | 2.40 |
| MeanFlow **[n=1 seed]** | 1 | `dpcc-t-tightened` | 6/6 | 0.0271 | 75.7 | 2.05 |

**Answer: at equal K, AlphaFlow does beat the diffusion baselines — but only narrowly, and only against `C109`.** Paired on seeds 7–10, AF K=1 vs C109 K=1 on episode planning cost:

| variant | AF K=1 | C109 K=1 | Δ episode cost | AF wins |
|---|---|---|---|---|
| `dpcc-t-tightened` | 24/24, 0.93 s | 24/24, 1.17 s | **−0.243 s [−0.305, −0.187]** | **12/12** |
| `dpcc-r-tightened` | 24/24, 1.09 s | 24/24, 1.18 s | −0.087 s [−0.160, −0.008] | 10/12 |
| `dpcc-c-tightened` | 21/24, 1.97 s | 24/24, 1.35 s | +0.616 s [+0.195, +0.985] | 2/12 |

AF K=1 wins on two of three projection variants (CI excludes 0 on both) and **loses** on the third. Against plain diffusion K=1 (CAND_8, 20/30) both flow arms win outright on quality.

> **AF K=1 + `dpcc-t-tightened` at 0.93 s/episode is the cheapest perfect configuration in the entire 106-candidate batch.** Note this is **K=1, not the K=2 that §§1–9 promoted** — see §10.3.

## 10.2 Baseline K-ladders — does DPCC/FM get better with more K?

**No. Quality is saturated at every K; only cost moves.** All 5 seeds, 30 episodes, best tightened variant:

| ladder | K | s+c | latency [s] | steps | episode cost [s] |
|---|---:|---|---:|---:|---:|
| **DPCC** GaussDiff aw10 thres0.5 | 10 | 30/30 | 0.3098 | 70.3 | 21.79 |
| **DPCC** GaussDiff aw10 thres0.5 | 20 | 30/30 | 0.5534 | 70.1 | **38.81** |
| FMv3sel GaussDiff aw1 | 1 | 30/30 | 0.0172 | 69.2 | **1.19** |
| FMv3sel GaussDiff aw1 | 5 (midpoint\*) | 30/30 | 0.1458 | 70.2 | 10.24 |
| FMv3sel GaussDiff aw1 | 10 (thr = 1\*\*) | 30/30 | 0.9400 | 63.8 | 59.97 |
| FMv3sel GaussDiff aw1 | 20 | 30/30 | 0.4470 | 62.9 | 28.13 |
| FM **FlowMatchingODE** aw10 | 20 | 30/30 | 0.4767 | 63.2 | 30.14 |
| FM **FlowMatchingODE** aw10 **[n=1 seed]** | 5 | 6/6 | 0.1114 | 68.2 | **7.60** |
| HardFlow mpc4 | 20 | 30/30 | 0.4709 | 63.2 | 29.78 |
| plain diffusion aw1 | 20 | 30/30 | 0.5973 | 78.7 | 46.99 |

\* `Mmidpoint` = midpoint solver, 2 network evals per step → 10 NFE, which is why its gen time/NFE is 17.8 ms vs ~9 ms elsewhere.
\*\* `T1` = projection threshold 1.0 → projects on **every** step; not a clean ladder point, kept for the §10.4 mechanism.

**Answer to "when DPCC/FM K rises, do MF/AF still beat them?" — yes, and the margin *grows*.** DPCC going K=10→20 costs 1.78× more time for **exactly zero** quality gain (30/30 → 30/30). Paired, 5 seeds:

| comparison | Δ s+c | sign p | speed-up | time wins |
|---|---|---:|---:|---|
| AF K=2 vs **DPCC K=20** | −0.067 [−0.167, 0.000] | 0.50 | **×25.7** (16.0–32.6) | **15/15** |
| MF K=2 vs **DPCC K=20** | −0.033 [−0.100, 0.000] | 1.00 | **×22.2** (17.9–26.5) | **15/15** |
| AF K=2 vs **FMv3sel-GD K=20** | −0.067 [−0.167, 0.000] | 0.50 | **×20.9** (8.4–29.7) | **15/15** |

Quality differences remain non-significant in every case (consistent with §9.2); the speed-up is 15/15 and grows from ×14.8 at K=10 to ×25.7 at K=20.

**But the flip side:** FM at **K=5** reaches 6/6 at 7.60 s/episode **[n=1 seed]**. This is the first direct evidence on the ablation flagged in §7.2 — **FM does not need K=20.** At K=5 it is already 4× cheaper than at K=20 with no visible quality loss. The K=20 FM baseline that §§1–9 measured against is therefore a **weak baseline**, and part of the headline ×22–30 speed-up is an artefact of comparing against an over-provisioned configuration. A properly tuned FM baseline would be at K=5, cutting the honest speed-up to roughly **×5–8**.

## 10.3 MF/AF K-ladders — high-K runs exist, and more K is strictly worse

**AlphaFlow ladder, same trained model, seeds 7–10 (24 episodes):**

| K | s+c | latency [s] | steps | episode cost [s] | robustness (min over 3 tightened variants) |
|---:|---|---:|---:|---:|---|
| **1** | 24/24 | 0.0140 | 77.7 | **1.09** | 21/24 |
| 2 | 24/24 | 0.0202 | 68.4 | 1.38 | **6/24** ← see §10.6 |
| 5 | 24/24 | 0.2010 | 69.6 | 13.98 | 22/24 |
| 10 | 24/24 | 0.3407 | 62.0 | 21.14 | 20/24 |

**MeanFlow ladder, same trained model [n=1 seed, seed 6]:**

| K | s+c | latency [s] | steps | episode cost [s] |
|---:|---|---:|---:|---:|
| 1 | 6/6 | 0.0271 | 75.7 | 2.05 |
| **2** | 6/6 | 0.0269 | 65.5 | **1.76** |
| 5 | 6/6 | 0.2220 | 67.2 | 14.91 |
| 10 | 6/6 | 0.3588 | 70.7 | 25.35 |
| 20 | 6/6 | 0.9739 | 67.0 | **65.25** |

Replicated on two independent older MF training runs (`Bf_Fix4`, `Bf_Fix5`) with the same shape — K=20 costs 63.8 s/episode on Fix5, 33× the K=2 cost, for identical 6/6.

**Answer to "are there high-K MF/AF runs?" — yes, K up to 20 for MF and K=10 (K=20 at one seed) for AF, and they are all a waste.** Paired, AF K=2 vs AF K=10 on seeds 7–10: Δ s+c +0.167 (favouring K=2, p = 0.25), speed-up **×18.3, 12/12 wins**. Quality is flat across the whole ladder for both families; cost rises ~20–35×.

**And K=2 is not the optimum.** AF K=1 vs AF K=2, paired, seeds 7–10: quality **identical on all 12 cells**, K=1 is **×1.4 faster, 12/12 wins**. The K=2 setting that §§1–9 promoted is dominated by K=1 on cost, and beaten on robustness too (21/24 vs 6/24 min). Its only advantage is 9 fewer environment steps (68.4 vs 77.7).

## 10.4 Where the wall time actually goes — this invalidates §3 and §9.6

§3 concluded "time scales cleanly with NFE (~9 ms per network call)" from the K-sweep of *generation-only* times, and §9.6/§9.7 built the latency argument on it. Decomposing projected time into generation + projection overhead across every ladder shows that is **wrong for the projected numbers, which are the ones that matter**:

| ladder | K | gen only [s] | projected [s] | overhead [s] | proj. calls | **per projection [ms]** | gen per NFE [ms] |
|---|---:|---:|---:|---:|---:|---:|---:|
| DPCC | 10 | 0.0888 | 0.3217 | 0.2329 | 5 | 46.6 | 8.88 |
| DPCC | 20 | 0.1793 | 0.5630 | 0.3837 | 10 | 38.4 | 8.97 |
| FMv3sel-GD | 1 | 0.0091 | 0.0173 | 0.0082 | 1 | 8.2 | 9.11 |
| FMv3sel-GD | 5 | 0.0892 | 0.1458 | 0.0566 | 3 | 18.9 | 17.84 |
| FMv3sel-GD (thr=1) | 10 | 0.0855 | 1.0564 | 0.9709 | 10 | 97.1 | 8.55 |
| FMv3sel-GD | 20 | 0.1826 | 0.4470 | 0.2644 | 10 | 26.4 | 9.13 |
| **AlphaFlow** | **1** | 0.0063 | 0.0141 | 0.0078 | **1** | 7.8 | 6.27 |
| **AlphaFlow** | **2** | 0.0122 | 0.0235 | 0.0113 | **1** | 11.3 | 6.10 |
| AlphaFlow | 5 | 0.0300 | 0.2010 | 0.1709 | 3 | 57.0 | 6.01 |
| AlphaFlow | 10 | 0.0577 | 0.3411 | 0.2834 | 5 | 56.7 | 5.77 |
| MeanFlow | 2 | 0.0178 | 0.0269 | 0.0091 | **1** | 9.1 | 8.89 |
| MeanFlow | 20 | 0.1648 | 0.9739 | 0.8091 | 10 | 80.9 | 8.24 |
| FM-ODE | 5 | 0.0457 | 0.1154 | 0.0697 | 3 | 23.2 | 9.13 |
| FM-ODE | 20 | 0.1844 | 0.4679 | 0.2835 | 10 | 28.4 | 9.22 |

Projection calls per replan is `K − int((1−threshold)·K)`, verified against the sampler loop (`flow_matcher_v3_alphaflow/models/af_diffusion.py:342-343`, `flow_matcher_v3_meanflow/models/mf_diffusion.py:284-285`, `flow_matcher_v3_ode_selectable/models/diffusion.py:207-208`).

**The generative model is 27–48 % of projected latency at K=1–2 and only 15–20 % at K≥5. The DPCC projection dominates, and its call count is tied to K by the threshold rule.** So:

> The real mechanism is **not** "MeanFlow/AlphaFlow generate faster". It is **"K=1–2 makes the DPCC projection run once per replan instead of K/2 times"**. Any generator that works at K≤2 gets the identical benefit — which is precisely why the K=1 GaussianDiffusion (C109) is competitive, and why §§1–9 could not explain it away.

At K=1–2 the projection fires exactly once, at the final iterate, so the overhead floor is one NLP solve (~8–11 ms). Everything above that is redundant projection calls on intermediate, noisier iterates — which are *also* individually more expensive (7.8 → 57 ms per call for AF as K goes 1 → 10), because the NLP starts further from feasibility.

**Consequence for the paper:** the contribution should be stated as *"MeanFlow/AlphaFlow retain trajectory quality at the K=1–2 operating point where the projection cost collapses to a single solve"*, not as a generation-speed result. That is still a real contribution — but the burden of proof moves to showing **FM/diffusion do *not* retain quality at K=1–2**, and §10.1 shows C109 does. This is now the central open question of the whole line of work.

## 10.5 MeanFlow vs AlphaFlow, head to head

**At K=2, full 5 seeds, paired over 15 (env, seed) cells:**

| variant | Δ s+c (AF − MF) | 95 % CI | W/L/T | sign p | AF speed-up | time wins |
|---|---|---|---|---:|---:|---|
| `dpcc-r-tightened` | +0.033 | [+0.000, +0.100] | 1/0/14 | 1.00 | ×1.2 (1.2–1.4) | **15/15** |
| `dpcc-t-tightened` | −0.033 | [−0.133, +0.067] | 1/2/12 | 1.00 | ×1.2 (0.6–1.4) | 13/15 |

**Quality: a dead tie** (§9.2 already showed MF − AF = −0.033, p = 1.00). **Cost: AF wins consistently but modestly** — ×1.2, from a cheaper backbone (SiT at 6.1 ms/NFE vs `mf_dit` at 8.9 ms/NFE), giving 1.38 vs 1.79 s/episode.

**Across the ladder AF is ahead at every K** (AF seeds 7–10 vs MF seed 6, *not* paired — different seeds, indicative only): AF 1.09/1.38/13.98/21.14 s per episode at K=1/2/5/10 against MF 2.05/1.76/14.91/25.35. The one place MF leads is **K=1**, where MF's episode cost (2.05 s) is worse than AF's (1.09 s) but MF K=2 (1.76 s) beats MF K=1 — i.e. **MF has an optimum at K=2, AF at K=1**. That is consistent with MeanFlow's one-step objective being trained for a *specific* step count while AlphaFlow's α-annealed objective degrades more gracefully.

**Verdict: prefer AlphaFlow.** Equal quality, ~20 % cheaper, better K=1 behaviour, and the more robust ladder. MF's only distinguishing result is that it survives §9.4's failure-mode test identically.

## 10.6 The `dpcc-c` collapse is a **K=2-specific** bug, not an MF/AF property

§9.8 called this an "interaction between the `-c` cost formulation and a 2-NFE sampler". The K-ladder pins it down much harder. `dpcc-c-tightened`, seed 6 (6 episodes), across **every** MF/AF training run in the batch:

| run | K=1 | **K=2** | K=5 |
|---|---|---|---|
| AF bbdit (U3) | 6/6 | **6/6** | 6/6 |
| AF bbsit (U3) | 6/6 | **0/6** | 6/6 |
| AF bbsit (fix4) | 6/6 | **0/6** | — |
| AF bbsit (final, CAND_32) | 6/6 | **0/6** | — |
| MF Fix4 | 2/6 | **0/6** | 6/6 |
| MF Fix5 | 2/6 | **0/6** | 6/6 |
| MF final (CAND_102) | 2/6 | **0/6** | 6/6 |

And on the AF ladder at seeds 7–10: K=1 → 21/24, **K=2 → 6/24**, K=5 → 23/24, K=10 → 24/24.

**It is exactly K=2, it reproduces across 4 independent AF runs and 3 independent MF runs, and it disappears at K=1 and K=5.** Both K=1 and K=2 make *exactly one* projection call (verified: `snapping_start_idx = int(0.5·K)` gives 0 for K=1 and 1 for K=2, both the final iterate), so this is **not** a projection-count effect — it is about the *iterate* the single projection is applied to. At K=2 the last velocity evaluation uses `tau = 0.5` and the projection lands on a half-integrated sample; at K=1 it lands on a fully-integrated one.

The one exception — **AF with the `bbdit` backbone is 6/6 at K=2** — says the failure is a K=2 × backbone interaction, not a property of the `-c` cost. Since `bbsit` is the backbone used by the headline candidate, this is a live bug in exactly the configuration §§1–9 recommends.

**This is now the highest-value debug target in the batch**: it is 100 % reproducible, isolated to one K, and has a working control (`bbdit`, and K=1/K=5) to diff against.

## 10.7 AlphaFlow backbone ablation **[n=1 seed]**

Seed 6, best tightened variant:

| backbone | K=2 | K=10 |
|---|---|---|
| `bbsit` (SiT) | **6/6, 1.30 s/ep** | 6/6, 26.94 s/ep |
| `bbdit` (DiT) | 6/6, 2.79 s/ep | 6/6, 28.66 s/ep |
| `bbunet` (U-Net) | **1/6, 17.64 s/ep** | **1/6, 145.84 s/ep** |

`bbsit` is the right default: same quality as `bbdit` at less than half the cost. **`bbunet` is broken for AlphaFlow** — 1/6 at both K, with a 13× latency penalty at K=10 (the projection cannot fix a bad trajectory, so it iterates). Do not run further AF experiments on the U-Net backbone.

## 10.8 Champion table

Best tightened variant per arm, **seeds 7–10, 24 episodes**, ranked by episode planning cost:

| rank | arm | K | variant | s+c | **episode cost [s]** | vs DPCC K=10 |
|---:|---|---:|---|---|---:|---:|
| 1 | **AlphaFlow** | **1** | `dpcc-t-tightened` | 24/24 | **0.93** | **×24** |
| 2 | GaussDiff-FMv3 (C109) | 1 | `dpcc-t-tightened` | 24/24 | 1.17 | ×19 |
| 3 | AlphaFlow | 2 | `dpcc-r-tightened` | 24/24 | 1.38 | ×16 |
| 4 | MeanFlow | 2 | `dpcc-r-tightened` | 24/24 | 1.79 | ×12 |
| 5 | DPCC | 10 | `dpcc-c-tightened` | 24/24 | 22.28 | 1.0 |
| 6 | FM-ODE | 20 | `dpcc-c-tightened` | 24/24 | 29.55 | ×0.75 |

Robustness across *all three* tightened variants (min /24): **C109 24**, AF K=5 22, DPCC K=10 22, AF K=1 21, FM K=20 21, AF K=10 20, **AF K=2 6, MF K=2 3**.

**The two rankings disagree, and that is the honest summary of this batch:** AF K=1 is the *cheapest* perfect arm, C109 K=1 is the *most robust* one, and the K=2 configurations that §§1–9 promoted are the *least robust* things in the table.

## 10.9 What §10 changes, and what to run

| §§1–9 said | §10 finds |
|---|---|
| K=2 MF/AF is the operating point | **K=1 dominates K=2 for AF** — same quality, ×1.4 faster, far more robust (21/24 vs 6/24) |
| Speed-up comes from fewer network evaluations | **False.** It comes from fewer *projection* calls; generation is 15–48 % of latency (§10.4) |
| FM K=20 is the FM baseline | **FM K=5 is 6/6 at 4× less cost [n=1 seed]** — K=20 is an over-provisioned baseline, honest speed-up drops to ≈×5–8 |
| Baselines might improve with more K | **They do not.** DPCC K=10→20 and FMv3sel-GD K=1→20 are all 30/30; only cost moves. MF/AF beat them by a *growing* margin |
| `dpcc-c` collapse is an "MF/AF × 2-NFE" interaction | **It is K=2-specific and backbone-specific**; K=1 and K=5 are clean, `bbdit` is clean. Reproduces on 7 independent training runs |
| MF vs AF not addressed | **Tied on quality, AF ~20 % cheaper**; MF optimum is K=2, AF optimum is K=1 |
| CAND_109 is the awkward rival | Still is — and is now the *most robust* arm in the batch (24/24 on all three variants) |

**Run queue, revised and re-prioritised:**

1. **FM-ODE and DPCC at K=1 and K=2, 5 seeds, 3 envs.** This is now the single decisive experiment. §10.4 says the entire advantage is "works at K≤2"; if FM/diffusion also work at K≤2 there is no flow-specific contribution, and if they do not, that *is* the contribution. Currently there is no FM run below K=5 and no diffusion K=2 run at all.
2. **Complete the AF ladder on seed 6** (CAND_30/31/33) so the AF K-sweep is 5-seed, and **run the MF ladder on seeds 7–10** — the MF ladder is currently one seed, and AF/MF ladders sit on *disjoint* seed sets, which blocks a paired MF-vs-AF K comparison.
3. **Debug the K=2 `dpcc-c` collapse** using `bbdit` @ K=2 and `bbsit` @ K=1/K=5 as working controls (§10.6).
4. **Re-run the headline at K=1**, not K=2 — AF K=1 is cheaper, more robust, and the fair same-K comparator against C109 (§10.1).
5. More rollouts (§7.1) — still binding: 24–30 episodes cannot separate 24/24 from 23/24.

---
---

# 11. HardFlow — how it works, and HardFlow vs DPCC

You asked for this because HardFlow is *theoretically* the better projector and you want to know whether the current numbers justify a threshold sweep. Short answer: **HF is theoretically better, is empirically not better here, and the reason it is not better tells you exactly where the optimisation space is — but it is not in the threshold, and lowering the threshold is the wrong direction.**

## 11.1 What HardFlow actually does

Source: `flow_matcher_v3_meanflow/sampling/hardflow_projection.py` (686 lines; the `flow_matcher_v3_hardflow` copy is the 624-line original). The sampler loop is `HardFlowSampler.sample()`, lines 490–523. Per ODE step `k` (τ_k = k·dt, dt = 1/K):

```
1.  V      = f(X, τ_k)                          # velocity, 1 network pass
2.  X_ref  = X + V·dt                           # plain Euler step
3.  active = (k >= (1 − threshold)·K) or (k == K − 1)
4.  if active:
5.      V_next = f(X_ref, τ_next)               # SECOND network pass
6.      X1_ref = X_ref + (1 − τ_next)·V_next    # extrapolate to the ENDPOINT x₁
7.      X1_proj = NLP.solve(X1_ref, τ_next)     # project the CLEAN trajectory
8.      X_next  = X_ref + τ_next·(X1_proj − X1_ref)   # τ-scaled blend back
9.  else:
10.     X_next  = X_ref
```

The two structural differences from DPCC — which projects the *current iterate* in place and keeps the full correction (`x, cost = projector.project(x, constraints)`, `models/diffusion.py:265-268`) — are:

- **What gets projected.** DPCC hands the NLP the half-integrated sample `x_τ`, which is a noisy object where "dynamics" and "obstacle clearance" are only approximately meaningful. HF first extrapolates to the predicted clean endpoint `x₁` and projects *that*, then maps the correction back. This is the same idea as projecting the x₀-prediction rather than the noisy latent in diffusion guidance.
- **How much of the correction is applied.** DPCC applies 100 % of it at every active step. HF scales by `τ_next`, so a correction at τ = 0.1 is applied at 10 % strength and one at τ = 1.0 at full strength. Early, unreliable corrections cannot wreck the sample.

The NLP itself (`HardFlowNLP`, lines 129–338) is a CasADi/IPOPT program over the whole horizon that enforces, simultaneously: the halfspace/obstacle constraint set, **linear transition dynamics** `A·s_t + B·a_t + c = s_{t+1}` (lines 300–308), and **input saturation** `−1 ≤ a_t ≤ 1` (lines 311–315). So it returns a *dynamically feasible* trajectory, not merely a geometrically feasible one. On failure it keeps the last IPOPT iterate rather than aborting (line 332–338) and increments `n_failures`.

## 11.2 The theoretical case for HF — it is sound

1. **The NLP sees a physically meaningful object.** Constraints and dynamics are evaluated on the predicted clean trajectory at every τ, not on a noisy iterate.
2. **Feasibility is enforced, not encouraged.** Hard constraints inside an IPOPT solve, including dynamics and actuator limits, versus DPCC's projection onto the constraint set alone.
3. **The τ-blend is a proper annealing schedule** — the correction strength grows with confidence in the sample.
4. **The forced final solve carries the safety guarantee** regardless of threshold (line 502, the `or (k == K − 1)` clause).

None of this is wrong. It is simply not the binding constraint on this benchmark.

## 11.3 The cost model — HF is structurally ~3× DPCC

Step 5 above costs an **extra network pass per active step**, so HF's NFE per replan is `K + n_active` against `K` for the plain sampler. On top of that the NLP is a full-horizon IPOPT solve with dynamics and saturation constraints, against DPCC's cheaper set projection. Measured, same generator, same seeds:

| generator | selection | DPCC s+c | DPCC lat. [s] | HF s+c | HF lat. [s] | **HF/DPCC** | HF `nfe_total` |
|---|---|---|---:|---|---:|---:|---:|
| AlphaFlow K=2 | `r-tightened` | **30/30** | 0.0202 | **30/30** | 0.0673 | **3.34×** | 755 |
| AlphaFlow K=2 | `t-tightened` | 28/30 | 0.0228 | **30/30** | 0.0686 | 3.00× | 751 |
| AlphaFlow K=2 | `c-tightened` | 6/30 | 0.0190 | **24/30** | 0.0667 | 3.50× | 1402 |
| MeanFlow K=2 | `r-tightened` | **29/30** | 0.0251 | 27/30 | 0.0788 | 3.14× | 771 |
| MeanFlow K=2 | `t-tightened` | **29/30** | 0.0253 | **29/30** | 0.0801 | 3.17× | 783 |
| MeanFlow K=2 | `c-tightened` | 3/30 | 0.0236 | **23/30** | 0.0790 | 3.34× | 1423 |

Per episode (`n_steps × latency`, 5 seeds, 30 episodes):

| generator | selection | DPCC | HardFlow | ratio |
|---|---|---|---|---:|
| AlphaFlow K=2 | `r-tightened` | 30/30, **1.36 s** | 30/30, 4.51 s | 3.31× |
| AlphaFlow K=2 | `t-tightened` | 28/30, **1.61 s** | 30/30, 4.58 s | 2.84× |
| AlphaFlow K=2 | `c-tightened` | 6/30, 3.43 s | **24/30**, 6.86 s | 2.00× |
| MeanFlow K=2 | `r-tightened` | 29/30, **1.77 s** | 27/30, 5.46 s | 3.08× |
| MeanFlow K=2 | `t-tightened` | 29/30, **1.73 s** | 29/30, 5.60 s | 3.24× |
| MeanFlow K=2 | `c-tightened` | 3/30, 4.40 s | **23/30**, 8.41 s | 1.91× |

**Verdict on the headline comparison: HF costs ~3× DPCC and buys nothing on the variants that already work.** On `r-tightened` and `t-tightened` the two are within ±2 episodes of each other out of 30 — statistically indistinguishable (§9.2's machinery applies) — while HF is 2.8–3.3× more expensive. If the goal is the cheapest safe controller, DPCC wins on this benchmark.

## 11.4 Where HF *is* clearly better — it rescues `dpcc-c`, and that identifies the K=2 bug

The one large, unambiguous HF win is on the `-c` selection rule, and it is large: **AF 6/30 → 24/30, MF 3/30 → 23/30.** This is the §10.6 collapse, and HF fixing it explains the mechanism.

`-r`/`-t`/`-c` are **candidate-selection rules**, not constraint sets (`scripts/eval.py:210-211`): `-t` = `temporal_consistency`, `-c` = `minimum_projection_cost`, `-r` = the default. `-c` ranks the candidate fan by the accumulated proximity cost `Σ‖x1_proj − x1_ref‖²` and keeps the **least-corrected** trajectory.

Under DPCC at K=2 that cost is measured on a **half-integrated iterate at τ = 0.5**. A candidate that barely moves is trivially near-feasible, so its projection cost is ≈ 0 and it wins the ranking — which is exactly the observed failure signature in §9.8: **199 steps (the cap), `n_success` = 0, `total_violations` = 0.0.** The selection rule is picking a stalled trajectory because standing still is cheap to project.

Under HF the same cost is measured on the **predicted clean endpoint x₁**, where a stalled trajectory is *not* cheap — it fails to reach the goal, so the NLP has to move it. The ranking signal becomes meaningful and the collapse disappears.

> **This is the strongest argument for HF in the whole batch, and it is not a speed or safety argument — it is that HF makes `minimum_projection_cost` a valid selection criterion at low K, which DPCC does not.** It also confirms §10.6's diagnosis and hands you the fix: either use HF for `-c`, or compute DPCC's ranking cost on an extrapolated endpoint instead of the current iterate (a much cheaper change than switching projectors).

## 11.5 HF needs high K — which is the wrong regime for this project

Standalone HF family (`flow_matching_v3_hardflow`, FlowMatchingODE generator), HF arm vs its own DPCC control arm on the identical model and seed. **[n=1 seed, seed 6, 6 episodes]** except CAND_42:

| cand | K | threshold | mpc | HF s+c | HF lat. [s] | HF ep [s] | DPCC s+c | DPCC lat. [s] | DPCC ep [s] |
|---|---:|---|---|---|---:|---:|---|---:|---:|
| CAND_39 | **2** | 0.0 | 1 | **2/6** | 0.0782 | 4.15 | **6/6** | 0.0261 | **1.91** |
| CAND_40 | **5** | 0.0 | 1 | **3/6** | 0.1888 | 10.76 | **6/6** | 0.1101 | **6.97** |
| CAND_35 | 10 | 0.0 | 1 | 6/6 | 0.3741 | 25.75 | 6/6 | 0.1996 | **12.61** |
| CAND_37 | 20 | 0.0 | 1 | 6/6 | 0.7460 | 51.47 | 6/6 | 0.4745 | **29.50** |
| CAND_36 | 20 | 0.5 | 1 | 6/6 | 0.4876 | 32.99 | 6/6 | 0.4751 | **29.53** |
| CAND_38 | 20 | 0.0 | 4 | 6/6 | 3.0081 | 207.56 | 6/6 | 0.4768 | **29.64** |
| CAND_44 | 20 | 0.5 | 4 | 6/6 | 1.0034 | 103.52 | 6/6 | 0.4733 | **29.42** |
| CAND_42 | 20 | 0.5 | 4 | **30/30** | 1.8225 | 182.92 | **30/30** | 0.4709 | **29.78** | *(5 seeds)* |

**HF collapses at K=2 (2/6) and K=5 (3/6) while DPCC is 6/6 on the identical model and seed.** It only reaches parity from K=10 up, and never beats DPCC on cost at any K.

The reason is line 6 of the algorithm: `X1_ref = X_ref + (1 − τ_next)·V_next`. At K=2 the first active step is at τ_next = 1.0 — fine — but at K=2 the *sample* reaching that point came from a single coarse Euler step, and at K=5 the extrapolation `(1 − τ)·V` spans up to 0.8 of the flow with a single velocity. **HF projects an endpoint estimate that is only as good as a first-order extrapolation, and that estimate degrades exactly as K falls.** The NLP then solves accurately — for the wrong target.

Since §10 established that the entire cost advantage of this line of work lives at **K = 1–2**, and HF is broken precisely there, HF as currently implemented is incompatible with the project's operating point.

## 11.6 The threshold — polarity, and why "lower" is the wrong direction

**Polarity first, because it is counter-intuitive** (`resolve_activation_threshold`, lines 345–370, and the gate at line 502):

> `threshold` = the fraction of the **late** trajectory over which the NLP is active. **Higher = MORE projection.** `1.0` → every step, `0.5` → last half, `0.0` → **terminal solve only**. Identical polarity to DPCC's `diffusion_timestep_threshold`.

So "trying a lower threshold" means **less** projection, not more. Three things follow from the data:

**(a) At K = 1–2, lowering the threshold does literally nothing.** The gate is `(k >= (1−thr)·K) or (k == K−1)`. The `or` clause forces the final step unconditionally, so at K=1 and K=2 the solve count is pinned at **exactly 1** for every threshold in [0, 1]. The current deployed value is 0.5 (`config/visual_aligning_eval.yaml`, and `diffusion_timestep_threshold: 0.5` in the eval yaml). **A threshold sweep at K=2 will produce identical numbers** — the same is true for the DPCC arm. Do not spend cluster time on it.

**(b) At high K, lowering the threshold makes HF *slower*, not faster.** This is the surprising one, and it is consistent across both mpc settings at K=20:

| K=20 | threshold 0.0 (terminal only, 1 solve) | threshold 0.5 (last half, 10 solves) | change |
|---|---:|---:|---|
| mpc1 | 0.7460 s/step | **0.4876 s/step** | **−35 %** |
| mpc4 | 3.0081 s/step | **1.0034 s/step** | **−67 %** |

Fewer solves cost *more* wall time. The explanation is solver conditioning: with terminal-only projection the NLP must drag a fully unconstrained endpoint into the feasible set in one shot, so IPOPT burns iterations (and `solve_limited` may bail, keeping a bad iterate, line 330–338). With projection over the last half, each solve starts near-feasible and converges quickly. **Gradual projection is cheaper per episode than one big projection.**

**(c) Quality is already saturated at 6/6 for every K≥10 threshold setting**, so there is no quality headroom for the threshold to recover either.

**Conclusion on your plan: the threshold sweep as described will not produce a result.** At K=2 it is a no-op by construction; at K=20 the direction you want to move (lower) is the direction the data says is worse. If you sweep anything, sweep **upward** — `threshold = 1.0` (project every step) is untested and is the only setting for which no run exists.

## 11.7 Where the optimisation space actually is

HF is not underperforming because of a tuning parameter. It is underperforming for two structural reasons, and both are fixable:

**1. The endpoint estimate is a first-order extrapolation — and MeanFlow makes it exact for free.**
Line 6 approximates `x₁ ≈ x_τ + (1−τ)·v(x_τ, τ)`, which is a one-step Euler jump across the remaining flow — accurate only for small `(1−τ)`, i.e. large K. **But MeanFlow's entire parameterisation is the average velocity `u(x, t, h)` over an interval `h`** (`mf_diffusion.py:165`, `_predict_velocity(x, cond, t, h=...)`; the engine takes `h` explicitly). Setting `h = 1 − τ` gives `x₁ = x_τ + (1−τ)·u(x_τ, τ, h=1−τ)` **exactly, by construction, in one network call** — no extrapolation error, at any K.

> This is the single highest-value idea in this section: **replace HF's Euler endpoint extrapolation with a MeanFlow one-shot endpoint jump.** It costs the same one extra network pass HF already pays, removes the error term that breaks HF at K=2–5, and it is only available because the generator is a MeanFlow. That is a real MF-specific contribution — arguably a better one than the speed claim in §§1–9, which §10.4 showed is really a projection-count effect available to any low-K generator.

**2. The NLP is doing more work than the benchmark requires.** Full-horizon IPOPT with dynamics and saturation, ~8–60 ms per solve, against DPCC's cheaper set projection at comparable quality. On `avoiding-d3il` the constraint set is simple enough that the extra machinery has no headroom to show value. If HF is to be justified it needs a benchmark where dynamic feasibility actually binds — UAV, or the obstacles/dynamics constraint types that are configured in `projection_eval.yaml` but not exercised in this halfspace-only batch.

**3. Cheap partial win available now:** the `-c` rescue in §11.4 does not require HF. Computing DPCC's `minimum_projection_cost` ranking on an extrapolated (or MeanFlow-exact) endpoint instead of the current iterate should fix the K=2 `-c` collapse at ~zero added cost, since the ranking is already computed — only the point at which it is evaluated changes.

## 11.8 Run queue for the HF thread

1. **Do not run the threshold sweep at K=2** — it is a no-op (§11.6a). If you want a threshold data point, run `threshold = 1.0` at K=10/20, the only untested setting.
2. **MeanFlow-exact endpoint in HF** (§11.7.1). Replace lines 5–6 of the sampler with a single `u(x, τ, h=1−τ)` call and re-run the K-ladder at K=1, 2, 5. **This is the experiment that would make HF viable at the project's operating point**, and it is a ~10-line change to `hardflow_projection.py`.
3. **Endpoint-based ranking for DPCC `-c`** (§11.7.3) — cheap, and directly targets the §10.6 bug.
4. **Complete the HF K-ladder on 5 seeds.** Everything below K=20 in §11.5 is one seed. The K=2 (2/6) and K=5 (3/6) failures are the load-bearing claims of this section and they rest on 6 episodes each.
5. **Test HF where dynamics bind** — `obstacles` / `dynamics` constraint types, or the UAV task. HF's advantage is dynamic feasibility, and the halfspace-only benchmark cannot show it.

**Bottom line:** HF is theoretically better, currently 3× more expensive for equal quality, and broken at K≤5 where this project operates. The threshold is not the lever — the endpoint extrapolation is, and MeanFlow happens to be the one generator that can make it exact.
