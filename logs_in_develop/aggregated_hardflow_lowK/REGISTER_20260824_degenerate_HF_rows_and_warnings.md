# Marking register — which HardFlow rows are NOT HardFlow, and which of them still mean something

**Date:** 2026-08-24 · **Type:** marking register (no run, no result changed)
**Scope:** AGGREGATED across every generation with a HardFlow arm — Gen12, Gen3v6, Gen3v7, Gen14,
Gen15 (UAV + visual aligning), Gen16.
**Companions:** [`CHANGELOG_20260824_hardflow_terminal_nfe_and_K1_guard.md`](./CHANGELOG_20260824_hardflow_terminal_nfe_and_K1_guard.md)
(the code side) · [`../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md`](../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md)
(the derivation).
**Nothing in the existing corpus was edited.** This register is the marking; it names rows, it does
not rewrite them.

---

## 0. The one-line rule

> A row is **HardFlow** only if `n_genuine ≥ 1`, where
> `n_genuine = max(K − floor((1−A)·K), 1) − 1`.
> `n_genuine == 0` ⇒ the row is `Π_S(Euler sample)` — **sample-then-project**, i.e. DPCC's
> algorithm with IPOPT instead of SLSQP.

Runs from 2026-08-24 onward record `n_active` / `n_genuine` / `degenerate` in the artifact and print
a `[hardflow][DEGENERATE]` banner. **Everything before that date must be classified by the table
below**, because the field does not exist in those artifacts. The derivation is exact — the gate is
pure arithmetic — so no re-run is needed to classify anything.

---

## 1. 🔴 The shipped `A` is NOT uniform across generations

This is the trap. "K ≤ 2 is degenerate" is **true for five generations and false for Gen12**, because
Gen12 ships `activation_threshold: 1.0` while everything else ships (or inherits) `0.5`.

| Gen | config carrying the default | shipped `A` | degenerate at |
|---|---|---|---|
| **Gen12** (FMv3ODE) | `config/hardflow_projection_eval.yaml:127` | **1.0** | **K = 1 only** |
| Gen3v6 (MeanFlow) | `config/meanflow_projection_eval.yaml:152` | 0.5 | K = 1, 2 |
| Gen3v7 (α-Flow) | `config/alphaflow_projection_eval.yaml:147` | 0.5 | K = 1, 2 |
| Gen14 / Gen15 visual aligning | `config/visual_aligning_eval.yaml:459` = `null` → inherits `diffusion_timestep_threshold` = 0.5 | 0.5 | K = 1, 2 |
| Gen15 UAV | `config/uav_mix.py:222` | 0.5 | K = 1, 2 |
| Gen16 visual avoiding | `config/visual_avoiding_mix_eval.yaml:139` | 0.5 | K = 1, 2 |

`HFFM_ACT_THRESHOLD` overrides all of these per job, and since Fix_9 (`808cb1a4`, 2026-08-07) it
lands in the results-folder name as an `A` token — so for post-Fix_9 folders you can read the true
`A` off the path. **Pre-Fix_9 folders carry no `A` token; use the generation default above, and see
warning W4.**

### Genuine-step count, both defaults

| K | A = 0.5 (five gens) `n_active` / `n_genuine` | A = 1.0 (Gen12) `n_active` / `n_genuine` |
|---|---|---|
| **1** | 1 / **0** ❌ | 1 / **0** ❌ |
| **2** | 1 / **0** ❌ | 2 / 1 ✅ |
| 3 | 2 / 1 ✅ | 3 / 2 ✅ |
| 5 | 3 / 2 ✅ | 5 / 4 ✅ |
| 10 | 5 / 4 ✅ | 10 / 9 ✅ |
| 20 | 10 / 9 ✅ | 20 / 19 ✅ |

`A = 0.0` (terminal-only) is degenerate at **every** K — it appears in the Gen12 `flow_matching_v3_hardflow`
ladder (CAND_35/37/39/40) and in the Gen3v6 threshold sweep.

---

## 2. The marking — result MDs carrying degenerate HardFlow rows

Classified by the §1 rule. "Degenerate rows" = the K values in that document whose HF arm has
`n_genuine == 0`. ✅ = the document also contains non-degenerate HF rows.

| result document | K values reported | assumed `A` | degenerate rows | still meaningful? |
|---|---|---|---|---|
| `Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ⚠️ partly — see §3 C2/C3 |
| `Gen3v6_MeanFlow/DA/DA_20260811_MF_UNet32_full5seeds_avoiding.md` | 1,2,5,10,20 ✅ | swept 0.0/0.1/0.5/1.0 | **K=1 (all A); K=2 at A≤0.5; every K at A=0.0** | ✅ yes — §3 C1; carries the headline row |
| `Gen3v6_MeanFlow/DA/DA_20260817_H16_horizon_MF_UNet.md` | 1,2,5,20 ✅ | 0.5 | **K=1, K=2** | ⚠️ §3 C2 |
| `Gen3v6_MeanFlow/DA/DA_20260818_H16_replan8_MF_UNet.md` | 1,2,5,10 ✅ | 0.5 | **K=1, K=2** | ⚠️ §3 C2 |
| `Gen3v6_MeanFlow/DA/DA_20260820_HF_lower_avgtime_batchsize_confound.md` | 1,2,10 ✅ | 0.5 | **K=1, K=2** | ✅ — it is *about* W3 |
| `Gen3v6_MeanFlow/fix_4/RESULTS_..._post_fix_K_sweep.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ⚠️ §3 C2 |
| `Gen3v6_MeanFlow/fix_5/RESULTS_..._verification_post_fix_sweep.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ⚠️ §3 C2 |
| `Gen3v6_MeanFlow/Fix_8_Unet/RESULTS_Fix_8_unet_width32_rerun_24317_24334.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ⚠️ §3 C2 |
| `Gen3v6_MeanFlow/U3/INSIGHT_..._hardflow_first_run_K2.md` | 1,2,5,20 | 0.5 | **K=1, K=2** | ❌ already void (W5 — pre-`fix_4` σ bug) |
| `Gen3v7_AlphaFlow/U3/RESULTS_Gen3v7_U3_hardflow_K_sweep.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ⚠️ §3 C2 |
| `Gen12/DA/DA_20260803_HardFlow_activation_threshold_0p1.md` | 1,2,5,10,20 ✅ | swept, incl. 0.1 / 0.0 | **K=1 always; K=2 at A≤0.5; all K at A=0.0** | ✅ §3 C1 |
| `Gen12/DA/DA_20260805_HardFlow_Pareto_Study.md` | 1,2,5,10,20 ✅ | 1.0 (Gen12 default) | **K=1 only** | ✅ §3 C1 |
| `Gen12/fix_3/RESULTS_Gen12_Ksweep_lowK.md` | 2,5,10,20 ✅ | 1.0 | *(none — K=2 at A=1.0 has 1 genuine step)* | ✅ but see §3 C4 |
| `Gen12/U4/RESULTS_Gen12_U4_threshold_K20.md` | 2,20 ✅ | swept | K=2 rows at A ≤ 0.5 | ✅ |
| `Gen13/*` (iMF-in-HardFlow: `INSIGHTS_first_run`, `U10_threshold_sweep`, `U11`, `CLOSURE`, `U9.2`) | 1,2,5,10,20 ✅ | per-run | **K=1 always; K=2 where A ≤ 0.5** | ⚠️ §3 C2 (iMF has the same two-time issue) |
| `Gen14/U7/DA_20260805_hardflow_first_run.md` | 2 | 0.5 | **all rows (K=2)** | ⚠️ §3 C3 — no non-degenerate row in the doc |
| `Gen14/U7/DA_20260805_n30_massive_K2_all_variants.md` | 2 | 0.5 | **all rows (K=2)** | ⚠️ §3 C3 |
| `Gen14/U7/DA_20260823_hardflow_vs_dpcc_visual_aligning.md` | 2 | 0.5 | **all rows (K=2)** | ⚠️ §3 C3 — title claims an HF-vs-DPCC verdict it cannot support |
| `Gen14/U8/DA_20260823_Gen14_U8_mf_dit_visual_aligning.md` | 2 | 0.5 | **all rows (K=2)** | ⚠️ §3 C3 + W6 (mf_dit provenance) |
| `Gen14/U7/DA_20260806_*`, `DA_20260809_*`, `FiLM_V2_DA/*` | 2, 10, 20, 100 ✅ | 0.5 | **K=2 rows** | ✅ |
| `Gen15/DA/DA_20260819_fm_vs_mf_3scenes_K10.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ✅ §3 C1 (`fm` engine — the clean one) |
| `Gen15/DA/DA_20260820_fm_K_sweep_corridor.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ✅ §3 C1 (`fm`) |
| `Gen15/DA/DA_20260824_af_sit_K_sweep_corridor.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ⚠️ §3 C2 (`af` — two-time confound) |
| `Gen15/DA/DA_20260816_*`, `DA_20260817_*` | 10 | 0.5 | *(none)* | ✅ |
| `Gen16/init/DA_20260823_Gen16_mf_visual_avoiding_first_results.md` | 2, 20 ✅ | 0.5 | **K=2 rows** | ⚠️ §3 C2 for the K=2 rows |
| `HF_Batch_Parity/DA_20260824_mpc1_parity_MF_vs_FM.md` | 2, 10, 20 ✅ | 0.5 | **K=2 rows** | ✅ — it is *about* W3 |
| `Data_Analysis/DA_Result_Curated_MD/DA_20260819_ntrials20_*.md` | 1,2,5,10,20 ✅ | 0.5 | **K=1, K=2** | ✅ §3 C1 |
| `Data_Analysis/DA_Result_Curated_MD/SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md` | headline is **K=1** | 0.5 | **the headline row** | ✅ number stands, label does not — §3 C1 |

---

## 3. Does a degenerate run still make sense? Four verdicts

### C1 — ✅ YES, and it is now a *better* experiment than it was

**Which:** `fm`-engine arms, and any single-engine arm-B-vs-arm-C comparison on the same checkpoint
and seed. Gen15 `fm` K-sweeps, Gen12 (no two-time field at all), the curated snapshot headline.

**Why it is valid:** at K=1 both arms compute `Π_S(one Euler step)` from the *same* sample, over the
*same* `constraint_list`. The comparison is real — it just measures **IPOPT vs SLSQP plus variable
scope**, not HardFlow. And since HFK1 (2026-08-24) removed the terminal lookahead call, arm C costs
**1 NFE at K=1, exactly like arm B** — so the comparison is matched-budget for the first time.

**How to label it:** *"IPOPT terminal projection"*, never *"HardFlow"*. The number stands; the
mechanism attribution does not. The gap is not noise either — the feasible set is **nonconvex**
(`sphere_outside`), so two solvers can land on different valid local minima. Read a gap as a
solver-choice effect, not as evidence about the algorithm.

### C2 — ⚠️ PARTLY: degenerate **and** confounded on `mf` / `af` / `imf`

**Which:** every `mf`, `af` and `imf` HardFlow row at K ∈ {1, 2}.

**Why:** the D3 confound. The engines' own samplers integrate the trained **interval** field
(`h = dt`); the HardFlow sampler deliberately queries the **instantaneous** field (`h = 0`). At K=20
the two integrators converge; at K=1 arm C throws away the entire reason MeanFlow exists and hands
the NLP a first-order Euler extrapolation of a curved flow across `Δt = 1`. So arm C is not even
projecting the same base trajectory as arms A/B.

**Consequence:** a low-K B-vs-C loss on `mf`/`af`/`imf` is **not** evidence that HardFlow is worse
than DPCC — two independent effects stack, and neither is the algorithm. Use these rows for the
engine comparison (A vs B), not for the projector comparison (B vs C). **`fm` is the only clean
engine for a low-K projector study.**

### C3 — ⚠️ WEAK: documents whose *only* HF rows are degenerate

**Which:** `Gen14/U7/DA_20260805_hardflow_first_run.md`, `DA_20260805_n30_massive_K2_all_variants.md`,
`DA_20260823_hardflow_vs_dpcc_visual_aligning.md`, `Gen14/U8/DA_20260823_*`.

**Why it matters more here:** these run at K=2 with A=0.5 and contain **no** non-degenerate HF row,
so any conclusion the document draws about HardFlow rests entirely on rows where HardFlow did not
run. `config/visual_aligning_eval.yaml:457-458` already warns that K=2 has almost no threshold
resolution. The measurements are fine; the *headline* is unsupported.

**What would fix them:** one K=5 (or K=3) run at the same checkpoint. Cheap, and it is the first
point where the claim these documents want to make becomes testable.

### C4 — ✅ YES and non-degenerate, but do not generalise it

**Which:** `Gen12/fix_3/RESULTS_Gen12_Ksweep_lowK.md` and other Gen12 rows at K=2.

Gen12 ships `A = 1.0`, so its K=2 has **1 genuine step**. It is not degenerate. But that single step
sits at `τ⁺ = 0.5` — the largest lookahead any non-terminal step can carry, and precisely the regime
the paper's Theorem 4 bound blows up in and its Remark 9 tells you to skip. **"Non-degenerate" is
not "trustworthy."** Treat a Gen12 K=2 HF row as the weakest possible positive evidence.

---

## 4. Warnings that must travel with any HardFlow row

Independent of degeneracy. Each one invalidates a *different* axis.

| # | warning | boundary | what it invalidates |
|---|---|---|---|
| **W1** | **Degeneracy** — `n_genuine == 0` | §1 table | the *label*, not the numbers. Row is sample-then-project. |
| **W2** | **NFE / wall-time re-baseline** | **2026-08-24** (HFK1) | arm C used to spend `K + n_active` NFE, now `K + n_active − 1`. **Any HF NFE or avg_time comparison crossing this date is invalid.** S&C numbers are unaffected — trajectories are bit-identical. |
| **W3** | **Batch parity** | **2026-08-20** (`0f1aa7fc`) | before it, `hardflow.batch_size` defaulted to 1 while the DPCC arms ran 4. Both loop serially around their CPU solve, so **every pre-2026-08-20 arm-B-vs-arm-C wall-clock number is meaningless** — arm C looked ~25 % cheaper while its per-solve cost is ~1.8–2.2× DPCC's. (`83471f8d`, 2026-08-23 added `FMPCC_MPC_BATCH` for the arm-A/B side.) |
| **W4** | **Provenance contamination** | **Fix_9 `808cb1a4`, 2026-08-07** | pre-Fix_9 HF folders carry **no `A`/`B` token in the path**, so runs with different `hf_batch` / `A` wrote into the *same* directory. Named casualties: the AF and `mf_dit` HardFlow folders (seed 6 at `B4,A0.5` vs seeds 7–10 at `B1,A1.0`), including the `4.58 s/ep` figure — already retracted. |
| **W5** | **Init-noise σ bug** | **`fix_4`, 2026-07-30** | Gen3v6 arm C sampled at σ=0.5 while arms A/B sampled at σ=1.0. Every `hardflow_new-*` number from jobs **23981 / 24021 / 24022 / 24023** is void. Arms A and B are unaffected. |
| **W6** | **Gate rounding** | **Gen12 `fix_8`, 2026-08-07** | the activation gate moved CEIL → FLOOR. No-op at integer `(1−A)·K`, so most rows are unmoved — but a pre-fix_8 row at non-integer `(1−A)·K` (e.g. K=5 A=0.5, K=5/10 A=0.9) ran **one fewer** projection step than its stated threshold implies. |
| **W7** | **`A` is per-generation** | always | Gen12 = 1.0, everything else = 0.5. Do not blanket-apply "K ≤ 2 is degenerate" — see §1. |

---

## 5. What I would actually do with this

1. **Relabel, don't delete.** Every degenerate row keeps its number and gets the tag
   *"terminal-only IPOPT projection (degenerate — n_genuine = 0)"*. That includes the curated
   snapshot's architecture-matched headline: the 1.000 / 63.77-steps / 2.64 s-per-episode figure is
   sound, the word "HardFlow" attached to it is not.
2. **Keep K=1 in the sweeps.** It is now matched-NFE against DPCC and is the cheapest safe operating
   point. It answers "which one-shot projector is better", which is a real question.
3. **Add K=3 or K=5 wherever a document's only HF rows are K ≤ 2** (§3 C3). One run turns four
   unsupported headlines into testable ones.
4. **Run the low-K projector study on `fm` only** (§3 C2). `mf`/`af`/`imf` cannot answer it.
5. **Two free reads, no cluster time** — both already in `infos`, both still undone:
   `nlp_failures` on Gen12 CAND_39/CAND_40 (the 2/6 and 3/6 terminal-only collapses: is IPOPT bailing
   and keeping an infeasible iterate?), and per-K `nlp_solves` against `n_active` on the `bbunet`
   ladder (confirms the degeneracy operationally).

---

## 6. Status

Register only — **no existing result document was edited**, no number was changed, nothing was run.
The classification in §1–§2 is exact arithmetic over `K` and `A` and needs no re-run to be trusted.
Whether to propagate the §5.1 relabelling into the curated snapshot and the Gen14 DA headlines is a
call I have not made.
