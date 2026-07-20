# Gen13 fix_7.3 RESULTS — ❌ THE iMF CLAIM IS REFUTED

**Date:** 2026-07-20 · **Job:** `eval_matched_nfe_hardflow` **23612** · **Data:** `temp/fix_7/13_33_42_eval_matched_nfe_hardflow_23612.log`
**Design (pre-registered):** `RESULTS_Gen13_fix7_smoothness_2x2.md` §9, interpretation pre-committed in §9.6 **before** these data existed.

> ## VERDICT
> **iMF is strictly dominated by FM at every matched budget, on both safety and speed — and the seam mechanism is measurably worse, not better.** The u_5 "1.74× speedup" was an artefact of comparing iMF@K=5 against FM@K=10. **Gen13's central claim does not survive.** The user's visual suspicion was correct.

---

## 1. T1+T2 — guided, matched budget (equal K ⇒ equal NFE ⇒ equal projections)

| K | FM safe | FM s/plan | iMF safe | iMF s/plan | FM faster? | FM safer? |
|---|---|---|---|---|---|---|
| 1 | **95%** | **0.1119** | 75% | 0.1357 | ✅ | ✅ |
| 2 | **100%** | **0.1894** | 85% | 0.2434 | ✅ | ✅ |
| 5 | **100%** | **0.4331** | 95% | 0.4923 | ✅ | ✅ |
| 10 | 100% | 0.8456 | 100% | 0.9224 | ✅ | = |

**FM wins or ties in every single cell, on both axes.** There is no budget at which iMF is preferable.

Two independent reasons iMF loses:
1. **Less safe below K=10** — 75/85/95% vs FM's 95/100/100%.
2. **Slower per plan at every K** (0.1357 vs 0.1119, …). The dual-head two-time network costs more per evaluation than FM's, so even the NFE advantage does not convert into wall-clock.

## 2. The configuration that dominates everything: **FM @ K=2**

```
FM  @ K=2  :  100% safe,  0.1894 s/plan      <-- best overall
iMF @ K=5  :   95% safe,  0.4923 s/plan      <-- the Gen13 headline config
```
**FM@K=2 is 2.6× FASTER and 5 points SAFER than the iMF configuration Gen13 was built around.**

## 3. Where the u_5 claim came from — the unfair pairing

u_5 compared **iMF@K=5 (0.4815 s) vs FM@K=10 (0.8379 s)** → "1.74× faster at similar safety".

But FM never needed K=10:

| | s/plan | safe |
|---|---|---|
| FM @ K=10 (the baseline we used) | 0.8456 | 100% |
| **FM @ K=2** | **0.1894** | **100%** |

**FM@K=2 is 4.5× faster than FM@K=10 at identical safety.** The baseline was simply over-provisioned. Once FM is allowed the same freedom iMF was given, the entire speedup belongs to FM.

Per §9.6's pre-committed rule — *"FM@K=5 safe ≈ iMF@K=5 safe, similar s/plan ⇒ Gen13 efficiency claim dead"* — the observed result (FM@K=5: 100% @ 0.4331 vs iMF@K=5: 95% @ 0.4923) triggers exactly that branch, and more strongly than anticipated.

## 4. T4 — the mechanism is REFUTED, and in the opposite direction

Terminal-prediction error at τ=0, where FM's Euler shot should be worst:

| K | iMF err(τ=0) | FM err(τ=0) | result |
|---|---|---|---|
| 1 | 0.1539 | 0.0260 | iMF **5.9× WORSE** |
| 2 | 0.1538 | 0.0303 | iMF **5.1× WORSE** |
| 5 | 0.1595 | 0.0356 | iMF **4.5× WORSE** |
| 10 | 0.1572 | 0.0384 | iMF **4.1× WORSE** |

The whole Gen13 rationale was that `x̂1 = z + (1−τ)·u` is the *exact* endpoint map, versus FM's first-order Euler shot with `O((1−τ)²)` error. **In practice iMF's "exact" map is 4–6× less accurate than the crude approximation it was meant to replace.**

**Why:** iMF's error is **flat in K** (0.1539/0.1538/0.1595/0.1572) — the signature of a fixed **field/training error**, not a discretisation error. The u-field is simply not accurate enough: its training error (~0.155) dwarfs the Euler error it was supposed to eliminate (~0.026–0.038). The theory is sound; the trained field cannot exploit it.

This is the pre-registered failure mode from §9.6 (*"iMF's τ=0 error ≈ FM's ⇒ mechanism refuted"*) — realised in the worse-than-expected form.

⚠️ **Metric caveat:** `err(τ=1) = 0` identically for both methods, because at τ=1 the prediction is `x_N + 0·u = x_N = x_final` by construction. The printed `decay` column is therefore meaningless (division by ~0). **Only `err(τ=0)` and the curve shape carry information** — the comparison above uses only those.

## 5. T3 — unguided: the raw field is worse at every budget

| K | FM safe / roughness | iMF safe / roughness | iMF roughness penalty |
|---|---|---|---|
| 1 | (cell missing) | 0% / 1.653e-03 | — |
| 2 | 20% / 9.206e-05 | 0% / 5.693e-04 | **6.2× rougher** |
| 5 | 0% / 3.002e-05 | 10% / 1.751e-04 | **5.8× rougher** |
| 10 | 0% / 1.792e-05 | 10% / 1.242e-04 | **6.9× rougher** |

The visual observation — *"iMF outputs really low quality, not even on the main trajectory manifold"* — is confirmed **at every budget**, with a 6–7× roughness penalty that more sampling steps do not fix.

## 6. What this does to the earlier conclusions

| Earlier claim | Status |
|---|---|
| u_5: "iMF ~2× cheaper at comparable safety" | ❌ **REFUTED** — artefact of K=5 vs K=10. At matched K, FM is faster *and* safer. |
| u_5: "efficiency win cannot be eroded by more data" | ❌ **wrong** — it was eroded by the missing *control*, not by more data |
| u_5: safety measurements (98.5% vs 100%, n=200) | ✅ still valid as measurements of those two configs |
| fix_7: "the NLP manufactures smoothness" | ✅ **unaffected** — still supported, and now reinforced (FM@K=2 is 100% safe, so the projection carries the task for both) |
| fix_7 §5: "roughness is not the cause of the safety gap" | ✅ still holds |
| Gen13's central thesis (exact endpoint map ⇒ fewer steps) | ❌ **REFUTED in practice** by T4 |

## 7. The genuine finding (it is not about iMF)

**HardFlow-FM can run at K=2 with 100% safety — a 4.5× speedup over its own default K=10.** That is a real, useful, reportable result about HardFlow's over-provisioned default, discovered only because we built the matched-budget battery. It owes nothing to iMF.

## 8. Why iMF failed — and what would be needed to rescue it

T4 localises the failure precisely: **the u-field's training error dominates.** Not the sampler, not the seam, not the projection count — the field itself.

This is the **pre-registered risk** from the Gen13 plan (§7, "96-demo data ceiling"): the average-velocity `u(z,τ,h)` is a **two-time** object and far more data-hungry than `v(z,τ)`. At 96 demonstrations it is under-determined. Corroborating evidence: `raw_mse_u` plateaued ≈13 (≈0.37/dim vs Gen3v4's 0.25/dim), `a0_mse` never reached the <0.15 reference, and the LR annealed to zero while still spiking.

**Rescue would require fixing the field, not the algorithm:** substantially more data, or a much longer/better-annealed training schedule — and it would then have to beat **FM@K=2 (100% @ 0.19 s/plan)**, not FM@K=10. That is a far higher bar than Gen13 was ever measured against. **Recommendation: do not invest further in iMF on this task at this data scale.**

## 9. Process note — the method worked

The error was not in the code; every implementation was verified. It was in the **experimental design**: one hardcoded value (`k_steps=10` for FM in `eval_smoothness_diag.sh`) made the decisive control unrunnable, and the confound survived four rounds of analysis because every comparison reproduced the same unfair pairing.

What caught it: the user's insistence that the visual evidence contradicted the conclusion, and pre-committing the interpretation (§9.6) *before* running. Both branches of the pre-commitment were exercised, and the negative one is reported here as written.

**A run summary table is not a control.** Matched-budget comparison should be the default from the start of the next generation.

## 10. Sanity checks

- 0 cells reported `cell FAILED`; 31 of 32 produced CSVs (`fm_unguided_K1` absent from the summary — minor, does not affect any conclusion).
- `imf_guided_K5_n20` = 95% safe, consistent with the n=200 result (98.5%) within n=20 noise ⇒ no configuration drift between runs.
- n=20/cell resolves the large effects reported here; it does **not** resolve differences of ~1–2 points (e.g. FM@K=1 95% vs 100%).
