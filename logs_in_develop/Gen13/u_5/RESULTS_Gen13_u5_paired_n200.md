# Gen13 u_5 — RESULTS of the decisive paired n=200 run

**Date:** 2026-07-20 · **Job:** `eval_paired_n200_hardflow` **23602**, node i6-gpu-1, git `cf23847`
**Arms:** A = iMF `hardflow_new_imf` K=5 (21m11s) · B = FM `hardflow_new` (30m16s) · total 51 min (2h requested)
**Data:** `H16_imf_hardflow_new_K5_n200/trajectories.csv`, `H16_1e6steps_hardflow_new_10steps_n200/trajectories.csv`
**Purpose:** resolve the question n=50 could not — is iMF's safety genuinely equal to FM's, or slightly worse?

> **One-line answer:** the efficiency win is confirmed and absolute; the safety gap is now **very likely real but small (~1.5 pts)** — evidence rose from 75%→**94%** that iMF is truly worse, though it still does not reach statistical significance (p=0.248). **Gen13 remains "not a strict win," and the honest reading has shifted against parity.**

---

## 1. Headline results

| | FM `hardflow_new` (B) | iMF `hardflow_new` K=5 (A) |
|---|---|---|
| Episodes | 200 | 200 |
| **Success** | **100.0%** | **98.5%** |
| **Safety** | **100.0%** | **98.5%** |
| **Violations** | **0** | **3** (run_ids 17, 160, 181) |
| Mean steps | 50.6 | 52.1 (+3%) |
| **Compute / plan** | 0.8379 ± 0.0406 s | **0.4815 ± 0.0309 s** |
| **NFE / plan** | ~41 | **21** |
| NLP failures | — | **0** |

## 2. ✅ Validity checks — both passed

**Regression check (the one flagged in u_5 §7).** Since `env.set_seed()` runs once before the episode loop, the first 50 episodes of each 200-run must reproduce the frozen n=50 results. They do, exactly:

| Arm | First 50 violations | Frozen n=50 | Match |
|---|---|---|---|
| iMF K5 | 1 | 1 | ✅ |
| FM | 0 | 0 | ✅ |

Determinism holds and no config drifted between runs — the new numbers are trustworthy.

**Solver health.** `nlp_failures = 0` across all 200 iMF episodes (7,030 NLP solves). The 3 violations are **not** solver failures — they are approximation error, as fix_4's instrumentation was designed to distinguish.

---

## 3. Efficiency — ✅ CONFIRMED, decisively (unchanged from n=50)

| Measure | Result |
|---|---|
| NFE/plan | **21 vs 41 = 1.95×** (deterministic count, no uncertainty) |
| Compute/plan | **0.4815 vs 0.8379 s = 1.74×**, permutation test **p = 0.0** (<1/20,000) |
| Distribution overlap | **NONE** — FM's fastest episode (0.754 s) is slower than iMF's slowest (0.580 s) |

**All 200 iMF episodes were faster than all 200 FM episodes.** This replicates the n=50 finding at 4× the sample size with complete separation. The efficiency claim is settled and cannot be overturned by more data.

---

## 4. Safety — ⚠️ the picture has shifted AGAINST parity

### The statistics

| Metric | n=50 (before) | **n=200 (now)** |
|---|---|---|
| Violations | 1 vs 0 | **3 vs 0** |
| Fisher exact p | 1.000 | **0.248** |
| **P(iMF truly worse)** | 75% | **94%** |
| iMF 95% CI upper | ≤10.6% | **≤4.32%** |
| FM 95% CI upper | ≤7.1% | **≤1.83%** |

Difference in violation rate (iMF − FM), Newcombe hybrid-score 95% CI:
```
point estimate = +1.50 pts      95% CI = [−0.63, +4.32] pts
```

### How to read this honestly

- **Not statistically significant.** p = 0.248; the CI still includes zero, so equality cannot be *rejected* at the 5% level.
- **But the evidence clearly leans against parity.** P(iMF worse) climbed **75% → 94%** with the extra power. That is the expected signature of a **small real effect** becoming visible, not of noise averaging out. Had the true rates been equal, more data should have pulled this toward 50%, not 94%.
- **Non-inferiority fails at every reasonable margin:**

| Margin | Verdict |
|---|---|
| 1 pt | ❌ FAIL (upper bound 4.32) |
| 2 pts | ❌ FAIL |
| 3 pts | ❌ FAIL |
| 5 pts | ✅ pass |

So iMF can only be declared "non-inferior" if one accepts a **5-percentage-point** safety margin — far too loose for a method whose selling point is *hard* constraint satisfaction.

**Best estimate: iMF K5 carries a true violation rate around 1.5–2%, versus FM's ≤1.8%.** The gap is small but probably real.

---

## 5. Verdict against the plan §5 criteria

| Criterion | Target | Result | Status |
|---|---|---|---|
| 1. Safety parity (**non-negotiable**) | 100%, 0 violations | 98.5%, **3 violations** | ❌ **FAIL** — and now with 94% confidence the gap is real, not noise |
| 2. Efficiency win | < B2 | 1.95× NFE, 1.74× compute, zero overlap | ✅ **PASS** |
| 3. Quality not degraded | steps ≈ 50.6 ±20% | 52.1 (+3%) | ✅ PASS |

> **Gen13 has NOT beaten the baseline.** Unlike after n=50 — where the fair summary was *"statistically indistinguishable at half the cost"* — the n=200 data no longer supports a parity claim. The defensible statement is now:
>
> **"iMF achieves ~2× the efficiency of FM at a small but likely-real safety cost (~1.5 pts, 94% posterior confidence, not significant at p<0.05)."**

This is a *trade-off* result, not a Pareto win. Whether the trade is acceptable depends on the application — but for HardFlow, whose entire premise is hard constraint satisfaction, a nonzero violation rate is a qualitative concession, not merely a quantitative one.

**Caveat that cuts the other way:** FM's 0/200 still only bounds its true rate at ≤1.83%; it is *not* a proven guarantee either. Neither method has a closed-loop guarantee — the NLP enforces constraints on the *plan*, while violations occur in *execution*. This is an empirical rate comparison, and the true gap could be as small as ~0 (CI lower bound −0.63).

---

## 6. Why the gap exists (mechanism, now better supported)

The n=200 data supports the **coarse-field floor** hypothesis from `../fix_3/INSIGHTS_Gen13_first_run.md` §14.4:

1. **Not solver failure** — 0/7,030 NLP failures. Definitively excluded.
2. **Not projection count** — K=5 already saturated the K-sweep (80→94→96→98% with returns +14/+2/+2). More projections cost FM-equivalent NFE, erasing the advantage.
3. **⇒ Prediction error.** The NLP guarantees the *predicted* endpoint is feasible (constraint violation ~1e-16), but iMF's field is measurably coarser (`raw_mse_u` ≈13 over 96 dims ≈ **0.37/dim**, vs Gen3v4's ≈0.25/dim on the easier H8 task). A less accurate endpoint prediction ⇒ the executed trajectory deviates from the certified plan ⇒ occasional violation. **This is a floor that more projections cannot remove — only a better field can.**

---

## 7. What to do next (revised priorities)

The bottleneck is now clearly identified as **field quality**, not the sampler or the seam.

1. **⭐ Improve training** — the highest-value lever, and it directly targets the §6 mechanism. Concrete evidence of untapped headroom from the first run: `a0_mse` settled at 0.2–0.35 vs the <0.15 reference, and the cosine LR annealed to 0 while `raw_mse_u` was still spiking (132.6 at step 98.4k) — the Gen3v4 "froze on a noisy plateau" pattern. Try a **constant-low-LR tail** and/or a longer budget, then re-run this exact paired comparison.
2. **The Newton/MF pull-back (THEORY Level 2)** — the one iMF advantage Gen13 has **not** cashed in. Gen13 built only the "Level 1" seam (plan D8); the pull-back still uses HardFlow's `τ` gain, which `THEORY_DeepMix_HF_iMF.md` shows delivers only ~11% of the requested correction at τ=0.1. The correct Jacobian `∇F = I + (1−τ)∇u` is available by JVP *precisely because we have `u`*. This improves correction accuracy **without** adding projections — exactly what a prediction-error-limited method needs.
3. **`value_objective="consistency"`** — built, wired, dormant (`../../HF_iMF/Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md` §4). Cheap, no retraining.
4. **Foresight-fan diagnostic** (now implemented, default-off) on the 3 violating episodes (run_ids **17, 160, 181**) — inspect whether the plan was already skirting the obstacle or whether execution diverged from a good plan. This would *visually confirm or refute* the prediction-error mechanism:
   ```bash
   IMF_PLOT_FAN=1 RANDOM_REPEAT=200 IMF_K=5 bash run_scripts/eval_hardflow_new_imf.sh
   ```

**Do NOT** raise K further — §9 saturation makes it self-defeating.

---

## 8. Bottom line

The decisive experiment did its job. **Efficiency: confirmed beyond doubt** (1.95× NFE, 1.74× compute, complete distribution separation at n=200). **Safety: the parity hypothesis weakened substantially** — 94% posterior confidence in a small real gap, non-inferiority failing at any margin tighter than 5 pts.

Gen13's result is therefore an honest **trade-off, not a win**: iMF buys ~2× efficiency for ~1.5 pts of safety. The mechanism is pinned down (prediction error from a coarse field, not solver failure or projection count), which makes the next step unusually well-targeted: **fix the field (training) or fix the correction (Newton pull-back)** — not the sampler.
