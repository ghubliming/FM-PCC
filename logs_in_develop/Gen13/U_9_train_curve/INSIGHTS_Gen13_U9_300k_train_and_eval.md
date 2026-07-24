# Gen13 U9 INSIGHTS — the 300k iMF training run: the field plateaued, but guided performance jumped

**Date:** 2026-07-21 · **Data:** `temp/gen13_u9/hf_results_20260721_095317.zip` (collector output) + jobs **23613** (train) / **23624** (eval)
**Context:** `WHERE_ARE_THE_TRAINING_CURVES.md` §4b flagged a **10× training-budget confound** (FM 1e6 steps vs iMF 1e5). This run tests it at 3× budget.

> ## HEADLINE
> **3× more training did NOT improve the field (`raw_mse_u` plateaued: 14.98 → 14.14, and *rose* +4.6% in the last quarter) — yet guided safety improved dramatically (K2: 94% → 99.5%).**
> iMF@300k is now **statistically indistinguishable from FM at matched K** (p=0.54 at K=1, p=1.00 at K=2) — but **still 1.13–1.24× slower per plan**, and **FM@K=2 remains the best configuration overall.**
> The fix_7.3 refutation **stands on speed**, but its "iMF is less safe" component **no longer holds**.

---

## 1. Training: 3× budget bought almost nothing

| window | 100k run: median `raw_mse_u` | 300k run: median `raw_mse_u` | median `a0_mse` (300k) |
|---|---|---|---|
| 0–25k | 18.47 | 19.09 | 0.4458 |
| 25–50k | 15.85 | 16.19 | 0.3549 |
| 50–100k | 14.98 | 15.54 | 0.2945 |
| 100–200k | — | **14.13** | 0.2583 |
| 200–300k | — | **14.14** | 0.2364 |

**The last two windows are identical (14.13 → 14.14).** The collector's plateau check reports **Δ = +4.6%** for the final quarter — the metric *increased*. Total gain from 3× compute: **14.98 → 14.14 ≈ 5.6%**.

`a0_mse` improved more (0.281 → 0.236, −16%) and `min` fell (8.54 → 6.37), so the model is marginally better — but the **median is flat**, which is what governs typical-case behaviour.

**Verdict: the 96-demo data ceiling is confirmed, not a training-length problem.** This closes the question `WHERE_ARE_THE_TRAINING_CURVES.md` §4b left open — a 1e6-step iMF run would not be worth the ~42 h.

## 2. Eval: guided performance improved a lot anyway

| config | n | safe | violations | s/plan | steps |
|---|---|---|---|---|---|
| iMF@100k K1 | 20 | 75.0% | 5 | 0.1357 | 56.4 |
| **iMF@300k K1** | 200 | **96.5%** | 7 | 0.1260 | 68.5 |
| iMF@100k K2 | 20 | 85.0% | 3 | 0.2434 | 48.5 |
| **iMF@300k K2** | 200 | **99.5%** | **1** | 0.2342 | 55.6 |
| iMF@100k K5 | 200 | 98.5% | 3 | 0.4815 | 52.1 |
| FM@1e6 K1 | 20 | 95.0% | 1 | **0.1119** | 58.2 |
| FM@1e6 K2 | 20 | 100.0% | 0 | **0.1894** | 51.3 |
| FM@1e6 K10 | 200 | 100.0% | 0 | 0.8379 | 50.6 |

**iMF@300k at K=2 (99.5%, n=200) beats iMF@100k at K=5 (98.5%, n=200) while using less than half the compute** (0.234 vs 0.482 s/plan). Better training moved the useful operating point from K=5 down to K=2.

⚠️ Note the summary table's `:.0f` rounding displayed 99.5% as "100" — the exact figures above come from the raw CSVs.

## 3. Matched-K comparison vs FM — the fix_7.3 verdict, revisited

| K | iMF@300k | FM@1e6 | Fisher p | speed |
|---|---|---|---|---|
| 1 | 193/200 = 96.5% | 19/20 = 95.0% | **0.540** | FM **1.13×** faster |
| 2 | 199/200 = 99.5% | 20/20 = 100.0% | **1.000** | FM **1.24×** faster |

**Safety: no significant difference at either budget.** fix_7.3's finding that iMF was *less safe* at matched K was a **training artefact** — it disappears with 300k. (Caveat: FM's arm is only n=20; its CIs are wide. A matched-n FM run would tighten this.)

**Speed: FM still wins at every K.** iMF's dual-head two-time network costs more per evaluation, so equal K never yields equal wall-clock.

**Best configuration overall is unchanged:**
```
FM  @ K=2      : 100.0% safe, 0.1894 s/plan
iMF @ K=2 (300k):  99.5% safe, 0.2342 s/plan   (1.24x slower)
```

## 4. Revised status of the Gen13 claims

| Claim | fix_7.3 verdict | after U9 (300k) |
|---|---|---|
| iMF is **less safe** at matched K | ❌ refuted iMF | ✅ **overturned — no significant difference** (p=0.54 / 1.00) |
| iMF is **slower** at matched K | ❌ refuted iMF | ❌ **still true** (1.13–1.24×) |
| iMF gives an **efficiency win** | ❌ refuted | ❌ **still refuted** — FM@K=2 dominates |
| The gap is a **training** problem | open | ❌ **no** — the field plateaued; 3× compute changed the median by 5.6% |
| The gap is a **data-ceiling** problem | suspected | ✅ **confirmed** |

**Net: Gen13's efficiency claim remains dead, but for a cleaner reason than fix_7.3 stated.** iMF is not worse at planning — it is *architecturally more expensive per evaluation*, and its field cannot be improved by more training at this data scale.

## 5. What did NOT improve: the raw field

| config | safe | plan roughness |
|---|---|---|
| iMF@100k K1 unguided | 0/20 | 1.653e-03 |
| **iMF@300k K1 unguided** | **0/200** | **1.807e-03** ← *worse* |
| iMF@100k K2 unguided | 0/20 | 5.693e-04 |
| **iMF@300k K2 unguided** | **1/200 (0.5%)** | **6.636e-04** ← *worse* |

Unguided iMF remains at **~0% success and slightly rougher** after 3× training — exactly consistent with the plateaued curve.

**So why did guided performance improve so much?** The warm-start's *raw* plan did get better (`plan_roughness_raw` at K=1: 3.744e-03 @100k → 1.922e-03 @300k, ~2×). The projection amplifies small improvements in the **warm-start** into large safety gains, while the **unguided sampler path** — a different code path with no NLP — is unchanged. This reinforces fix_7's central finding: **the NLP, not the field, determines task outcome**, and the field only needs to be good enough to land in the right basin.

## 6. Anomaly worth noting

`diag_smooth_imf_guided_K1_n20` reports **`nlp_failures = 2`** — the first non-zero solver-failure count observed in Gen13. Every other cell is 0, including all n=200 runs. Most likely K=1 (a single projection from a very rough warm-start) occasionally hands IPOPT an infeasible start. Not affecting conclusions (that cell is superseded by the n=200 K=1 run, which had 0 failures), but worth watching if K=1 is used further.

## 7. Recommendations

1. **Stop training iMF.** §1 settles it: the field is data-limited, not compute-limited. A 1e6-step run (~42 h, over the 24 h cap) is not justified.
2. **If Gen13 is continued, the target is per-evaluation cost, not accuracy.** iMF is now safety-equivalent; it loses purely on network cost (dual head + two-time conditioning). A cheaper backbone — or the un-built **Newton pull-back** (`THEORY` Level 2, plan D8), which improves correction *without* extra projections — are the only remaining levers.
3. **Cheap loose end:** run **FM at K=1,2 with n=200** to match the iMF arms. FM's numbers rest on n=20; the comparison in §3 deserves equal footing before anything is written up.
4. **Report FM@K=2 as the genuine finding.** 100% safe at 0.1894 s/plan — a **4.5× speedup over HardFlow's own K=10 default**, discovered only because Gen13 forced the matched-budget battery.

## 9. Training-curve internals (300k run) — four findings

Beyond the plateau in §1, the full curve exposes structure worth recording.

### 9.1 The **v-head is also bad** — and it is plain flow matching

`ImfMatcher` trains two heads. The **u-head** learns the two-time average velocity; the **v-head** is an auxiliary that regresses *directly* to `v_target = x₁ − x₀` — i.e. **vanilla flow matching**, with none of the iMF-specific machinery.

| window | median `raw_mse_u` | median `raw_mse_v` | u/v |
|---|---|---|---|
| 0–50k | 17.33 | 14.76 | 1.17 |
| 50–150k | 15.29 | 12.15 | 1.26 |
| 150–300k | **13.92** | **10.45** | **1.33** |

Per-dim (96 dims, normalized `[-1,1]`):

| head | median | per-dim RMS | % of full range | variance explained |
|---|---|---|---|---|
| u | 14.34 | 0.386 | **19%** | 89% |
| v | 10.24 | 0.327 | **16%** | 92% |

**The v-head — pure flow matching on the same data, same backbone — is only marginally better than the u-head, and is itself at 16% of the normalized range.** That is a much stronger clue than anything previous: if a *vanilla FM regression* inside our pipeline is this inaccurate, the problem may not be the average-velocity idea at all.

Caveat: the v-head is an auxiliary term sharing a backbone and the adaptive weighting, so it is **not** a clean FM baseline. But it is close enough that its poor quality shifts weight toward "pipeline problem" and away from "the 2-time object is just hard".

### 9.2 The u-head falls **further behind** the v-head over training

u/v grows monotonically **1.17 → 1.26 → 1.33**. The two-time object is not merely harder — the gap *widens* with training, consistent with the u-head chasing a noisier target (the JVP tangent) while the easier v-head converges.

### 9.3 Instability **escalates**

| window | median | p90 | max | frac > 3× median |
|---|---|---|---|---|
| 0–100k | 16.43 | 28.5 | 331 | 3.8% |
| 100–200k | 14.13 | 30.2 | **1,117** | **7.8%** |
| 200–300k | 14.14 | 31.5 | **7,548** | 6.2% |

The worst spike grows **23× over the run** (331 → 7,548 = **500× the median**), and the spike *rate* roughly doubles after 100k. Training is becoming less stable as it proceeds, not settling. Known JVP predicted-v-tangent variance is the expected source — but a 500× outlier is large enough to warrant gradient clipping (currently none) before any further iMF training.

### 9.4 The adaptive loss is confirmed useless as a signal

`1.9980 → 1.9963` — a **0.09% change over 300,000 steps**. Exactly the Gen3v4 warning, now quantified: anyone reading `loss` alone would conclude the model never trained at all.

---

## 10. ⚠️ Is the raw field "coarse", or is something actually BROKEN?

Framing it as a "data ceiling" has been too generous. Physically:

| | iMF | FM |
|---|---|---|
| x̂1 endpoint error at τ=0 (T4) | **15.4 cm** | 2.6 cm |
| as % of the 60 cm workspace | **26%** | 4% |
| vs obstacle radius (~5 cm) | **3.1×** | 0.5× |

**A terminal prediction wrong by ~3 obstacle radii is not "slightly rough" — it is unusable for obstacle avoidance.** Combined with 0% unguided success at *every* K and 6–10× roughness, the raw generative output is, bluntly, not a trajectory model anyone would ship.

**Correction to T4 (owed):** it measures `‖x̂1 − x_final‖` where `x_final` is the *chain endpoint*, which on the **guided** path has already been NLP-projected. Since the projection moves iMF 96× vs FM's 7.4× (fix_7), part of that 15.4 cm is the projection's own correction, not pure prediction error — **T4 is inflated for iMF.** The **unguided** evidence (0% at every K, 6–10× roughness, §5) is unconfounded and stands on its own.

### 10.1 The decisive bug-vs-ceiling tests

Two experiments separate "iMF genuinely can't learn this from 96 demos" from "our pipeline is broken". **Until one is run, every Gen13 conclusion — including the fix_7.3 refutation — rests on an unvalidated field.**

**Test A — sample from the v-head (CHEAP, no retraining).**
The v-head *is* a flow-matching model living in the existing 300k checkpoint. Sampling with it (Euler, like FM) uses identical weights, backbone, data and normalizer — only the head changes.

| outcome | conclusion |
|---|---|
| v-head samples look like FM's (~cm-level, unguided success > 0) | pipeline **fine** ⇒ the two-time `u` object is what fails ⇒ genuine ceiling |
| v-head samples are **also garbage** | **pipeline BUG** (data / normalizer / backbone / adaptive loss) ⇒ Gen13's conclusions are void |

Needs a small additive sampler that calls the v-head — no training, minutes to run.

**Test B — retrain with `IMF_DATA_PROPORTION=1.0` (thorough, ~4 h).**
Forces `h=0` on every sample, collapsing the objective to exactly flow matching:
```
h=0 ⇒ V = u − 0·D_tot = u ⇒ loss = ‖u − (x₁ − x₀)‖²      ← plain FM
```
Same everything, two-time part removed.
```bash
IMF_DATA_PROPORTION=1.0 N_TRAIN_STEPS=100000 IMF_EXP_NAME=H16_imf_fmmode_100k \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/train_imf_hardflow.sh
```

**Recommended order: A first** (cheap, and §9.1 already suggests what it will show), then B only if A is ambiguous.

### 10.2 What this does to the current verdict

| statement | status |
|---|---|
| iMF's raw field is unusable (~3 obstacle radii of error) | ✅ **established** |
| The NLP rescues it to ~FM-level task performance | ✅ established (fix_7, §3) |
| The cause is the 96-demo data ceiling | ⚠️ **assumed, not proven** — §9.1's weak v-head is evidence *against* it |
| Gen13's efficiency claim is refuted | ⚠️ holds on *measured speed*, but rests on a field that may be broken |

**If Test A shows a pipeline bug, the honest position is that Gen13 has not yet been evaluated at all** — neither the refutation nor the earlier positive results would mean anything, because the model under test was defective.

## 11. Bottom line (revised)

The 300k run answered the confound cleanly: **more training does not fix iMF's field** (plateau; median improved 5.6% for 3× compute), **but it did fix iMF's safety deficit** (K=2: 94% → 99.5%, now indistinguishable from FM). What remains is a pure architecture-cost gap — iMF is 1.13–1.24× slower per plan at every matched budget — so **FM@K=2 is still the configuration to use**, and the Gen13 efficiency thesis stays refuted.

**Added after the curve/field analysis (§9–§10):** the above stands on *measured* quantities (speed, safety rates), but the iMF field itself is now in question — its **v-head, which is plain flow matching, is nearly as inaccurate as the u-head** (16% vs 19% of the normalized range). Until **Test A** (§10.1) rules out a pipeline bug, treat every Gen13 verdict — the refutation included — as provisional.
