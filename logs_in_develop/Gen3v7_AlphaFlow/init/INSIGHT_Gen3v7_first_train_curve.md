# INSIGHT — Gen3v7 α-Flow, first training run (job 23759): "the MSE curve is TERRIBLE"

**Date:** 2026-07-24 · **Type:** first-look insight (NO code changes) · **Status:** ✅ **RESOLVED with `losses.pkl` — see §7 (read that first)**
**Evidence:** `temp/Gen3V7/14_17_55_af_train_23759.log` (train) · `temp/Gen3V7/00_36_18_eval_alphaflow_23786.log` (eval, **seed 6, n=2**, K∈{1,2,5,10}) · `temp/Gen3V7/alpha_flow_losses.pkl` (**seed 6**, full h-stratified + grad-norm history)
**Reader's report:** *"first sight of MSE loss in the train curve is TERRIBLE."* — Confirmed, but §7 shows the homotopy is actually **working** where it should; the damage is localized to two specific things.

> **⚠️ §1–§6 below were the pre-`pkl` provisional read. §7 supersedes them.** They are kept for the reasoning trail. The one-line update: the aggregate MSE *was* misleading (§3.1 was right), the field *does* train (b0 falls ~25×), the α-homotopy *does* make mid-h queries learnable (b1/b2 → ~0.001 as α→0) — but two real problems remain: **the gradient clip is saturated on 100 % of steps**, and **the few-NFE bucket (b3) is under-sampled and blows up in the α=0 tail**.

---

## 0. TL;DR (post-`pkl`, resolved — full detail in §7)

1. ✅ **The machinery works.** α annealed 1.0→0.0 exactly as designed; `discrete_frac` tracked it and hit 0 at ~72k. The #1 silent-failure trap (dead schedule) did **not** happen.
2. 🔴 **The "terrible" aggregate MSE was ~half a metric artefact.** At α=1 the u-head's target is deliberately `v` (unmatchable at large h), so a high flat `raw_mse_u` there is expected, not a bug (§3.1, now confirmed by the buckets).
3. ✅ **The field DOES train.** `h_mse_b0` (h=0, always-learnable) falls **~25×** (58.6 → ~2.1). It is *not* the catastrophic "even FM won't converge" case.
4. ⭐ **The homotopy WORKS where it can be measured.** As α→0, the mid-h buckets **b1/b2 collapse to ~0.001** — the α-Flow bootstrapped target makes the previously-blind mid-h queries learnable, which the MeanFlow JVP could not. First positive evidence for the mechanism in this project.
5. 🔴 **Two real, separable defects remain:** (a) the **gradient clip is saturated on 100 % of steps** (pre-clip norm median 25, max 59, vs clip 1.0) → the whole curve is throttled and noisy; (b) the **few-NFE bucket b3 (h∈[0.6,1]) is under-sampled and unstable**, spiking to **269** in the α=0 tail — exactly the regime K=1–2 depends on.
6. 🟢 **The planner is not terrible.** K=1, seed 6: goal success 1.0 everywhere; DPCC-projected variants ~1.0 while unprojected baselines sit at 0.0 — the **raw-rough / projected-fine** inversion from Gen13 CLOSURE I.

**Verdict: not broken, and the mechanism shows signs of working — but do not celebrate yet.** Fix the grad-clip saturation and the b3 under-sampling before trusting any few-NFE number. Still **seed 6 only, n=2 eval**; the matched-K table over all seeds and the endpoint metric are the real deliverables and are not in these files.

---

## 1. What the curve actually shows

`raw_mse_u` = **per-sample SUM over H·D = 8·6 = 48 dims** (not a mean). `per_dim_rms_u = sqrt(raw_mse_u/48)` is the comparable-across-gens number.

| phase | steps | α | mean `raw_mse_u` | `per_dim_rms_u` | shape |
|---|---|---|---|---|---|
| **FM-dominated** | 0–30k | 1.0 | **5.0** | ~0.32 | high, flat, noisy (3–9) |
| **anneal** | 30k–72k | 1.0→0.0 | **2.4** | ~0.22 | **improving** |
| **pure MeanFlow** | 72k–100k | 0.0 | **4.1** | ~0.24 | noisy, **spikes to 33.7** (per-dim 0.84) |

Best point seen: `raw_mse_u=0.56` (per-dim 0.108) around step 54k, mid-anneal. Worst: 33.7 at step 81k, α=0.
⚠️ This log **interleaves 5 seeds**; the table is indicative of typical behaviour, not one clean seed-6 curve. The qualitative shape (high→dip→spiky) holds across seeds.

**The single most telling fact: the curve is best in the *middle* of the anneal and degrades as α→0.** α-Flow ends *as* MeanFlow — and MeanFlow is exactly where it gets worse.

## 2. The schedule fired correctly (rule this out first)

From the pre-flight banner and the live telemetry:

```
alpha:  step 0→1.000  30k→0.993  40k→0.924  50k→0.500  60k→0.076  70k→0.007  ≥72k→0.000
discrete_frac: ~0.5 while α>0   →   0.000 once α snaps to 0
```

So the α=0 constructor assert, the sigmoid port, the clamp snap, and `set_train_step` are all doing their job. **None of the Gen3v7-specific plumbing is the cause of the bad curve.** (This is the good news buried in an ugly plot.)

## 3. Why the curve looks terrible — ranked

### 3.1 🔴 STRUCTURAL, not a bug: at α=1 the target is unmatchable at large h
While α=1, **both** live branches regress `u` to `v`:
- FM anchors (h=0): target `v` — correct, learnable.
- discrete branch (h>0, α=1 short-circuit): target `v` **exactly**, but the query is `u(z_r, r, h>0)`.

So half the batch asks the u-head to output the *instantaneous* velocity `v` at *large* h, where the true average velocity `u*(z,r,h) ≠ v`. That error is **irreducible by design** — it is what "α=1 = pure flow matching" means. Averaged over random h each batch, `raw_mse_u` is floored at a few units and looks flat. **This is not the field failing to train; it is the metric measuring an intentionally-wrong target.** As α→0 the target becomes the real `u*`, which *is* matchable — and indeed the curve drops (§1).
👉 **The h=0 bucket (`h_mse_b0`) is the only honest convergence signal at α=1, and it is not in these logs.**

### 3.2 ⭐ SCIENTIFIC: the α=0 regime is where it destabilises
Every large spike (33.7 at 81k) and the climb from 2.4→4.1 happen **after α=0** (steps >72k), i.e. once the pure-MeanFlow JVP target takes over and the well-posed bootstrapped target is gone. This is precisely the **blind-direction / self-referential-target** pathology (PLAN §0, COMPARE §8.2) that motivated the whole generation — reproduced here, on cue. It is a *result*, not just noise.
⚠️ Note the spike survived `gradient_clip=1.0`. The plan predicted the discrete branch would be *calmer* than a JVP; the data agree (the calm stretch is mid-anneal), and the spikes cluster exactly where the JVP is alone.

### 3.3 METRIC noise: random (r,h) per batch + 5 seeds interleaved
`raw_mse_u` is a single-batch estimate with a fresh (r,h) draw every step and no fixed eval set, so batch-to-batch swings of 2–3× are baked in even for a healthy field. This log also concatenates seeds 6–10, inflating the apparent jitter. **Read `val/raw_mse_u` (held-out running mean) and the h-buckets, not the train postfix.**

### 3.4 Possible contributor: the h-only DiT (see the backbone note)
`imf_dit_trajectory`'s DiT conditions on **h only** — it never sees the query time `r` (`dit_condition_on_t=False`). At α=1 the u-head must represent "v at all h from a fixed z", and at α=0 the MeanFlow identity loses its `∂_r u` term. This restricts the function class in exactly the regime that hurts here. Shared with Gen3v6, so not unique to α-Flow, but it caps the best achievable `raw_mse_u`. See [`NOTE_backbone_fidelity_Gen3v4_v6_v7.md`](NOTE_backbone_fidelity_Gen3v4_v6_v7.md) §4 / R1.

## 4. The eval says the planner is OK (K=1, seed 6, n=2)

| variant group | constraint-sat | reading |
|---|---|---|
| **goal (all variants)** | **1.0** | the generative field reaches the target even at K=1 (1-step) |
| DPCC-projected (`dpcc-c/-r/-t` ± tightened) | mostly **1.0**, some 0.5 | the "physical brakes" recover a rough field |
| `dpcc-c-tightened` | **1.0** both maps | the config's headline variant is clean |
| unprojected (`model_free`, `gradient`, `diffuser`) | **0.0**, ~26 violations, total ~10 | the **raw field is rough** — consistent with the bad MSE |

This is the **raw-rough / projected-fine inversion** from Gen13 CLOSURE I: the unguided field is poor (matching the MSE) yet the DPCC-projected planner is fine. **"Terrible MSE" and "terrible planner" are different claims, and only the first is true so far.**
⚠️ **n=2 trials, one seed, K=1 only extracted.** This is a smell test, not a result. The matched-K table over all 5 seeds is the real deliverable and is not in these files.

## 5. What I'd need to turn this from "insight" into "verdict"

You offered other files — these, in priority order:

1. 🔴 **`losses.pkl`** (or `losses.json`) from the seed-6 savepath —
   `logs/.../ai1.0_ae0.0_ag25.0_rf0.5/6/losses.pkl`. Gives `h_mse_b0..b3`, `val/raw_mse_u`, `grad_norm_history`, and the per-seed `alpha` series. **This is the single most useful file** — it settles §3.1 (is `h_mse_b0` actually converging while the aggregate is floored?) and §3.2 (does `grad_norm` spike at α=0?).
2. **The full eval log / aggregated table** for **all seeds and all K** (the `load_results` output, not just K=1 seed 6). Needed for the matched-K safety-vs-s/plan table — the actual success criterion (PLAN §8).
3. **`endpoint_error.json`** if `endpoint_error_alphaflow.py` was run — the decisive `err(τ=0)` vs K number. If flat in K, the objective change did not fix the field.
4. **Gen3v6's and Gen3v4's `raw_mse_u`/`per_dim_rms_u`** at the same points, to calibrate whether 0.2–0.3 per-dim is anomalous or just what this metric looks like on this task.

## 6. Provisional reading (to be confirmed, NOT acted on)

- The bad-looking curve is **~60% expected metric artefact** (§3.1, §3.3) and **~40% a real α=0 instability** (§3.2) that is itself the phenomenon under study.
- **Nothing here says the run is broken.** The schedule is correct, the planner reaches the goal, projection recovers constraints.
- **The one thing that would be genuinely bad** — and which the pkl will reveal — is if `h_mse_b0` (the h=0, always-learnable bucket) is *also* flat. That would mean even plain flow matching isn't converging, pointing at LR / clip / the h-only backbone rather than at the α-Flow idea. **Check that first.**
- Do **not** change hyper-parameters off this log alone. Get `losses.pkl`.

---

## 7. ✅ RESOLVED with `alpha_flow_losses.pkl` (seed 6) — the real diagnosis

The h-stratified residual (`h_mse_b0..b3`) and `grad_norm_history` settle every open question in §5. Buckets: **b0 = h==0**, b1 = (0, 0.3), b2 = [0.3, 0.6), **b3 = [0.6, 1.0]** (the extreme few-NFE regime).

### 7.1 The field DOES train — the aggregate metric was hiding it (§3.1 confirmed)

| bucket | α=1 (0–30k) | anneal (30–72k) | α=0 (72–100k) | reading |
|---|---|---|---|---|
| **b0** (h=0, always-learnable) | 58.6 → mean **6.7** | **2.9** | **2.1** (min 0.58) | ✅ trains ~**25×**; the "flat" aggregate was masking a real, if noisy, descent |
| **b1** (small h) | bouncing 3–15 | falling | **~0.001–0.05** | ⭐ collapses as α→0 |
| **b2** (mid h) | bouncing 3–21 | falling | **~0.001–0.006** | ⭐ collapses as α→0 |
| **b3** (h∈[0.6,1]) | mostly **empty** (n=30/100) | erratic 0.02–2.1 | **0.5–1.9, spike to 269** | 🔴 the problem child |

**§3.1 was correct:** at α=1 the mid/large-h buckets are high because their target is deliberately `v` (unmatchable at large h). As α→0 the target becomes the true `u*` and **b1/b2 drop to ~0.001** — i.e. **the α-Flow homotopy does exactly what it was designed to do: it makes the previously-blind mid-h queries learnable.** That is a genuine positive result, invisible in the aggregate `raw_mse_u`.

### 7.2 🔴 Problem 1 — the gradient clip is saturated on 100 % of steps
`grad_norm` (pre-clip): first 2.8, **median 24.8, max 59.4**, and **> 5 at every single logged step** with `gradient_clip=1.0`. So the optimiser spends the entire run in "emergency brake": the true gradient is **25–60× the clip threshold**, always. That is the mechanistic source of the noise — the effective step is a unit-norm direction with a permanently throttled magnitude, so the loss cannot settle. grad_norm also **rises in the α=0 tail (40–60)** vs the anneal (20–30).
→ **This is the most actionable finding.** Candidates: LR 5e-4 is too high for this objective; and/or the b3 spikes (below) inject the gradient. Worth a short LR/clip sweep — but see §7.4 first.

### 7.3 🔴 Problem 2 — the few-NFE bucket (b3) is under-sampled AND unstable
b3 (h∈[0.6,1]) is **empty on 70 % of logged steps** — the logit-normal × minmax τ-pair sampling rarely produces a large interval. When it *is* populated it is erratic, and in the pure-MeanFlow tail it produces the catastrophic outliers: **b3 = 269 at step 98k**, b2 = 49.6 at 96k. The `af_clamp_utgt=4.0` clamps the *target*, so a loss of 269 means the **prediction** diverged, not the target — the field is genuinely unstable at large h once α=0. **This is the exact regime K=1–2 sampling depends on**, and it is both the least-trained and the least-stable part of the field. It is also the pre-registered failure signature (PLAN §8, COMPARE §8.2), now localized to a bucket.

### 7.4 Verdict (supersedes §6)

- ✅ **Not broken, and not the catastrophic case.** b0 trains ~25×; the homotopy makes b1/b2 learnable. The "terrible" aggregate was ~half metric artefact (§3.1) as predicted.
- ⭐ **The mechanism works where it can be measured** (b1/b2 → ~0.001). That is the first evidence in this project that the bootstrapped target does something the MeanFlow JVP could not.
- 🔴 **Two concrete, separable defects remain:** (1) permanent grad-clip saturation → the whole curve is throttled/noisy; (2) the few-NFE bucket b3 is under-sampled and unstable, blowing up in the α=0 tail — precisely where the paper's claim lives.
- ⚠️ **Still seed 6 only, n=2 eval.** Per-seed b0 floors ~2 (per-dim ~0.2) need the Gen3v6/Gen3v4 comparison (§5.4) to know if that is anomalous or normal for this metric on this task.

### 7.5 What this implies for "train the UNet?" (your question)

The diagnosis makes the UNet run **more** worth doing, not less, and for a sharper reason:
- b0 plateauing at ~2 (per-dim ~0.2) and grad-norm pinned at 25–60× **on the h=0 bucket** — the simplest possible target — hints the strain is partly in the **backbone/optimisation**, not only the objective. The h-only DiT (backbone note §4) is a prime suspect.
- The UNet is the backbone this project's **trusted 100 %-safe FM baseline** used, so it is the natural control.
- b3's under-sampling and the grad-clip saturation are **objective/sampling** issues that will follow the UNet too — so a UNet run also cleanly separates "backbone problem" from "objective problem": if grad-norm is still pinned at 25–60× on the UNet, the clip/LR/sampling is the cause, not the DiT.

**Suggested order now:**
1. 🔴 **One cheap knob first, before any new full run:** since the clip bites 100 % of steps, do a short probe — LR 5e-4 → 2e-4 (or clip 1.0 → 5.0) for ~10k steps — and watch whether grad-norm comes off the ceiling and b0 drops below ~2. This is the highest-information-per-GPU-hour move.
2. **Then the UNet train + eval** (flip `imf_backbone` in both config blocks; folders auto-separate). Run `gates_alphaflow.py` on it first (G2 = JVP-on-UNet parity).
3. Keep the DiT run for the A/B; get Gen3v6/Gen3v4 b0 for calibration.
