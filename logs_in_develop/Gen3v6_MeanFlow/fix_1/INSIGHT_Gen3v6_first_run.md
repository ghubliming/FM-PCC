# INSIGHT — Gen3v6 (MeanFlow) first successful run

**Date:** 2026-07-24 · **Type:** results insight · **Status:** functional success, NOT a benchmark yet
**Runs:** train `23745` (`mf_train`, i6-gpu-1, started 2026-07-23 09:19 UTC) · eval `23777` (`eval_meanflow`)
**Source logs:** `temp/Gen3v6/11_19_31_mf_train_23745.log`, `temp/Gen3v6/23_41_30_eval_meanflow_23777.log`, `temp/Gen3v6/mf_losses.pkl` (seed 6, full 100-point curves)
**W&B:** `FMPCC-MeanFlow/runs/l95cfsjk` (`MeanFlow-seed-6-slurm-23745`)
**Follows:** [`CHANGELOG_fix1_diffuser_namespace_shim.md`](CHANGELOG_fix1_diffuser_namespace_shim.md)

---

## 0. TL;DR

The `fix_1` shim worked — Gen3v6 trained end-to-end for the first time. **MeanFlow (analytic-v
JVP) learns a goal-reaching field, and MeanFlow + DPCC hard projection produces safe control at
K=2 NFE.** The headline eval cell `dpcc-c-tightened` is **100 % goal + 100 % constraint, 0
violations on all three obstacle halfspaces**.

But this is a **functional smoke, not a scientific result**: **one seed (6), two rollouts per
cell**. Every SR is a multiple of 0.5. No A/B against Gen3v4-iMF or FMv3ODE was run. Treat the
numbers as "the pipeline produces sane behaviour," not "MeanFlow beats X."

The pre-registered h-stratified kill criterion (PLAN §7) did **not** cleanly fire, but it surfaced
the real subtlety: **the large-h regime is barely trained because it is barely sampled.**

---

## 1. What actually ran

| | |
|---|---|
| **`fix_1` shim** | ✅ confirmed on cluster — log shows `Config: <class 'flow_matcher_v3_meanflow.models.mf_engine.MeanFlowEngine'>` and training proceeded past the step-0 crash point |
| **Seeds trained** | **only seed 6** completed (100 epochs / 100 k steps, ~11 h). Seed 7 reached epoch 9, then the job was cancelled 2026-07-23 21:38 UTC |
| **Why not 5 seeds** | 5 × ~11 h ≫ the 24 h wall. The `--seeds 6 7 8 9 10` loop is serial; it can never fit. **This needs fixing** (one seed per job / array job) before a multi-seed result exists |
| **Eval** | seed 6 only, EMA weights, K=2 (`H8_K2_Meuler_T0.5`), all 13 projection variants × 3 halfspaces, **n_trials = 2** |
| **gradient_clip** | ✅ active and biting hard: final pre-clip `train/grad_norm = 28.4` against `clip = 1.0` (≈28× reduction). The fix is doing real work |

---

## 2. Training convergence (seed 6)

Read `raw_mse_u` and the h-buckets, **never** `loss`/`diffusion_loss` — the adaptive loss sat pinned
at its ceiling (`diffusion_loss ≈ 1.86–1.99`, `train/loss 0.965`), exactly as designed.

| signal | step 0 | best / final | reading |
|---|---|---|---|
| `train/raw_mse_u` (per-sample SUM over 48 dims) | ~9 | min ~1.0, final 1.67 | converges, but **very spiky** (bursts to 32–68 recur all through training) |
| `train/per_dim_rms_u` | 0.44 | ~0.145 best, 0.187 final | ~2–3× per-dim error reduction |
| `val/raw_mse_u` | █ (high) → ▁ | 9.02 final | drops ~8× on the sparkline, then noisy |
| `test/loss` | 1.00 | 0.967 | monotone but **almost flat** (3 % over 100 epochs) — adaptive, so uninformative by design |
| `train/fm_frac` | — | 0.5625 | ✅ `meanflow_data_proportion = 0.5` is working |
| `train/h_mean` | — | 0.105 | ⚠️ **most sampled intervals are small** — see §3 |

The recurring `raw_mse_u` spikes happen **with gradient clipping on**, so they are not gradient
explosion — they are the measured error on hard `(t, r, h)` draws where the JVP target `v + h·du/dr`
is large. The clip bounds the *update*; it cannot shrink the *measured* error on an outlier batch.
The `losses.pkl` curves confirm this is a **general JVP-target property, not a b3 quirk**: every
non-anchor bucket spikes occasionally (`train/h_mse_b1` max **504**, `b2` max **685**, `raw_mse_u`
max **144**) while its final value stays low (1.7–1.8). `grad_norm` itself *grows* over training
(2.8 → 28.4, always ≥ the clip of 1.0), so the clip is active every step from start to finish.

## 3. The h-stratified metric — the real finding (⭐ new this generation)

This is the metric COMPARE §7.4.1 asked for a month ago and that no prior generation had. First
time we can see *where in h* the field is good. From the W&B run summary:

| bucket (h range) | first (step 0) | final | trend |
|---|---|---|---|
| `h_mse_b0` (h == 0, FM anchors) | 58.6 | **1.60** | ✅ ~37× drop |
| `h_mse_b1` (0, 0.3) | 56.5 | **1.73** | ✅ ~33× drop |
| `h_mse_b2` [0.3, 0.6) | 49.3 | **1.84** | ✅ ~27× drop |
| `h_mse_b3` [0.6, 1.0] | 2.42 | **1020** (train) / **2.06** (val) | ⚠️ see below |

**b0–b2 train cleanly and land at essentially the same low error** — the field is well-fit for
h < 0.6, which covers K≥2 sampling (each K=2 step has h=0.5).

**The `train/h_mse_b3 = 1020` "final" is a single-batch outlier, not a training failure.** The
`losses.pkl` now makes this decisive: (a) `train/h_mse_b3` has **only 30 of 100 logged points**
(step 12 k onward) — the h≥0.6 bucket is so rarely drawn it is empty/NaN-filtered ⅔ of the time —
and it swings from **min 0.063 to max 3380**, so any single value (incl. the last, 1020) is noise;
(b) `test/h_mse_b3` is logged all **100** points (the test pass averages 100 batches, so the bucket
is always populated) and **ends at its minimum, 2.06** — held-out large-h error is at its *best* at
the end, the opposite of divergence; (c) `h_mean = 0.10`, `fm_frac = 0.56` confirm most intervals
are tiny.

So the pre-registered kill criterion ("b3 flat while b0 drops 10×, ⇒ stop") is **not cleanly
triggered** — b3 is *noisy*, not flat, and validation says the large-h field is okay. But the metric
did its job and exposed the genuine issue underneath:

> **With `dp=0.5` plus two logit-normals massed near the data end, the sampler almost never draws
> large h. The h∈[0.6,1.0] regime — exactly where a 1-NFE jump lives — is trained on a handful of
> samples and its error estimate is too noisy to trust.** This is precisely the motivation for
> Gen3v7 (AlphaFlow's α-anneal reshapes the h distribution), and it is now *measured*, not assumed.

**Actionable:** for a 1-NFE claim, either widen the h schedule or lower `dp`, and log the b3
*sample count* alongside its mean so a noisy bucket is not mistaken for a trained one.

## 4. Eval — MeanFlow + DPCC projection at K=2 (seed 6, n=2/cell)

`g` = SR(goal), `b` = SR(goal ∧ constraints), `v` = avg # violations. **n_trials = 2**, so each
number is 0.0 / 0.5 / 1.0 — directional only.

| variant | both-hard | top-left-hard | top-right-hard |
|---|---|---|---|
| **dpcc-c-tightened** | **g1 / b1 / v0** | **g1 / b1 / v0** | **g1 / b1 / v0** |
| dpcc-r-tightened | g1 / b.5 / v.5 | g1 / b1 / v0 | g1 / b1 / v0 |
| post_processing-tightened | g1 / b.5 / v.5 | g1 / b1 / v0 | g1 / b1 / v0 |
| dpcc-t-tightened | g1 / b1 / v0 | g1 / b1 / v0 | g.5 / b.5 / v0 |
| dpcc-c | g1 / b1 / v0 | g1 / b.5 / v.5 | g.5 / b.5 / v2.5 |
| dpcc-t | g1 / b1 / v0 | g1 / b.5 / v2 | g.5 / b.5 / v2.5 |
| dpcc-r | g1 / b0 / v3.5 | g1 / b.5 / v1.5 | g.5 / b.5 / v0 |
| **diffuser** (unconstrained) | g1 / b.5 / v5 | g1 / **b0** / v14.5 | g1 / **b0** / v20 |
| **gradient** (unconstrained) | g1 / b.5 / v5 | g1 / **b0** / v14.5 | g1 / **b0** / v20 |
| **model_free** (unconstrained) | g1 / b.5 / v6 | g1 / **b0** / v15 | g1 / **b0** / v20.5 |

Computation time ≈ **0.03–0.04 s/step** (per-step, EMA weights, K=2).

**Read:**
1. **The generative brain reaches the goal** — `g = 1.0` in nearly every cell, including all
   unconstrained variants. MeanFlow learned goal-directed trajectories.
2. **Unconstrained variants are unsafe** — `diffuser`/`gradient`/`model_free` plow through the
   obstacle (`b=0`, 14–20 violations) in the top-left/top-right halfspaces. Expected: no projection.
3. **DPCC hard projection restores safety** — the tightened constraint-projection variants
   (`dpcc-c-tightened` especially) hit `b=1.0, v=0` almost everywhere. This is the DPCC design
   intent working on a MeanFlow backbone: *generate freely, project to safety.*
4. `top-right-hard` is the hardest cell (several `g0.5`), consistent with prior generations.

## 5. Caveats — do not over-read (each will bite a reviewer)

- 🔴 **One seed, two trials per cell.** No confidence intervals exist. This is a smoke test.
- 🔴 **Seeds 7–10 never trained** — the serial 5-seed loop cannot fit 24 h. No multi-seed number.
- 🔴 **No comparator.** This is the MeanFlow arm *in isolation*. The whole point of Gen3v6 (the
  analytic-vs-predicted-v A/B against Gen3v4-iMF, and the FM/FMv3ODE reference) is **not yet run**.
- ⚠️ **Window-level train/test split leak (inherited, POST_U10_III §4.2).** At H=8 adjacent windows
  share 7/8 frames, so `val/*` is effectively a train metric. The §3 val numbers are optimistic.
- ⚠️ **Comp time is per-step, not per-plan.** It is *not* directly comparable to fix_7.3's
  `0.1894 s/plan` FM bar. A matched per-plan measurement is still owed.
- ⚠️ **`h_mse_b3` summary is a noisy single point** (§3). Do not quote 1020 as "the field diverges."
- ⚠️ tqdm progress bars leaked into the batch log (visible as `Epoch N: 100%|…`). Cosmetic, but the
  repo rule is no live bars in batch logs — worth silencing.

## 6. Next steps (in priority order)

1. **Make seeds runnable.** Switch `train_meanflow.sh` to one seed per job (or a Slurm array) so
   seeds 6–10 each complete. Without this there is no multi-seed result, ever.
2. **Run the A/B.** Same eval harness, matched K ∈ {1, 2, 5, 10}, against Gen3v4-`imf_official`
   (predicted v_c) and the FMv3ODE / FM reference. That comparison *is* Gen3v6's reason to exist.
3. **Fix the large-h blind spot** before any 1-NFE claim: widen the h schedule or lower `dp`, and
   log per-bucket sample counts so a starved bucket is not read as a trained one.
4. **Raise n_trials** in `projection_eval.yaml` to something statistically meaningful for the final
   table; label the current 2-trial pass as a smoke everywhere it is cited.
5. Silence the tqdm bars in the batch log path.

## 7. One-line verdict

Gen3v6 is **alive and behaving correctly** — MeanFlow + DPCC = safe K=2 control on one seed — and
its new h-stratified metric already earned its keep by localising the field's weak spot at large h.
It is **not yet evidence for or against iMF**; that needs the multi-seed A/B, which needs the
per-seed job fix first.
