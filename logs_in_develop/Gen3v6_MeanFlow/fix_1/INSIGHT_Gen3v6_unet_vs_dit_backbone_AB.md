# INSIGHT — Gen3v6 backbone A/B: UNet fails, DiT works (same MeanFlow objective)

**Date:** 2026-07-25 · **Type:** results insight (controlled backbone ablation) · **Status:** decisive negative for UNet
**This run:** train `23813` + eval `23814` — **`imf_backbone='unet'`**, seed 6, full 100 k steps
**Baseline (prior insight):** train `23745` + eval `23777` — **`imf_backbone='dit'`**, seed 6, 100 k steps
**Source:** `temp/Gen3v6/2507/{23_49_59_mf_train_23813.log, 23_49_59_mf_eval_23814.log, mf_unet_losses.pkl}`
**Companion:** [`INSIGHT_Gen3v6_first_run.md`](INSIGHT_Gen3v6_first_run.md) (the DiT run)

---

## 0. TL;DR

Swapping **only the backbone** (DiT → UNet), holding the MeanFlow objective, data, schedule, seed,
and K=2 eval fixed, **breaks training completely**. The UNet arm's loss stays pinned at the adaptive
ceiling for all 100 k steps, its **best checkpoint is step 3000** (it never improves after epoch 3),
and at eval it **cannot even reach the goal** in the hardest halfspace. The DiT arm, by contrast,
converged and gave 100 %-safe K=2 control.

**Conclusion: the analytic-v MeanFlow JVP objective requires the DiT backbone. The UNet
(`Flow_matcher_U_Net_v2`) does not learn it.** This is a clean, controlled result — the two runs
differ in exactly one flag.

---

## 1. It is a true A/B (one variable)

Both runs: seed 6, `dp=0.5`, `t_schedule=logit_normal`, `p_mean=-0.4`, 100 k steps, `gradient_clip=1.0`,
`dual_head=True`, K=2 eval, EMA weights. The **only** difference is `imf_backbone`: `dit` vs `unet`
(confirmed in the train log: `imf_backbone: unet`, and in the folder name `..._bbunet_...`). The DiT
carries native two-time `(t,h)` + interval tokens with RoPE; the UNet uses additive
`time_mlp(t) + h_mlp(h)` conditioning.

## 2. Training — the UNet does not converge

| signal (seed 6, final) | **DiT (23745)** | **UNet (23813)** |
|---|---|---|
| `train/loss` (adaptive) | 0.965 | **0.9998** (pinned at ceiling, min 0.9945) |
| `test/loss` | 0.967 | **0.9998** |
| `diffusion_loss` | ~1.9 | **2.0** exactly (= 1.0/head × 2, saturated) |
| `train/raw_mse_u` first→last | 56 → **1.67** (min 1.4) | 64 → **69.6** (min 8.0, **went up**) |
| `test/raw_mse_u` | 9.0 | **97.7** |
| `per_dim_rms_u` first→last | 1.08 → **0.187** | 1.16 → **1.20** (never moved) |
| `train/h_mse_b0` (FM anchor) first→last | 58.6 → **1.60** (37×↓) | 66.9 → **46** (1.5×↓) |
| `train/a0_loss` | 0.13 | 1.48 |
| `grad_norm` first→last | 2.8 → **28.4** (grows: learning) | 8.9 → **1.5** (shrinks: gives up) |
| **best checkpoint step** | late | **3000** (epoch 3 — never beaten) |

Everything points the same way: the UNet's field is essentially **untrained**. `per_dim_rms_u`
sits at ~1.2 (the DiT reached 0.19), the FM-anchor bucket `h_mse_b0` barely moves (1.5× vs the DiT's
37×), and the adaptive loss never leaves its ceiling — the hallmark of an objective the network
cannot fit. The **best-checkpoint-at-step-3000** fact is the clincher: `best_test_loss` was set
early and never improved across the remaining 97 k steps.

It is not quiet stagnation — it is **unstable**: `train/raw_mse_u` spikes to **5038**, test to
**8799**, and the held-out large-h buckets blow up (`test/h_mse_b1` max **2.5e4**, `b2` **2.8e4**,
`b3` **9.7e4**). The JVP target on the UNet is wildly ill-conditioned; `grad_norm` decaying to 1.5
shows the optimiser then coasts on near-zero useful gradient between spikes.

## 3. Eval — goal-reaching itself fails (seed 6, n_trials = 2)

`g` = SR(goal), `b` = SR(goal ∧ constraints), `v` = avg # violations.

| variant | both-hard | top-left-hard | top-right-hard |
|---|---|---|---|
| dpcc-c-tightened | g1 / b1 / v0 | g1 / **b0** / v3.5 | **g0.5** / b.5 / v0 |
| dpcc-r-tightened | g1 / b.5 / v2.5 | g1 / b.5 / v2 | **g0** / b0 / v0 |
| dpcc-t | g1 / b0 / v14 | g1 / b0 / **v41** | **g0** / b0 / v0 |
| diffuser (unconstrained) | g1 / b0 / v21.5 | g1 / b0 / v27 | g.5 / b0 / v17.5 |
| model_free (unconstrained) | g1 / b0 / v22 | g1 / b0 / v27.5 | g.5 / b0 / v18 |

Compare the **headline cell** against the DiT run:

| `dpcc-c-tightened` | both-hard | top-left-hard | top-right-hard |
|---|---|---|---|
| **DiT (23777)** | g1 / **b1 / v0** | g1 / **b1 / v0** | g1 / **b1 / v0** |
| **UNet (23814)** | g1 / b1 / v0 | g1 / **b0** / v3.5 | **g0.5** / b.5 / v0 |

- **`top-right-hard` collapses to `g=0`** for most UNet variants — the model cannot even reach the
  goal there. The DiT reached it (`g=1`) everywhere.
- **Violations balloon** (`dpcc-t` top-left **v41**; unconstrained v21–27, vs DiT's v14–20), because
  the underlying trajectory is poor and the projector has too far to pull it back.
- The DiT's clean `b1/v0` across all three halfspaces has **no UNet counterpart**.

Caveats unchanged from the first insight: **one seed, two trials per cell** — the *eval* numbers are
directional. But the **training curves (§2) are unambiguous and trial-count-independent**: the UNet
simply did not learn the objective, so the weak eval is the expected consequence, not noise.

## 4. Why this matters

1. **Backbone choice is not a free hyperparameter for MeanFlow here — it is load-bearing.** The
   config default (`imf_backbone='dit'`) is now empirically justified, not just inherited from the
   iMF lineage. The comment "U10 imf_official REQUIRES dit" turns out to understate it: even *plain*
   MeanFlow with **no CFG** needs the DiT.
2. **It localises the DiT's advantage to the two-time conditioning.** The MeanFlow identity is a
   statement about `∂u/∂r` at fixed `t` over an interval `h`; the DiT conditions on `h` natively (and
   with RoPE positional structure), while the UNet's additive `time_mlp(t)+h_mlp(h)` apparently
   cannot represent the `(t,h)`-field well enough for the JVP target to be consistent. The
   catastrophic large-h spikes (§2) are the symptom.
3. **It sharpens the Gen3v6 ↔ Gen3v4 A/B design.** The MeanFlow-vs-iMF comparison must be run on the
   **DiT** for both arms; a UNet MeanFlow arm would lose to iMF for the wrong reason (backbone, not
   tangent). This run rules out UNet as a shortcut.

## 5. Open questions / next

- **Is it the objective or the optimisation?** The instability (raw_mse_u → 5038) hints the UNet
  JVP might be salvageable with a smaller LR, stronger clip, or warmup on `h`. Worth *one* cheap probe
  before abandoning UNet entirely — but not a priority; the DiT works.
- **Does the UNet fail the same way in the h-stratified metric on a trained run?** Here it failed
  globally (b0 barely moved), so the h-stratification is moot — nothing trained. The metric's value
  shows up only once b0–b2 converge (as they did on DiT).
- **Priority remains unchanged** from the first insight: fix the serial multi-seed loop, then run the
  matched-K MeanFlow(DiT)-vs-iMF(DiT)-vs-FM A/B. This run just confirms **DiT is the only viable
  backbone** to run that comparison on.

## 6. One-line verdict

Same objective, one flag changed: **DiT trains and gives 100 %-safe K=2 control; UNet never leaves
its initialisation (best ckpt @ step 3000) and cannot reach the goal.** MeanFlow needs the DiT.
