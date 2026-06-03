# iMF Train-Log Analysis — `temp/imf_fix3/imf_wandb`

**Date written**: 2026-06-02
**Log span**: 2026-05-31 11:39:57 → 14:52:21 UTC (≈ 3 h 12 m, 100 epochs × 1000 steps = 100k total steps)
**Checkpoint scope**: this is the training run behind the **current** Gen3v4u2 iMF checkpoint — *not* retrained, just re-analyzed at the user's request.
**Fix-1 status**: commit `0a628c1` ("Gen3v4u2F1fix") landed **2026-05-31 11:37:38** — i.e. **2 minutes 19 seconds before** training started. The corrected u-target `(x_t − x_r) / (h + 1e-8)` is active throughout this run. Anything we see below is **not** the legacy `(1−r)/h · v_const` over-scaling.

---

## 1. TL;DR

The user's gut read is qualitatively correct but quantitatively softer than classical double descent:

- **Phase A (E0 → E3) — fast descent.** `loss` 0.308 → 0.179, `diffusion_loss` 0.667 → 0.39. Healthy.
- **Phase B (E4) — single-epoch spike.** Train `diffusion_loss` **0.39 → 1.57** (4× jump) inside epoch 4, train `loss` **0.179 → 0.719**. **Test loss lags one epoch** (still 0.134 at E4) — the spike is a train-side event that hadn't yet damaged the weights enough to ruin the held-out batch averaged across the whole epoch.
- **Phase C (E5) — test catches up.** Train `loss` 0.565, test `loss` **0.681** (worst test of the entire run).
- **Phase D (E6 → E99) — slow shallow recovery.** Test `loss` drifts down from **0.681 → 0.459** over the next 94 epochs. That's the "slight drop after the rise" the user noticed. It's real but **never returns to the E2 minimum** of test `loss` = 0.164.
- **Generalization gap at end**: train `loss` = 0.451 vs test `loss` = 0.459 → gap ≈ 0.008 (≈ 1.7%). Essentially zero. The model is well-generalized; the problem isn't overfitting, it's that the basin it converged into is worse than the basin it briefly occupied around E2.

So the shape is **"descent → spike → slow re-descent to a worse plateau"** — not the classical interpolation-threshold double descent where the second descent matches or beats the first. The post-spike curve is monotone-ish but never recovers the early floor.

---

## 2. Numerical Timeline

Key metric extracted at each phase boundary:

| Phase | Epoch | LR | `loss` (train) | `loss_test` | `diffusion_loss` (train) | `a0_loss` (train) | `a0_loss_test` |
|---|---|---|---|---|---|---|---|
| A start | 0 | 5.00e-4 | 0.308 | 0.914 | 0.667 | 0.105 | 0.168 |
| A end | 3 | 4.99e-4 | 0.179 | **0.147** ← test floor | 0.386 | 0.060 | 0.041 |
| **B** spike | **4** | 4.98e-4 | **0.719** | 0.134 | **1.57** | 0.089 | 0.037 |
| C catch-up | 5 | 4.97e-4 | 0.565 | **0.681** ← test peak | 1.24 | 0.039 | 0.092 |
| D mid-1 | 30 | 4.01e-4 | 0.525 | 0.482 | 1.15 | 0.032 | 0.023 |
| D mid-2 | 60 | 1.68e-4 | 0.545 | 0.476 | 1.19 | 0.052 | 0.020 |
| D end | 99 | 0 | 0.451 | **0.459** | 0.986 | 0.011 | 0.0175 |

**Total loss decomposition (from `imf_diffusion.py:319`):**

```
total_loss = u_mix · main_loss(u-head) + aux_loss_weight · aux_loss(v-head)
           = u_mix · diffusion_loss     + aux_loss_weight · aux_loss
```

Across the log, `loss ≈ 0.46 · diffusion_loss` consistently (E0: 0.308/0.667 = 0.462; E50: 0.505/1.10 = 0.459; E99: 0.451/0.986 = 0.457). That means **`u_mix ≈ 0.46` is doing essentially all the work** in the reported `loss`, and the aux/v term contributes a near-constant small offset (~0.0–0.01). The aux head is well-fit; the u head is the one that plateaued at ~1.0.

---

## 3. The E4 spike — root-cause hypotheses

`diffusion_loss` is the **u-head MSE** against target `u_target = (x_t − x_r) / (h + 1e-8)`. Going from 0.39 → 1.57 in one epoch means the u-target was **suddenly hard** for that epoch's batches, or the model state slipped into a regime where the prediction error against the same distribution of targets quadrupled. The candidates:

### Hypothesis I — small-h noise amplification (most physically grounded)

For the linear interpolant `q_sample(τ) = (1−τ)·noise + τ·x_data`, the exact average velocity over `[r, t]` is `v_const = x_data − noise`, **bounded and independent of h**. The training target `(x_t − x_r) / h` is numerically equal to `v_const` *only* if `x_t` and `x_r` are computed from the same `(noise, x_data)` pair. They are (within one `p_losses` call). So mathematically the target is bounded.

**But:** `apply_conditioning(u_target, cond, …, noise=True)` overrides the action-anchor entries of `u_target` with sampled noise, and that noise is **not divided by h**. If `h` is sampled near 0 anywhere in the batch and the *non-conditioned* coordinates carry small finite-precision residual, `(small / 1e-8)` produces big spikes in those residuals. The `1e-8` floor in `(h + 1e-8)` doesn't help here — it bounds `h=0` exactly but not `h=1e-5`.

If the `t, r` sampler can produce `h = t − r` in the 1e-3 to 1e-5 range and that range is rare, the model can ignore it for the first 3 epochs (a few unlucky batches dilute into the running mean). Around E4 the cosine LR is still close to peak (4.98e-4) and one truly bad batch can move the weights into a regime where the gradient on every-other batch is also off — explaining the sustained jump rather than a one-epoch blip.

### Hypothesis II — u_mix / aux interaction

The total gradient is `u_mix · ∇main + aux_weight · ∇aux`. The aux target `v_target = x_start − x_base` is bounded and easy. If the u-head is briefly destabilized (Hypothesis I) and ∇main becomes large, `u_mix ≈ 0.46` keeps it dominant, while the v-branch keeps converging cleanly. That's consistent with what we see: `a0_loss` (a u-head diagnostic) jumps with `diffusion_loss` at E4 (0.06 → 0.089) and recovers; v-branch never visibly perturbs.

### Hypothesis III — data-side artifact

A1 D3IL avoiding/aligning trajectories can have rare high-velocity segments. If those land in E4's shuffled order in a cluster, gradients spike. Hard to distinguish from I/II without a per-batch log.

### What I'd rule out

- **Pre-fix_1 over-scaling**: ruled out by timing (training started 139 s after the fix_1 commit; nothing pre-fix_1 ran).
- **LR warmup ending**: cosine schedule is decreasing monotonically from step 0; no warmup phase to "end" at E4.
- **EMA shadowing**: would smooth, not spike. Not a candidate for a *positive* jump.

---

## 4. The plateau — what it implies for inference

After E5, `diffusion_loss` (u-head MSE) **never goes below ≈ 0.92** and spends most of the run between 0.95 and 1.18. Compare to the Phase A floor of 0.283 at E2. **The u-head is structurally undertrained relative to what was achievable.** It has roughly 3× the residual MSE the brief E2–E3 window touched.

Meanwhile `a0_loss_test` (a u-head per-action-dim diagnostic) does keep dropping: 0.041 (E3) → 0.0175 (E99). So *along certain coordinate dims* the u-head is improving; the global u-MSE is bottlenecked by other coordinates.

**Direct inference consequence** — bridges this analysis to [`../CHANGELOG.md`](../CHANGELOG.md)'s deviation findings:

- **Deviation A (we drop aux contribution at inference)**: the **u-head is the one we now rely on at inference**, and the u-head is the one with the unrecovered residual. Removing the aux from inference is the architecturally correct move per reference iMF, but it does *unmask* the u-head's training quality. Symptom: trajectories "no longer explode (fix_1) but also not smooth" — exactly the user's description.
- **Deviation B (freeze t = 0.5 at inference)**: orthogonal to what this log tells us. The t-conditioning issue is a sampling-time choice and isn't visible in train metrics.

In short: fix_3's two deviations are correctly identified and correctly applied, **but the u-head's training-side residual is what's preventing post-fix_3 trajectories from being as smooth as the old visual FM model.** That's a training-process issue, not a fix_3 inference issue.

---

## 5. Gen-gap audit

| Epoch | train `loss` | test `loss` | gap |
|---|---|---|---|
| 3 | 0.179 | 0.147 | -0.032 (test *lower*) |
| 5 | 0.565 | 0.681 | +0.116 |
| 30 | 0.525 | 0.482 | -0.043 |
| 60 | 0.545 | 0.476 | -0.069 |
| 99 | 0.451 | 0.459 | +0.008 |

By the end of training the gap is **essentially zero** (and was *negative* in the middle of Phase D — test slightly below train, common with dropout/noise during train forward). So the model is not overfitting. It's at a stable, well-generalized solution — just a mediocre one for u-head MSE.

---

## 6. Recommendations

### Option A — Keep current checkpoint, ship fix_3 inference

What the user is currently doing. fix_3's architectural deviations are correct. Inference quality limit = current u-head residual. Expected outcome: trajectories that don't explode, don't crash, but aren't as smooth as well-trained DDPM/FM baselines. Acceptable as a milestone; not the final answer.

### Option B — Retrain with stability guardrails (recommended if you can afford a 3-hour run)

Address the E4-spike hypothesis directly:

1. **Clamp `h` away from 0** at sample time: replace `(h + 1e-8)` with `(max(h, h_min))`, `h_min = 1e-3` or so. Keeps the target bounded by `||x_t − x_r|| / 1e-3` rather than `/ 1e-8`.
2. **Add gradient clipping** (`max_norm = 1.0`) if not already present. One bad batch shouldn't be able to ratchet the model into a different basin.
3. **Lower the LR ceiling** to ~2e-4 (current peak 5e-4 is aggressive for a 100k-step run on this data scale). Slower descent, less chance of skating past the Phase A basin.
4. Keep all the iMF architecture / fix_1 / fix_3 code as-is.

Expected outcome: Phase A floor (test `loss` ≈ 0.15) becomes the converged value rather than a transient touched at E2.

### Option C — Train longer at the current LR floor

Cheaper than Option B. The cosine schedule hits LR = 0 at E99. If you extend with another 50 epochs at LR ≈ 1e-5 starting from the current checkpoint, you might recover another 5–10% on test `loss`. But the basin is what it is; this won't move test `loss` from 0.46 down to 0.15. Diminishing returns.

### Option D — Switch to canonical reference iMF code

If post-fix_3 inference is still unacceptable, the next escalation in [`../CHANGELOG.md`](../CHANGELOG.md) §"Why This Doesn't Require Retraining" was already named: "retrain a model with `time_mlp` removed entirely (so the model is structurally h-only-conditioned, matching reference iMF exactly)." That's a bigger lift than Option B but addresses the architecture rather than the training stability.

---

## 7. What this log does **not** answer

- **Whether the inference jitter the user reported post-fix_1 is fully explained by the u-head residual** — would need a per-NFE inference-quality sweep (1, 2, 5, 10, 20 steps) to confirm. The expectation is that with proper u-head training (Option B), even NFE=1 should produce reasonable trajectories because iMF is designed for that.
- **Whether Hypothesis I (small-h amplification) is the actual root cause** — would need a per-batch log with `h.min()` distribution. Cheap test: re-run training with `h_min = 1e-3` and check if the E4 spike disappears.
- **Whether the checkpoint's aux head is "hiding" real u-head problems** — was the basis for fix_3's Deviation A. Now that we've disabled aux at inference, the next eval run will tell us. If trajectories are visibly worse than pre-fix_3, the aux head was load-bearing and the only path forward is Option B/D.

---

## 8. Cross-References

- [`../CHANGELOG.md`](../CHANGELOG.md) — fix_3's two architectural deviations and why no retrain was needed for *those*. This file argues a retrain may be needed for the *u-head training residual* — a separate concern.
- [`../../fix_1/INVESTIGATION.md`](../../fix_1/INVESTIGATION.md) — the corrected target formula `(x_t − x_r) / h`. Active throughout this training run.
- [`../../fix_2/REFERENCE_IMF_AUDIT.md`](../../fix_2/REFERENCE_IMF_AUDIT.md) — reference iMF audit that identified Deviations A and B.
- `flow_matcher_v3_imeanflow/models/imf_diffusion.py:303-322` — loss composition used to interpret the log.
- `flow_matcher_v3_imeanflow/models/helpers.py:180-188` — `a0_loss` definition (first-action MSE diagnostic, NOT the aux head).

---

## 9. One-line summary

Training shows a **single-epoch instability spike at E4** (u-head MSE 0.39 → 1.57) that the model never fully recovers from; the rest of training drifts down from the post-spike peak (test `loss` 0.681 → 0.459) but never returns to the Phase A floor (0.147). The end state has near-zero generalization gap but a u-head MSE ~3× worse than what was briefly achievable, which is the most plausible reason post-fix_3 trajectories still look "not smooth." Recommendation: retrain with `h_min` clamp + gradient clipping + reduced LR ceiling (Option B); fix_3 inference code stays as-is.
