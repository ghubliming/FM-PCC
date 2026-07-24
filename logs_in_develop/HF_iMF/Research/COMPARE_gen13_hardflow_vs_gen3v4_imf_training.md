# Comparing the two iMF trainings: Gen13 (HardFlow/UNet) vs Gen3v4 (FMPCC/DiT)

**Date:** 2026-07-21 · **Type:** discussion / diagnosis, **no code change**
**Sources:** `temp/23_22_28_imf_train_23552.log` (Gen3v4 seed 6, job 23552) ·
`temp/gen13_u9/hf_results_20260721_095317.zip → train/H16_imf_{100k,300k}__metrics.csv`
**Supersedes parts of:** `Gen13/U_9_train_curve/CHANGELOG_Gen13_U9.2_training_instability_fix.md` §1 (see §5 below)

---

## 0. TL;DR

1. **Both trainings converge, and by comparable amounts (~10×).** My earlier claim that "Gen13 barely learns" was **wrong** — an artifact of measuring from a 25k-step window median instead of from step 0. Retracted in §2.
2. The flat `loss` curve in W&B is **saturation of the adaptive objective, by design** — not non-convergence. Never read it. §1.
3. **HardFlow has no train/validation split at all** — confirmed, zero occurrences in `run/train.py` *and* `run/train_imf.py`. Gen3v4 does. §3.
4. Both runs show the **same spike pathology** and **neither has gradient clipping**. §4.
5. **Adam is invariant to a constant loss rescale**, which undercuts the U9.2 "effective LR is 14–27× too hot" arithmetic. §5.
6. Consequence: **the bad raw trajectory is not explained by a training failure.** The suspicion moves to the inference/seam path or the data. §6.
7. ⭐ **The measured `h`-coverage gap** — the K=1 sampler operates at `h=1.0`, which receives **0.11% of training samples**; 25% of every batch is the trivial `h=0` case. `raw_mse_u` converging and the trajectory being garbage are **measurements of different regions of the same function**, not a contradiction. §7.3
8. **The instrument is broken, not the model.** Highest-value next step is an `h`-stratified `raw_mse` — nearly free, no retraining, and it proves or kills §7.3 outright. §7.4
9. ⭐⭐ **Why iMF fails where FM works, in theory.** FM is supervised regression; iMF is a **differential-constraint residual** with a blind direction of width `h` — errors satisfying `δ_u = h·δ_D` are *invisible* to the loss, yet the sampler uses `u` alone. Conditioning is perfect only at `h→0`, where iMF **is** FM; it is degenerate at `h=1`, which is exactly where 1-NFE lives. **A structural tension, not a bug.** §8
10. Corollary: `raw_mse_u` and `a0_mse` are computed on the **residual `V`, not on `u`** — in *both* codebases. "Converged" has only ever meant "self-consistent". §8.3

---

## 1. The flat `loss` curve is a saturated metric, not a stalled model

Both codebases use the official iMF adaptive objective:

```python
adp(L) = L / sg(L + 0.01)**1.0        # ≡ ~1.0 whenever L ≫ 0.01
loss   = (adp(loss_u) + adp(loss_v)).mean()
```
Two heads ⇒ **hard ceiling 2.0**. Observed:

| | start | end |
|---|---|---|
| Gen3v4 `diffusion_loss` | 1.99 | 1.92 |
| Gen3v4 `loss_test` | 1.000 | 0.975 |
| Gen13 `loss` | 1.9998 | 1.9966 |

It is pinned against its own ceiling and **cannot** descend. It moves only when a sample's error approaches `eps=0.01`. This is why `train_imf.py`'s docstring says *"judge convergence on raw_mse_*, NEVER on the adaptive loss"*. Both repos implement the identical construct — Gen3v4 `imf_diffusion.py::_p_losses_imf_official`, Gen13 `imf/imf_matcher.py`.

## 2. ⚠️ Retraction — both converge, ~10× each

I previously wrote "Gen13 1.35× vs Gen3v4 4.1×" and concluded Gen13 barely moves off initialization. **That was wrong.** It compared Gen13's *first-quartile median* (18.5, already ~25k steps in) against its end, discarding the entire initial transient — while comparing Gen3v4 from its epoch-0 value. Measured consistently from step 0:

| | step 0 | end (last-10% median) | **drop** |
|---|---|---|---|
| **Gen13** `raw_mse_u` (300k) | **139.4** | **14.04** | **9.9×** |
| **Gen13** `raw_mse_u` (100k) | 139.4 | 15.18 | 9.2× |
| **Gen3v4** `raw_mse` | ~12.7 (epoch-0 avg) | 3.07 (final 1.52) | 4.1× (8.4× to final) |

And the **most directly comparable metric of all** — `a0`, plain MSE on the first action, same normalized units, same meaning at any horizon:

| | start | end | drop |
|---|---|---|---|
| **Gen13** `a0_mse` | **1.570** | **0.212** | **7.4×** |
| **Gen3v4** `a0_loss_test` | **1.4929** | ~0.19 | ~7.8× |

**These two curves are nearly identical.** You are right: on W&B both converged. Gen13's training is as healthy as Gen3v4's by every scale-free measure available.

### 2.1 What survives: a modest absolute-quality gap

Both use `LimitsNormalizer` (verified in both configs), so per-dim numbers are on a common [-1,1] scale:

| | dims (H×6) | final `raw_mse_u` | **per-dim** | per-dim RMS |
|---|---|---|---|---|
| Gen3v4 | 8×6 = 48 | 3.07 | 0.064 | 0.253 |
| Gen13 | 16×6 = 96 | 14.04 | **0.146** | **0.383** |

Gen13 ends ~2.3× worse per dimension. **But this is a soft comparison**: the regression target `u = v + h·D_tot` has a scale that depends on the `h` distribution, and the two runs sample `h` differently (Gen3v4: two independent logit-normals, D3, 50% FM anchors; Gen13: `p_mean=-0.4, p_std=1.4`, 25% anchors, logged `h_mean≈0.19–0.22`). H16 is also a strictly harder regression than H8. **Do not read this as "Gen13 is 2.3× worse."**

## 3. Yes — HardFlow has no train/validation split, at all

Verified:

```
grep -cE "\b(val|valid|test|holdout)\b"  HardFlow/run/train.py      -> 0
grep -cE "\b(val|valid|test|holdout)\b"  HardFlow/run/train_imf.py  -> 0
```
`hardflow/datasets/sequence.py::SequenceDataset` has no split parameter. **This is upstream HardFlow's own design, not something Gen13 broke** — our `train_imf.py` is an additive sibling of `train.py` and inherited it. HardFlow evaluates on rollout success, never on held-out likelihood.

Gen3v4 by contrast has `Trainer(train_test_split=..., split_seed=42)` with a seeded `random_split`, and job 23552 logged `loss_test` / `a0_loss_test` throughout.

### Why this matters here

- Every "Gen13 converged" statement in §2 is **train-set only**. We cannot distinguish *underfit* from *memorized-96-demos*.
- Gen3v4 gives us the missing evidence by proxy: **its test curve tracks its train curve** (`a0_loss_test` 1.49 → ~0.19, monotone, no divergence). At 96 demos with the same objective family, iMF was **not** overfitting there. That is mild evidence against a pure-overfit story for Gen13, but it is *not* a substitute for measuring it.
- Consequence: on the "data ceiling vs pipeline bug" question, **Gen13 currently has no instrument that can answer it.** Adding a held-out split would be the cheapest way to get one.

## 4. What the two runs genuinely share: spikes, and no clipping

`raw_mse` outliers against the run median:

| | median | max spike | ratio |
|---|---|---|---|
| Gen3v4 (50k) | ~5 | **327** (ep 22), also 108 (ep 7), 36.5, 23.2, 20.6 | ~65× |
| Gen13 (100k) | ~15 | **557** | ~37× |
| Gen13 (300k) | ~14 | **7,548** | ~500× |

**Neither codebase clips gradients** (`grep clip_grad` → nothing in `flow_matcher_v3_imeanflow/utils/training.py`; absent in HardFlow before U9.2). Both use Adam. Gen3v4 ran at **LR 5e-4** — hotter than Gen13's 2e-4, and 25× its own `Trainer` default of `train_lr=2e-5`.

This is the one part of the U9.2 fix that survives intact, and it is now **independently corroborated by a second codebase**: JVP-based MeanFlow losses spike, and clipping is the standard, cheap guard. It would likely help Gen3v4 too.

## 5. ⚠️ Correction to U9.2 §1 — Adam absorbs a constant loss rescale

The U9.2 changelog argued: `adp(SUM)` gives gradient scale `1/(err+eps) ≈ 1/14` vs the reference's `1/96`, hence "effective LR 14–27× too hot."

**Both trainings use Adam, whose update is invariant to multiplying the loss by a constant** — the factor cancels in the second-moment normalization. So that arithmetic overstates the case badly.

Gen3v4 is the direct counter-example:

| | adp denominator | LR | combined |
|---|---|---|---|
| Gen13 | ~1/14 | 2e-4 | 1× |
| **Gen3v4** | **~1/4** (3.5× hotter) | **5e-4** (2.5× hotter) | **~9× hotter** |

Gen3v4 is ~9× "hotter" by that metric **and converges at least as well**. The strong form of the hypothesis does not survive.

**What does survive:** Adam's scale-invariance breaks for *transient* outliers — a single 65–500× gradient poisons the running second moment for ~1/(1−β₂) ≈ 1000 steps. That is a real, local instability, and clipping fixes exactly it.

**Therefore `IMF_LR=2e-5` is now a weak bet**, not the headline fix. `IMF_GRAD_CLIP=1.0` remains well-motivated and near-free.

## 6. Where this leaves the "pure BS raw trajectory"

The training is not the culprit. Both models learned; their `a0` curves are near-identical. So the ~15.4 cm x̂1 error and 0% unguided success must come from somewhere else:

| candidate | status | how to test |
|---|---|---|
| **Inference / seam path** (`x̂1 = z + (1−τ)·u`, τ-schedule, sign/direction at sampling time) | **now the leading suspect** — training is exonerated | **Test A**: sample from the **v-head** through the identical sampler. The v-head is plain FM with none of the iMF machinery; if *it* also produces garbage, the bug is in the shared sampler, not in iMF. |
| Train/eval convention mismatch (τ=1−t, u_HF=−u_iMF) | gates G0/G1 passed on synthetic data only — they never exercised the real sampler on real data | compare a training-time `u` prediction against the sampler's `u` call for identical inputs |
| Data ceiling (96 demos) | **weakened** — Gen3v4's test curve did not diverge at the same data scale | needs a held-out split in Gen13 (§3) |
| Configuration gap vs Gen3v4 | untested | `imf_data_proportion` **0.25 → 0.5**; add guided `v_g` target + null token (D2) |
| Backbone | UNet-3.69M (Gen13) vs **DiT** (Gen3v4) | last resort — expensive, and the a0 curves say capacity is not the binding constraint |

**Recommended ordering:** Test A first (cheap, decisive, isolates sampler from objective), then the held-out split, then `data_proportion=0.5`. The LR sweep drops to last.

## 7. Quick recap — what each curve actually tells you, and what to do next

### 7.1 The "never-converging" `loss` curve — a broken *display*, not a broken *model*

**The failure mode:** `adp(L) = L / sg(L + 0.01)` divides the loss by its own detached value. The result is ≈ 1.0 for *any* `L ≫ 0.01`, so `loss = adp(u) + adp(v)` sits at its ceiling of 2.0 forever. This is **intentional** — the division is a per-sample *gradient reweighting* scheme (it equalises hard and easy samples). The number that gets printed is a side effect nobody was meant to read.

**So "how do I make it converge?" is the wrong question.** You do not fix the model; you fix the metric. Three options, in order:

| option | verdict |
|---|---|
| Log `raw_mse_u/v` and read those | ✅ **already done** in both codebases — this is the answer |
| Also log **per-dim RMS** = `sqrt(raw_mse_u / (H·6))` | ✅ cheap, and makes H8 vs H16 numbers directly comparable (§2.1's problem) |
| Raise `eps` so `adp` stops saturating | ❌ **never** — that silently changes the objective to make a plot look nicer |

**Do not try to make `loss` descend.** A descending `adp` curve would mean errors approaching 0.01, i.e. the reweighting has switched itself off — that is not a goal.

### 7.2 "`raw_mse` is converging" — what it buys you, and the three things it does *not*

It means: **the network fits its training regression target, averaged over the `(τ, h)` pairs it happens to sample.** That is a real result — it rules out "the model is broken/not learning". It is *not* nothing. But it is much weaker than it looks, for three separate reasons:

1. **It is a train-set number** (§3). No val ⇒ cannot separate generalisation from memorising 96 demos.
2. **It averages over the wrong `h` distribution** — see §7.3. This is the big one.
3. **MSE on `u` is not the quantity that matters.** What matters is the endpoint map `x̂1 = z + (1−τ)·u`. An error in `u` at `τ=0.9` is multiplied by 0.1; the same error at `τ=0` is multiplied by 1.0. `raw_mse_u` weights them identically, so it systematically **under-reports** exactly the errors that wreck the trajectory.

**Answer to "means nothing, right?":** it means *something* — just not what we were reading it as. It certifies the optimiser worked. It certifies nothing about sampling quality.

### 7.3 ⭐ The measured coverage gap — training and sampling live in different regimes

`convention.py::sample_tau_h` with Gen13's settings (`p_mean=-0.4, p_std=1.4, data_proportion=0.25`), simulated at N=200,000:

```
h_mean 0.2203   h_median 0.1594        (matches the logged h_mean 0.19–0.22 ✓)
P(h = 0)    = 0.250      <- FM anchors: u ≡ v, the trivial case
P(h ≥ 0.3)  = 0.323
P(h ≥ 0.5)  = 0.140
P(h ≥ 0.7)  = 0.036
P(h ≥ 0.9)  = 0.0011
```

Now what the **sampler** actually asks for:

| sampler | queries | training mass there |
|---|---|---|
| **K=1** | `(τ=0, h=1.0)` — one giant jump | **P(h ≥ 0.9) = 0.11%** — about **1 in 900** |
| **K=2** | `(τ=0, h=0.5)`, `(τ=0.5, h=0.5)` | P(h ∈ [0.45,0.55]) = **7.1%** |

**25% of every batch is `h=0`, where `u ≡ v` and iMF degenerates to plain flow matching.** The bulk of the remaining mass sits near `h ≈ 0.16`. So `raw_mse_u` is dominated by the *easy, small-h* regime — while every trajectory we generate depends entirely on `h = 0.5–1.0` accuracy.

**This reconciles the two facts with no contradiction:** `raw_mse_u` converged 10× *and* the raw trajectory is garbage, because they are measurements of different regions of the function. It also explains why 300k steps barely beat 100k (§2) — the extra compute went almost entirely into the region we never query.

> **Honest counter-evidence:** official MeanFlow achieves genuine 1-NFE image generation using a comparably-shaped logit-normal, so coverage alone cannot be the *whole* story — `u` is smooth in `h` and the network does interpolate. Treat §7.3 as the **leading testable hypothesis**, not a proven cause. §7.4 settles it cheaply.

### 7.4 How to make the metric real: measure what you sample

Three additions, cheapest first. All are **log-only** — no objective change, no retraining needed for (1).

1. **`h`-stratified `raw_mse_u`** — bucket the *existing* per-sample errors by `h` into `[0]`, `(0,0.3)`, `[0.3,0.6)`, `[0.6,1.0]` and log four numbers instead of one. **Zero extra compute** (the per-sample errors already exist before `.mean()`). If bucket 4 is flat at ~139 while bucket 1 fell to ~2, §7.3 is confirmed and the diagnosis is finished.
2. **True endpoint error at the sampler's own grid** — once per log interval, evaluate `‖x̂1 − x1‖` at exactly `(τ=0,h=1)` and the K=2 pair. This is *the* number that predicts trajectory quality. ~1 extra forward per 200 steps.
3. **Per-dim RMS** instead of a summed MSE, so numbers are horizon- and codebase-comparable.

**(1) is the single highest-value change in this document** — it is nearly free and it either proves or kills the leading hypothesis.

### 7.5 Do we need a real validation set? Yes — but it is *second*, and it must be episode-level

**Why yes:** 96 demos is small. Without a held-out set we cannot answer "does this generalise or has it memorised", and that question will be asked of any paper claim.

**Why second:** a val `raw_mse` would inherit **the exact same `h`-distribution mismatch** from §7.3. A beautifully converging val curve would be just as uninformative as the train curve is now. Fix the *metric* before adding a second copy of it.

**⚠️ Critical implementation detail — split by EPISODE, not by window.** `SequenceDataset` generates *overlapping* windows: with H=16 over 200-step episodes, adjacent windows share 15 of 16 frames. A naive `random_split` over the index list puts near-duplicates on both sides and reports a val loss that is **essentially a train loss**. Gen3v4's `train_test_split` uses `torch.utils.data.random_split` over the dataset — worth checking whether it has this same leak before treating its test curve (§3) as strong evidence.

The correct form: hold out ~19 whole episodes (80/20), build indices only from the remaining 77. Purely additive to `train_imf.py`; touches no frozen HardFlow file.

### 7.6 Do we need to redesign the objective?

**Not yet — and not first.** Ordered by evidence-per-hour:

| # | action | cost | what it decides |
|---|---|---|---|
| 1 | **`h`-stratified `raw_mse`** (§7.4.1) | ~free, no retrain | Is the field bad *only* at large `h`? |
| 2 | **Test A** — sample from the v-head through the identical sampler | 1 eval run | Is the bug in the **sampler** (shared) or the **objective** (iMF-only)? |
| 3 | **Endpoint error at the sampler grid** (§7.4.2) | 1 train run | Direct quality number, trackable during training |
| 4 | **Episode-level val split** (§7.5) | 1 train run | Generalise vs memorise |

Only if 1–3 come back clean does objective redesign become the right move. If §7.3 **is** confirmed, the redesign is small and targeted — not a rewrite:

- **Train where you sample:** force a fixed fraction of each batch onto the sampler's exact grid `{(0,1), (0,0.5), (0.5,0.5)}`. ~5 lines in `sample_tau_h`, directly closes the 0.11% gap.
- **Re-tune `p_mean`/`p_std`** so `h_mean` lands near 0.5 rather than 0.22. Blunter — `τ` and `h` are order statistics of the same two draws, so they move together.
- **`imf_data_proportion` 0.25 → 0.5** (Gen3v4's value). ⚠️ **Note the tension:** anchors are `h=0`, so raising this adds *more easy samples* and makes the coverage gap relatively worse — yet Gen3v4 uses 0.5 and trains better. The anchors are what make `u` well-posed at all. Do not change this on intuition; change it only after (1) tells you where the error actually lives.

**Bottom line:** the model is not broken and the optimiser is not broken. The *instrument* is broken — it reports the average of a function over a region we never evaluate. Fix the instrument (§7.4.1) before touching anything else.

## 8. Theory — "FM works, iMF is a strict generalisation of it, so why does iMF fail?"

The premise is correct, and that is what makes the answer interesting. **The representation theory is fine**: a `u` satisfying the MeanFlow identity exists, is unique given `v` and the boundary condition, and reduces to `v` as `h → 0`. Nothing is wrong there. The failure is not in *what is represented* — it is in *what kind of learning problem we posed*.

### 8.1 FM is supervised regression; iMF is a differential-constraint residual

| | target | nature |
|---|---|---|
| **FM** | `v_θ(z,τ) → x1 − x0` | the target is **in the data**: fixed, unbiased. Minimising it provably yields the conditional mean. Nothing self-referential. |
| **iMF** | `u_θ − h·sg(D_tot u_θ) → x1 − x0` | the data target is still real, but it is matched against a **combination of the network's output and the network's own JVP**. This is a PDE residual, not a pointwise regression. |

`imf_matcher.py:102-106`:
```python
V = u - pad_t_like_x(h, u) * du_tot.detach()   # du_tot = JVP of the model through itself
err_u = ((V - v_target) ** 2).sum(dim=(1, 2))
```

### 8.2 ⭐ The objective has a blind direction, and its width is exactly `h`

Perturb the network's `u` by `δ_u` and its total derivative by `δ_D`. The residual `V − v_target` sees only

```
δ_u − h · δ_D
```

**Any error satisfying `δ_u = h·δ_D` is completely invisible to the loss.** But **the sampler uses `u` alone** — `x̂1 = z + (1−τ)·u`, `z_{τ+h} = z + h·u`. The loss is blind to precisely the quantity that generates trajectories.

How blind, as a function of `h`:

| regime | to hide an error `δ` in `u` you need `δ_D =` | consequence |
|---|---|---|
| `h → 0` | `δ/h → ∞` | impossible ⇒ **well-conditioned** — and this regime *is* plain FM |
| `h = 0.5` (K=2) | `2δ` | partially degenerate |
| `h = 1.0` (K=1) | `δ` | **free** ⇒ maximally degenerate |

> **iMF is FM plus a differential constraint whose conditioning degrades with `h`, and whose only perfectly-conditioned regime is exactly the one where it reduces to FM.** Its entire selling point — 1–2 NFE, i.e. large `h` — lives in its worst-conditioned regime. That is a **structural tension, not a bug.**

This **subsumes and strengthens §7.3**: even with perfect `h` coverage, the large-`h` constraint is close to rank-deficient. Coverage is a contributing factor; conditioning is the mechanism.

### 8.3 Consequence: we have never measured what we thought we were measuring

`raw_mse_u` (line 106) is `‖V − v_target‖²` — **the residual, not the accuracy of `u`.** Likewise `a0_mse` (line 115) is computed on `V`, not on `u`.

So **"raw_mse converged" literally means "the network became self-consistent."** Self-consistent ≠ correct — §8.2 says a whole family of wrong `u` fields are perfectly self-consistent at large `h`.

The same holds for Gen3v4: its `a0` is `(u_pred − u_target)` where `u_target = (v_g + h·du_dr).detach()` is bootstrapped. **Neither codebase has ever measured `u` accuracy directly.** This weakens the §2 `a0` comparison — the two curves match, but they are matched residuals, not matched accuracies.

### 8.4 Where the grounding actually comes from, and why it decays

The `h = 0` anchors (25% of every batch, where `u ≡ v`) are the **boundary condition** of the PDE. Everything at `h > 0` is determined by propagating that boundary through a *learned* derivative. Error compounds with distance from `h = 0`, and §8.2 says the residual cannot see the compounding. At `h = 1` we are maximally far from the only fully-grounded data in the objective.

This also explains the weak v-head (§1.2 of the U9.2 changelog) without invoking learning rates: it shares a backbone whose representation is shaped mostly by a weakly-constrained objective.

### 8.5 So why does MeanFlow work on ImageNet?

1. **Batch 256–1024 vs our 32.** The JVP is a high-variance estimator of `D_tot`; averaging is what tames it. We have 96 demonstrations total.
2. **Error tolerance — probably the largest transfer gap.** FID rewards *plausible* samples: a slightly-wrong velocity field still produces a good-looking image. A slightly-wrong trajectory hits the obstacle. Our measured x̂1 error is **15.4 cm against a 5 cm obstacle radius**. Image generation is forgiving of exactly the error mode §8.2 hides; **constrained control is not.**
3. Image velocity fields are smoother and more locally structured than a 96-demo multimodal trajectory distribution.

### 8.6 ⚠️ Do not let this theory close the investigation

This explains every observation — but **a plain implementation bug (sign, τ-convention, seam) would produce identical symptoms.** An elegant story is not evidence. The distinguishing tests are unchanged:

| test | what it decides |
|---|---|
| **h-stratified residual** (§7.4.1) | does quality degrade with `h` as §8.2 predicts? |
| **Test A** — v-head through the identical sampler | the v-head is pure FM with **none** of this machinery. If it is *also* garbage, it is a shared-sampler bug and §8 is moot. |
| **Endpoint error `‖x̂1 − x1‖` at the sampler's grid** (§7.4.2) | the first metric in this project that measures **`u`** rather than the residual — see §8.3 |

**If §8 holds up it is a legitimate, publishable negative result:** *MeanFlow's few-NFE advantage does not transfer to low-data constrained control, because the identity's conditioning degrades exactly in the large-`h` regime that few-NFE requires.* This is consistent with the fix_7.3 finding that **FM@K=2 (100% safe, 0.1894 s/plan) beats iMF at every matched K**.

## 9. Open items / caveats

> **⚠️ CORRECTIONS FILED LATER** — see [`Gen3v4_imf/U10/debug_notes/POST_U10_III_large_batch_test_and_theory_corrections.md`](../../Gen3v4_imf/U10/debug_notes/POST_U10_III_large_batch_test_and_theory_corrections.md):
> - **§3 is RETRACTED in part** — Gen3v4's val split **leaks** (`random_split` over *overlapping windows*; adjacent windows share 7 of 8 frames at H=8). Its `loss_test` is effectively a train loss, so it is **not** evidence against overfitting. Both codebases have measured generalisation: never.
> - **§8.5 point 1's mechanism is wrong** — the JVP is *exact*, not a noisy estimator. Batch size helps by ordinary gradient-noise reduction. The prediction survives; the stated reason does not.
> - **§4's "batch 32" understates Gen3v4** — `gradient_accumulate_every: 2` makes the effective batch **64**.
> - **§4 addendum** — Gen3v4's `gradient_clip: 1.0` is a **dead config key**, never read by `utils/training.py`. §4's conclusion stands, but the config misleads any auditor.


- Gen3v4's exp folder name says `aw10` while the printed config shows `meanflow_aux_weight: 0.05` — unresolved; the printed block may belong to a different config section. Does not affect any conclusion above.
- Gen3v4 `raw_mse` is read from **epoch-end tqdm postfix values** (per-epoch snapshots, n=50), not a per-step CSV, so its spike count is a lower bound — the true per-step maximum is likely higher than 327.
- Gen13 has no step-0 equivalent for Gen3v4 to compare against directly; Gen3v4's 12.7 is an epoch-0 average over 1000 steps and therefore already post-transient. The `a0` comparison in §2 is the sound one.
- No claim here has been re-run; this is analysis of two existing logs.
- **§8 is theory, not measurement.** The blind-direction argument (§8.2) is exact algebra on the loss as written, but the claim that it is *the operative cause* of the bad trajectories is a hypothesis. §8.6 lists the tests that would confirm or refute it.
- §8.3 retroactively weakens §2: the Gen13-vs-Gen3v4 `a0` agreement is an agreement between two *residual* metrics, not between two accuracy metrics. The §2 retraction (both trainings converge) still stands — "converge" simply means less than we assumed.
- The `h`-coverage simulation in §7.3 reproduces `sample_tau_h` in pure Python (no torch in this container). It matches the logged `h_mean` 0.19–0.22 to 3 significant figures, but it is a re-implementation, not the code path itself.
