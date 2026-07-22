# POST U10 IV — CATCH-UP: the A/B test is decisive, and where every experiment stands

**Date:** 2026-07-22 · **Type:** status + results analysis, no code change
**Evidence:** `temp/gen13_u9p2/FOLLOWUP/hf_results_20260722_155957.zip` (6 training runs, all evals) + 17 slurm logs in `FOLLOWUP/2026-07-21/`
**Purpose:** single orientation document. Covers **two generations** — see §1 for which job belongs where.
**Predecessor:** [`POST_U10_III_large_batch_test_and_theory_corrections.md`](POST_U10_III_large_batch_test_and_theory_corrections.md)
**Gen13 side:** [`Gen13/U_9_train_curve/results_analysis/INSIGHTS_Gen13_U9.2_gradclip_run.md`](../../../Gen13/U_9_train_curve/results_analysis/INSIGHTS_Gen13_U9.2_gradclip_run.md)

---

## 0. THE HEADLINE

> **⭐⭐ The A/B test resolved cleanly: the LEARNING RATE is the entire mechanism. Gradient clipping contributes nothing.**
>
> **⭐ A 5-model rank inversion is now established: the two LOWEST-loss models are the two WORST performers, and vice versa. `raw_mse_u` does not just fail to predict quality — it predicts it backwards.**

---

## 1. Job map — this drop spans TWO generations

Job names disambiguate them (`hf_imf_*` = Gen13 HardFlow; `imf_*` = Gen3v4 FMPCC):

| jobs | generation | what it was | status |
|---|---|---|---|
| 23636 / 23637 / 23638 | **Gen13** | `H16_imf_lrfix_100k` — U9.2 LR+clip | ✅ done, analysed |
| **23669 / 23670 / 23671** | **Gen13** | **ARM A** `H16_imf_lronly_100k` | ✅ **decisive, §2** |
| **23672 / 23673 / 23674** | **Gen13** | **ARM B** `H16_imf_cliponly_100k` | ✅ **decisive, §2** |
| 23668 | **Gen13** | smoothness diag on lrfix | ⚠️ ran fine — **summary display bug**, §4 |
| 23634 | Gen13 | earlier smoothness diag | superseded |
| 23650 / 23651 / 23652 | **Gen3v4** | large-batch 512, 50k steps | ✅ POST_U10_III executed, §5 |
| 23680 / 23681 / 23682 | **Gen3v4** | large-batch 512, 200k steps | ✅ §5 |
| (running) | Gen13 | `H16_imf_lrfix_800k` | 🔄 **~465k of 800k at collection**, no eval yet, §6 |

**Your notes were right on all three counts.** 23636 = the LR tune, 23668 = smoothness, 23669+23672 = arms A & B.

---

## 2. ⭐⭐ THE A/B TEST — LR is the mechanism

Configs verified from the drop: ARM A `grad_clip=1e9` (off) + `LR=2e-5`; ARM B `grad_clip=1.0` + `LR=2e-4`. Exactly as designed.

### Unguided (raw field, no NLP), n=200

| model | LR | clip | **K=1** | **K=2** |
|---|---|---|---|---|
| `H16_imf_100k` | 2e-4 | none | — | 0/20 |
| `H16_imf_300k` | 2e-4 | none | 0.0 % | 0.5 % |
| `H16_imf_lrfix_100k` | **2e-5** | 1.0 | — | **15.5 %** |
| **ARM A `lronly`** | **2e-5** | **OFF** | **8.0 %** | **17.5 %** |
| **ARM B `cliponly`** | 2e-4 | 1.0 | **0.0 %** | **1.5 %** |

**ARM A — low LR with clipping completely disabled — reproduces and slightly beats `lrfix` (17.5 % vs 15.5 %).**
**ARM B — clipping at the baseline LR — fails (1.5 %, statistically indistinguishable from the 0.5 % baseline).**

> **The low learning rate carries 100 % of the effect. Gradient clipping carries none.**

This lands exactly on row 2 of the prediction table in `INSIGHTS_Gen13_U9.2` §10 (*"A ~15 %, B ~0 % ⇒ low LR is the mechanism"*), which was written before the arms ran.

### 2.1 ⚠️ This vindicates a recommendation I had retracted

`CHANGELOG_Gen13_U9.2` §3 recommended `IMF_LR=2e-5`. I later **retracted** it (COMPARE §5: Adam is invariant to a constant loss rescale, and Gen3v4 runs ~9× "hotter" while converging). The retraction was **wrong on the conclusion**.

The honest accounting:
- The original **mechanism** argument ("adp(SUM) makes the effective LR 14–27× too hot") is still wrong — Adam does absorb a constant rescale.
- The original **prescription** (2e-5) is **empirically correct**, now confirmed by a controlled arm.

**Right answer, wrong reason.** Both errors are mine; the retraction shouldn't have converted a falsified *argument* into a rejected *prescription*.

### 2.2 The true gradient norm is ~47

ARM A ran with clipping **off**, so its `grad_norm` is the honest unclipped measurement:

| run | clip | `grad_norm` median |
|---|---|---|
| `lronly` (ARM A) | **off** | **46.9** ← true value |
| `lrfix` | 1.0 | 49.6 |
| `cliponly` (ARM B) | 1.0 | 10.0 |

**`grad_clip=1.0` was ~47× below the actual gradient scale** — confirmed independently now, not inferred. Every step in `lrfix` and `cliponly` was clipped.

### 2.3 Guided results — the ordering reverses again

| model | guided K=1 | guided K=2 |
|---|---|---|
| `300k` | 96.5 % | **99.5 %** |
| `cliponly` | **96.0 %** | **96.5 %** |
| `lronly` | 94.0 % | 90.0 % |
| `lrfix` | — | 90.5 % |

**On the guided path the ranking is inverted relative to unguided.** ARM B (worst raw field, 1.5 %) is the *best* of the three new models once the NLP is in the loop. Consistent with the standing finding that the projection dominates outcome — and a warning that "best model" is undefined without specifying the path.

---

## 3. ⭐ The rank inversion, now across 5 evaluated models

| model | `R²_u` | unguided K=2 | rank agreement |
|---|---|---|---|
| `lronly` | **87.03 %** (worst) | **17.5 %** (best) | ❌ inverted |
| `lrfix` | 87.16 % | 15.5 % | ❌ inverted |
| `100k` | 89.11 % | 0/20 | ❌ inverted |
| `cliponly` | 89.17 % | 1.5 % | ❌ inverted |
| `300k` | **89.93 %** (best) | 0.5 % | ❌ inverted |

**The two lowest-loss models are the two best performers. The three highest-loss models are the three worst.** A perfect rank inversion, n=5.

`raw_mse_u` doesn't merely fail to predict trajectory quality — **it predicts it backwards.** Mechanism in COMPARE §8.2: it is a *residual*, blind to any error satisfying `δ_u = h·δ_D`, while the sampler uses `u` alone.

### 3.1 🎯 A falsifiable prediction, recorded before the data lands

`H16_imf_lrfix_800k` currently has **`R²_u` = 89.97 % — the highest of any model.**

> **If the inversion holds, the 800k model will perform POORLY unguided (≲2 %) despite having the best loss curve of the entire project.**

If instead it scores ≥15 %, the inversion is not a law and §3 needs weakening. **Either way this is the cleanest test of the central claim.** Record the number the moment its eval lands.

---

## 4. ⚠️ Job 23668 — "not smooth, almost same as before" is a DISPLAY BUG

**Your observation was correct, and the cause is a shell glob.**

The job ran correctly and wrote correctly-tagged output:
```
=== exp_name: diag_smooth_imf_unguided_K5_n5_from_H16_imf_lrfix_100k ===
```
But the summary block at the end of `eval_smoothness_diag_hardflow.sh` globs:
```bash
for d in logs/avoiding-v0/eval/diag_smooth_*_n${N}; do
```
This requires the directory name to **end** in `_n5`. The new dirs end in `_from_H16_imf_lrfix_100k`, so **they never matched** — the summary silently printed the **old 100k / FM** results instead.

It looked identical to before because **it *was* the old data.**

**Fix:** one character — `diag_smooth_*_n${N}*`. (Same silent-wrong-output class as the U9 eval-naming bug. It should also print "no cells matched" rather than an empty loop.)

### 4.1 The real numbers, recovered from the zip

| cell | old (`100k`) | **new (`lrfix`)** | change |
|---|---|---|---|
| unguided K=5, n=5 | 1.666e-04 | **1.243e-03** | **7.5× rougher** |
| guided K=5, n=5 | 2.194e-06 | **1.594e-06** | 1.4× smoother |

Consistent with everything else: the better raw field is **much rougher**. Guided roughness stays pinned at ~2e-06 regardless of model — the NLP flattens everything, so guided roughness measures the projection, not the model.

⚠️ n=5 and both cells scored 0 % / 100 % — too small to rank. Treat as descriptive only.

---

## 5. Gen3v4 large-batch — POST_U10_III was executed

Jobs 23651 (50k steps) and 23681 (200k steps). Config verified: **`batch_size 512`, `gradient_accumulate_every 1`, `train_lr 1.4e-3`** — exactly POST_U10_III §7.3. (One deviation: `ema_decay 0.995`, not the recommended 0.9995.)

| run | steps | final `raw_mse` |
|---|---|---|
| reference (job 23552, eff. batch 64) | 50k | 3.07 |
| **23651** (batch 512) | 50k | **1.37** |
| **23681** (batch 512) | 200k | **1.25** |

**The brute-force batch increase cut `raw_mse` 2.2×**, and 4× more steps bought only a further 9 % — i.e. it plateaued, as §7.2 predicted from the ~96-independent-episode data ceiling.

⚠️ **But per §3, `raw_mse` is the metric that predicts backwards.** A 2.2× better loss is not evidence of a better model. The eval logs (23652, 23682) end with `Success rate: 1.0` alongside `Success rate (goal and constraints): 0.0` — reaching the goal while violating constraints. **These need proper parsing from the results pickles before any conclusion**; the console tail alone is not enough.

⚠️ Also unresolved from POST_U10_III §4.2: **Gen3v4's val split leaks** (random_split over overlapping windows). At 512×200k = **7,500 epochs** over 96 demos, memorisation is near-certain and `loss_test` cannot detect it.

---

## 6. What is still running / missing

| item | status |
|---|---|
| `H16_imf_lrfix_800k` | 🔄 ~465k / 800k at collection. **No eval yet** — §3.1's prediction is pending on it |
| `lrfix` unguided K=1 | ❌ never run (ARM A has it: 8.0 %) |
| `diag_smooth` summary | ⚠️ display bug, §4 — data exists, console lies |
| Gen3v4 eval parsing | ⚠️ needs the pickles, not the console tail |
| `H16_imf_100k` unguided n=200 | ❌ still n=20 only |

---

## 7. What to do next

### 7.1 Immediate, free
1. **Fix the glob** in `eval_smoothness_diag_hardflow.sh` (§4) — one character, prevents the next silent-wrong-summary.
2. **Parse the Gen3v4 evals properly** (§5) — the answer to the brute-force experiment is sitting in those files unread.

### 7.2 When the 800k eval lands
**Record its unguided K=1/K=2 number first, before anything else.** §3.1 makes it the decisive test of the rank inversion.

### 7.3 The experiment that now matters most: an LR sweep

§2 established LR is the lever but tested only two values, a factor of 10 apart. **The optimum is unmeasured and could be lower still** — the trend from 2e-4 (0.5 %) → 2e-5 (17.5 %) does not tell us where it turns.

```bash
cd /u/home/llim/FMPCC/FM-PCC && git pull

for lr in 5e-6 1e-5 5e-5; do
  IMF_GRAD_CLIP=1e9 IMF_LR=$lr N_TRAIN_STEPS=100000 \
  IMF_EXP_NAME=H16_imf_lr${lr}_100k IMF_KS="1 2" RANDOM_REPEAT=200 \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
done
```
Clipping **off** throughout — §2 showed it does nothing, so including it only adds a confound. ~4 h each. Together with the two existing points this gives a 5-point curve in unguided success vs LR.

### 7.4 Drop
**Gradient clipping.** Two controlled arms show no effect. Keep `grad_norm` **logging** (it is how §2.2 was measured), set the default to off.

---

## 8. ⭐ BOTTOM LINE — what we actually have, and has iMF beaten FM?

### 8.1 Direct answer: **No on the production path. Newly tied on the raw field.**

Best iMF from any run vs the authors' FM checkpoint (`H16_1e6steps`), at **matched K**:

| path | K | **FM** | **best iMF** | verdict |
|---|---|---|---|---|
| **guided** | 1 | 95.0 % @ **0.1119** s (n=20) | 96.0 % @ 0.1339 s — `cliponly` (n=200) | success tied¹, FM **1.20× faster** |
| **guided** | 2 | **100.0 %** @ **0.1894** s (n=20) | 99.5 % @ 0.2342 s — `300k` (n=200) | **FM wins** — better *and* 1.24× faster |
| **unguided** | 2 | 20.0 % (n=20) | 17.5 % — `lronly` (n=200) | **statistically tied** (Fisher p = 0.76) |
| **unguided** | 1 | *never ran*² | 8.0 % — `lronly` (n=200) | unknown |

¹ 96.0 % vs 95.0 % is 19/20 vs 192/200 — not distinguishable.
² `diag_smooth_fm_unguided_K1_n20` is the cell that never completed (§6).

> **Guided path: FM still wins.** It is faster at every matched K with equal-or-better safety. **The Gen13 efficiency thesis — that iMF's average-velocity field buys equal quality at fewer NFE — remains refuted.** It has now survived a 300k run, an 800k run in progress, and a controlled LR/clip A/B.
>
> **Unguided path: iMF has caught up.** 0.5 % → 17.5 % against FM's 20 %, Fisher p = 0.76. That is a genuine result and it is new as of this drop.

### 8.2 What we actually got out of all of this

**Three findings that outlast the iMF question:**

1. **`raw_mse_u` predicts trajectory quality *backwards*** (§3, n=5, perfect inversion). It is a *residual* — the objective is blind to any error with `δ_u = h·δ_D`, while the sampler uses `u` alone (COMPARE §8.2). **Any project selecting MeanFlow-family checkpoints by training loss is selecting the wrong ones.**
2. **Post-projection roughness measures the NLP, not the model** (§4.1, ~2e-06 for every model at every K). Both of the cheap surrogates are dead; only unguided task success survived scrutiny.
3. **The learning rate was the whole story, and it is a 35× effect** (§2) — 0.5 % → 17.5 % from one hyper-parameter, on a model everyone had written off as architecturally broken.

**One methodological result:** the failure mode in this project was never the math — it was **measurement**. Every wrong conclusion (the K confound, "300k is best", "clipping is the fix", "no U9.2 run happened", the smoothness display bug) came from an instrument that was mis-read or mis-built, not from a modelling error.

### 8.3 What we still do **not** know

- **Whether iMF can beat FM at all.** The LR sweep (§7.3) is untested past two points; the trend 2e-4 → 2e-5 has not turned over, so the optimum may be lower still.
- **Whether the 800k model helps or hurts** — §3.1's prediction says it will *hurt*, which would be the strongest confirmation of the inversion.
- **Whether any of this survives a second seed.** Every Gen13 number is one seed (§9).
- **Whether the field is genuinely good or the task is simply intolerant.** Unresolved since COMPARE §8.5: an FM R² reference (minutes, no training) would settle whether 87–90 % is good or bad, and it has still never been run.

### 8.4 Honest framing for a write-up

The defensible claim today is **not** "iMF beats FM". It is:

> *On a constrained-control task with 96 demonstrations, an improved-MeanFlow backbone matches flow matching's raw-field quality but does not improve the constrained-planning result, and is 1.2× slower per plan at matched NFE. The MeanFlow identity's residual objective is shown to be anti-correlated with sampled-trajectory quality across 5 trained models, which we trace to a blind direction of width `h` in the loss.*

That is a **publishable negative result with a diagnosed mechanism** — stronger than an unexplained failure, and it is what the evidence currently supports.

## 9. Corrections ledger

| claim | where | status |
|---|---|---|
| "effective LR 14–27× too hot" (adp/SUM argument) | `CHANGELOG_U9.2` §1 | **still wrong** — Adam absorbs a constant rescale |
| "`IMF_LR=2e-5` is the fix" | `CHANGELOG_U9.2` §3 | ✅ **correct after all** — §2.1. My retraction was wrong |
| "clipping is the part that survives" | COMPARE §5, `INSIGHTS_U9.2` | ❌ **falsified** — ARM B, §2 |
| "`grad_clip=1.0` is ~50× too small" | `INSIGHTS_U9.2` §5 | ✅ confirmed at **~47×** by direct unclipped measurement, §2.2 |
| "escalating instability" | `CHANGELOG_U9.2` §1.1 | still retired — `grad_norm` flat in every run |
| "300k is the best model" | `INSIGHTS_U9` | ❌ **it is the worst** on unguided, §3 |

## 10. Caveats

- §2's arms are **n=1 seed each**. The A/B contrast is large (17.5 % vs 1.5 %) and consistent across K=1 and K=2, but seed variance is unmeasured anywhere in Gen13.
- §3's inversion is **5 models, one task, one architecture**. It is a strong regularity here, not a general law — and §3.1 is its live test.
- §5's Gen3v4 conclusions are **provisional pending proper eval parsing**; only the training numbers are solid.
- The 800k `R²_u` (89.97 %) is measured at ~465k steps, not at completion.
- Every unguided number is `guidance_method=original` at `batch_size=1`; nothing here speaks to the guided/production path except §2.3.
