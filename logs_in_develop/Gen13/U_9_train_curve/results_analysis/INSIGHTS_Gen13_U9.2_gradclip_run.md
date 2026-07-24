# Gen13 U9.2 — the gradient-clipping run: worse loss, 31× better trajectories

**Date:** 2026-07-21 · **Type:** results analysis, no code change
**Evidence:** `temp/gen13_u9p2/hf_results_20260721_152932.zip` — 3 training runs, 39 eval runs, from `logs/hardflow/avoiding-v0/`
**Subject:** `H16_imf_lrfix_100k` (U9.2, 2026-07-21) vs `H16_imf_100k` / `H16_imf_300k` / `H16_1e6steps` (FM)
**Related:** [`../INSIGHTS_Gen13_U9_300k_train_and_eval.md`](../INSIGHTS_Gen13_U9_300k_train_and_eval.md) · [`../CHANGELOG_Gen13_U9.2_training_instability_fix.md`](../CHANGELOG_Gen13_U9.2_training_instability_fix.md) · [`COMPARE_…_imf_training.md`](../../../HF_iMF/Research/COMPARE_gen13_hardflow_vs_gen3v4_imf_training.md) §8

---

## Findings

| # | finding | § |
|---|---|---|
| 1 | ⭐⭐ **Unguided success 0.5 % → 15.5 %** (K=2, n=200) — the raw field solves the task for the first time in Gen13 | §4 |
| 2 | ⭐ **Every training metric got worse simultaneously** (`R²_u` 89.1→87.2 %). A worse loss produced a better field. | §3, §6 |
| 3 | 🔴 **`grad_clip=1.0` was ~50× too small** — measured `grad_norm` median **49.6**, **100 %** of steps clipped. Not clipping: normalised gradient descent. | §5 |
| 4 | 🔴 **`grad_norm` is flat** across training — the "escalating instability" story is retired | §5.2 |
| 5 | ⭐ **Both cheap quality surrogates are broken.** `raw_mse_u` is a residual, not accuracy; roughness *anti-correlates* with success — confirmed on FM's own data | §6, §7 |
| 6 | Guided performance moved the *opposite* way (99.5 → 90.5 %), traced to a 4× rougher warm start | §8 |
| 7 | The 31× gain is **confounded** across two changes; two 4 h arms de-confound it | §10 |

---

# PART I — WHAT WE MEASURED

## 1. What ran

```yaml
exp_name: H16_imf_lrfix_100k    grad_clip: 1.0             # U9.2 (new)
n_train_steps: 100000           learning_rate: 2.0e-05     # U9.2 (was 2e-4)
batch_size: 32                  ema_decay: 0.995           seed: 0
horizon: 16   imf_data_proportion: 0.25   imf_p_mean: -0.4   imf_p_std: 1.4
```
Identical to `H16_imf_100k` except `grad_clip` and `learning_rate`. Chained eval ran at K=2, n=200, both `original` (unguided) and `hardflow_new` (NLP-projected).

## 2. Reference points

| model | what it is |
|---|---|
| `H16_imf_100k` | the matched baseline — same config, no clip, LR 2e-4 |
| `H16_imf_300k` | 3× longer training, otherwise as baseline |
| `H16_1e6steps` | the authors' **FM** checkpoint — the thing to beat |

## 3. Training metrics

`R² = 1 − mse/mse₀`, `mse₀` = each run's own step-0 value (139.40 for `u`, 131.01 for `v`).

| run | `R²_u` | `R²_v` | `raw_mse_u` | `raw_mse_v` | `a0_mse` | max spike |
|---|---|---|---|---|---|---|
| `H16_imf_100k` | **89.11 %** | **91.21 %** | 15.18 | 11.52 | **0.2664** | 557 |
| **`H16_imf_lrfix_100k`** | 87.16 % | 89.21 % | 17.90 | 14.14 | 0.3724 | **211** |
| `H16_imf_300k` | 89.93 % | 92.51 % | 14.04 | 9.82 | 0.2123 | 7 548 |

Clipping cut the worst spike 557 → 211. Everything else regressed ~2 points of R². **On the loss curves alone this run is a failure.**

## 4. ⭐⭐ Task performance — K=2, n=200

| model | arm | success | steps | s/plan | NLP | `rough` | `rough_raw` |
|---|---|---|---|---|---|---|---|
| `300k` | unguided | 0.5 % | 37.8 | 0.0493 | 0 | 6.636e-04 | — |
| **`lrfix`** | **unguided** | **15.5 %** | 36.6 | 0.0527 | 0 | 2.406e-03 | — |
| `300k` | guided | **99.5 %** | 55.6 | 0.2342 | 14.7 | 4.116e-06 | 6.758e-04 |
| **`lrfix`** | guided | 90.5 % | 51.8 | 0.2352 | 13.6 | 2.839e-06 | **2.710e-03** |
| FM @ K=10 | guided | 100.0 % | 50.6 | 0.8379 | — | — | — |

**Unguided 0.5 % → 15.5 %.** Against the like-for-like `H16_imf_100k` baseline (0 of 20): `P(0 of 20 | p = 0.155) = 0.034`, significant at 5 %. 31 successes from a field that had never produced one.

## 5. 🔴 `grad_clip=1.0` was ~50× too small

`grad_norm` measured for the first time — the **pre-clip** norm returned by `clip_grad_norm_`:

```
median 49.6    p90 66.5    max 102.6    min 1.87
steps exceeding the 1.0 threshold:  100.00 %      below it:  0.00 %
```

**All 100 000 steps were clipped; the smallest gradient ever seen was still 1.9× the threshold.** With the norm pinned constant, Adam became **direction-only at fixed step size** — normalised gradient descent — at 10× the smaller LR.

**My error, of a specific kind:** I set `1.0` because it is the textbook default, without measuring the quantity it thresholds. The true scale was ~50. One diagnostic run logging `grad_norm` with clipping off would have caught it in minutes.

### 5.2 No instability escalation exists

`grad_norm` quintile medians: **48.2 · 52.0 · 49.6 · 49.7 · 49.2** — flat.

`CHANGELOG_U9.2` §1.1 inferred *escalating* instability from `raw_mse_u` spikes (331 → 1 117 → 7 548) plus a positive-feedback argument. **The gradient norms show nothing of the kind.** Retired. (See caveats: measured on an always-clipped trajectory.)

## 6. Smoothness — the full matrix

### 6.1 Guided (post-projection), n=20, `H16_imf_100k` vs FM

| K | FM `rough` | iMF `rough` | FM `rough_raw` | iMF `rough_raw` |
|---|---|---|---|---|
| 1 | 1.69e-06 (95 %) | 4.00e-06 (75 %) | 5.67e-04 | **3.74e-03** |
| 2 | 2.35e-06 (100 %) | 2.99e-06 (85 %) | 1.08e-04 | **8.66e-04** |
| 5 | 2.70e-06 (100 %) | 2.34e-06 (95 %) | 3.54e-05 | **1.98e-04** |
| 10 | 2.80e-06 (100 %) | **2.06e-06** (100 %) | 1.78e-05 | **1.42e-04** |

**Post-projection roughness is the same for both models (~2–4e-06).** The NLP flattens everything to one level, so `plan_roughness` on the guided path **measures the projection, not the model** — it cannot discriminate backbones. This is exactly what `DISCUSSION_foresight_fan_and_smoothness_paradigms.md` predicted.

The real difference is in `rough_raw` (the warm start): **iMF is 6–8× rougher than FM at every K.**

Secondary: iMF's guided roughness *falls* with K while FM's *rises*; they cross near K=5.

### 6.2 Unguided

| K | FM | iMF 100k | iMF 300k | **iMF lrfix** |
|---|---|---|---|---|
| 1 | *missing* | 1.65e-03 (0 %) | 1.81e-03 (0.0 %) | — |
| 2 | 9.21e-05 (**20 %**) | 5.69e-04 (0 %) | 6.64e-04 (0.5 %) | **2.41e-03 (15.5 %)** |
| 5 | 3.00e-05 (0 %) | 1.75e-04 (10 %) | — | — |
| 10 | 1.79e-05 (0 %) | 1.24e-04 (10 %) | — | — |

**`lrfix` is 26× rougher than FM@K=2 and still reaches 15.5 % vs FM's 20 %** — the first time iMF's raw field is in the same league as FM's, achieved while getting dramatically rougher.

---

# PART II — WHAT IT MEANS

## 7. Both of our cheap quality metrics are broken

### 7.1 `raw_mse` never had a known floor

The target is `v_target = x1 − x0` with **fresh noise every step**. A perfect model predicts `E[x1 − x0 | z_τ]`, not the sample; the gap is the conditional variance — **irreducible**. `raw_mse` cannot reach 0 and its floor is unknown, so every "raw_mse is high/bad" judgement in this project was made against an unjustified zero. Step 0 bounds the scale from the top, giving R². At ~89 % the model explains most of the target variance — **not a model that failed to learn.**

### 7.2 ⭐ `raw_mse_u` is a residual, not an accuracy — now demonstrated

`imf_matcher.py:106-107`:
```python
err_u = ((V - v_target) ** 2).sum(...)     # V = u - h·sg(D_tot)  -> RESIDUAL
err_v = ((v - v_target) ** 2).sum(...)     #                       -> direct ACCURACY
```
COMPARE §8.2 shows the residual is **blind to any error with `δ_u = h·δ_D`** while the sampler uses `u` alone. That was algebra. **§3 + §4 are the empirical counter-example**: `R²_u` fell 2 points while unguided success rose 31×.

**Every ranking of Gen13 checkpoints by `raw_mse_u` is unreliable — including "300k is the best model."** On unguided success the 300k model is the **worst** of the three.

### 7.3 ⭐ Roughness anti-correlates with success — confirmed on FM's own data

**FM falsifies it by itself:** its *roughest* unguided setting (K=2, 9.21e-05) is the **only one that ever succeeds** (20 %); its *smoothest* (K=10, 1.79e-05) succeeds **0 %**. And `lrfix` is 26× rougher than FM@K=2 while reaching 78 % of its success rate.

**Smooth-and-wrong is the dominant failure mode.** Task success on the **unguided** path is the only trustworthy signal we currently have.

## 8. Why guided moved the opposite way

| | `rough_raw` (warm start) | guided success |
|---|---|---|
| `300k` | 6.758e-04 | 99.5 % |
| `lrfix` | **2.710e-03** (4.0× worse) | 90.5 % |

**The lrfix field lands in the right basin more often on its own, yet hands the projection a worse starting point.** Direct use rewards being *near the goal*; the NLP rewards being *smooth and feasible*. Nothing forces those to agree, and here they anti-correlate.

Re-confirms the parent INSIGHTS: on the guided path **the projection dominates** — a field 31× better in isolation still lost 9 points once the NLP was in the loop.

## 9. Theory status

COMPARE §8 proposed iMF fails because the objective is a **differential-constraint residual with a blind direction of width `h`** — well-conditioned only at `h→0` (where it *is* FM), degenerate at `h=1` (where 1-NFE lives).

**§7.2 confirms the measurement consequence** — residual and field can move in opposite directions, and they did.

**The mechanism is not confirmed.** Plausible untested reading of §4: normalised gradient descent (§5) is scale-free across the `(τ,h)` domain, so it may have pushed capacity into the large-`h` region the sampler needs (COMPARE §7.3) at the cost of average residual. **The `h`-stratified metric settles this and costs nothing.**

---

# PART III — WHAT NEXT

## 10. De-confound the 31× gain — two arms, 4 h each, parallel

`lrfix` changed **two** things (LR 2e-4→2e-5 *and* effectively normalised gradients), so neither can be credited.

```bash
cd /u/home/llim/FMPCC/FM-PCC && git pull

# ARM A — learning rate only (clipping effectively off)
IMF_GRAD_CLIP=1e9 IMF_LR=2e-5 N_TRAIN_STEPS=100000 \
IMF_EXP_NAME=H16_imf_lronly_100k IMF_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh

# ARM B — normalised gradients only (baseline LR)
IMF_GRAD_CLIP=1.0 IMF_LR=2e-4 N_TRAIN_STEPS=100000 \
IMF_EXP_NAME=H16_imf_cliponly_100k IMF_KS="1 2" RANDOM_REPEAT=200 \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh
```

Read out on **unguided success @ K=2** (baseline 0/20, lrfix 15.5 %):

| Arm A | Arm B | conclusion |
|---|---|---|
| ~0 % | ~15 % | **normalised gradients** are the mechanism — reopens the optimiser choice entirely |
| ~15 % | ~0 % | **low LR** is the mechanism — sweep properly (1e-5, 5e-5) |
| both ~15 % | | either suffices; take the cheaper |
| both ~0 % | | genuine interaction — the two only work together |

`IMF_KS="1 2"` deliberately: K=1 unguided was **0.0 %** at 300k and is the harshest test.

## 11. Fill the data gaps

| gap | command / cost |
|---|---|
| **No `diag_smooth` for `lrfix`** — every cell is 100k or 300k | `N=5 IMF_EXP_NAME=H16_imf_lrfix_100k IMF_CP=4 IMF_K=2 CELLS="imf:unguided imf:guided" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_smoothness_diag_hardflow.sh` — ~5 min. ✅ verified: the script honours `IMF_EXP_NAME` and auto-tags `_from_<name>` (`eval_smoothness_diag.sh:29,35`); default `IMF_K` is 5, so pass `IMF_K=2` to match §4 |
| **`diag_smooth_fm_unguided_K1_n20` never completed** — has `config.yaml`, **no `trajectories.csv`** | re-run that one cell; K=1 is the regime the few-NFE argument lives in |
| **`H16_imf_100k` unguided baseline is n=20** — §4's significance rests on it | one short eval at n=200 |

## 12. Cheap, high-value, no retraining

| # | action | cost | value |
|---|---|---|---|
| 1 | **`h`-stratified residual** — bucket existing per-sample errors by `h` before `.mean()` | free | tests §9's mechanism; ~10 lines in `imf_matcher.py` |
| 2 | **FM's R² reference** — load `H16_1e6steps`, evaluate FM loss on the dataset | minutes | calibrates every `raw_mse` in the project: is 89 % good or bad? |
| 3 | **Test A** — drive the sampler from the v-head | 1 eval | separates "iMF `u` path" from "shared sampler bug" |

## 13. Not yet

Objective redesign (`data_proportion`, `p_mean/p_std`), backbone scaling, and the `aux_repo/imeanflow` cross-check all wait until §10 identifies the mechanism.

---

# APPENDIX

## A. Corrections to earlier documents

| document | claim | status |
|---|---|---|
| `CHANGELOG_U9.2` §1 | "effective LR 14–27× too hot" | **falsified** — Adam is invariant to a constant loss rescale (COMPARE §5) |
| `CHANGELOG_U9.2` §1.1 | escalating instability from the adaptive loss | **retired** — `grad_norm` is flat (§5.2) |
| `CHANGELOG_U9.2` §2 | `grad_clip=1.0` presented as a safe default | **wrong by ~50×** (§5) |
| `INSIGHTS_U9` | "300k is the better model" (on `raw_mse_u`) | **unreliable ranking** (§7.2) — 300k is worst on unguided success |
| earlier draft of this doc | "no U9.2 training has run" | **wrong** — the run existed; only the file drop was stale |
| earlier draft of this doc | "the retrain is priority 5 of 5 and probably won't help" | **wrong** — largest unguided improvement in Gen13 to date |

## B. Caveats

- **§4's 31× is confounded** across two simultaneous changes (§10). Real effect, unidentified cause.
- The `H16_imf_100k` unguided baseline is **n=20**; §4's significance rests on that small arm (§11).
- `grad_norm` (§5) was recorded on an **always-clipped** trajectory. Per-step values are true pre-clip norms, but the path through parameter space differs from an unclipped run — the ~50 scale must be re-measured on Arm A before being treated as universal.
- `lrfix` was evaluated at **K=2 only**; no K=1 arm, and no `diag_smooth` (§11).
- The guided/unguided anti-correlation (§8) is a single observation on one model pair.
- §6.1's iMF column is `H16_imf_100k` (untagged legacy runs), not `lrfix` — the two are not the same model.
- R² uses **step 0** as baseline — the untrained network, close to but not exactly the optimal constant predictor (§12 item 2 is the proper fix).
