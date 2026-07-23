# Gen13 CLOSURE I — Does improved-MeanFlow beat flow matching inside HardFlow?

**Date:** 2026-07-23 · **Status:** CLOSURE · **Type:** paper-ready results analysis
**Task:** D3IL `avoiding-v0`, horizon 16, receding-horizon MPC (replan every 8 steps), obstacle radius 5 cm
**Models:** 6 iMF trainings + the authors' FM checkpoint `H16_1e6steps`
**Evaluations:** 28 runs, n=200 per arm on all decisive comparisons
**Evidence:** `temp/Gen13_Closure/hf_results_20260723_102914.zip` · `00_01_01_hf_imf_train_23684.log` · `23_48_48_hf_imf_eval_23734.log`

---

## Abstract

We replaced HardFlow's flow-matching (FM) backbone with an improved-MeanFlow (iMF) average-velocity backbone, hypothesising that iMF's exact endpoint map `x̂1 = z + (1−τ)·u` would improve HardFlow's constrained sampler over FM's Euler shot, enabling equal planning quality at fewer network evaluations (NFE).

**The hypothesis is refuted.** At matched NFE, FM equals or beats iMF on constrained planning while running **1.09–1.24× faster per plan**. iMF's raw (unguided) field reaches parity with FM's only after a hyper-parameter correction, and never exceeds it.

The investigation produced a result we consider more valuable than the original question. Across five iMF models spanning a 35× range in raw-field quality, **unguided and guided task success are perfectly rank-inverted (Spearman ρ = −1.00, n = 5)**. The model with the best raw trajectories (17.5 %) is the worst planner (90.0 %); the model with the worst raw trajectories (0.5 %) is the best planner (99.5 %). The training loss `raw_mse_u` correlates **+0.90 with guided** performance and **−0.90 with unguided** performance.

**Generating good trajectories and generating good warm starts for a constrained optimiser are conflicting objectives.** We trace the mechanism to warm-start smoothness (`plan_roughness_raw`), which predicts guided success at ρ = −0.90 while raw success predicts it at ρ = −1.00 in the wrong direction.

---

## 1. Method

### 1.1 What was implemented

iMF was added to the vendored HardFlow repository as a **purely additive** package (`hardflow/models_flow/imf/`); no pre-existing HardFlow file was modified.

| file | role |
|---|---|
| `imf/convention.py` | sole owner of time-convention logic; `sample_tau_h()`, `jvp_tangents()` |
| `imf/imf_matcher.py` | the training objective (§1.2) |
| `imf/temporal_imf_unet.py` | `TemporalImfUnet` — 8 blocks imported from the frozen `unet.py`, plus an `h` embedding and a dual `(u,v)` head; 3.69 M parameters |
| `imf/imf_flow_policy.py` | `ImfFlowPolicy`; the sampler seam |
| `run/train_imf.py`, `run/eval_imf.py` | additive siblings of `run/train.py`, `run/eval.py` |

### 1.2 The objective

FM regresses instantaneous velocity onto a **data-supplied** target (`hardflow/models_flow/flow_matcher.py:39-56`):

```
v_θ(z,τ) → x₁ − x₀ ,        L_FM = mean‖v − (x₁−x₀)‖²
```

iMF regresses the **average** velocity `u(z,τ,h)` over an interval of width `h`, via the MeanFlow identity `u = v + h·D_tot u` (`imf/imf_matcher.py:99-109`):

```
V     = u − h·sg(D_tot u)              # D_tot from torch.func.jvp, tangents (v_c, +1, −1)
L_iMF = adp(‖V − v_target‖²) + adp(‖v − v_target‖²),      adp(L) = L / sg(L+ε)
```

with `ε = 0.01`, `p = 1`, per-sample **sum** over the 96 trajectory dimensions, 25 % `h = 0` flow-matching anchors, and `(τ,h)` drawn from paired logit-normals (`p_mean = −0.4`, `p_std = 1.4`).

### 1.3 The seam — why iMF was expected to win

HardFlow's constrained sampler needs a terminal prediction at every ODE step:

| backbone | terminal prediction | error |
|---|---|---|
| FM | `x̂₁ = z + (1−τ)·v` | Euler shot, `O((1−τ)²)` |
| **iMF** | `x̂₁ = z + (1−τ)·u` | **exact endpoint map, zero truncation error** |

Since `u` is by construction the average velocity over `[τ,1]`, `z + (1−τ)·u` is *exact*, not an approximation. **This is the entire theoretical case for Gen13.** §5 evaluates whether it materialised.

### 1.4 Evaluation protocol

`K ≡ ode_t_steps` controls **both** NFE **and** the number of NLP projections — it is not a free axis, and all comparisons below are at **matched K**. Two paths are reported throughout:

- **unguided** (`guidance_method=original`) — the raw generative field, no projection. Measures the model.
- **guided** (`hardflow_new`) — the full constrained sampler. Measures the deployed system.

`batch_size = 1` (no candidate fan), n = 200 per arm unless stated. Success = reached goal **and** violated no constraint.

---

## 2. Experimental matrix

### 2.1 Training runs

`R² = 1 − mse/mse₀`, `mse₀` = each run's own step-0 value (139.40 for `u`, 131.01 for `v`).

| run | LR | grad clip | steps | `R²_u` | `R²_v` | `a0` | `grad_norm` med |
|---|---|---|---|---|---|---|---|
| `H16_imf_100k` | 2e-4 | none | 100 k | 89.11 % | 91.21 % | 0.2664 | not logged |
| `H16_imf_300k` | 2e-4 | none | 300 k | **89.93 %** | **92.51 %** | 0.2123 | not logged |
| `H16_imf_lrfix_100k` | **2e-5** | 1.0 | 100 k | 87.16 % | 89.21 % | 0.3724 | 49.6 |
| **`H16_imf_lronly_100k`** (ARM A) | **2e-5** | **off** | 100 k | **87.03 %** | 89.39 % | 0.3902 | **46.9** |
| **`H16_imf_cliponly_100k`** (ARM B) | 2e-4 | 1.0 | 100 k | 89.17 % | 91.50 % | 0.2852 | 10.0 |
| `H16_imf_lrfix_800k` ⚠️ | 2e-5 | 1.0 | **550 400** | 89.88 % | 91.69 % | 0.2555 | 33.4 |

⚠️ **The run named `800k` reached 550,400 steps**, not 800,000 — it was cancelled at the 24 h partition cap (measured rate 23,294 steps/h ⇒ 800 k needs 34 h). It is evaluated at checkpoint 22 (step 550,000). **Read every "800k" label as ~550k.**

### 2.2 The decisive evaluation table

All n = 200. FM baseline from `H16_1e6steps` (the authors' 1e6-step checkpoint).

| model | K | unguided | guided | s/plan | `rough_raw` | NLP solves |
|---|---|---|---|---|---|---|
| `300k` | 1 | 0.0 % | **96.5 %** | 0.1260 | 1.922e-03 | 9.0 |
| `cliponly` | 1 | 0.0 % | 96.0 % | 0.1339 | 2.446e-03 | 9.0 |
| `lronly` | 1 | **8.0 %** | 94.0 % | 0.1362 | 8.260e-03 | 7.9 |
| `550k` | 1 | 2.0 % | 75.5 % | 0.1727 | 2.429e-03 | 7.1 |
| `300k` | 2 | 0.5 % | **99.5 %** | 0.2342 | 6.758e-04 | 14.7 |
| `cliponly` | 2 | 1.5 % | 96.5 % | 0.2373 | 7.521e-04 | 14.2 |
| `550k` | 2 | 4.5 % | 95.5 % | 0.2923 | 8.153e-04 | 14.1 |
| `lrfix` | 2 | 15.5 % | 90.5 % | 0.2352 | 2.710e-03 | 13.6 |
| **`lronly`** | 2 | **17.5 %** | **90.0 %** | 0.2291 | 2.498e-03 | 13.3 |

FM baselines (n = 20 for K ∈ {1,2,5}; n = 200 at K = 10):

| model | K | unguided | guided | s/plan |
|---|---|---|---|---|
| FM | 1 | — ¹ | 95.0 % | **0.1119** |
| FM | 2 | 20.0 % | **100.0 %** | **0.1894** |
| FM | 5 | 0.0 % | 100.0 % | 0.4331 |
| FM | 10 | 0.0 % ² | 100.0 % | 0.8456 |

¹ `diag_smooth_fm_unguided_K1_n20` never completed (config written, no `trajectories.csv`) — the single gap in the matrix.
² 4.0 % at n = 50 in `H16_1e6steps_original_10steps`; 0/20 in the `diag_smooth` cell. Both consistent with a near-zero rate.

---

## 3. Q1 — Did iMF beat FM?

### 3.1 Guided (the deployed system): **No.**

| K | FM | best iMF | success | speed |
|---|---|---|---|---|
| 1 | 95.0 % @ **0.1119 s** | 96.5 % @ 0.1260 s (`300k`) | tied (Fisher p = 0.583 vs `cliponly`) | **FM 1.13–1.20× faster** |
| 2 | **100.0 %** @ **0.1894 s** | 99.5 % @ 0.2342 s (`300k`) | tied (p = 1.000) | **FM 1.24× faster** |
| 10 | 100.0 % @ 0.8456 s | 100.0 % @ 0.9224 s | tied | **FM 1.09× faster** |

**At every matched K, FM matches or exceeds iMF's success while running faster.** No iMF configuration at any budget reverses this.

The speed gap is architectural, not algorithmic: at matched K both backbones use identical NFE and identical NLP counts (K=2: 14.7 vs ~14 solves). The overhead is `TemporalImfUnet`'s `h` embedding and dual `(u,v)` head — iMF pays for machinery it does not convert into accuracy.

### 3.2 Unguided (the raw field): **parity, not superiority**

| model | unguided K=2 | vs FM (20.0 %, n=20) |
|---|---|---|
| `H16_imf_300k` | 0.5 % | far below |
| **`H16_imf_lronly_100k`** | **17.5 %** | **tied — Fisher p = 0.761** |

iMF's raw field went from *unusable* (0.5 %) to *statistically indistinguishable from FM* (17.5 % vs 20.0 %) — **entirely through a learning-rate change**, not through any property of the average-velocity formulation. It never exceeds FM.

### 3.3 The efficiency thesis is refuted

Gen13's premise was equal quality at fewer NFE. Measured: FM at **K=2** achieves **100 % safety at 0.1894 s/plan**. The best iMF result at any budget is 99.5 % at 0.2342 s/plan. **There is no operating point at which iMF is preferable.**

---

## 4. ⭐ The central finding: unguided and guided success are perfectly anti-correlated

Across the five models evaluated at K = 2, n = 200:

| model | `R²_u` | unguided | guided | `rough_raw` |
|---|---|---|---|---|
| `300k` | 89.93 % | **0.5 %** (worst) | **99.5 %** (best) | 6.758e-04 |
| `cliponly` | 89.17 % | 1.5 % | 96.5 % | 7.521e-04 |
| `550k` | 89.88 % | 4.5 % | 95.5 % | 8.153e-04 |
| `lrfix` | 87.16 % | 15.5 % | 90.5 % | 2.710e-03 |
| `lronly` | 87.03 % | **17.5 %** (best) | **90.0 %** (worst) | 2.498e-03 |

**Spearman rank correlations (n = 5):**

```
unguided  vs  guided     ρ = −1.00      ← perfect inversion
R²_u      vs  unguided   ρ = −0.90
R²_u      vs  guided     ρ = +0.90
rough_raw vs  guided     ρ = −0.90
```

### 4.1 Interpretation

**The training loss is not broken — it measures the wrong path.** `raw_mse_u` predicts *guided* performance correctly (ρ = +0.90) and *unguided* performance backwards (ρ = −0.90). Since the deployed system is guided, **`raw_mse_u` was a valid model-selection metric all along** — but only for the constrained sampler, and only by accident of what it happens to correlate with.

**The mechanism is warm-start smoothness.** `plan_roughness_raw` — the roughness of the plan handed to the NLP — orders guided success at ρ = −0.90. The two model families differ by ~3.5× in this quantity:

- **smooth-warm-start family** (`300k`, `cliponly`, `550k`): `rough_raw` ≈ 7–8e-04, guided 95.5–99.5 %
- **rough-warm-start family** (`lrfix`, `lronly`): `rough_raw` ≈ 2.5–2.7e-03, guided 90.0–90.5 %

A prox-NLP warm-started from a rough plan converges to a worse local solution — and it does so **regardless of whether that plan is closer to the goal**.

> **The two objectives conflict.** Direct trajectory generation rewards proximity to the goal. Constrained planning rewards a smooth, feasible, easily-projected initialisation. Optimising for one degrades the other. **This is the closure finding of Gen13.**

### 4.2 The confirmed prediction

Before the `550k` evaluation ran, `POST_U10_IV` §3.1 recorded: *"If the inversion holds, the 800k model will perform POORLY unguided (≲2 %) despite having the best loss curve of the entire project."*

**Measured: 2.0 % at K=1, 4.5 % at K=2** — against `lronly`'s 17.5 % from the *same configuration* trained 5.5× less (Fisher **p = 4.2e-05**).

**Training the winning configuration 5.5× longer destroyed 74 % of its raw-field performance** while improving its loss from 87.16 % to 89.88 % R². The inversion is not an artefact of hyper-parameter differences: it reproduces *within a single configuration* as a function of training duration.

### 4.3 Reframing the A/B result

Arms A and B (`lronly` vs `cliponly`) showed the learning rate carried the entire effect and gradient clipping none. §4.2 refines this: **the operative variable is total optimisation, not the learning rate per se.** Both high-performing raw-field models are the two *least converged*. Low LR at 100 k steps is an undertrained model; 550 k steps at the same LR converges to where the high-LR models already sat.

**iMF's raw field is best when the iMF objective is least optimised** — which is itself evidence for §6's account.

---

## 5. Q2 — Did HardFlow's math help the iMF raw field?

Two distinct questions.

### 5.1 Does the projection rescue the raw field? **Yes, overwhelmingly — for both backbones.**

| model | K | unguided → guided | lift |
|---|---|---|---|
| `H16_imf_300k` | 1 | 0.0 % → 96.5 % | from zero |
| `H16_imf_300k` | 2 | 0.5 % → 99.5 % | **199×** |
| `cliponly` | 1 | 0.0 % → 96.0 % | from zero |
| **FM** | **5** | **0.0 % → 100.0 %** | **from zero** |
| **FM** | **10** | **0.0 % → 100.0 %** | **from zero** |
| FM | 2 | 20.0 % → 100.0 % | 5× |

Roughness tells the same story: guided `plan_roughness` is **2.0–5.1e-06 for every model at every K**, spanning 0 %–99.5 % raw quality. Unguided roughness spans 5.3e-04 to 8.3e-03 — a 16× range that the projection collapses to a 2.5× range.

**HardFlow's projection determines the outcome; the generative field determines only the starting point.** A field that solves the task **0 times in 200** becomes a 96–100 % controller. This holds for FM exactly as much as for iMF — at K=5 and K=10, FM's unguided field is *also* at 0 %.

**⚠️ Consequence for the field:** results reported on the guided path are **not** evidence about generative model quality. Gen13's guided numbers (75.5 %–99.5 %) span models whose raw fields differ by 35×.

### 5.2 Did iMF's exact endpoint map help HardFlow? **No measurable benefit.**

This is the specific theoretical claim of §1.3: `x̂₁ = z + (1−τ)·u` is exact, FM's is `O((1−τ)²)`. The truncation error is largest at small τ and at small K, so the advantage should be **maximal at K=1**.

**At K = 1 — the most favourable possible test:**

| | success | s/plan |
|---|---|---|
| FM | 95.0 % (19/20) | **0.1119** |
| iMF `300k` | 96.5 % (193/200) | 0.1260 |
| iMF `cliponly` | 96.0 % (192/200) | 0.1339 |

Fisher p = 0.583. **The predicted advantage does not appear.** iMF is 1.13–1.20× slower for a statistically indistinguishable success rate.

**Why the exact endpoint map does not help:** exactness of the *map* is worthless if the *field* `u` is inaccurate. §6 shows the iMF objective cannot measure `u`'s accuracy, so the exactness guarantee applies to a quantity of unverified quality. FM's Euler shot is inexact but built on a field trained against an unbiased data target.

---

## 6. Why: the objective has a blind direction of width `h`

The training residual is `V − v_target` where `V = u − h·sg(D_tot u)`. Perturb the network's output by `δ_u` and its derivative by `δ_D`; the loss sees only

```
δ_u − h·δ_D
```

**Any error satisfying `δ_u = h·δ_D` is invisible to the loss.** But the sampler uses `u` alone (`x̂₁ = z + (1−τ)·u`, `z_{τ+h} = z + h·u`).

| regime | to hide error `δ` in `u`, need `δ_D =` | conditioning |
|---|---|---|
| `h → 0` | `δ/h → ∞` | well-conditioned — **and here iMF *is* FM** |
| `h = 0.5` (K=2) | `2δ` | degrading |
| `h = 1.0` (K=1) | `δ` | **free — degenerate** |

**iMF is flow matching plus a differential constraint whose conditioning degrades with `h`, and whose only well-conditioned regime is the one where it reduces to flow matching.** Its entire selling point — few-NFE sampling, i.e. large `h` — lives in its worst-conditioned regime. This is structural, not an implementation defect.

### 6.1 Coverage compounds it

Simulating `sample_tau_h` at the deployed settings (N = 200 000, reproduces the logged `h_mean` 0.19–0.22 to 3 s.f.):

```
P(h = 0)   = 0.250     ← FM anchors; u ≡ v, the trivial case
P(h ≥ 0.5) = 0.140
P(h ≥ 0.9) = 0.0011    ← the K=1 sampler operates HERE: ~1 sample in 900
```

The K=1 sampler queries `h = 1.0`, which receives **0.11 %** of training mass. Training and sampling occupy different regions of the same function.

### 6.2 What this predicts, and what was observed

| prediction | observation |
|---|---|
| the residual can improve while `u` degrades | ✅ §4.2 — R² 87.16 → 89.88 %, unguided 17.5 → 4.5 % |
| more optimisation of the residual → worse field | ✅ §4.3 — the two best raw fields are the least converged |
| the v-head (plain FM, no blind direction) should be healthier | ✅ `R²_v` exceeds `R²_u` in **all six** runs, and the gap widens with training (1.4 → 2.6 points on `300k`) |

**⚠️ Not confirmed:** these are consistent with §6 but do not exclude an implementation defect (sign, τ-convention, or seam). The decisive test — driving the sampler from the v-head, which contains none of the iMF machinery — was designed but **never run**. §8.

---

## 7. Threats to validity

| # | threat | severity | detail |
|---|---|---|---|
| 1 | **Single seed** | **high** | Every Gen13 training is seed 0. Seed variance is unmeasured project-wide. §4's ρ = −1.00 rests on 5 single-seed points. |
| 2 | **FM baseline arms are n = 20** at K ∈ {1,2,5} | medium | Only K=10 has n=200. The parity claim (§3.2) and the K=1 claim (§5.2) rest on n=20 FM arms. |
| 3 | **No held-out validation, anywhere** | **high** | `SequenceDataset` has no split (verified: 0 occurrences of val/test/holdout in `run/train.py` and `run/train_imf.py`). With 96 demonstrations, generalisation vs memorisation was never measured. |
| 4 | `550k` ≠ 800 k | medium | Truncated at the 24 h cap; evaluated at cp 22. Every "800k" label is ~1.8× the 300 k budget, not 2.7×. |
| 5 | **`raw_mse_u` has no known floor** | medium | The target `x₁ − x₀` uses fresh noise each step, so the irreducible conditional variance is unknown. R² is normalised against step 0 (untrained ≈ predict-zero), which bounds the scale from above only. **The FM checkpoint's R² — minutes of compute — would calibrate this and was never measured.** |
| 6 | One task, one architecture | medium | `avoiding-v0`, `TemporalUnet`-family, H=16. No claim generalises beyond this. |
| 7 | `diag_smooth_fm_unguided_K1_n20` missing | low | The one gap in the matrix; blocks a direct FM/iMF unguided comparison at K=1. |

---

## 8. What was never run

| experiment | cost | what it would settle |
|---|---|---|
| **Test A** — drive the sampler from the v-head | 1 eval | Separates "iMF `u` path is at fault" (§6) from "shared sampler bug". **The single largest open question.** |
| **FM checkpoint R²** | minutes, no training | Whether 87–90 % is good or bad (threat 5) |
| **`h`-stratified residual** | free — bucket existing per-sample errors before `.mean()` | Directly tests §6.1's coverage argument |
| **LR sweep** (5e-6, 1e-5, 5e-5) | 4 h each | §4.3 tested two points 10× apart; the optimum is unmeasured |
| **Checkpoint sweep** on `lronly` (cp 1–4) | ~30 min | §4.3 implies an early-stopping optimum below 100 k steps |
| Second seed on any arm | 4 h | Threat 1 |

---

## 9. Conclusions

1. **iMF does not beat FM inside HardFlow.** At matched NFE, FM matches or exceeds iMF's constrained-planning success while running 1.09–1.24× faster. The efficiency thesis is refuted. (§3)
2. **iMF's exact endpoint map yields no measurable benefit**, even at K=1 where its theoretical advantage is maximal. (§5.2)
3. **HardFlow's projection helps enormously — and helps FM equally.** It lifts both backbones from 0 % to 96–100 %. It is the controller, not an accessory. (§5.1)
4. ⭐ **Unguided and guided success are perfectly rank-inverted (ρ = −1.00).** Generating good trajectories and generating good warm starts are conflicting objectives, mediated by warm-start smoothness. (§4)
5. **The mechanism is a blind direction of width `h` in the MeanFlow residual**, degenerate precisely in the large-`h` regime few-NFE sampling requires. Three independent predictions of this account were confirmed; it remains unconfirmed against the implementation-bug hypothesis. (§6)
6. **Guided-path results are not evidence about generative model quality.** Models spanning 35× in raw field quality produce guided results within 9 points of each other. (§5.1)

### 9.1 Recommended framing

> *On a constrained-control task with 96 demonstrations, an improved-MeanFlow backbone matches flow matching's raw-field quality but yields no improvement in constrained planning, at 1.2× the per-plan cost. We further show that raw-field quality and constrained-planning quality are anti-correlated across five trained models (ρ = −1.00), mediated by warm-start smoothness — indicating that generative quality and optimiser-warm-start quality are conflicting objectives in projection-based planners.*

**This is a negative result with a diagnosed mechanism plus an unexpected positive finding** (§4), which is a stronger contribution than the original hypothesis would have been.

### 9.2 Recommendation

**Use FM at K=2 (100 % safe, 0.1894 s/plan).** Do not deploy iMF in this pipeline.

**Do not close the mechanism question.** §8's Test A costs one evaluation and determines whether §6 is the explanation or whether a latent implementation bug has been mistaken for a theoretical limit. Until it runs, §6 is well-supported theory, not established fact.

---

## Appendix A — file reference index

| artefact | path |
|---|---|
| results archive (all runs) | `temp/Gen13_Closure/hf_results_20260723_102914.zip` |
| final training log (550 k) | `temp/Gen13_Closure/00_01_01_hf_imf_train_23684.log` |
| final eval log (cp 22) | `temp/Gen13_Closure/23_48_48_hf_imf_eval_23734.log` |
| iMF objective | `HardFlow/hardflow/models_flow/imf/imf_matcher.py:99-109` |
| FM objective (reference) | `HardFlow/hardflow/models_flow/flow_matcher.py:39-56` |
| time convention, `sample_tau_h` | `HardFlow/hardflow/models_flow/imf/convention.py:65-103` |
| the seam | `HardFlow/hardflow/models_flow/imf/imf_flow_policy.py` |
| `hardflow_new` sampler | `HardFlow/hardflow/models_flow/flow_policy.py:1286-1321` |
| guidance-method dispatch | `HardFlow/run/eval.py:578-637` |
| training entry | `HardFlow/run/train_imf.py` |
| cluster bridge | `Slurm_Codes/sbatch/hardflow/_hardflow_common.sh` |
| pipeline (train→eval chain) | `Slurm_Codes/sbatch/hardflow/imf_pipeline_hardflow.sh` |
| results collector | `Slurm_Codes/sbatch/hardflow/collect_hf_results.py` |

**Analysis lineage:** `Gen13/U_9_train_curve/INSIGHTS_Gen13_U9_300k_train_and_eval.md` → `.../results_analysis/INSIGHTS_Gen13_U9.2_gradclip_run.md` → `Gen3v4_imf/U10/debug_notes/POST_U10_IV_catchup_AB_test_decisive.md` → **this document**.
**Theory:** `HF_iMF/Research/COMPARE_gen13_hardflow_vs_gen3v4_imf_training.md` §8.

## Appendix B — corrections ledger (claims retracted during Gen13)

| claim | where | outcome |
|---|---|---|
| "effective LR is 14–27× too hot" (adp/SUM argument) | `CHANGELOG_U9.2` §1 | **wrong** — Adam absorbs a constant loss rescale |
| "`IMF_LR=2e-5` is the fix" | `CHANGELOG_U9.2` §3 | **correct**, though for the wrong stated reason — §4.3 |
| "gradient clipping is the part that survives" | COMPARE §5 | **falsified** by ARM B |
| "`grad_clip=1.0` is a safe default" | `CHANGELOG_U9.2` §2 | **~47× too small** — measured unclipped `grad_norm` = 46.9 |
| "escalating instability" | `CHANGELOG_U9.2` §1.1 | **retired** — `grad_norm` flat in every run |
| "300k is the better model" | `INSIGHTS_U9` | **inverted** — worst raw field, best planner (§4) |
| "roughness is a quality proxy" | multiple | **falsified** — guided roughness measures the NLP; FM's roughest unguided setting is its only successful one |
| iMF@K=5 vs FM@K=10 comparisons | pre-`fix_7` | **invalid** — K controls NFE *and* NLP count; all conclusions re-derived at matched K |
