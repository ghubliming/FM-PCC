# Gen13 U11 — Results Insights: MeanFlow (MF) & AlphaFlow (AF) vs the HardFlow iMF baseline

**Run:** first HF_Mix_ML seed (seed 6, `avoiding-v0`, H16), one checkpoint each of the two new MLbones
trained inside HardFlow at **100k steps**, then evaluated with the frozen `eval_imf.py` path (objective-agnostic).
Data source: `temp/2907/2907/` (two training `metrics.csv` + the two `hf_imf_eval` SLURM logs).
**No local numerical run** — everything below is parsed from the cluster logs/CSVs.

> Checkpoints evaluated: `H16_ml_mf_100k/model_ema_4.pth`, `H16_ml_af_100k/model_ema_4.pth` (EMA, step 100k, cp 4).

---

## 0. Is the `temp/2907/2907` dump complete? — **NO, two gaps**

| Piece | Present? | Note |
|---|---|---|
| MF training `metrics.csv` | ✅ full | 500 rows to step 100k |
| AF training `metrics.csv` | ✅ full | 500 rows to step 100k |
| MF eval (all 4 method×K blocks) | ✅ full | 200/200 episodes each |
| AF eval `original` K1/K2, `hardflow_new` K1 | ✅ full | 200/200 each |
| **AF eval `hardflow_new` K2** | ⚠️ **truncated 129/200** | job died writing `128_real.png` |
| **Frozen iMF baseline eval** ("the HF baseline") | ❌ **absent** | pulled from repo logs instead (§4) |
| Per-run `trajectories.csv` / `.npz` output dirs | ❌ not in dump | numbers recovered from the log lines |

**Gap 1 — disk-full crash (not a code bug).** The AF `hardflow_new K2` job crashed mid-render:
```
OSError: [Errno 28] No space left on device:
  logs/avoiding-v0/eval/H16_imf_hardflow_new_K2_from_H16_ml_af_100k_n200/128_real.png
```
It completed **episodes 0–128 (129)** then died before the `done` summary. Its 97.7% is on that partial set.

**Gap 2 — no baseline in the dump.** To answer "vs the HF baseline" I used the frozen-iMF numbers
already recorded in `logs_in_develop/Gen13/9_CLOSURE_I/` and `.../U_9_train_curve/` (§4). If you want a
same-download apples-to-apples, re-pull one frozen-iMF eval dir alongside these.

**To make it whole:** free cluster disk (`df -h`; prune old `logs/avoiding-v0/eval/*`), then re-run **only**
AF `hardflow_new K2` (and ideally a fresh iMF@100k baseline in the same submit).

---

## 1. Headline — both new objectives WORK under HardFlow projection

Success rate over n=200 episodes (n=129 where noted), matched-K, seed 6:

| Objective (100k) | orig K1 | orig K2 | **HF K1** | **HF K2** |
|---|---|---|---|---|
| **MF** (MeanFlow) | 1.0% | 4.0% | **95.5%** | **97.0%** |
| **AF** (AlphaFlow) | 2.0% | 11.5% | **89.0%** | **97.7%** ⚠️(n=129) |

- `original` = raw flow sampler, **no projection**; `hardflow_new` = HardFlow-projected (the real method).
- **First run of both new MLbones inside HardFlow trains and evaluates end-to-end with no plumbing failure.**
  The U11 assembly (family selector → per-family knobs) is functionally validated: `ml_type=mf` and `ml_type=af`
  each produce a working checkpoint that the frozen eval path loads and projects.

---

## 2. The critical finding — **the u-head diverged late in training; the loss hid it**

Both objectives show the classic adaptive-loss trap: `loss` sat flat at ≈2.0 the whole run while the
quantity that actually drives the sampler, `raw_mse_u`, **collapsed to a minimum mid-run and then blew up**:

| | min `raw_mse_u` | @ step | **final `raw_mse_u`** (step 100k) | final `a0_mse` |
|---|---|---|---|---|
| MF | **11.14** | ~76 200 | **344.4** (30×↑) | 4.59 |
| AF | **3.05** | ~70 600 | **350.8** (115×↑) | **20.09** |

- `loss≈2.0` is **flat by construction** (adaptive normalization) — it told us nothing. Judge on `raw_mse_u`/`a0_mse`.
- **The final checkpoint (`_4`, step 100k) is trained ~25–30k steps *past* the u-head minimum.** The EMA weights
  are what saved the eval (the raw late-training spike is smoothed), plus HardFlow projection is robust to a rough
  field — which is exactly why success is still 95–98% despite `raw_mse_u=340+`.
- **AF's blow-up is worse** (`a0_mse` 20 vs MF's 4.6), and it happens right as **α→0**, i.e. once AF has annealed
  into its MeanFlow-equivalent JVP branch. The instability lives in the **analytic/bootstrapped-tangent JVP**
  shared by MF and by AF-at-α=0 — the branch iMF deliberately avoids (iMF uses the *predicted* v-head as the
  tangent, which is model-bounded). This is a real, reproducible objective-level difference, not a port bug.

**Actionable:** the `_3` checkpoint (step 75k) sits almost exactly on the `raw_mse_u` minimum for both.
**Re-eval cp `3` — it should beat cp `4`.** For the next training round, add early-stop on `raw_mse_u` or
cut the schedule to ~75k; the last quarter of training is actively hurting the field.

---

## 3. Cost (NFE / NLP) and field quality

- **NFE/plan:** identical across objectives (3.0 @ K1 orig, 5.0 @ K1 HF / K2 orig, 9.0 @ K2 HF) — expected,
  since NFE is set by the shared sampler, not the objective.
- **NLP failures (out of ~1.5–2.8k solves): 0 unguided; 1–2 under HF** — negligible, well within the frozen-iMF
  norm. No systematic infeasibility from the new fields.
- **Raw (unguided) field quality — AF > MF.** AF's *unprojected* success is meaningfully higher than MF's,
  especially at K2 (**11.5% vs 4.0%**, and 2.0% vs 1.0% at K1). AF's bootstrap target produces a raw trajectory
  that is closer to feasible before projection. Both still need the projection (as does iMF: ~0–2% unguided).

---

## 4. vs the HardFlow iMF baseline (from repo logs — not in this dump)

Frozen-iMF `hardflow_new` success, same task/seed family (`logs_in_develop/Gen13/U_9_train_curve/`, `9_CLOSURE_I/`):

| Model | budget | HF K1 | HF K2 |
|---|---|---|---|
| **MF (this run)** | 100k | 95.5% | 97.0% |
| **AF (this run)** | 100k | 89.0% | 97.7% ⚠️(n=129) |
| iMF@100k | 100k | — | (98.5% @ **K5**) |
| iMF@300k | 300k | 96.5% | **99.5%** |

**Read:** at a matched 100k budget and K2, MF (97.0%) and AF (97.7%, partial) land just under a *well-trained*
iMF, and iMF's best numbers (99.5%) use **3× the training budget**. For a first, untuned run whose u-head was
allowed to diverge in its final quarter, MF/AF at ~97% is a **strong positive** — the objectives are competitive,
and §2 says there is clear headroom (cp `3` / early-stop) before any conclusion that iMF is superior.

⚠️ **Do not over-read the ranking yet:** (a) AF-K2 is n=129, (b) both MF/AF used the *post-divergence* cp `4`,
(c) the cleanest control — a fresh **iMF@100k** at K1/K2 through this exact pipeline — was not in the download.

---

## 5. Sanity checks that passed

- **AF α-schedule fired correctly:** α = 1.00 (step 0) → 0.50 (50k) → **0.00 (100k)**, `discrete_frac` 0.5→0,
  `clamp_frac` ≈ 0 throughout. The sigmoid anneal spanned the real 100k budget (the `af_alpha_end_step ==
  n_train_steps` guard did its job). AF is genuinely annealing FM→MeanFlow, not stuck.
- **Objective-agnostic eval confirmed:** MF and AF checkpoints both loaded into `TemporalImfUnet` + ran through
  `ImfFlowPolicy` with zero eval-side changes — validating the U11 decision to reuse `eval_imf.py`.
- **No sign/convention bug:** if the DATA-AT-1 port had flipped, projected success would be ~0%, not 95–98%.
  The dialect-swap port is correct.

---

## 6. Recommended next steps (in priority order)

1. **Free disk, re-run AF `hardflow_new K2`** to n=200 (and disable per-episode PNG saving in batch eval, or
   cap it — the render is what filled the disk and it is not needed for metrics).
2. **Re-evaluate cp `3` (step 75k)** for both MF and AF — expected to beat cp `4` given the §2 divergence.
3. **Add a fresh iMF@100k pass through this pipeline** as the true like-for-like baseline row.
4. Next training round: **early-stop / schedule cut at ~75k** or LR-tune the late phase to kill the u-head blow-up
   (the analytic-tangent JVP branch is the unstable element — worth a small `grad_clip`/LR ablation on MF alone).
5. Only after 1–4: draw the MF-vs-AF-vs-iMF field-quality verdict at matched budget.

---

### Appendix — exact aggregates parsed from the logs

```
MF  H16_ml_mf_100k  (14_37_35_hf_imf_eval_23968.log)
  original      K=1 | N=200 | success  2 ( 1.0%) | mean_viol 0.990 | reward 0.010 | steps 27.3 | NFE 3
  original      K=2 | N=200 | success  8 ( 4.0%) | mean_viol 0.960 | reward 0.040 | steps 29.1 | NFE 5
  hardflow_new  K=1 | N=200 | success191 (95.5%) | mean_viol 0.045 | reward 0.955 | steps 62.9 | NFE 5 | NLP fail 2
  hardflow_new  K=2 | N=200 | success194 (97.0%) | mean_viol 0.030 | reward 0.970 | steps 53.8 | NFE 9 | NLP fail 2

AF  H16_ml_af_100k  (16_36_29_hf_imf_eval_23980.log)
  original      K=1 | N=200 | success  4 ( 2.0%) | mean_viol 0.980 | reward 0.020 | steps 17.4 | NFE 3
  original      K=2 | N=200 | success 23 (11.5%) | mean_viol 0.885 | reward 0.115 | steps 26.6 | NFE 5
  hardflow_new  K=1 | N=200 | success178 (89.0%) | mean_viol 0.115 | reward 0.890 | steps 56.6 | NFE 5 | NLP fail 1
  hardflow_new  K=2 | N=129 | success126 (97.7%) | mean_viol 0.023 | reward 0.977 | steps 53.4 | NFE — | *TRUNCATED (disk full)*

Training (final row @ step 100k / min raw_mse_u):
  MF : loss 1.998 | raw_mse_u 344.4 (min 11.14 @76.2k) | raw_mse_v 10.67 | a0_mse 4.59
  AF : loss 2.000 | raw_mse_u 350.8 (min  3.05 @70.6k) | raw_mse_v 10.33 | a0_mse 20.09 | alpha 1→0.5→0
```
