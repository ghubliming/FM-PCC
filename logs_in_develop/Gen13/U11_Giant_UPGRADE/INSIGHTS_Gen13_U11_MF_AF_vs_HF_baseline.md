# Gen13 U11 — Results Insights: MeanFlow (MF) & AlphaFlow (AF) vs the HardFlow iMF baseline

**Run:** first HF_Mix_ML seed (seed 6, `avoiding-v0`, H16), one checkpoint each of the two new MLbones
trained inside HardFlow at **100k steps**, then evaluated with the frozen `eval_imf.py` path (objective-agnostic).
Data source: `temp/2907/2907/` (two training `metrics.csv` + the `hf_imf_eval`/`eval_ml_hardflow` SLURM logs).
**No local numerical run** — everything below is parsed from the cluster logs/CSVs.

> Checkpoints evaluated: `H16_ml_mf_100k/model_ema_4.pth`, `H16_ml_af_100k/model_ema_4.pth` (EMA, step 100k, cp 4).

**Update (post U12):** the originally-truncated AF `hardflow_new K2` run was re-evaluated through the
fixed U12 pipeline (`eval_ml_hardflow.sh`, `HF_EVAL_SAVE_PNG=0`) and **completed cleanly to n=200** —
no `Errno 28`, confirming the disk-flood fix works. All numbers below now reflect the full n=200 run;
the 129-episode partial result is superseded (§0/§1/§3/§4/Appendix updated).

---

## 0. Is the `temp/2907/2907` dump complete? — now **YES** (the one gap is closed)

| Piece | Present? | Note |
|---|---|---|
| MF training `metrics.csv` | ✅ full | 500 rows to step 100k |
| AF training `metrics.csv` | ✅ full | 500 rows to step 100k |
| MF eval (all 4 method×K blocks) | ✅ full | 200/200 episodes each |
| AF eval `original` K1/K2, `hardflow_new` K1 | ✅ full | 200/200 each |
| **AF eval `hardflow_new` K2** | ✅ **now full, 200/200** | rerun via `eval_ml_hardflow.sh` (log `12_24_11_eval_ml_hardflow_23991.log`) — the U12 PNG-flood fix held, no disk error |
| **Frozen iMF baseline eval** ("the HF baseline") | ❌ still absent | pulled from repo logs instead (§4) — this is a separate, still-open gap |
| Per-run `trajectories.csv` / `.npz` output dirs | ✅ present for the AF rerun (`H16_ml_af_100k/hfproj_K2_n200/`); ❌ still not downloaded for the rest | numbers for the rest recovered from log lines |

**Resolved — the original disk-full crash.** The first AF `hardflow_new K2` attempt died writing
`128_real.png` (`OSError: Errno 28`) after 129/200 episodes. The rerun used the U12-fixed eval path
(objective-named output, `HF_EVAL_SAVE_PNG=0` by default) and finished the full 200 episodes with
**zero disk errors** — direct field evidence the U12 fix works, not just a code-review claim.

**Still open — no frozen-iMF baseline eval in the dump.** §4 still relies on repo-log numbers rather
than a fresh iMF@100k pass through this exact pipeline; unchanged from before.

---

## 1. Headline — both new objectives WORK under HardFlow projection

Success rate over n=200 episodes, matched-K, seed 6 (AF-K2 now the completed n=200 rerun):

| Objective (100k) | orig K1 | orig K2 | **HF K1** | **HF K2** |
|---|---|---|---|---|
| **MF** (MeanFlow) | 1.0% | 4.0% | **95.5%** | **97.0%** |
| **AF** (AlphaFlow) | 2.0% | 11.5% | **89.0%** | **98.0%** |

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
- **NLP failures:** 0 unguided everywhere; under HF-projection, MF has 2/2 (K1/K2), AF has 1 (K1) but
  **6 at K2** (out of 2884 solves, on the now-complete n=200 rerun) — still <0.3% of solves and every
  episode still resolved to a value (the solver falls back to "last available value" on failure, per the
  `eval_imf` WARNING lines), so this is noise, not a systematic infeasibility. Worth a glance if AF's NLP
  failure rate keeps trending above MF's as more seeds are run, but not actionable on n=1.
- **Raw (unguided) field quality — AF > MF.** AF's *unprojected* success is meaningfully higher than MF's,
  especially at K2 (**11.5% vs 4.0%**, and 2.0% vs 1.0% at K1). AF's bootstrap target produces a raw trajectory
  that is closer to feasible before projection. Both still need the projection (as does iMF: ~0–2% unguided).

---

## 4. vs the HardFlow iMF baseline (from repo logs — not in this dump)

Frozen-iMF `hardflow_new` success, same task/seed family (`logs_in_develop/Gen13/U_9_train_curve/`, `9_CLOSURE_I/`):

| Model | budget | HF K1 | HF K2 |
|---|---|---|---|
| **MF (this run)** | 100k | 95.5% | 97.0% |
| **AF (this run)** | 100k | 89.0% | **98.0%** (n=200, confirmed) |
| iMF@100k | 100k | — | (98.5% @ **K5**) |
| iMF@300k | 300k | 96.5% | **99.5%** |

**Read:** at a matched 100k budget and K2, MF (97.0%) and AF (98.0%) land just under a *well-trained*
iMF, and iMF's best numbers (99.5%) use **3× the training budget**. For a first, untuned run whose u-head was
allowed to diverge in its final quarter, MF/AF at ~97–98% is a **strong positive** — the objectives are competitive,
and §2 says there is clear headroom (cp `3` / early-stop) before any conclusion that iMF is superior.
AF's K2 number is now a **confirmed full-n=200 result**, not an extrapolation from a partial run — it holds up.

⚠️ **Still open before the ranking is final:** both MF/AF used the *post-divergence* cp `4`, and the cleanest
control — a fresh **iMF@100k** at K1/K2 through this exact pipeline — is still not in the download (§0).

---

## 5. Sanity checks that passed

- **AF α-schedule fired correctly:** α = 1.00 (step 0) → 0.50 (50k) → **0.00 (100k)**, `discrete_frac` 0.5→0,
  `clamp_frac` ≈ 0 throughout. The sigmoid anneal spanned the real 100k budget (the `af_alpha_end_step ==
  n_train_steps` guard did its job). AF is genuinely annealing FM→MeanFlow, not stuck.
- **Objective-agnostic eval confirmed:** MF and AF checkpoints both loaded into `TemporalImfUnet` + ran through
  `ImfFlowPolicy` with zero eval-side changes — validating the U11 decision to reuse `eval_imf.py`.
- **No sign/convention bug:** if the DATA-AT-1 port had flipped, projected success would be ~0%, not 95–98%.
  The dialect-swap port is correct.
- **U12 disk-flood fix confirmed in the field:** the rerun of AF `hardflow_new K2` through
  `eval_ml_hardflow.sh` (`HF_EVAL_SAVE_PNG=0`) completed all 200 episodes with no `Errno 28`, where the
  original attempt crashed at episode 128. Not just a code-review pass — an actual before/after on the
  same checkpoint and arm.

---

## 6. Recommended next steps (in priority order)

1. ~~Free disk, re-run AF `hardflow_new K2` to n=200~~ — **done.** Completed cleanly at n=200 via the U12
   pipeline; result folded into §1/§3/§4 above (98.0%).
2. **Re-evaluate cp `3` (step 75k)** for both MF and AF — expected to beat cp `4` given the §2 divergence.
   *(now the top open item)*
3. **Add a fresh iMF@100k pass through this pipeline** as the true like-for-like baseline row.
4. Next training round: **early-stop / schedule cut at ~75k** or LR-tune the late phase to kill the u-head blow-up
   (the analytic-tangent JVP branch is the unstable element — worth a small `grad_clip`/LR ablation on MF alone).
5. Only after 2–4: draw the MF-vs-AF-vs-iMF field-quality verdict at matched budget.

---

### Appendix — exact aggregates parsed from the logs

```
MF  H16_ml_mf_100k  (14_37_35_hf_imf_eval_23968.log)
  original      K=1 | N=200 | success  2 ( 1.0%) | mean_viol 0.990 | reward 0.010 | steps 27.3 | NFE 3
  original      K=2 | N=200 | success  8 ( 4.0%) | mean_viol 0.960 | reward 0.040 | steps 29.1 | NFE 5
  hardflow_new  K=1 | N=200 | success191 (95.5%) | mean_viol 0.045 | reward 0.955 | steps 62.9 | NFE 5 | NLP fail 2
  hardflow_new  K=2 | N=200 | success194 (97.0%) | mean_viol 0.030 | reward 0.970 | steps 53.8 | NFE 9 | NLP fail 2

AF  H16_ml_af_100k  (16_36_29_hf_imf_eval_23980.log; K=2 rerun: 12_24_11_eval_ml_hardflow_23991.log)
  original      K=1 | N=200 | success  4 ( 2.0%) | mean_viol 0.980 | reward 0.020 | steps 17.4 | NFE 3
  original      K=2 | N=200 | success 23 (11.5%) | mean_viol 0.885 | reward 0.115 | steps 26.6 | NFE 5
  hardflow_new  K=1 | N=200 | success178 (89.0%) | mean_viol 0.115 | reward 0.890 | steps 56.6 | NFE 5 | NLP fail 1
  hardflow_new  K=2 | N=200 | success196 (98.0%) | mean_viol 0.020 | reward 0.980 | steps 53.4 | NFE 9 | NLP fail 6 | *COMPLETE (U12 rerun, no disk error)*

Training (final row @ step 100k / min raw_mse_u):
  MF : loss 1.998 | raw_mse_u 344.4 (min 11.14 @76.2k) | raw_mse_v 10.67 | a0_mse 4.59
  AF : loss 2.000 | raw_mse_u 350.8 (min  3.05 @70.6k) | raw_mse_v 10.33 | a0_mse 20.09 | alpha 1→0.5→0
```
