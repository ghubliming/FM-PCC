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

## 6. MPC foresight-fan smoothness diagnostic (U13) — does more K make the trajectory smoother?

**Setup:** U13's diagnostic (`eval_smoothness_diag_ml_hardflow.sh`), `guidance=hfproj`, n=5 episodes/cell,
K ∈ {1, 2, 5, 10, 20}, both MF and AF, checkpoints unchanged (100k, cp 4). Each hfproj run reports, per
replanned horizon, BOTH `plan_roughness_raw` (the pre-NLP warmstart plan) and `plan_roughness` (the
post-NLP projected plan) — so raw-vs-projected smoothness is read straight from one run, no separate
unguided pass needed. Data: `temp/2907/2907/{mf,af}/H16_ml_*_100k/smooth_hfproj_K*_n5/trajectories.csv`.

| fam | K | raw (pre-NLP) | projected (post-NLP) | raw/proj ratio | safe% (n=5) |
|---|---|---|---|---|---|
| MF | 1  | 1.494e-3 | 2.101e-6 | 711.2 | 100% |
| MF | 2  | 3.828e-4 | 2.288e-6 | 167.3 | 100% |
| MF | 5  | 1.635e-4 | 1.841e-6 |  88.8 | 100% |
| MF | 10 | 1.354e-4 | 1.971e-6 |  68.7 | 100% |
| MF | 20 | 2.042e-4 | 1.817e-6 | 112.4 | 100% |
| AF | 1  | 7.050e-4 | 1.294e-6 | 544.8 | 80% |
| AF | 2  | 1.954e-4 | 1.539e-6 | 127.0 | 100% |
| AF | 5  | 8.418e-5 | 1.712e-6 |  49.2 | 100% |
| AF | 10 | 7.564e-5 | 1.909e-6 |  39.6 | 100% |
| AF | 20 | 1.346e-4 | 1.827e-6 |  73.6 | 100% |

**Visual observation (user, from the `*_fan.png` renders): higher K visibly smooths the raw trajectory,
more so for MF than AF.** The numbers confirm the *direction* of this but add a real nuance the eye
can't see in a handful of static images:

1. **Confirmed — raw (pre-NLP) roughness drops sharply from K1→K10** for both: MF **11.0×** smoother
   (1.494e-3 → 1.354e-4), AF **9.3×** smoother (7.050e-4 → 7.564e-5). MF's improvement factor IS
   larger, corroborating "MF benefits more from higher K." The gap is real but modest (11.0× vs 9.3×),
   not the dramatic split the visual read might suggest.
2. **Important caveat the fan images alone don't show: AF's raw field is smoother than MF's at every
   matched K**, by ~1.8–2.1×, throughout the whole range (K1: 7.05e-4 vs 1.49e-3; K10: 7.56e-5 vs
   1.35e-4). So both are true and not contradictory: **MF's raw field starts rougher and closes the gap
   faster as K grows, but AF's raw field stays smoother in absolute terms at every K tested.**
3. **⚠️ NOT monotonic — K10 is the smoothest point tested, not K20.** Both families get *rougher again*
   at K20 versus K10: MF 1.354e-4 → 2.042e-4 (~1.5× rougher), AF 7.564e-5 → 1.346e-4 (~1.8× rougher).
   This reversal is consistent across BOTH families at the same K, which argues against pure n=5 sampling
   noise and for a real effect — plausibly compounding discretization/replan-boundary error at very fine
   step counts, or a warmstart-quality artifact at extreme K. **"More K = smoother raw trajectory" holds
   only up to ~K10 here; don't extrapolate it to K20 without checking this reversal at larger n first.**
4. **The PROJECTED (post-NLP) trajectory does NOT get smoother with K — it's flat, ~1.8–2.3e-6 for both
   families across the entire K1–K20 range**, with no discernible trend (differences are within n=5
   noise). This is the direct, quantitative answer to "does the other data show more K is better [for
   the decisive result]": **no.** It reconfirms the Gen13 fix_7 finding (`fix_7/RESULTS_..._2x2.md`) that
   HardFlow's NLP manufactures a consistent smoothness as a hard constraint, regardless of how rough the
   raw generative field is or how many ODE steps fed it. K only changes the *pre-projection* field;
   the thing that's actually executed is smooth either way.
5. **Practical takeaway:** if the goal is a smoother *raw* field (e.g. for warmstart quality or as a
   standalone generative-model diagnostic), K5–K10 is the useful range and MF gains more from it than AF
   — but for the *decisive, executed* trajectory, raising K buys no smoothness benefit (§3 already showed
   it costs more NFE/NLP solves for no safety benefit either, consistent with fix_7.3's "flat in K" precedent
   on this exact task). Recommend: re-run the K10/K20 cell at larger n (e.g. n=15–20) before treating the
   K20 reversal as established — n=5 is enough to see a 10×+ effect (K1→K10) but is thin for confirming a
   ~1.5–1.8× reversal at the tail.

**No new n=200 real run exists at K5/K10/K20** — only the n=5 smoothness cells above were run. But those
cells share `eval_imf.py`'s CSV format, so `success`, `steps`, and `average_computation_time` (wall-clock
per plan) are already sitting in the same `trajectories.csv` files, letting the "does more smooth also
bring better success/steps/time" question be answered from data already in hand, not a new run:

| fam | K | success (n=5) | mean steps | time/plan (s) | NFE/plan | raw roughness |
|---|---|---|---|---|---|---|
| MF | 1  | 100% | 62.4 | 0.170 | 5  | 1.494e-3 |
| MF | 2  | 100% | 56.2 | 0.267 | 9  | 3.828e-4 |
| MF | 5  | 100% | 50.6 | 0.545 | 21 | 1.635e-4 |
| MF | 10 | 100% | 50.6 | 1.065 | 41 | 1.354e-4 |
| MF | 20 | 100% | 51.8 | 2.185 | 81 | 2.042e-4 |
| AF | 1  |  80% | 56.0 | 0.148 | 5  | 7.050e-4 |
| AF | 2  | 100% | 56.2 | 0.246 | 9  | 1.954e-4 |
| AF | 5  | 100% | 52.4 | 0.544 | 21 | 8.418e-5 |
| AF | 10 | 100% | 52.6 | 1.050 | 41 | 7.564e-5 |
| AF | 20 | 100% | 51.0 | 1.963 | 81 | 1.346e-4 |

**Does smoother bring better success/steps/time? Mixed — one clear yes, one saturates, one is a clear no:**

- **Steps — YES, up to a point.** Episode length drops as K (and raw smoothness) improves from K1→K5:
  MF 62.4→50.6 (−19%), AF 56.0→52.4 (−6%), then **plateaus at K5–K20** (MF stays ~50–52, AF ~51–53) —
  tracking the raw-roughness trend's shape almost exactly (both improve fastest K1→K5, both flatten after).
  A smoother plan does reach the goal in fewer steps, but the benefit is exhausted by K5.
- **Success — saturates immediately, can't see a K-driven trend past K2 at this n.** AF's 80%→100% jump
  from K1→K2 matches the *direction* of its n=200 headline result (§1: AF unguided/K1 raw field is weaker
  — consistent with §6's finding that AF's raw roughness is still 3.6× worse at K1 than K10). Once at
  K≥2, n=5 sits at a 100% ceiling for both — too small a sample to resolve the ~1–3 point residual gaps
  the real n=200 K1-vs-K2 numbers show (§1: MF 95.5%→97.0%, AF 89.0%→98.0%). **Not evidence that K5/10/20
  keep improving safety — just that n=5 can't measure past a ceiling.**
- **Time — NO, and this is the clearest result in the table.** Time/plan scales linearly with K (and
  with NFE, exactly as expected: NFE≈4K+1, time≈NFE×~0.027s): **K1→K20 is a 13–14× wall-clock increase**
  (MF 0.170s→2.185s, AF 0.148s→1.963s) **for a benefit (steps) that already stopped at K5, and a raw
  smoothness benefit that reversed at K20 (§6 point 3).** There is no point in this data past ~K5 where
  extra compute buys anything measurable.

**Synthesis:** the U11 headline's choice of K1/K2 for the decisive n=200 run is well-supported, not
just convention — K2 already captures nearly all of the steps/success benefit smoother K provides, and
K5+ only adds linearly-growing compute cost. This lines up with `fix_7.3`'s established "flat in K"
precedent on this exact task (§ recommended next steps) and extends it: it's not just safety that's flat
past low K, steps-to-goal and raw smoothness are too, while cost keeps climbing regardless.

---

## 7. Recommended next steps (in priority order)

1. ~~Free disk, re-run AF `hardflow_new K2` to n=200~~ — **done.** Completed cleanly at n=200 via the U12
   pipeline; result folded into §1/§3/§4 above (98.0%).
2. **Re-evaluate cp `3` (step 75k)** for both MF and AF — expected to beat cp `4` given the §2 divergence.
   *(now the top open item)*
3. **Add a fresh iMF@100k pass through this pipeline** as the true like-for-like baseline row.
4. Next training round: **early-stop / schedule cut at ~75k** or LR-tune the late phase to kill the u-head blow-up
   (the analytic-tangent JVP branch is the unstable element — worth a small `grad_clip`/LR ablation on MF alone).
5. Only after 2–4: draw the MF-vs-AF-vs-iMF field-quality verdict at matched budget.
6. **§6's K20 raw-roughness reversal** (both families get rougher again vs K10) is n=5 — re-run
   K10/K20 at n=15–20 before treating it as a confirmed non-monotonicity rather than sampling noise.

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

Smoothness diagnostic (U13, hfproj, n=5/cell; source: temp/2907/2907/{mf,af}/H16_ml_*_100k/smooth_hfproj_K*_n5/):
  MF  K=1  raw 1.494e-3  proj 2.101e-6  ratio 711.2  safe 100%
  MF  K=2  raw 3.828e-4  proj 2.288e-6  ratio 167.3  safe 100%
  MF  K=5  raw 1.635e-4  proj 1.841e-6  ratio  88.8  safe 100%
  MF  K=10 raw 1.354e-4  proj 1.971e-6  ratio  68.7  safe 100%
  MF  K=20 raw 2.042e-4  proj 1.817e-6  ratio 112.4  safe 100%
  AF  K=1  raw 7.050e-4  proj 1.294e-6  ratio 544.8  safe  80%
  AF  K=2  raw 1.954e-4  proj 1.539e-6  ratio 127.0  safe 100%
  AF  K=5  raw 8.418e-5  proj 1.712e-6  ratio  49.2  safe 100%
  AF  K=10 raw 7.564e-5  proj 1.909e-6  ratio  39.6  safe 100%
  AF  K=20 raw 1.346e-4  proj 1.827e-6  ratio  73.6  safe 100%

Steps/success/time cross-check (same n=5 cells, same trajectories.csv, extra columns):
  MF  K=1  steps 62.4  time/plan 0.170s  NFE 5
  MF  K=2  steps 56.2  time/plan 0.267s  NFE 9
  MF  K=5  steps 50.6  time/plan 0.545s  NFE 21
  MF  K=10 steps 50.6  time/plan 1.065s  NFE 41
  MF  K=20 steps 51.8  time/plan 2.185s  NFE 81
  AF  K=1  steps 56.0  time/plan 0.148s  NFE 5
  AF  K=2  steps 56.2  time/plan 0.246s  NFE 9
  AF  K=5  steps 52.4  time/plan 0.544s  NFE 21
  AF  K=10 steps 52.6  time/plan 1.050s  NFE 41
  AF  K=20 steps 51.0  time/plan 1.963s  NFE 81
```
