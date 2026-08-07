# DA — Gen14 mf/af massive run vs the Gen7 (FM) and Gen6V4 (diffusion) runs

**Date:** 2026-08-06
**Question:** can the Gen14 mf/af massive K=2 run
(`DA_20260805_n30_massive_K2_all_variants.md`) be compared against the old Gen7 and Gen6V4
runs, and if so, what does the comparison say?
**Data:** `temp/0608/batch_va2_20260806_204620/` — one DA_VA_v2 batch, 9 candidates, 198 units,
**3174 rollouts**, all from `logs/aligning-d3il-visual/plans` on i6-gpu-1, all seed 6.
**Figures:** `figs/fig1_gen_cmp_dead_run.png`, `figs/fig2_gen_cmp_cost.png`,
`figs/fig3_gen_cmp_availability.png`
**Script:** `da_20260806_gen14_vs_gen7_gen6v4.py` (scratchpad venv — this container has no
project env)
**Companion:** `CODE_STUDY_20260806_determinism_and_gen6v4_gen7_equivalence.md` settles the
same question from the code side. This is the data side. They agree.

> The batch CSVs live under `temp/`, which is **gitignored** — local only.

### Candidate → generation map

| cand | generation | tree | engine | K | source |
|---|---|---|---|---|---|
| 1, 2, 3, 4 | **Gen7** (Visual Flow Matching) | `fm_visual_aligning/` | `VisualFlowMatching` | 20 | MASTER_TEST_HISTORY line 30 |
| 9 | **Gen6V4** (Visual DPCC 9D) | `visual_aligning_dpcc/` | `VisualGaussianDiffusion` | 20 @ steps400 | `config/aligning-d3il-visual.py:344` ("Gen6V4 — Visual-DPCC") |
| 6, 8 | **Gen14** (Visual-Mix-ML) | `mix_visual_aligning_{af,mf}/` | `VisualAlphaFlow` / `VisualMeanFlow` | 2 | the massive run |
| 5, 7 | Gen14 | same | same | 100 | partial (41 rollouts) |

Gen7 appears as **four** candidate folders: `c1` = `(Bf_U8_Legacy_complex_struct)`,
`c2` = `(legacy_correcct_besides_bounds_for_comapre)`, `c3` = the current
`steps1000_filmv2`, `c4` = `steps900_filmv1`. c1 and c2 carry legacy markers in their folder
names; **c3 is the Gen7 candidate to compare against**, with c4 as the filmv1 sibling.

---

## 0. TL;DR

1. **Yes, a comparison is possible — but only on 3 contexts.** Gen14's 30-context data is on the
   **train** split; Gen7's and Gen6V4's 30-context data is on the **test** split, with a
   different context set (verified: the test contexts do not match Gen14's at any
   `rollout_idx`). The only ground all nine candidates share is **train contexts 0, 1, 2**,
   where the context values match to 0.000e+00. Everything else is unpaired across both a split
   boundary and a context boundary. See §1.
2. **The Gen6V4 run is dead and should not be quoted as a baseline at all.** In **89% of its
   test rollouts and 87% of its train rollouts the box never moves** (final offset = initial
   offset to <1 mm); for its projected variants that reaches **93–97%**. Its median physical
   tracking error is **1.80 m** — the arm is nowhere near where it is commanded. Its
   "0.382 m distance" is just the initial box offset. See §2.
3. **Gen7 cannot be compared on the constraint axis.** Its constraint fields are absent from
   most units: **1% / 6% / 38% / 87%** availability for c1 / c2 / c3 / c4. On the paired
   3-context basis, Gen7-c3 has **zero** rollouts with violation data. See §3.
4. **Neither old generation ever completes the task.** **0 goal successes in 535 Gen7 rollouts
   and 0 in 277 Gen6V4 rollouts.** Gen14 K=2 gets 42 in 2280 — 1.8%, which is dismal but not
   zero. This is the only quality statement in this document that rests on more than 3 contexts.
   See §4.1.
5. **On the paired 3 contexts, Gen14 mf/af beats both old generations on every axis that has
   data** — distance, violated steps, physical tracking error and cost. But **n = 3**, and the
   08-05 DA measured the noise floor on a 30-rollout mean at ±0.135 m; at n=3 it is roughly
   ±0.43 m, which swallows every distance difference below. Treat §4 as directional only.
6. **Cost is the one robust axis, and the margin is large.** Matched variants, same contexts:
   Gen14 K=2 runs at **28–56 ms/replan** against Gen6V4 at **19×** and Gen7-c3 at **21×**
   (medians over 15 shared variants; worst individual case Gen7-c3 `bounds_free` at
   **19.3 s/replan**, 460×). Cost does not depend on the split, so this comparison is clean.
   See §5.

---

## 1. What is comparable, and what is not

### 1.1 Inventory

| generation | candidate | split | geometries | variants | rollouts | n per variant |
|---|---|---|---|---|---|---|
| **Gen14** | 8 (mf), 6 (af) | train | `combined_5`, `-tightened` | 19 | 1140 each | **30** |
| Gen14 | 7 (mf), 5 (af) | train | `combined_5` | 2 | 41 each | 30 / 11 |
| **Gen6V4** | 9 | **test** | `combined_5` | 7 | 187 | 30 (×6), 7 (×1) |
| Gen6V4 | 9 | train | both | 15 | 45 each | **3** |
| **Gen7** | 3 (filmv2) | **test** | `combined_5` | **1** | 30 | 30 |
| Gen7 | 3 | train | both | 15 / 3 | 45 / 7 | **3** |
| Gen7 | 4 (filmv1) | **test** | `combined_5` | 11 | 303 | 30 (×10), 3 (×1) |
| Gen7 | 4 | train | 3 geos | 5 | 15 each | **3** |
| Gen7 | 1, 2 | train only | — | 2–11 | 70 / 35 | 1–3 |

*(fig3 renders this as a coverage heatmap.)*

### 1.2 The two things that block a 30-context comparison

**Split.** Gen14's massive run is `results_train_set` — in-distribution, 30 train contexts.
Gen6V4's and Gen7-c4's 30-context data is `test`. Comparing in-distribution rollouts against
held-out rollouts measures generalization gap, not engine quality.

**Contexts.** The test contexts are a different draw. Checking box init xy, target xy and box
angle at matching `rollout_idx` against Gen14's train contexts:

| candidate | split | n | overlap | max &#124;Δ context&#124; | |
|---|---|---|---|---|---|
| Gen7-c1/c2/c3/c4, Gen6V4 | train | 3 | 3 | **0.000e+00** | **same contexts** |
| Gen7-c3/c4, Gen6V4 | test | 30 | 30 | 1.715e+02 | **different contexts** |

So the paired basis is **train split, contexts 0–2**, and nothing larger exists in this batch.

### 1.3 And even there, it is not a controlled comparison

Per `CODE_STUDY_20260806_determinism_and_gen6v4_gen7_equivalence.md`:

- Gen14 loads weights from **its own checkpoint tree** (`mix_visual_aligning_{engine}/…`), not
  Gen6V4's or Gen7's (§4.1 of that study). Different weights.
- Gen14's `diffusion` arm runs `mpc_batch_size` **4** against Gen6V4's **1** — a different MPC
  controller (§4.2). The `fm` arm does match Gen7's planning block.
- The projected variants are not reproducible run-to-run anyway: the projector's breaker and
  solve deadline branch on wall-clock (§2 #7, #8).

Gen14's K also differs from both (2 vs 20), which is the point of Gen14, not a confound to
remove — but it does mean "Gen14 vs Gen7" is never a single-variable comparison.

---

## 2. Gen6V4 (candidate 9) is a dead run

A rollout in which `final_xy_dist_m` equals `context_init_xy_dist` is one where the box ended
exactly where it started. Counting those:

| candidate | split | rollouts | **box never moved** | median displacement | median `phys_err_m` | goal successes |
|---|---|---|---|---|---|---|
| **Gen6V4** | test | 187 | **89.3%** | **0.0000 m** | 0.245 | **0** |
| **Gen6V4** | train | 90 | **86.7%** | **0.0000 m** | **1.795** | **0** |
| Gen7-c4 | train | 45 | 44.4% | 0.098 | 0.577 | 0 |
| Gen7-c1 | train | 70 | 27.1% | 0.060 | 0.474 | 0 |
| Gen7-c4 | test | 303 | 26.7% | 0.299 | 0.049 | 0 |
| Gen7-c3 | train | 52 | 23.1% | 0.078 | 0.470 | 0 |
| Gen7-c3 | test | 30 | 20.0% | 0.041 | 0.054 | 0 |
| Gen7-c2 | train | 35 | 8.6% | 0.187 | 0.377 | 0 |
| **Gen14mf-K2** | train | 1140 | 26.8% | 0.231 | 0.048 | 29 |
| **Gen14af-K2** | train | 1140 | 26.5% | 0.189 | 0.057 | 13 |

Per-variant on Gen6V4's test split, where its 30-rollout cells live:

| variant | n | box never moved | mean dist | `phys_err_m` | ms/replan |
|---|---|---|---|---|---|
| `model_free` | 7 | **100.0%** | 0.324 | 2.519 | 2028 |
| `dpcc-c` | 30 | **96.7%** | 0.415 | 0.343 | 1825 |
| `dpcc-t` | 30 | **96.7%** | 0.441 | 0.149 | 1585 |
| `dpcc-r` | 30 | **93.3%** | 0.412 | 0.556 | 1776 |
| `post_processing` | 30 | **93.3%** | 0.413 | 1.838 | 391 |
| `diffuser` | 30 | 76.7% | 0.429 | 2.719 | 339 |
| `gradient` | 30 | 76.7% | 0.405 | 1.996 | 357 |

Three things follow:

1. **Gen6V4's distance numbers carry no information.** On the paired train contexts every
   variant reports 0.364–0.406 m, and `final_xy_dist_m` (0.451) equals `context_init_xy_dist`
   (0.451) exactly. That is the box sitting still, not a policy performing at 0.38 m. Any table
   that ranks Gen6V4 at "0.382 m" against Gen14's 0.348 m is comparing a number to a constant.
2. **Its constraint numbers are the opposite of good.** A policy that does not move mostly
   cannot violate a geometry constraint — and yet Gen6V4 still logs **282.7 violated steps** for
   `diffuser` and 87–327 across its variants on the paired contexts, against Gen14 mf's 90.3.
   So it is both inert *and* in violation, which points at the commanded trajectory diverging
   from the arm rather than at the arm doing something wrong.
3. **`phys_err_m` of 1.80 m median (train) confirms that reading.** The commanded position and
   the actual position are ~1.8 m apart at peak, against 0.05 m for Gen14. The controller is
   emitting trajectories the arm cannot follow, so the arm stalls and the box is never touched.

**Gen6V4 in this batch is a broken run, not a weak baseline.** It needs to be re-run before it
can be compared to anything. Whether the fault is the checkpoint, the eval wiring or the
`mpc_batch_size`/config drift noted in §1.3 is not answerable from these CSVs.

---

## 3. Gen7's constraint metrics are mostly missing

Fraction of rollouts with `constraint_exec_*` populated:

| candidate | `mean_dist_m` | `phys_err_m` | `avg_time_ms` | `n_success` | **constraint fields** |
|---|---|---|---|---|---|
| Gen14 (all 4) | 100% | 100% | 100% | 100% | **100%** |
| Gen6V4 | 100% | 100% | 100% | 100% | **100%** |
| Gen7-c4 | 100% | 100% | 100% | 100% | 87% |
| Gen7-c3 | 100% | 100% | 100% | 100% | **38%** |
| Gen7-c2 | 100% | 100% | 100% | 100% | **6%** |
| Gen7-c1 | 100% | 100% | 100% | 100% | **1%** |

On the paired 3-context basis, Gen7-c3 contributes **no** violated-step data at all in
`combined_5`, and only `dpcc-c` (n=3) in the tightened geometry. Gen7-c4 contributes `dpcc-c`
only. **The Gen14-vs-Gen7 constraint comparison cannot be made from this batch** — it would need
a Gen7 re-run with the current eval, which writes those fields unconditionally.

Distance, `phys_err_m` and timing *are* available for all Gen7 units, so §4 and §5 use those.

---

## 4. The paired comparison — train contexts 0–2

**Read this section as directional only.** n = 3. The 08-05 DA measured the run-to-run 95%
half-width on a 30-rollout distance mean at ±0.135 m (mf) / ±0.068 m (af); scaled to n=3 that is
roughly **±0.43 m / ±0.22 m**, wider than almost every gap in the tables below.

### 4.1 The one statement that does not depend on n=3

| generation | rollouts in batch | goal successes | rate |
|---|---|---|---|
| **Gen7** (c1+c2+c3+c4) | 535 | **0** | 0.00% |
| **Gen6V4** | 277 | **0** | 0.00% |
| **Gen14** (all four candidates) | 2362 | **45** | 1.91% |
| Gen14 K=2 only (cands 6+8) | 2280 | 42 | 1.84% |

Zero successes in 812 old-generation rollouts against 42 in 2280 is not an n=3 argument. It is
also not a good result for Gen14 — 1.8% is a failing policy — but the old generations do not
solve this task at all in any data present in this batch.

### 4.2 `combined_5`, contexts 0–2

`mean_dist_m` (lower better):

| variant | Gen14mf-K2 | Gen14af-K2 | Gen6V4 | Gen7-c3 |
|---|---|---|---|---|
| `diffuser` | 0.622 | **0.213** | 0.382 † | 0.722 |
| `bounds_free` | **0.293** | 0.324 | 0.382 † | 0.743 |
| `dpcc-c` | 0.392 | **0.268** | 0.364 † | 0.632 |
| `dpcc-r` | 0.349 | 0.364 | 0.382 † | 0.700 |
| `dpcc-t` | 0.411 | **0.247** | 0.382 † | 0.373 |
| `dpcc-c-dt0p5` | **0.293** | 0.270 | 0.381 † | 0.691 |
| `gradient` | **0.253** | 0.372 | 0.374 † | 0.366 |
| `post_processing` | **0.349** | 0.361 | 0.382 † | 0.830 |
| `model_free` | 0.589 | **0.299** | 0.383 † | 0.822 |

† Gen6V4's column is the initial box offset — see §2. It is printed for completeness, not as a
performance number.

`phys_err_m` (lower better) — the axis with the widest generational gap:

| variant | Gen14mf-K2 | Gen14af-K2 | Gen6V4 | Gen7-c3 |
|---|---|---|---|---|
| `diffuser` | 0.567 | 0.825 | **3.213** | 0.290 |
| `bounds_free` | **0.103** | 0.152 | 0.407 | 0.777 |
| `dpcc-c` | **0.049** | 0.046 | 0.454 | 0.427 |
| `dpcc-c-dt0p5` | **0.040** | 0.052 | 2.520 | 0.749 |
| `dpcc-r` | **0.135** | 0.116 | 1.165 | 1.223 |
| `dpcc-t` | **0.050** | 0.049 | 0.154 | 1.419 |
| `gradient` | **0.040** | 0.856 | 2.489 | 1.156 |
| `post_processing` | **0.135** | 0.116 | 2.366 | 0.297 |

Gen14's projected variants sit at **0.03–0.16 m** where Gen6V4 sits at **0.15–3.2 m** and Gen7
at **0.29–1.42 m**. That is a 3–60× gap and it points the same way in every row that has data.

Violated steps — Gen7 has no data (§3), so this is Gen14 vs a run that does not move (§2):

| variant | Gen14mf-K2 | Gen14af-K2 | Gen6V4 |
|---|---|---|---|
| `diffuser` | 90.3 | 106.3 | **282.7** |
| `bounds_free` | **1.7** | 24.0 | 143.0 |
| `dpcc-c` | 27.0 | 21.3 | 96.7 |
| `dpcc-c-dt0p5` | **10.3** | 16.3 | 133.7 |
| `dpcc-r` | 28.7 | 30.0 | 136.7 |
| `dpcc-t` | 78.0 | **2.0** | 76.0 |
| `gradient` | **1.3** | 151.3 | 326.7 |
| `post_processing` | 28.7 | 30.0 | 240.7 |

### 4.3 `combined_5-tightened`, contexts 0–2

Gen7 contributes almost nothing here (c2: 2 variants, c3: 3, c4: 5). Gen14 vs Gen6V4 on
violated steps:

| variant | Gen14mf-K2 | Gen14af-K2 | Gen6V4 |
|---|---|---|---|
| `dpcc-t` | **0.0** | **0.0** | 0.0 |
| `dpcc-c` | **0.0** | **0.0** | 0.0 |
| `dpcc-c-dt0p25` | **0.0** | **0.0** | 120.3 |
| `dpcc-c-dt0p5` | **0.0** | **0.0** | 193.3 |
| `bounds_free` | **0.0** | 2.3 | 25.0 |
| `dpcc-r` | **0.0** | 11.0 | 0.0 |
| `hardflow_new-t` | **0.0** | **0.0** | n/a |
| `diffuser` | 96.0 | 101.7 | 282.7 |

Gen6V4's zeros in this table are the "did not move" zeros of §2 — its `dpcc-t`/`dpcc-c`/`dpcc-r`
cells are 93–97% inert. Gen14's zeros come with a median box displacement of 0.27–0.42 m in the
same cells, i.e. the policy moved the box **and** stayed clean.

---

## 5. Cost — the one clean comparison

Timing does not depend on the split or on which contexts were drawn, so this comparison is not
subject to §1.2. Matched variants, same 3 contexts, `combined_5`, ms/replan:

| variant | Gen14mf-K2 | Gen14af-K2 | Gen6V4 | Gen7-c3 | Gen7-c4 | Gen7-c2 |
|---|---|---|---|---|---|---|
| `diffuser` | **30** | 28 | 323 | 334 | 289 | 326 |
| `dpcc-c` | **49** | 42 | 1642 | 3873 | 17635 | 13524 |
| `dpcc-r` | **47** | 55 | 989 | 11407 | — | — |
| `dpcc-t` | **52** | 46 | 1463 | 2923 | 3815 | 26701 |
| `bounds_free` | **42** | 40 | 1198 | **19275** | — | — |
| `model_free` | **40** | 47 | 1827 | 488 | 374 | 438 |
| `post_processing` | **47** | 56 | 382 | 335 | 289 | 322 |

Median slowdown against Gen14-K2 over the shared variants:

| | vs Gen14-K2 | shared variants |
|---|---|---|
| Gen7-c3 (current filmv2) | **21.3×** | 15 |
| Gen6V4 | **19.4×** | 15 |
| Gen7-c1 | 63.4× | 2 |
| Gen7-c2 | 11.2× | 5 |
| Gen7-c4 (filmv1) | 10.0× | 5 |
| *Gen14 K=100 (for reference)* | *164–198×* | *2* |

Gen14 K=2 is the only configuration in the batch that lands near the 30 Hz budget: **28–56 ms**
for everything except the HardFlow variants (134–184 ms). Gen6V4 and Gen7 sit at
**0.3–19.3 s/replan** — 10× to 580× over budget. *(fig2.)*

Note this is **not** purely the K=20 → K=2 sampler change. Gen7-c3's unprojected `diffuser` is
334 ms against Gen14's 30 ms (11×, which is roughly the NFE ratio), but its `bounds_free` is
19 275 ms against 42 ms (460×). The projector, not the sampler, is where the old generations
lose most of the time — consistent with the super-linear projector cost measured inside Gen14
itself between K=2 and K=100.

---

## 6. What this does and does not settle

**Settled:**

- The Gen6V4 run in this batch is broken (§2). It cannot serve as a baseline until re-run.
- Gen7's constraint metrics are not recorded well enough to compare (§3).
- Gen7 and Gen6V4 score **zero** goal successes across 812 rollouts (§4.1).
- Gen14 K=2 is 10–21× cheaper per replan than either, and the gap is dominated by the projector
  (§5).

**Not settled — and not settleable from this batch:**

- Whether Gen14 mf/af is *better* than Gen7 on goal distance. The paired basis is 3 contexts,
  the noise floor at n=3 is ~±0.43 m, and the 30-context data sits on different splits.
- Whether Gen14 is better on constraints than Gen7. No Gen7 data (§3).
- Any Gen6V4 comparison at all until it is re-run.
- Generalization. Gen14 has **no test-split data** in this batch; Gen7-c4 and Gen6V4 have no
  train-split data beyond 3 contexts. The two generations were evaluated on opposite splits.

**What would make the comparison real** (in priority order):

1. **Run Gen14 mf/af on the test split, 30 contexts, same variant list.** This alone converts
   the Gen7-c4 test cells (11 variants × 30) into a like-for-like comparison and costs a
   fraction of the 08-05 run, because Gen14 K=2 is 20× faster.
2. **Re-run Gen6V4** and check the box-displacement rate before analysing anything else. The
   `unmoved_%` column in `da_20260806_gen14_vs_gen7_gen6v4.py` is a one-line sanity gate worth
   putting in DA_VA_v2 itself.
3. **Re-run Gen7-c3 on the current eval** so the constraint fields are written.
4. If a controlled Gen14-vs-Gen6V4 comparison is ever wanted, set
   `'mpc_batch_size': 1` in `plan_mix_visual_aligning_diffusion`
   (`config/aligning-d3il-visual.py:1048`) per §4.2 of the code study — otherwise the
   controllers differ regardless of the engine.

**No code was changed for this DA.**

---

## 7. Reproduction

```bash
# all tables and figures in this document (scratchpad venv with pandas/numpy/matplotlib)
python logs_in_develop/Gen14/U7/da_20260806_gen14_vs_gen7_gen6v4.py \
       temp/0608/batch_va2_20260806_204620/per_rollout_detail.csv \
       logs_in_develop/Gen14/U7/figs
```

The batch came from DA_VA_v2 on the cluster against `logs/aligning-d3il-visual/plans`; see
`temp/0608/18_19_25_run_da_batch_va_v2_24330.log` for the discovery/loading trace
(198 units, 0 failed).
