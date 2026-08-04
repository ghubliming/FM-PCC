# Insight — HardFlow activation threshold 0.1: the projection schedule is a U-curve, and 0.5 was on the wrong side of it

**Date:** 2026-08-03
**Subject:** `hardflow_new-*` activation threshold sweep at K=20
**New data:** `temp/0308/K20_thres0.1_mpc4_n2/` + `temp/0308/16_26_05_eval_fmv3_hardflow_job_24179.log` (job 24179, git `ccae7d4`)
**Baselines re-read from:** `temp/2026-08-02/batch_avoiding_combined_20260802_092307/candidates_multidimensional_raw.csv` (CAND_36/37/38/44)

> **This document retracts the central conclusion of §11.6 of**
> `logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md`.
> That section said lowering the threshold makes HardFlow slower. That was inferred from a
> **two-point** sweep, `0.0` and `0.5`, whose endpoints happen to straddle the optimum.
> With `0.1` filled in, the curve is U-shaped: **0.1 is 3× faster than 0.5 and 8.6× faster
> than 0.0**, and at 0.1 HardFlow beats DPCC on wall clock at identical quality.
> §11.1–11.5 and §11.7 stand. §11.6 and the §11.8 run queue are superseded by this file.

---

## 1. What was run

Single evaluation, seed 6, `n_trials=2`, all three halfspace envs (6 episodes per arm):

| field | value |
|---|---|
| generator | `models.diffusion.FlowMatchingODE`, `H8_..._a1.5_b1.0_aw10`, ckpt step 98000 |
| K (`flow_steps_v3`) | 20, matched across every arm (log l.41) |
| MPC batch | 4 (`mpc4`) |
| HF activation threshold | **0.1** (`act_thr=0.1` in every HF line) |
| arms | `diffuser`, `dpcc-c-tightened`, `hardflow_new-{r,c,t}[-tightened]` |
| code | `flow_matching_v3_hardflow` (untagged = current), git `ccae7d4` |

The comparison point is **CAND_44** — same generator, same seed, same K, same `mpc4`, same
untagged `flow_matching_v3_hardflow` folder, threshold **0.5**. This is a clean one-variable
contrast; the only difference between the two runs is the threshold.

Also on the ladder, from the older `flow_matching_v3_hardflow(Gen12_Bf_U5)` tree (same
generator/seed, single fused `hardflow_new` arm only):

- **CAND_37** — K20, thr **0.0**, mpc**1**
- **CAND_36** — K20, thr **0.5**, mpc**1**
- **CAND_38** — K20, thr **0.0**, mpc**4**

CAND_42 (`Gen12_bf_Fix6_wrong_batch_parallel`) is excluded — the folder name says the batch
parallelism was wrong.

---

## 2. What the threshold actually does, measured

`hardflow_projection.py:512` gates the NLP:

```python
active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)
```

so the number of projected ODE steps is

```
n_active(thr, K) = #{k in 0..K-1 : k >= (1-thr)*K}   , floored at 1 by the `or` clause
```

At K=20: **thr 0.0 → 1, thr 0.1 → 2, thr 0.5 → 10, thr 1.0 → 20.**

This is not a reading of the source — the new log *measures* it. NLP solves and NFE are
instrumented per arm, and both match the closed form exactly:

```
NLP solves  = n_active × batch × n_planning_calls
NFE         = (K + n_active) × batch × n_planning_calls
```

Check on `both-hard / hardflow_new-r-tightened`: 1136 solves, 12496 NFE.
1136 / (2×4) = 142 planning calls → 12496/142 = **88 = (20+2)×4**. Exact.
Every other HF line in the log closes the same way.

Two consequences worth stating separately:

- **The generator overhead of HF is tiny at low threshold.** NFE ratio vs unprojected is
  `(K+n_active)/K` = **1.10** at thr 0.1, vs 1.50 at thr 0.5. HF's "extra velocity pass"
  is a 10% network cost, not a 2× cost.
- **The NLP count is the whole story.** thr 0.5 → 40 solves/step; thr 0.1 → 8 solves/step.

---

## 3. Headline result

Seed 6, 6 episodes/arm, `q` = `n_success_and_constraints`, `s/step` and `ep_s` averaged over
the three envs (`ep_s` = mean over envs of `n_steps × avg_time`, i.e. planning wall clock per
episode):

| arm | q@0.1 | q@0.5 | s/step @0.1 | s/step @0.5 | speed-up | ep_s @0.1 | ep_s @0.5 |
|---|---|---|---|---|---|---|---|
| `diffuser` (no projection) | 0/6 | 0/6 | 0.171 | 0.176 | 1.03× | 11.1 | 11.4 |
| `dpcc-c-tightened` | **6/6** | **6/6** | 0.472 | 0.473 | 1.00× | 29.0 | 29.1 |
| `hardflow_new-t-tightened` | **6/6** | **6/6** | **0.353** | 1.105 | **3.13×** | **21.6** | 67.6 |
| `hardflow_new-r-tightened` | **6/6** | **6/6** | 0.349 | 1.070 | 3.07× | 24.4 | 73.0 |
| `hardflow_new-c-tightened` | **6/6** | **6/6** | 0.335 | 1.003 | 2.99× | 34.4 | 103.5 |
| `hardflow_new-r` | 2/6 | 5/6 | 0.347 | 1.041 | 3.00× | 16.5 | 70.5 |
| `hardflow_new-c` | 3/6 | 3/6 | 0.333 | 0.998 | 3.00× | 22.3 | 105.6 |
| `hardflow_new-t` | 1/6 | 4/6 | 0.355 | 1.073 | 3.02× | 17.3 | 67.8 |

The `diffuser` and `dpcc-c-tightened` rows are the control: they are the *same* arms run in
both jobs and they reproduce to within 1% (0.171 vs 0.176; 0.472 vs 0.473). So the 3× on the
HF rows is the threshold, not machine noise or a code-rev artefact.

**Three things fall out.**

**(a) On the tightened arms, dropping 0.5 → 0.1 is a free 3× speed-up.** All three tightened
arms are 6/6 at both thresholds — no quality is traded away. Zero violations, zero NLP
failures on `r`/`c`.

**(b) HardFlow now beats DPCC.** `hardflow_new-t-tightened` at thr 0.1 is **21.6 s/episode
against DPCC's 29.0** — 26% cheaper at equal 6/6 quality, and it also uses slightly fewer
control steps (61.2 vs 62.2 mean). At thr 0.5 the same arm cost 67.6 s, i.e. **2.3× worse
than DPCC**. This is a sign flip on the core HF-vs-DPCC verdict, produced entirely by the
threshold.

**(c) HF's cost is flat across envs; DPCC's is not.** Paired per-env episode cost:

| env | HF `t-tightened` @0.1 | DPCC `c-tightened` | ratio |
|---|---|---|---|
| top-right-hard | 21.0 s | 20.3 s | 1.04 |
| top-left-hard | 21.8 s | 29.7 s | 0.74 |
| both-hard | 21.9 s | 37.1 s | **0.59** |

HF pays a fixed, predictable price (21.0/21.8/21.9 — essentially constant), because it solves
a fixed number of well-posed endpoint problems. DPCC's in-place projection on a noisy iterate
gets steadily more expensive as the constraint set bites — 20.3 → 37.1 s from the easiest to
the hardest env. **HF's advantage grows with problem difficulty.** On the hardest env it is
already 1.7× cheaper than DPCC.

---

## 4. Why 0.1 beats both 0.5 and 0.0 — the per-solve cost model

Decompose measured `s/step` into generation + NLP, using the measured `diffuser` time
(0.171 s/step for K=20 at batch 4) scaled by the known NFE ratio:

```
t_step ≈ t_gen × (K + n_active)/K  +  n_solves_per_step × c_solve
n_solves_per_step = n_active × batch
```

Solving for `c_solve`:

| run | thr | mpc | n_active | solves/step | measured s/step | implied **c_solve** |
|---|---|---|---|---|---|---|
| **new (this job)** | 0.1 | 4 | 2 | 8 | 0.3487 | **20.0 ms** |
| CAND_44 | 0.5 | 4 | 10 | 40 | 1.0577 | **20.0 ms** |
| CAND_36 | 0.5 | 1 | 10 | 10 | 0.4876 | **23.1 ms** |
| CAND_37 | **0.0** | 1 | **1** | 1 | 0.7460 | **566 ms** |
| CAND_38 | **0.0** | 4 | **1** | 4 | 3.0081 | **707 ms** |

**The per-solve cost is 20 ms whenever `n_active ≥ 2`, and 566–707 ms — a 28–35× blow-up —
when `n_active = 1`.** The 20.0 / 20.0 agreement across a 5× difference in solve count is
about as clean as this kind of decomposition gets, and CAND_36/37 are the same code rev as
each other, so the cliff is not a rev artefact either.

**Mechanism.** `hardflow_projection.py:327` seeds IPOPT with the extrapolated endpoint itself:

```python
self.opti.set_initial(self.x1, x1_ref)
```

There is no cross-step solver warm start — but there is a *data* warm start. Once the k=18
solve has run, the state `X` carried into k=19 is already close to feasible, so `x1_ref` at
k=19 is a good initial guess and IPOPT converges in a handful of iterations. At `n_active=1`
the single solve fires at k=K−1 with `τ_next = 1.0`, meaning `X1_ref = X_ref` — the raw,
completely unprojected ODE endpoint, which for `both-hard` sits well inside the obstacle set.
IPOPT starts infeasible and far, and burns through restoration-phase iterations
(`solve_limited()`, l.330, default `max_iter`). One cold solve costs more than ten warm ones.

So the threshold trades two opposing terms:

```
cost(n_active) = [linear ↑ in n_active]  +  [conditioning penalty, explodes at n_active = 1]
```

The minimum is at **n_active = 2** — the smallest schedule that still gives the terminal solve
a warm predecessor. That is exactly `thr = 0.1` at K=20.

`thr = 1.0` (n_active = 20) is still untested but the model predicts ≈ 0.171×2 + 80×0.020 =
**1.94 s/step** — twice as bad as 0.5. There is no reason to run it except to confirm the
linear branch.

---

## 5. The part that is *not* free: untightened arms regress

The three untightened arms lose quality going 0.5 → 0.1:

| arm | q@0.5 | q@0.1 |
|---|---|---|
| `hardflow_new-r` | 5/6 | 2/6 |
| `hardflow_new-t` | 4/6 | 1/6 |
| `hardflow_new-c` | 3/6 | 3/6 |
| **pooled** | **12/18** | **6/18** |

Fisher exact on the pooled counts: **p = 0.094** — suggestive, not significant, on 18 vs 18
episodes from one seed.

But the *pattern* is more alarming than the count. On `top-right-hard`, **all three**
untightened arms report `Success rate: 0.0` — not a constraint trip, an outright failure to
reach the goal. (`Avg number of steps: 0.00` is an artefact: `eval_FM_v3_hardflow.py:482`
averages steps over successful episodes only. The NLP counters prove the episodes really ran
— e.g. `hardflow_new-r` logged 664 solves ≈ 83 planning calls ≈ 41 steps/episode.) At thr 0.5
those same three arms were all `Success rate: 1.0` on that env.

Reading: with only 2 projected steps and no tightening margin, the correction applied at
k=18/19 is large and late, and the resulting trajectory is feasible-but-bad — it satisfies the
halfspaces while failing to make progress toward the goal. The tightening margin absorbs this
(all tightened arms stay 6/6). **The safe conclusion is that thr 0.1 is only validated
together with tightening.**

Corroborating signal: the first **NLP failures** in this family appear at thr 0.1 —
`top-left-hard / hardflow_new-t`: **27 failures / 1152 solves (2.3%)**, and `t-tightened`:
4/968. Both on the `-t` (temporal-consistency) selection, both on `top-left-hard`, zero
elsewhere. `nlp_failures_total` is not populated for CAND_44 in the batch CSV, so this cannot
be compared against thr 0.5 — it needs a re-run to attribute.

---

## 6. What this means for the K=2 collapse

§11.5 of the Gen3v6 DA reported HardFlow collapsing at low K on this same generator and seed:
**K=2 → 2/6, K=5 → 3/6, against DPCC's 6/6**, reaching parity only at K=20. That was
attributed to Euler extrapolation error over a long `(1−τ)` interval.

The `n_active = 1` cliff gives a second, simpler, and more likely explanation. At thr 0.5:

| K | (1−thr)·K | active k | **n_active** |
|---|---|---|---|
| 2 | 1.0 | k=1 | **1** ← terminal-only, cold |
| 5 | 2.5 | k=3,4 | 2 |
| 10 | 5.0 | k=5..9 | 5 |
| 20 | 10.0 | k=10..19 | 10 |

**K=2 at any threshold lands in the pathological single-cold-solve regime**, because
`(1-thr)*K ≥ K-1` for every `thr ≤ 0.5` and the `or (k == K-1)` clause floors it at one. This
is the same failure mode as thr 0.0 at K=20 — and thr 0.0 at K=20 is where HF cost 3.0 s/step
and where the endpoint handed to IPOPT is fully unprojected.

This matters because **K=2 is the operating point the MF/AF line cares about** (CAND_32 and
CAND_102 both carry the full `hardflow_new-*` arm set at K=2). If the K=2 HF collapse is the
cold-solve cliff rather than extrapolation error, it is fixable with a scheduling floor rather
than a new endpoint estimator.

**Proposed change (not applied — needs the go-ahead):** floor the schedule at two active steps.

```python
# hardflow_projection.py, sample() — current
active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)
# proposed
n_min   = min(2, K)                     # never leave the terminal solve cold
active  = (k >= (1.0 - self.activation_threshold) * K) or (k >= K - n_min)
```

At K=20/thr0.1 this is a no-op (already 2). At K=2 it makes both steps active. At K=1 it is
unchanged. Cheap to test and it discriminates the two hypotheses directly: if K=2 HF jumps
from 2/6 toward parity, the cliff explains it and §11.7's MeanFlow-exact endpoint jump is a
refinement rather than a prerequisite; if it does not move, the extrapolation-error story
survives and §11.7 becomes the main lever.

---

## 7. Caveats

1. **One seed (6), two trials, 6 episodes per arm.** Everything here is single-seed. The speed
   results are near-deterministic instrument readings and I would defend them; the quality
   results (§5 especially) are not powered.
2. **thr 0.0 comes from a different code tree** (`Gen12_Bf_U5`, CAND_37/38). The within-rev
   mpc1 pair CAND_37 vs CAND_36 controls for this and shows the same cliff, so the conclusion
   holds, but a thr-0.0 run on `ccae7d4` would close the gap properly.
3. **Only `dpcc-c-tightened` was run as the DPCC arm** in this job. §9.8/§10.6 of the Gen3v6
   DA found `-c` pathological at K=2 specifically; at K=20 it is well-behaved (6/6, 0
   violations), so it is a fair baseline here — but `dpcc-r-tightened` / `dpcc-t-tightened`
   are missing and DPCC's best arm may be faster than 29.0 s/episode.
4. **`c_solve` is inferred, not measured.** The decomposition assumes NLP time is the only
   non-generation cost and that generation scales exactly with NFE. The 20.0/20.0 agreement
   across the two independent mpc4 points is strong support, but it is still a model.
5. The `-t` NLP failures (§5) are unexplained and have no thr-0.5 comparison point.

---

## 8. Run queue

Ordered by information per GPU-hour.

1. **`thr=0.1`, K=20, mpc4, seeds 7–10, n_trials≥5.** The single highest-value run in this
   file. Everything above rests on seed 6. Add `dpcc-r-tightened` and `dpcc-t-tightened` so
   the DPCC baseline is its best arm, and make sure `nlp_failures_total` is exported for the
   DPCC arms too.
2. **`thr ∈ {0.05, 0.15, 0.25}`, K=20, mpc4, seed 6.** Locate the U-minimum properly. thr 0.05
   and 0.15 both still give n_active = 2 at K=20 — if they reproduce 0.348 s/step exactly,
   that independently confirms n_active (not the threshold value) is the sole control
   variable, which is a cleaner statement for the paper than "0.1 is good".
3. **`thr=0.0`, K=20, mpc4, on `ccae7d4`/current code.** Closes caveat 2 and pins the cliff on
   the current tree.
4. **`min_active_steps=2` patch, K=2 and K=5, seed 6, HF on FlowMatchingODE.** The
   discriminating experiment for §6. Small run, large consequence for the MF/AF line.
5. **`thr=1.0`, K=20, mpc4, seed 6.** One run to confirm the linear branch (predicted ≈ 1.94
   s/step). Low priority — only worth it if the U-curve goes in the paper.
6. Once 1 and 4 land: **HF at thr 0.1 on the MF (CAND_102) and AF (CAND_32) K=2 generators**,
   which is the actual target configuration.

---

## 9. One-liners for the paper

- HardFlow's cost is set by the number of projected ODE steps, `n_active = ⌈thr·K⌉ ∨ 1`, at a
  near-constant 20 ms per NLP solve — provided `n_active ≥ 2`.
- A single terminal projection is 28–35× more expensive per solve than a warm one, because it
  hands IPOPT the fully unprojected ODE endpoint; the cost curve in `thr` is therefore
  U-shaped with a minimum at `n_active = 2`.
- At its optimal schedule (K=20, thr 0.1), HardFlow reaches 6/6 goal-and-constraint success at
  **21.6 s of planning per episode against DPCC's 29.0 s**, and its cost is independent of
  constraint difficulty where DPCC's grows 1.8× from the easiest to the hardest environment.

---

## 10. The Pareto question — at equal success+constraints, do we get fewer steps AND less time?

### 10.0 How these arms may and may not be compared

`-r`, `-c`, `-t` are **not three samples of a method.** They are three candidate-selection
rules (`scripts/eval.py:210-211`: `-t` = `temporal_consistency`, `-c` =
`minimum_projection_cost`, `-r` = default/random) that sit *downstream* of the projector, and
at deployment you pick exactly one. The same three rules exist in both families. Therefore:

- **Pooling across suffixes is invalid.** "HF 12/18 vs DPCC 8/18" is a meaningless statistic —
  it counts episodes from configurations you would never run simultaneously. *An earlier draft
  of §10.3 did exactly this; it has been removed.*
- **HF `-c` succeeding where DPCC `-r` fails says nothing about HF.** The honest response to
  that observation is "then use HF `-r`", not "HF is better".
- Only two comparisons carry information: **matched suffix** (HF `-r` vs DPCC `-r`, etc.),
  which isolates the projector, and **best-of-family vs best-of-family**, which is what you
  would actually deploy — stated as such, and discounted for the fact that picking the best of
  three on 6 episodes is itself a selection over noise.

Second framing point, which decides what this whole section can and cannot claim:

> **The tightened metric is saturated.** Every `-tightened` arm in both families scores 6/6.
> A saturated metric carries **zero quality information** — it cannot rank projectors, only
> costs. So §10.1 is a **cost** result at a quality ceiling, and the actual quality question
> can only be asked on the **untightened** arms, where the safety margin is removed and the
> projector has to do the work itself. That is §10.3, and it is where the real claim would
> have to come from.

All numbers: seed 6, K=20, mpc4, same `FlowMatchingODE` generator, 6 episodes per cell.
DPCC arms other than `c-tightened` come from **CAND_105** (same generator/seed/K), rescaled
×0.965 so its `c-tightened` matches this job's clock. `n_steps` is averaged over successful
episodes only (`eval_FM_v3_hardflow.py:482`) — full average for 6/6 arms, biased low otherwise.

---

### 10.1 Cost, at the saturated tightened ceiling — matched suffix

| suffix | | steps | episode time | q |
|---|---|---|---|---|
| **`-t-tightened`** | **HF @0.1** | **61.2** | **21.6 s** | 6/6 |
| | DPCC | 63.5 | 28.5 s | 6/6 |
| | | **HF wins both** | **−24%** | |
| **`-r-tightened`** | **HF @0.1** | **69.8** | **24.4 s** | **6/6** |
| | DPCC | 71.5 | 47.5 s | 5/6 |
| | | **HF wins both** | **−49%** | |
| **`-c-tightened`** | HF @0.1 | 102.7 | 34.4 s | 6/6 |
| | **DPCC** | **62.2** | **29.0 s** | 6/6 |
| | | **DPCC wins both** | **−16%** | |

**HF wins 2 of 3 suffixes and loses 1.** Globally the Pareto frontier over all eligible arms
is still the single point HF `t-tightened` @0.1 (61.2 steps, 21.6 s — no 6/6 arm anywhere in
the dataset has fewer of either). But the matched view is the one that means something, and it
says the advantage is **not uniform across selection rules**: under `-c`, DPCC is better on
both axes, by 40 steps and 16%.

Two caveats on the margins:

- The **step** margins on `-t` (−2.3) and `-r` (−1.7) are **inside the noise** — per-env step
  spreads in the log are ±2.5 to ±12.5 on 6 episodes. Read those two rows as *tie on steps,
  large win on time*. The `-c` step gap (−40) is far outside noise and is real.
- DPCC `-r-tightened`'s 47.5 s is inflated by one bad environment (top-left, 1.055 s/step
  against 0.36–0.61 elsewhere) which is also where it drops to 5/6. Treat the −49% as soft.

Reference floor: unprojected `diffuser` is 65.0 steps at 0/6. `-t` and `-c` arms in both
families come in *below* it — projection shortens the path as well as taxing it.

---

### 10.2 Does more time buy fewer steps? **No — it buys nothing**

Within each suffix (so this comparison is clean), thr 0.1 → 0.5, i.e. `n_active` 2 → 10:

| suffix | time | steps | q |
|---|---|---|---|
| `-t-tightened` | 21.6 → 67.6 s (**3.13×**) | 61.17 → 61.17 (**+0.0%**) | 6/6 → 6/6 |
| `-r-tightened` | 24.4 → 73.0 s (2.99×) | 69.83 → 68.17 (−2.4%) | 6/6 → 6/6 |
| `-c-tightened` | 34.4 → 103.5 s (3.01×) | 102.67 → 103.17 (+0.5%) | 6/6 → 6/6 |

**Tripling the projection budget moves the step count by −0.6% on average, sign inconsistent.**
`-t-tightened` reproduces to two decimals. There is no steps-for-time trade at the tightened
ceiling — past `n_active = 2` the extra 46 s/episode re-solves an already-solved problem.

---

### 10.3 The relaxed (untightened) problem — the only place a real claim could live

Remove the tightening margin and the projector has to enforce the constraints on its own. This
is the meaningful test, and the metric is no longer saturated. Matched suffix, `q` out of 6,
per-env breakdown `(top-right, top-left, both-hard)`:

| suffix | HF @ thr **0.5** | HF @ thr **0.1** | DPCC | matched verdict |
|---|---|---|---|---|
| `-r` | **5/6** (1.0, 1.0, 0.5) | 2/6 (0, 1.0, 0) | 2/6 (0, 1.0, 0) | HF@0.5 **+3**, HF@0.1 tie |
| `-t` | **4/6** (1.0, 0.5, 0.5) | 1/6 (0, 0, 0.5) | 2/6 (0, 0.5, 0.5) | HF@0.5 **+2**, HF@0.1 **−1** |
| `-c` | 3/6 (0.5, 1.0, 0) | 3/6 (0, 1.0, 0.5) | **4/6** (0.5, 1.0, 0.5) | **DPCC +1** at both thresholds |
| **best of family** | **5/6** (`-r`) | 3/6 (`-c`) | 4/6 (`-c`) | HF@0.5 **+1 episode** |

**What this does and does not support:**

- **No leap has happened.** Nothing solves the relaxed problem. The best untightened result
  anywhere in the dataset is **HF `-r` at thr 0.5, 5/6** — one episode short, and it costs
  70.5 s/episode against DPCC `-r`'s 29.6 s. The relaxed problem remains open.
- **Best-of-family is 5/6 vs 4/6 — a one-episode margin**, on 6 episodes, from one seed, after
  picking the best of three rules in each family. That is not evidence of anything. It is
  certainly not the 12/18-vs-8/18 the earlier draft claimed; that number was an artefact of
  pooling and is withdrawn.
- **Matched by suffix the sign is not consistent**: HF ahead on `-r` (+3) and `-t` (+2), behind
  on `-c` (−1). If a single number had to be quoted it would be the `-r` cell, since `-r` is
  the neutral rule (no selection heuristic), and there HF@0.5 is 5/6 against DPCC's 2/6 — the
  strongest cell in the table and the one worth trying to reproduce first.
- **Difficulty is concentrated in two envs.** Every untightened arm in every config scores 1.0
  on `top-left-hard`. All the discrimination is in `top-right-hard` and `both-hard`. Six
  episodes per cell is really two informative episodes per cell.

**What the extra time actually buys.** Compare within HF, matched suffix: `-r` goes 2/6 → 5/6
and `-t` goes 1/6 → 4/6 when the schedule goes from `n_active` 2 to 10. So the 3× time is *not*
wasted on the relaxed problem — it is only wasted at the tightened ceiling (§10.2). **The trade
is time-for-margin-free-robustness, and it is the only real trade in this dataset.** `-c` is
flat (3/6 → 3/6), consistent with `-c` being broken in a way projection effort does not fix.

**The configuration that follows.** `thr = 0.15` → `n_active = 3` at K=20. From the §4 model
(fit 18/18 HF cells within ±6%): **0.437 s/step, ≈26.6 s/episode — still under DPCC's 28.5 s.**
It is the last setting cheaper than DPCC (0.2 → 32.0 s), so it is the only candidate for buying
relaxed robustness back without giving up §10.1. Run it on `-r` first.

---

### 10.4 What sets the step count: the selection rule, not the projector

| selection rule | HF @0.1 | HF @0.5 | DPCC |
|---|---|---|---|
| `-t` `temporal_consistency` | **61.2** | **61.2** | 63.5 |
| `-r` `random` | 69.8 | 68.2 | 71.5 |
| `-c` `minimum_projection_cost` | **102.7** | **103.2** | **62.2** |

`-t` is cheapest in both families at both thresholds — staying consistent with the previous
plan avoids step-wasting course changes. `-r` costs ~8 more. **`-c` costs 40 more under
HardFlow but not under DPCC**, and it is the suffix where HF loses on both axes (§10.1) and on
relaxed quality (§10.3). It reaches 6/6 tightened, so this is not the K=2 `-c` collapse from
§10.6 of the Gen3v6 DA — but ranking candidates by projection cost on the *predicted clean
endpoint* evidently rewards trajectories that are cheap to project over ones that make
progress. **The `-c` pathology changed shape rather than disappearing: from failing to
dawdling.** `-c` should be dropped from the HF line unless it is fixed.

---

### 10.5 Why HF is faster despite a *more expensive* solve

Per unit of projection work HardFlow is strictly worse than DPCC, and the data confirms it:

- **+10% network** — `(K + n_active)/K` = 22/20, the extra velocity pass for the endpoint
  extrapolation.
- **A heavier NLP** — backing out the measured `diffuser` time (0.171 s/step): **HF 20.0 ms per
  solve, DPCC 3.6–11.6 ms.** HF's problem carries linear dynamics equalities and input
  saturation (`hardflow_projection.py:300-315`); DPCC projects onto halfspaces only.
- **Head to head at the same schedule** (`n_active` = 10): **HF 1.058 s/step vs DPCC 0.472 —
  2.24× slower.** Theory confirmed.

HF wins anyway only because it runs **8 solves/step against DPCC's 40**. That is the entire
margin. Two supporting facts:

**(a) HF's solve cost is constant; DPCC's is not.**

| | top-right | top-left | both-hard | spread |
|---|---|---|---|---|
| HF (all 6 arms) | 17.4–19.9 ms | 17.7–22.5 ms | 19.0–21.2 ms | **1.15×** |
| DPCC `c-tightened` | 3.55 ms | 7.43 ms | 11.57 ms | **3.26×** |

HF always poses the same problem — pull a predicted clean endpoint onto the feasible set from a
warm predecessor. DPCC poses a harder one as the constraints bind. Hence HF's flat episode cost
(21.0 / 21.8 / 21.9 s) vs DPCC's climb (20.3 / 29.7 / 37.1 s), and **DPCC wins the easiest
environment by 4%** — the §10.1 `-t` win is carried by the two harder envs.

**(b) The cost model holds for HF and fails for DPCC.** `t = t_gen·(K+n)/K + n·batch·c_solve`
with one constant reproduces **all 18 HF cells within −6.1% … +5.7%** (mean |err| 2.8%) and
misses DPCC by **−33.5% … +34.6%**. Constant per-solve cost is a property of HardFlow, not of
constrained projection in general — HF's latency is predictable, DPCC's is not.

---

### 10.6 What would overturn §10.1

**DPCC was never given the short schedule.** The `thres` flag gates HardFlow only — proof:
`dpcc-c-tightened` is unchanged to within 0.4% across the thr-0.5 and thr-0.1 runs (0.636 →
0.634, 0.470 → 0.468, 0.314 → 0.314 s/step) with identical step counts (58.5 / 63.5 / 64.5).
So DPCC ran at `n_active = 10` in both — the schedule HF was just shown to be 3× too slow at.
At `n_active = 2` DPCC would do 8 solves/step at 3.6–11.6 ms → **0.20–0.26 s/step, ≈13–16 s per
episode**, which beats HF's 21.6 and flips every row of §10.1. Whether DPCC tolerates it is
unknown; it corrects the noisy iterate in place and plausibly needs the passes, whereas HF
corrects a predicted clean endpoint. That is an argument, not data.

**Until that run exists, §10.1 is HF tuned against DPCC untuned.**

Secondary: **DPCC's 40 solves/step is inferred** — `nlp_solves_total = 0` for every DPCC arm,
the counters live only in `HardFlowSampler`. If DPCC solves once per projected step for the
whole batch, `c_dpcc` becomes 14–46 ms and §10.5's per-solve story inverts. §10.1–§10.3 use
only measured wall clock, steps and success, so they stand either way.

Third: **seed 6 only, 2 trials, 6 episodes per cell**, and on the untightened arms effectively
2 informative episodes per cell (§10.3). Timing figures are near-deterministic instrument
readings; step and quality figures are not powered.

---

### 10.7 Verdict

| question | answer |
|---|---|
| Tightened, matched suffix: fewer steps than DPCC? | **Tie on `-t`/`-r`** (1–2 steps, inside noise); **lose by 40 steps on `-c`**. |
| …and less time? | **Yes on `-t` (−24%) and `-r` (−49%, soft); no on `-c` (+16%).** 2 of 3 suffixes. |
| Is the tightened result a quality claim? | **No.** Everything is 6/6 — the metric is saturated and ranks nothing. It is a cost result only. |
| Relaxed (untightened): have we made the leap? | **No.** Best anywhere is HF `-r` @0.5 at **5/6**, one episode short, at 2.4× DPCC `-r`'s time. |
| Relaxed: do we beat DPCC? | **Unresolved.** Best-of-family 5/6 vs 4/6 = one episode. Matched suffix: HF +3 on `-r`, +2 on `-t`, −1 on `-c`. Nothing significant. |
| Does more time buy fewer steps? | **No** — 3× time, +0.0% steps at the tightened ceiling. |
| Does more time buy anything? | **Yes, but only on the relaxed problem**: `-r` 2/6 → 5/6, `-t` 1/6 → 4/6 going `n_active` 2 → 10. |
| Is HF intrinsically slower, as theory says? | **Yes** — 2.24× at matched schedule, 20.0 vs 3.6–11.6 ms per solve. It wins on schedule, never on solve. |
| Is the comparison fair? | **Not yet** — DPCC never ran at `n_active = 2`. |

---

### 10.8 Added to the run queue (§8), at the top

0. **`dpcc_threshold ∈ {0.1, 0.2}`, K=20, mpc4, seed 6, all six DPCC arms**, alongside HF at
   thr 0.1. The only thing between §10.1 and a publishable cost claim. Instrument
   `nlp_solves_total` / `nlp_failures_total` on the DPCC path in the same job.
0b. **`thr = 0.15` (`n_active = 3`), untightened arms, `-r` first, seeds 6–10, n_trials ≥ 5.**
   The relaxed problem is where the real claim lives and it is currently decided by ~2
   informative episodes per cell. This run does double duty: locates the robustness/time knee
   *and* gives the untightened comparison enough episodes to mean something.
0c. Drop `-c` from the HF sweep set until §10.4 is understood — it loses on every axis.

---
---

# Part II — the DPCC short-schedule run (added 2026-08-04)

> **This part resolves §10.6 and overturns §10.1.** §10.8 item 0 asked for DPCC on the short
> schedule, and called it "the only thing between §10.1 and a publishable cost claim". That run
> now exists. DPCC **tolerates the short schedule completely** (6/6 on all three tightened arms
> down to a single projected step) and at matched `n_active` it is **1.8× faster than HardFlow,
> winning all three suffixes on time and tying or winning on steps**. §10.1's "HF wins 2 of 3"
> is withdrawn. §3(c) and §10.5(a) ("HF's cost is flat, DPCC's is not") are also withdrawn —
> DPCC's cost is flat too, once it is on the schedule HF was being run at.
> §4, §5, §10.2, §10.3 and §10.4 stand and are extended below.

**New data:** `temp/0408/FMv3ODE/` — four jobs, all git `1b3c080`, node i6-gpu-1:

| job | savepath tag | K | `diffusion_timestep_threshold` |
|---|---|---|---|
| 24210 | `H8_K20_Meuler_T0.1_D…FlowMatchingODE` | 20 | 0.1 |
| 24207 | `H8_K20_Meuler_T0.05_D…FlowMatchingODE` | 20 | 0.05 |
| 24196 | `H8_K10_Meuler_T0.1_D…FlowMatchingODE` | 10 | 0.1 |
| 24198 | `H8_K10_Meuler_T0.05_D…FlowMatchingODE` | 10 | 0.05 |

---

## 11. What was run, and why it is comparable

Evaluation script is the **FMv3ODE sibling** (`eval_flow_matching_v3_ode_selectable.py`), not
the Gen12 HardFlow script. That is the point: the FMv3ODE path reads
`diffusion_timestep_threshold` from `config/projection_eval.yaml` and feeds it to `Projector`,
so the DPCC schedule is actually settable there (this is the wiring Gen12's port dropped — see
`logs_in_develop/Gen12/fix_8/`). fix_8 is therefore **irrelevant to these four runs**: it
changes the Gen12 HF gate and the Gen12 DPCC threshold plumbing, neither of which is on this
code path.

Everything else is matched to Part I: generator `models.diffusion.FlowMatchingODE`,
`H8_…_a1.5_b1.0_aw10`, **ckpt step 98000**, **seed 6**, `n_trials=2`, the same three halfspace
envs, and **`batch_size: 4`** (`config/avoiding-d3il.py:1053`, `plan_fm_v3_ode_selectable`) —
the same B=4 as the HF job's `mpc4`. Full arm set: all six `dpcc-*`, `diffuser`, `gradient`,
`post_processing`, `model_free`.

**Cross-script control.** The `diffuser` arm is the same computation in both scripts and it
reproduces across them:

| | HF job (Part I) | FMv3ODE K=20 job | Δ |
|---|---|---|---|
| `diffuser` s/step | 0.171 / 0.176 | 0.174 / 0.170 | **< 2.5%** |

So wall clocks from Part I and Part II are directly comparable.

### 11.1 The gate arithmetic — and a free confirmation of the §8-item-2 hypothesis

`flow_matcher_v3_ode_selectable/models/diffusion.py:207-208`:

```python
snapping_start_idx = int((1.0 - projector.diffusion_timestep_threshold) * self.flow_steps_v3)
near_end = (loop_idx >= snapping_start_idx) or (loop_idx == self.flow_steps_v3 - 1)
```

| K | T | `int((1−T)·K)` | **n_active** |
|---|---|---|---|
| 20 | 0.5 | 10 | 10 (Part I) |
| 20 | **0.1** | 18 | **2** |
| 20 | **0.05** | 19 | **1** |
| 10 | **0.1** | 9 | **1** |
| 10 | **0.05** | 9 | **1** |

Note the last two rows collapse to the same schedule. And they produce **bit-identical
results**: every `n_steps`, `n_success_and_constraints`, `n_violations` and `total_violations`
array agrees exactly across all 7 arms × 3 envs of the K=10 T0.1 and T0.05 runs (21/21 cells,
checked directly in the `.npz`). At K=20 the same pair of thresholds gives `n_active` 2 vs 1
and **17/21 cells differ**.

> **§8 item 2 is answered without running it: `n_active` — not the threshold value — is the
> sole control variable.** Two different thresholds that floor to the same schedule are the
> same run, to the last bit. This is also an independent confirmation that the gate is
> integer-floored (the premise of fix_8), measured rather than read off the source.

---

## 12. Headline — DPCC on the short schedule

Seed 6, 6 episodes/arm, `s/step` and `ep_s` averaged over the three envs. `n=10` column is
Part I (§3, §10.1); `n=2` and `n=1` are new.

| arm | q @ n=10 | q @ n=2 | q @ n=1 | s/step n=10 | s/step **n=2** | s/step **n=1** |
|---|---|---|---|---|---|---|
| `diffuser` (K=20) | 0/6 | 0/6 | 0/6 | 0.176 | 0.174 | 0.170 |
| `dpcc-r-tightened` | 5/6 | **6/6** | **6/6** | 0.664 † | **0.188** | **0.178** |
| `dpcc-c-tightened` | **6/6** | **6/6** | **6/6** | 0.473 | **0.189** | **0.177** |
| `dpcc-t-tightened` | 6/6 | **6/6** | **6/6** | 0.449 | **0.189** | **0.180** |

† `n=10` values for `-r`/`-t` are `ep_s / steps` from §10.1 (CAND_105, rescaled ×0.965); only
`-c-tightened` was run natively in the Part I job. `-r-tightened`'s 0.664 is inflated by one bad
env (§10.1) — the same env where it drops to 5/6. `-c-tightened`'s 0.473 is the clean number.

**DPCC tolerates the short schedule completely.** All three tightened arms hold 6/6 at
`n_active = 2` *and* at `n_active = 1`, with zero violations. §10.6's open question — "whether
DPCC tolerates it is unknown; it corrects the noisy iterate in place and plausibly needs the
passes" — is answered: **it does not need the passes.**

**§10.6's prediction was conservative.** It predicted 0.20–0.26 s/step and 13–16 s/episode at
`n_active = 2`. Measured: **0.189 s/step, 11.8–13.0 s/episode.**

### 12.1 Head to head with HardFlow at matched `n_active = 2` — every §10.1 row flips

All 6/6, so this is a pure cost comparison at a saturated quality ceiling (§10's framing rule
still applies: the tightened metric ranks costs, not projectors).

| suffix | | steps | episode time | s/step |
|---|---|---|---|---|
| **`-t-tightened`** | HF @0.1 | 61.2 | 21.6 s | 0.353 |
| | **DPCC @0.1** | 62.3 | **11.8 s** | **0.189** |
| | | tie (+1.1, noise) | **DPCC −45%** | **1.87×** |
| **`-r-tightened`** | HF @0.1 | 69.8 | 24.4 s | 0.349 |
| | **DPCC @0.1** | **69.0** | **13.0 s** | **0.188** |
| | | **DPCC** −0.8 | **DPCC −47%** | **1.86×** |
| **`-c-tightened`** | HF @0.1 | 102.7 | 34.4 s | 0.335 |
| | **DPCC @0.1** | **67.0** | **12.6 s** | **0.189** |
| | | **DPCC −35.7** | **DPCC −63%** | **1.77×** |

**DPCC wins 3 of 3 suffixes on time and 2 of 3 on steps (third is a tie inside noise).**
§10.1 reported the mirror image of this table. The difference is entirely that DPCC was
previously pinned at `n_active = 10` while HF ran at 2 — exactly the unfairness §10.6 flagged.

> **§10.1 was HF tuned against DPCC untuned. Tuned against tuned, DPCC wins.**

### 12.2 The intrinsic penalty is unchanged — and larger than §10.5 measured

Backing the NLP term out of each measured `s/step` (`t − t_gen`, ÷ `n_active` ÷ B=4), using
each run's **own** `diffuser` time and full `.npz` precision:

| schedule | DPCC ms/solve | HF ms/solve | **ratio** |
|---|---|---|---|
| `n_active = 10` (θ=0.5) | 3.57 / 7.47 / 11.62 (env) | 20.0 | 1.7–5.6× |
| **`n_active = 2`** | **1.68–2.16** | 20.0 | **≈ 10×** |
| **`n_active = 1`, K=20** | **1.50–3.29** | — | — |
| **`n_active = 1`, K=10** | **1.61–2.48** | — | — |

At the schedule where both methods are cheapest, **HardFlow's NLP costs ~10× DPCC's per
solve**, not the 1.7–5.6× §10.5 reported. HF's problem carries linear dynamics equalities and
input saturation; DPCC's late-schedule problem is a nearly-satisfied halfspace projection.
§10.5's closing line — "it wins on schedule, never on solve" — survives, but **the schedule
advantage was never real**: both methods run the same 8 solves/step at `n_active = 2`, so
there is nothing left for HF to win on.

### 12.3 The generator constant is portable

| run | K | `diffuser` s/step | implied **a** (ms per batched net call) |
|---|---|---|---|
| FMv3ODE K=20 T0.1 | 20 | 0.17370 | **8.69** |
| FMv3ODE K=20 T0.05 | 20 | 0.17005 | **8.50** |
| FMv3ODE K=10 T0.1 | 10 | 0.08617 | **8.62** |
| Part I HF job | 20 | 0.171 | 8.55 |

`a = 8.6 ± 0.1 ms` across a 2× change in K and across both eval scripts. Generation is exactly
linear in K. This is the one constant in the whole analysis that is directly measured and
genuinely portable (cf. Test_NFE §2c, which found the same for `a` and the opposite for the
solver constants).

---

## 13. NFE-eq closes on all four schedules

Using the Test_NFE vocabulary (`logs_in_develop/Gen12/Test_NFE/PLAN_hardflow_vs_dpcc_equal_cost_test.md` §5):
`NFE-eq = N_batched_net_calls + (b/a)·NPE`, with `a = 8.6 ms`, `NPE = n_active × B`, and
`N_net = K` for DPCC (post-hoc, no extra velocity pass) vs `K + n_active` for HF.

| config | net calls | NPE | b (ms/solve) | **NFE-eq** | **predicted s/step** | **measured** |
|---|---|---|---|---|---|---|
| DPCC K20 n=10 | 20 | 40 | 7.43 | 54.6 | 0.469 | **0.473** |
| HF K20 n=10 | 30 | 40 | 20.0 | 123.0 | 1.058 | **1.058** |
| **DPCC K20 n=2** | 20 | 8 | 1.80 | **21.7** | **0.186** | **0.189** |
| **HF K20 n=2** | 22 | 8 | 20.0 | **40.6** | **0.349** | **0.348** |
| **DPCC K20 n=1** | 20 | 4 | 1.75 | **20.8** | **0.179** | **0.177** |
| **DPCC K10 n=1** | 10 | 4 | 1.75 | **10.8** | **0.093** | **0.093** |

All six within 1.6%. Two things follow.

- **The additive `a·N_net + b·NPE` model is right**; what is *not* right is treating `b` as a
  single number. Given the correct `b` for the schedule, the model is essentially exact.
- **The HF/DPCC ratio in NFE-eq units** is 2.25× at `n_active = 10` and **1.87× at
  `n_active = 2`** — it does not cancel, because HF's extra velocity pass and its 10× solve
  both scale differently from DPCC's.

---

## 14. Why `b_scipy` is not a constant — it is a decreasing function of τ

This is the DPCC counterpart of §4's HardFlow U-curve, and it runs the **opposite** way.

Per-projected-step NLP cost, `both-hard`, `dpcc-c-tightened`, K=20:

| schedule | projected steps k | mean ms per solve |
|---|---|---|
| `n_active = 10` | k = 10…19 (τ = 0.50…0.95) | **11.62** |
| `n_active = 2` | k = 18, 19 (τ = 0.90, 0.95) | **1.79** |
| `n_active = 1` | k = 19 (τ = 0.95) | **1.72** |

The two terminal solves cost 1.7–1.8 ms each. The ten-step schedule averages 11.6 ms. So the
**marginal** cost of the eight early solves (k = 10…17) is `(10×11.62 − 2×1.79)/8 =` **14.1 ms
each — 8× the terminal ones.**

**Mechanism.** DPCC projects the *iterate in place*. At k=10 the iterate is halfway through
the ODE and sits well outside the feasible set, so SLSQP takes a large step through many
active-set changes. By k=19 the trajectory is nearly converged *and* already feasible from the
preceding solves, so the projection is almost a no-op. Cost is monotone decreasing in τ, and
the short schedule keeps only the cheap tail.

**This reconciles three previously discordant results:**

1. **§10.5(b)** — "the cost model reproduces all 18 HF cells within ±6% and misses DPCC by
   ±34%." It missed DPCC because it assumed a constant `b_scipy` over a schedule where the
   per-solve cost varies 8×. With the schedule-correct `b`, §13 closes DPCC to 1.6%.
2. **Test_NFE §2c** — "`b_scipy` is portable (7.4 → 7.8 → 8.3 ms), `b_ipopt` is not." Those
   three points were all at `θ_eff = 0.5`, i.e. schedules that all *start* at τ = 0.5. `b_scipy`
   was portable across K at fixed τ-range, not across τ-range. **The unified statement is that
   both solvers' costs depend on how far the seed is from feasible** — for DPCC via the
   iterate's τ, for HF via whether a predecessor solve has already run (§4). The two families
   differ only in which direction the schedule moves them.
3. **§4's K=2 puzzle.** At K=2 the single DPCC solve fires at k=1, τ=0.5 — the *expensive* end
   of the curve. The Test_NFE K=2 measurement of 7.8–8.3 ms/solve is therefore not in conflict
   with the 1.7 ms measured here at K=20, τ=0.95; they are two points on the same decreasing
   curve. *(Caveat: those K=2 points are MeanFlow/AlphaFlow checkpoints, so iterate quality is
   a confound. This remains the reason to run the same-checkpoint K sweep.)*

---

## 15. `-c` step inflation is a function of projection budget, in both families

§10.4 found `-c` (`minimum_projection_cost`) costs +40 steps under HardFlow but not under
DPCC, and concluded the pathology was HF-specific. With the schedule swept, it is not:

| config | `n_active` | `dpcc-c-tightened` mean steps |
|---|---|---|
| K=20, θ=0.5 | 10 | 62.2 |
| K=20, T=0.1 | 2 | 67.0 |
| K=20, T=0.05 | 1 | **94.2** |
| K=10, T≤0.1 | 1 | **117.8** |

A clean monotone ladder: **62 → 67 → 94 → 118 steps as the projection budget falls.** `-t` and
`-r` are flat over the same sweep (62.3/63.5/64.8 and 69.0/71.0/70.3).

Reading: ranking candidates by projection cost rewards trajectories that are cheap to project,
which correlates with not making progress. The bias is always present; a long projection
schedule *masks* it by correcting the dawdling trajectory anyway. Shorten the schedule and it
surfaces — under DPCC exactly as it does under HF. **§10.4's "the `-c` pathology is HF-specific"
is withdrawn; it is a selection-rule pathology that any short schedule exposes.**
§10.8 item 0c (drop `-c`) should be applied to **both** families.

---

## 16. The relaxed (untightened) problem — the trade is symmetric

Best-of-family `q` out of 6, untightened arms only (the only non-saturated metric):

| `n_active` | DPCC best | HF best |
|---|---|---|
| 10 | 4/6 (`-c`) | **5/6** (`-r`) |
| **2** | **3/6** (`-c`) | 3/6 (`-c`) |
| **1** (K=20) | 2/6 (`-r`/`-c`) | — |
| **1** (K=10) | 2/6 | — |

Per suffix at K=20:

| suffix | DPCC n=10 | **DPCC n=2** | **DPCC n=1** | HF n=10 | HF n=2 |
|---|---|---|---|---|---|
| `-r` | 2/6 | 2/6 | 2/6 | **5/6** | 2/6 |
| `-t` | 2/6 | 2/6 | **0/6** | 4/6 | 1/6 |
| `-c` | **4/6** | 3/6 | 2/6 | 3/6 | 3/6 |

**§10.3's conclusion holds and generalises.** "The trade is time-for-margin-free-robustness"
is not a HardFlow property — DPCC degrades the same way, 4/6 → 3/6 → 2/6 as the schedule
shortens. At matched `n_active = 2` the two families **tie at 3/6**.

So the ranking on the relaxed problem is unchanged from §10.3: the single best untightened
result anywhere is still **HF `-r` at θ=0.5, 5/6** — and it still costs 70.5 s/episode against
DPCC `-r` @n=10's 29.6 s and DPCC `-r` @n=2's ~13 s at 2/6. **Nothing solves the relaxed
problem, and the one point that comes closest is by far the most expensive in the dataset.**
That is the honest state of play, and it is unchanged by Part II.

---

## 17. The cheapest 6/6 in the dataset is K=10, `n_active = 1`

| config | arm | q | steps | **ep_s** |
|---|---|---|---|---|
| K=10, T≤0.1 | `dpcc-t-tightened` | **6/6** | 64.8 | **6.1 s** |
| K=10, T≤0.1 | `dpcc-r-tightened` | **6/6** | 70.3 | **6.6 s** |
| K=20, T=0.05 | `dpcc-t-tightened` | **6/6** | 63.5 | 11.4 s |
| K=20, T=0.1 | `dpcc-t-tightened` | **6/6** | 62.3 | 11.8 s |
| K=20, θ=0.5 | `dpcc-c-tightened` | 6/6 | 62.2 | 29.0 s |
| K=20, θ=0.1 | **HF** `-t-tightened` | 6/6 | 61.2 | 21.6 s |

**DPCC at K=10 with a single projected step reaches 6/6 at 6.1 s/episode — 3.5× cheaper than
HardFlow's best (21.6 s) and 4.8× cheaper than the K=20 θ=0.5 DPCC baseline (29.0 s).** The
projection itself costs 0.6 s of that 6.1 s; the `diffuser` floor at K=10 is 5.5 s at 1/6.

Step counts across all six rows span 61.2–70.3 — a 15% band, with the K=10 rows at the top of
it. So the K reduction is close to free on this task at this quality ceiling. Two caveats:

- **The tightened metric is saturated** (§10's framing rule). This is a cost result. It says
  K=10/n=1 is *not worse* on the measured axes, not that it is equally good.
- **`n_active = 1` at K=10 is one solve from τ=0.9.** §14 says that is a cheap, well-conditioned
  solve; §4 says HardFlow's `n_active = 1` is a catastrophically expensive one. The two
  families' cliffs are at opposite ends, which is worth stating explicitly because it is the
  single most counter-intuitive fact in this document.

---

## 18. Updated verdict (supersedes §10.7)

| question | Part I answer | **Part II answer** |
|---|---|---|
| Tightened, matched suffix: does HF use fewer steps than DPCC? | tie on `-t`/`-r`, −40 on `-c` | **No.** At matched `n_active=2`: tie on `-t`, DPCC −0.8 on `-r`, **DPCC −35.7 on `-c`**. |
| …and less time? | HF −24%/−49%, +16% | **No — DPCC wins all three: −45% / −47% / −63%.** |
| Does DPCC tolerate the short schedule? | unknown | **Yes, completely** — 6/6 on all three tightened arms at `n_active` = 2 **and** 1. |
| Is HF intrinsically slower at matched schedule? | yes, 2.24× | **Yes, and worse than measured: 1.86× wall clock, ≈10× per solve.** |
| Is the comparison fair? | **no** | **Yes now.** Matched K, B, seed, generator, checkpoint, `n_active`; `diffuser` control agrees to 2.5% across scripts. |
| Is `n_active` or the threshold the control variable? | assumed `n_active` | **`n_active`, proven** — two thresholds with the same floor give bit-identical results (21/21 cells). |
| Is `b_scipy` a constant? | assumed yes | **No** — 1.7 ms at τ=0.95, ~14 ms at τ=0.5. Decreasing in τ. |
| Is the `-c` pathology HF-specific? | claimed yes | **No** — DPCC `-c` inflates 62→118 steps as the budget falls. |
| Relaxed: have we made the leap? | no | **Still no.** Best anywhere is HF `-r` @θ=0.5 at 5/6, and it is the most expensive point in the dataset. |
| Relaxed: does HF beat DPCC? | unresolved (5/6 vs 4/6) | **Still unresolved, and now a tie at matched schedule** (3/6 vs 3/6 at `n_active`=2). |
| **Should the paper's cost claim be HF-favourable?** | §10.1 said yes | **No. Withdraw it.** |

---

## 19. Updated run queue (supersedes §8 and §10.8)

1. **Seeds 7–10, `n_trials ≥ 5`, K=20, both families at `n_active = 2`, all six arms in each.**
   Everything in Parts I and II is seed 6. §12.1 is the headline result of this whole
   investigation and it currently rests on 6 episodes per cell. Highest value by a wide margin.
2. **K sweep on the *same* FMv3ODE checkpoint: K ∈ {2, 5, 10, 20} at `n_active = 1`.** Settles
   §14's τ-dependence of `b_scipy` free of the MeanFlow/AlphaFlow confound, and tests §17 —
   whether the K=10/n=1 result keeps going down or falls off a cliff. Cheap: `n_active=1` runs
   are the fastest in the dataset.
3. **HF at `n_active = 1`, K=20, on the current tree** (§8 item 3, still open). Closes the U-curve
   with the same code rev as everything else and confirms §17's "opposite cliffs" claim
   directly.
4. **Untightened arms, seeds 6–10, `n_trials ≥ 5`, both families at `n_active` ∈ {2, 10}.**
   §16 is decided by ~2 informative episodes per cell. This is where the only real quality
   claim could live and it is currently unpowered in both directions.
5. **Drop `-c` from both sweep sets** until §15 is understood (revises §10.8 item 0c, which
   only dropped it from HF).
6. Once 1 and 2 land: **the MF/AF K=2 generators (CAND_32/CAND_102) at `n_active` = 1 and 2**,
   which is the actual target configuration for the Gen3v6/v7 line.

**Dropped from the old queue:** §8 item 2 (threshold sweep to locate the U-minimum) — answered
by §11.1, `n_active` is the control variable and the thresholds that matter are only those that
change it. §10.8 item 0 — done, this is it.

---

## 20. Caveats specific to Part II

1. **Seed 6, 2 trials, 6 episodes per cell** — same as Part I. Timing figures are
   near-deterministic instrument readings and I would defend them to 2%; step and quality
   figures are not powered.
2. **HF numbers are re-used from Part I**, i.e. a different job on a different day, on the same
   node and code rev. The `diffuser` control (§11) makes the wall clocks comparable to ~2.5%,
   which is far below the 1.8× effect, but it is not a same-job comparison.
3. **`nlp_solves_total` is still not instrumented on the DPCC path** — the counters live in
   `HardFlowSampler` only. Every DPCC per-solve number in §12.2 and §14 is inferred as
   `(t − t_gen)/(n_active × B)` with **B = 4 assumed** from `config/avoiding-d3il.py:1053`. If
   DPCC batches the four candidates into one SLSQP call, every "per solve" figure quadruples
   and §12.2's 10× ratio becomes 2.5×. **§12.1, §12.3, §13 and §17 use only measured wall
   clock, steps and success, and stand either way.** Instrumenting this is cheap and would
   remove the last inferred quantity from the analysis.
4. **`n_active = 1` at K=20 (T=0.05) is only one solve from τ=0.95** and still holds 6/6. That
   is a strong result but it is also the configuration closest to "no projection at all", and
   the untightened arms do degrade there (`-t` drops to 0/6, §16). Do not read §17 as "one
   solve is always enough".
5. **K=10 and K=20 are different generators in effect** — same weights, different ODE
   discretisation. §17 compares them at a saturated quality metric, which is the weakest kind
   of "no worse".

---
---

# Part III — the DPCC-baseline (diffusion generator) run (added 2026-08-04)

> **Read this first: the run does not measure what it was submitted to measure.**
> `scripts/eval.py` never passes `diffusion_timestep_threshold` to `Projector`, so both jobs ran
> the constructor default **θ = 0.5** regardless of the YAML. The two savepath tags `T0.1` and
> `T0.05` are cosmetic. The runs are bit-identical to each other (39/39 cells) — a duplicate,
> not a sweep.
>
> It is still worth three things, and they are what §21–§25 report:
> **(a)** a free end-to-end determinism control that puts an error bar under every number in
> Parts I and II; **(b)** the first **baseline-generator** (GaussianDiffusion) numbers on this
> benchmark, which show the *projector* cost is generator-agnostic while the *generator* is not;
> **(c)** two config-vs-code defects — the orphaned threshold and an unimplemented
> `post_processing` variant — both **inherited from upstream DPCC**, both still live, and
> neither touched by fix_8. Call the repair **fix_9**.

**New data:** `temp/0408/dpcc/`, node i6-gpu-1, seed 6, `n_trials = 2`, three halfspace envs,
13 arms:

| job | savepath tag | git rev | wall |
|---|---|---|---|
| 24215 | `H8_K20_T0.1_Dmodels.GaussianDiffusion` | `1b3c080` | 10:07 → 10:46 UTC |
| 24226 | `H8_K20_T0.05_Dmodels.GaussianDiffusion` | `3e84451` | 13:29 → 14:08 UTC |

Both loaded `logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10/6`,
**checkpoint step 91000**, `n_diffusion_steps = 20`, `batch_size: 4` (`plan` block,
`config/avoiding-d3il.py`). The two commits differ only by Gen14 U6 (`--flow-steps` for the
mf/af visual eval), which is not on this code path. The config snapshots are byte-identical
except the one YAML line:

```
30c30
< diffusion_timestep_threshold: 0.1
---
> diffusion_timestep_threshold: 0.05
```

---

## 21. The two defects

### 21.1 `diffusion_timestep_threshold` is orphaned in `scripts/eval.py`

The gate reads the value **off the projector**, not off `args`:

```python
# diffuser/models/diffusion.py:179, 186
if projector is not None and projector.gradient       and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
if projector is not None and not projector.gradient   and t <= projector.diffusion_timestep_threshold * self.n_timesteps:
```

and the projector is built without it:

```python
# scripts/eval.py:205-206
projector = Projector(horizon=args.horizon, ..., variant=diffuser_variant, dt=delta_t,
                      cost_dims=None, device=args.device, solver='scipy')
#                     ^ no diffusion_timestep_threshold -> falls back to the constructor
#                       default 0.5 (diffuser/sampling/projection.py:8)
```

Meanwhile `config/avoiding-d3il.py` **does** put the YAML value into `args`
(`'diffusion_timestep_threshold': _yaml_threshold`) and **does** watch it in `exp_name`
(`:831`), which is why the folder is named after a number the sampler never saw.

> **The savepath tag is not evidence of what ran.** `H8_K20_T0.05_…` here means θ = 0.5.

**Evidence, not inference.** All 39 arm × env cells of the two jobs agree exactly on
`n_success`, `n_success_and_constraints`, `n_steps`, `n_violations`, `total_violations` and
`collision_free_completed`. A 3× change in projection budget (see §21.2) cannot leave a
stochastic 200-step MPC rollout bit-identical.

**Provenance.** `aux_repo/dpcc/scripts/eval.py:151-152` has the identical omission, and
upstream's `config/projection_eval.yaml:26` pins `diffusion_timestep_threshold: 0.5` — exactly
the constructor default, so the bug is invisible upstream. It became reachable only when
FM-PCC started varying the YAML. **This is not a Gen12 regression and fix_8 does not touch it**
(fix_8 repaired the Gen12 HardFlow eval script; this is the Gen0 baseline script).

**Blast radius.** Every FM-PCC run through `scripts/eval.py` with a YAML threshold ≠ 0.5 is
mislabeled and actually ran at 0.5. The FMv3ODE and Gen12-post-fix_8 paths are unaffected —
they pass the value explicitly (`eval_flow_matching_v3_ode_selectable.py:241-242`). Fix is one
keyword argument.

### 21.2 The baseline gate is off by one from the FM gate

Even once wired, the two families would not agree, because they count differently.

**Baseline** (`diffuser/models/diffusion.py:175-186`): `t` runs **down** from `K−1` to `0` and
the test is `t <= T·K` on floats, so

```
n_active = floor(T·K) + 1
```

**FMv3ODE** (`flow_matcher_v3_ode_selectable/models/diffusion.py:207-208`, §11.1):

```
n_active = K − int((1−T)·K)
```

| K | T | T·K | baseline `n_active` | FMv3ODE `n_active` |
|---|---|---|---|---|
| 20 | 0.5 | 10 | **11** | 10 |
| 20 | 0.1 | 2 | **3** | 2 |
| 20 | 0.05 | 1 | **2** | 1 |
| 10 | 0.1 | 1 | **2** | 1 |
| 10 | 0.05 | 0.5 | 1 | 1 |

They agree **iff `T·K` is not an integer**, and differ by exactly one when it is. Every
threshold used in Parts I–III hits the integer case except K=10/T=0.05.

> Consequence: a naive "matched threshold" DPCC-baseline vs FMv3ODE comparison at K=20, T=0.1
> would silently be **3 solves vs 2** — a 50% budget gap dressed as a matched condition. This
> is the same class of defect fix_8 repaired between HardFlow and DPCC, one layer up. **fix_9
> must fix the wiring and the arithmetic together, or fixing the wiring alone makes the
> comparison wrong in a new way.**

It also means Part I's DPCC-on-FM at θ=0.5 (`n_active = 10`) and this run's θ=0.5
(`n_active = 11`) are **10 vs 11**, not equal. §24 corrects for it.

### 21.3 `post_processing` is an unimplemented alias of `dpcc-r`

`scripts/eval.py` branches on `gradient`, `model_free`, `dpcc-t` and `dpcc-c` (`:183-211`).
There is **no branch for `post_processing`**, so it falls through to exactly the `dpcc-r`
configuration: `gradient=False`, `trajectory_selection='random'`, same projector, same
threshold.

Verified at the byte level — `sha256(obs_all.npy)`:

| env | `dpcc-r` | `post_processing` | |
|---|---|---|---|
| top-right-hard | `26896f11bc2afddb` | `26896f11bc2afddb` | identical |
| top-left-hard | `87c4847e1734ed86` | `87c4847e1734ed86` | identical |
| both-hard | `75ecaf727e9c4dc9` | `75ecaf727e9c4dc9` | identical |

The same holds for the `-tightened` pair, and — checked separately — for **all four
`temp/0408/FMv3ODE` runs**, both suffixes, all three envs. The FM sibling scripts have no
`post_processing` branch either.

The vestige is visible in the projector itself: `diffuser/sampling/projection.py:14` is
`# self.only_last = only_last`, commented out — and commented out identically in **every**
sibling (`flow_matcher_v3*/sampling/projection.py:14`) and **in upstream DPCC**. The feature
`post_processing` names has never existed in this lineage; only the config entry does.

> **Consequence:** `post_processing` and `post_processing-tightened` are duplicate `dpcc-r`
> columns in every results matrix, LaTeX table and `all_seeds` figure this repo has produced.
> Any table presenting post-processing as a distinct baseline is reporting `dpcc-r` twice.
> Decide in fix_9: implement it (project only the final sample, i.e. `n_active = 1` with no
> in-loop passes) or delete it from `projection_variants`.

---

## 22. The free determinism control

The accidental duplicate is the cleanest control in this document: two jobs, **different GPUs**
(`CUDA_VISIBLE_DEVICES=1` vs `2`), **different commits**, 3.4 h apart, same everything else.

- **Trajectories: 39/39 cells bit-identical.** Steps, successes, violations, total violations
  all agree exactly. The `torch.manual_seed(i)` per trial (`scripts/eval.py:241`) is doing its
  job end to end, through MuJoCo, the sampler and SLSQP.
- **Wall clock, 78 episodes**, ratio job-24226 / job-24215:

| min | median | mean | max |
|---|---|---|---|
| 0.9996 | 1.0100 | 1.0097 | 1.0134 |

So timing repeats to **±1%, with a ~1% systematic offset between jobs** (24226 uniformly
slower — different GPU, different node load). That is the honest error bar for §12.3's
`a = 8.6 ± 0.1 ms`, and for every `s/step` in Parts I–III.

> **Every step-count and quality delta in Parts I and II is signal.** Run-to-run scatter on
> this harness is exactly zero for those metrics, so the differences in §11.1 (17/21 cells
> differ at K=20) and §12.1 are caused by the configuration change and nothing else. And
> §12.1's effects are 45–63% against a 1% timing noise floor.

---

## 23. Baseline DPCC at θ = 0.5 — the reference numbers

GaussianDiffusion, K = 20, `n_active = 11` (§21.2), ckpt 91000, seed 6, 6 episodes/arm,
averaged over the three envs. Full `.npz` precision.

| arm | q | steps | s/step | ep_s |
|---|---|---|---|---|
| `diffuser` | 0/6 | 58.8 | 0.1776 | 10.4 |
| `gradient` | 0/6 | 60.8 | 0.1914 | 11.6 |
| `gradient-tightened` | 0/6 | 62.2 | 0.1917 | 11.9 |
| `model_free` | 0/6 | 67.5 | 0.2607 | 17.6 |
| `model_free-tightened` | 0/6 | 64.7 | 0.2823 | 18.3 |
| `dpcc-r` | 1/6 | 67.2 | 0.4057 | 27.3 |
| `dpcc-t` | 1/6 | 67.8 | 0.5347 | 36.3 |
| **`dpcc-c`** | **6/6** | 65.7 | 0.5060 | 33.2 |
| `dpcc-r-tightened` | 5/6 | 65.3 | 0.5609 | 36.6 |
| **`dpcc-t-tightened`** | **6/6** | 62.0 | 0.5324 | 33.0 |
| **`dpcc-c-tightened`** | **6/6** | 61.5 | 0.5718 | 35.2 |
| `post_processing` | 1/6 | 67.2 | 0.4046 | 27.2 | 
| `post_processing-tightened` | 5/6 | 65.3 | 0.5610 | 36.7 |

(The last two rows are §21.3 duplicates of `dpcc-r` / `dpcc-r-tightened`; the sub-1% time
differences are the §22 noise floor. They are listed once for the record and are excluded from
every comparison below.)

**The generator constant transfers across engines.** `diffuser` gives
`a = 8.90 / 8.83 / 8.90 ms` per batched net call across the three envs — against
`8.6 ± 0.1 ms` for FlowMatchingODE (§12.3). **3% apart**, which is about what the extra work in
`p_sample` (noise draw + posterior variance, `diffuser/models/diffusion.py:155-161`) should
cost over a bare Euler step. §12.3 can be strengthened: `a` is portable across K, across eval
scripts, **and across the generative engine.**

The unguided step counts are not the same, though — 58.8 (diffusion) vs 65.0 (FM, §12) — so
this is a different policy, not a re-parameterisation of the same one.

---

## 24. The projector cost belongs to the constraint set, not the generator

Backing the NLP term out as in §12.2, with each run's own `diffuser` time and
`n_active × B = 11 × 4 = 44`:

| arm | top-right | top-left | both | mean ms/solve |
|---|---|---|---|---|
| `dpcc-t-tightened` | 8.51 | 8.66 | 7.02 | **8.06** |
| `dpcc-r-tightened` | 8.27 | 10.81 | 7.05 | **8.71** |
| `dpcc-c-tightened` | 8.05 | 9.00 | 9.83 | **8.96** |
| `dpcc-c` | 7.57 | 6.55 | 8.28 | 7.46 |
| `dpcc-t` | 10.89 | 6.57 | 6.88 | 8.12 |
| `dpcc-r` | 2.41 | 7.35 | 5.79 | 5.19 |
| `gradient[-tightened]` | 0.32 | 0.33 | 0.30 | **0.31** |
| `model_free[-tightened]` | 1.60–2.38 | 2.17–2.58 | 1.90–2.18 | 1.89 / 2.38 |

Against the FM generator at `n_active = 10` (§12.2): **3.57 / 7.47 / 11.62 ms per env, mean
7.43**. The two are inside each other's env-to-env spread.

> **`b_scipy` is a property of the constraint set and the τ-range, not of the generator.**
> Same halfspace constraints, same SLSQP, same τ window (0.50…0.95 vs 0.45…0.95) → same
> per-solve cost, whether the iterate came from a diffusion posterior or an ODE step.
> This is the cross-engine version of §14 and it is consistent with §14's mechanism: cost is
> set by how far outside the feasible set the iterate sits, and both generators are equally far
> out at τ = 0.5.

**NFE-eq across generators.** Predicting the baseline from the *FM* constants — `a = 8.88 ms`
(this run's own generator, §23) and `b = 7.43 ms` (the FM run's, §13) — with `N_net = 20`,
`NPE = 44`:

```
predicted  =  20 × 8.88 ms  +  44 × 7.43 ms  =  177.5 + 326.9  =  504 ms/step
measured   =  519 ms/step   (mean of the six dpcc arms)          →  −2.7%
```

Per-arm the spread is −20% (`dpcc-r`, dragged down by one env that terminates in 31.5 steps) to
+13% (`dpcc-c-tightened`). So `b` transfers across generators to about ±10% at fixed schedule,
which is far better than it transfers across schedules (§14: 8×). **Schedule is the first-order
variable; generator is a rounding error.**

`gradient`'s 0.31 ms/solve is worth recording separately: it is 24× cheaper than the SLSQP
arms and buys 0/6 — the cost floor of "touching the trajectory at all", and the reason
`gradient` sits within 8% of `diffuser` on time.

---

## 25. The only 6/6 on the **relaxed** problem in this entire document

`dpcc-c` — **untightened** — is **6/6**, zero violations in all three envs.

| generator | schedule | best untightened arm | q |
|---|---|---|---|
| **GaussianDiffusion** | `n_active = 11` | **`dpcc-c`** | **6/6** |
| FlowMatchingODE | `n_active = 10` | `dpcc-r` | 4/6 |
| FlowMatchingODE | `n_active = 2` | `dpcc-c` | 3/6 |
| FlowMatchingODE | `n_active = 1` (K=20) | `dpcc-r` / `dpcc-c` | 2/6 |
| HardFlow | `n_active = 10` (θ=0.5) | `dpcc-r`-style `-r` | 5/6 |
| HardFlow | `n_active = 2` | — | 3/6 |

§16 concluded "no leap — best anywhere is HF `-r` @θ=0.5 at 5/6." **That conclusion is
superseded on the relaxed axis: the baseline diffusion generator with `dpcc-c` solves it
outright.** §16's *comparative* claim (the FM-vs-HF trade is symmetric, tie at 3/6 at matched
schedule) is untouched — this row is neither of those two arms.

Two further observations:

- **The arm ordering inverts between generators.** On the diffusion generator the untightened
  ranking is `-c` (6/6) ≫ `-r` ≈ `-t` (1/6). On the FM generator (§10.3, §16) `-r` led and `-c`
  trailed. `minimum_projection_cost` selection is doing real work here and nothing on the FM
  side — which is the opposite of §10.4/§15, where `-c` was the arm that *inflated* step counts.
  It does not inflate them here: 65.7 steps, mid-pack.
- **Do not read this as "diffusion beats flow matching."** Three things differ at once —
  engine, checkpoint (91000 vs 98000, different training runs entirely), and schedule (11 vs
  10 solves). It is a single seed, 6 episodes. What it *does* establish is that **the relaxed
  problem is solvable on this benchmark at this budget**, which §16 left open, and that makes
  §19 item 4 considerably more valuable than it looked.

**What it costs:** 33.2 s/episode, against 6.1 s for the cheapest 6/6-*tightened* point (§17,
FM K=10 `n_active=1`). **5.4×** — for a result on a strictly harder problem. That is the real
Pareto frontier of this dataset, and both ends of it are DPCC.

---

## 26. What Part III does and does not license

**Does not:**

- Provide a DPCC-*baseline* short-schedule datapoint. §12.1 remains an FM-generator result.
  The K=20/`n_active`∈{2,3} baseline row is still missing and this run did not produce it.
- Settle any generator comparison. §25 is confounded three ways.

**Does:**

- Put a ±1% error bar and a zero-scatter guarantee under Parts I–II (§22).
- Extend §12.3 (`a` portable) and §14 (`b` set by schedule, not engine) across generators (§23, §24).
- Establish that the relaxed problem is solvable at all (§25).
- Find fix_9 (§21).

### Additions to the §19 run queue

- **0. fix_9, before anything else re-runs through `scripts/eval.py`.** Three parts:
  (a) pass `diffusion_timestep_threshold=args.diffusion_timestep_threshold` at
  `scripts/eval.py:206`; (b) reconcile the gate arithmetic with the FM path (§21.2) — floor to
  the *same* `n_active`, and tag it `[Gen12fix9]` alongside the fix_8 comments; (c) decide
  `post_processing` — implement or delete. Then **re-run these two jobs**, which are currently
  a datapoint we believed we had and do not.
- **7.** After fix_9: baseline DPCC at `n_active = 2` and `10`, to put the diffusion generator
  on §12.1's axes. Pair it with the FM run at `n_active = 11` (or the baseline at 10) so the
  §25 comparison is schedule-matched.
- **8.** Audit which published/exported tables in `Data_Analysis/` and the DA-v3 Visualizer
  carry a `post_processing` column, and which carry a `T`-tagged savepath from
  `scripts/eval.py`. Both are wrong in the ways §21.1 and §21.3 describe.

### Caveats specific to Part III

1. Everything is seed 6, `n_trials = 2`. §25 rests on 6 episodes.
2. `n_active = 11` is inferred from the source (§21.2), not instrumented. §20 caveat 3 applies
   in full: `B = 4` is read from the config, not counted, so every ms/solve in §24 is
   `(t − t_gen)/(11 × 4)`. §22, §23 and the NFE-eq total in §24 use only measured wall clock.
3. The checkpoint is **91000**, not the FM line's 98000, and comes from a separate training
   run. No conclusion here should cross that boundary.
4. The two jobs sit on different commits. They differ only in Gen14 U6, which touches
   `mix_visual_aligning` only — but the §22 determinism claim technically covers "these two
   revs", not "any two revs".
