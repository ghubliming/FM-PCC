# HardFlow — Pareto study: does HF buy fewer steps at equal success+constraints in equivalent time?

**Date:** 2026-08-05
**Source:** `temp/2026-08-02/batch_avoiding_combined_20260802_092307/candidates_multidimensional_raw.csv`
**Scope:** `avoiding-d3il`, H=8, 3 halfspace envs × 6 selection variants × 6 generator/K settings = **93 comparison cells**
**Supersedes:** §11 of `logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md` — that section was
built on one candidate at one seed and its cost conclusions are wrong. Nothing in §11 should be cited.
**Related:** `DA_20260803_HardFlow_activation_threshold_0p1.md` (threshold sweep; still valid)

---

## 0. The question and the decision rule

> **Does HardFlow deliver fewer control steps, at the same or better goal-and-constraint
> success, in equivalent wall clock — compared to DPCC on the same generator?**

Three axes per cell: **quality** `n_success_and_constraints` (higher better), **steps**
(lower better), **episode time** = `n_steps × avg_time` (lower better).

A cell is one **(generator/K, environment, selection variant)** triple, comparing
`hardflow_new-X` against `dpcc-X` — matched suffix, so the projector is the only thing
that changes. Each cell is classified:

| verdict | meaning |
|---|---|
| **HF DOMINATES** | no axis worse, at least one better |
| **non-dominated** | mixed — better on some axes, worse on others. **Acceptable.** |
| **HF dominated** | no axis better, at least one worse. **The only failure mode.** |

Per the brief: partial Pareto optimality counts as a pass; only being beaten on everything
is a fail.

**Answer up front.** Across 93 cells: **15 HF DOMINATES, 52 non-dominated, 26 HF dominated.**
So HF is not strictly worse in 72% of cells. But the structure matters far more than the
tally, and it has two parts:

- **HF's Pareto standing is entirely a function of K**, and at the low-K operating point this
  project targets (K=1–2), **HF costs 2.1–3.3× DPCC's wall clock** — it fails "equivalent
  time" outright, whatever it does to steps.
- **The one K where HF dominates most broadly (K=5) is confounded by a gate-rounding bug that
  gave HF 33% fewer NLP solves than DPCC**, fixed in `924db516` *one day after this data was
  generated*. §5 quantifies this; K=5 must be re-run before it is cited.

The result that survives both is narrower and is in §4.

---

## 1. Data

Candidates carrying the full `hardflow_new-{r,c,t}[-tightened]` arm set:

| generator | K | candidate | seeds | episodes/env-cell |
|---|---|---|---|---|
| AlphaFlow `bbsit` | 1 | CAND_31 | 7–10 | 8 |
| **AlphaFlow `bbsit`** | **2** | **CAND_32** | **6–10** | **10** |
| **MeanFlow `bbmf_dit`** | **2** | **CAND_102** | **6–10** | **10** |
| AlphaFlow `bbsit` | 5 | CAND_33 | 7–10 | 8 |
| AlphaFlow `bbsit` | 10 | CAND_30 | 7–10 | 8 |
| FlowMatchingODE | 20 | CAND_42 * | 6–10 | 10 |

`*` **CAND_42 sits in `flow_matching_v3_hardflow(Gen12_bf_Fix6_wrong_batch_parallel)`** — the
folder name declares the batch parallelism was wrong. It carries only `-c-tightened` in this
batch. It is reported in §3 for completeness and **excluded from every conclusion**.

MeanFlow exists at full seed count **only at K=2** — there is no MF K-ladder, so §4's
K-dependence is AlphaFlow's alone.

**Backbone note.** Every row above is a transformer backbone (`bbsit` / `bbmf_dit`); FM and DPCC
use the UNet. This does **not** confound anything in this document — every comparison here is
`hardflow_new-X` against `dpcc-X` **on the same generator**, so the backbone is held fixed within
each cell and cancels. It does mean the absolute times are not comparable to a UNet baseline:
per network evaluation, SiT costs 6.17 ms, `mf_dit` 8.45 ms and the UNet 8.97 ms
(§7.2 of `DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md`). Since HF's overhead is dominated by
the NLP rather than the network, a UNet generator would shift §3's time ratios *toward* HF, not
away from it — the K≤2 penalty reported there is if anything conservative.

`n_steps` is averaged over successful episodes only (`eval_FM_v3_hardflow.py:482`), so step
comparisons between cells with very different success rates are optimistic for the weaker cell.
This matters in §6 and is flagged there.

---

## 2. Cell-by-cell verdicts

| generator | HF DOMINATES | non-dominated | **HF dominated** | not-worse rate |
|---|---|---|---|---|
| AF K=1 | 0 | 13 | 5 | 72% |
| **AF K=2** | **0** | **12** | **6** | 67% |
| **MF K=2** | **0** | **12** | **6** | 67% |
| **AF K=5** | **9** | **9** | **0** | **100%** |
| AF K=10 | 6 | 6 | 6 | 67% |
| FM K=20 * | 0 | 0 | 3 | 0% |
| **total** | **15** | **52** | **26** | **72%** |

**HF is never dominated at K=5, and never dominant at K=1 or K=2.** No cell at K≤2, on either
generator, has HF winning on all three axes.

---

## 3. The claim as stated, tested directly

Cells where **quality is equal or better AND steps are lower** — the literal claim — plus what
the wall clock did in exactly those cells:

| generator | K | claim holds | **time ratio HF/DPCC in those cells** |
|---|---|---|---|
| AF K=1 | 1 | 10/18 | **1.47×** [1.05–2.12] |
| AF K=2 | 2 | **11/18** | **2.25×** [1.03–3.36] |
| MF K=2 | 2 | 9/18 | **2.12×** [1.50–3.14] |
| **AF K=5** | 5 | 9/18 | **0.63×** [0.39–0.73] |
| **AF K=10** | 10 | 6/18 | **0.76×** [0.62–0.84] |
| FM K=20 * | 20 | 0/3 | — |

**The "fewer steps" half of the claim holds broadly — in 45 of 93 cells.** The "equivalent
time" half is where it splits, and it splits cleanly on K.

Median time ratio over **all** cells, by K:

| generator | K | median HF/DPCC time | range |
|---|---|---|---|
| AF | 1 | **2.11×** | [1.05, 2.84] |
| AF | 2 | **3.28×** | [1.03, 3.72] |
| MF | 2 | **2.63×** | [1.28, 4.24] |
| AF | 5 | **0.79×** | [0.39, 0.98] |
| AF | 10 | **0.97×** | [0.62, 1.25] |
| FM * | 20 | 6.65× | [5.25, 7.19] |

**At K=1–2 HardFlow costs 2.1–3.3× DPCC. It reaches parity only at K=5–10.** Both halves of
the claim are never simultaneously satisfied below K=5.

The mechanism is in the companion threshold doc: at K=2 the gate `k >= int((1−thr)·K) or
(k == K−1)` yields **one** active projection step for both projectors, so HF and DPCC solve the
same number of NLPs — but HF's NLP additionally carries the linear dynamics equalities and
input saturation (`hardflow_projection.py:300-315`), and HF spends an extra velocity evaluation
on the endpoint extrapolation. Same solve count, heavier solve, extra network pass. **At K=2
HardFlow has no schedule advantage to trade against its higher per-solve cost.**

---

## 4. What survives — the `-r` result at K=10

Breaking AF K=10 down by selection rule (this is the highest K with a clean gate — see §5):

| suffix | top-right | top-left | both |
|---|---|---|---|
| **`-r`** | **HF DOMINATES** | **HF DOMINATES** | **HF DOMINATES** |
| **`-r-tightened`** | **HF DOMINATES** | **HF DOMINATES** | **HF DOMINATES** |
| `-c` | HF dominated | non-dominated | HF dominated |
| `-c-tightened` | HF dominated | non-dominated | HF dominated |
| `-t` | HF dominated | HF dominated | non-dominated |
| `-t-tightened` | non-dominated | non-dominated | non-dominated |

**On the neutral selection rule, HardFlow strictly dominates DPCC in all six cells** — better
or equal quality, fewer steps, less time. Representative numbers:

| cell | quality | steps | episode time |
|---|---|---|---|
| `-r`, top-right | HF 5/8 vs DPCC 2/8 | 67.4 vs 83.0 (**−18.8%**) | 20.2 vs 26.5 s (**0.76×**) |
| `-r`, both | HF 4/8 vs DPCC 2/8 | 61.6 vs 94.0 (**−34.4%**) | 18.4 vs 26.5 s (**0.69×**) |
| `-r-tightened`, top-right | HF 8/8 vs DPCC 4/8 | 80.8 vs 95.6 (**−15.6%**) | 25.7 vs 41.5 s (**0.62×**) |
| `-r-tightened`, both | HF 8/8 = DPCC 8/8 | 60.2 vs 64.8 (−6.9%) | 17.9 vs 21.3 s (0.84×) |

This is the claim, satisfied: **same-or-better quality, 7–34% fewer steps, 16–38% less time.**

It is also the whole of it. Under `-t` HF is dominated or non-dominated in every cell; under
`-c` it is dominated in four of six. **HardFlow's Pareto advantage is specific to the neutral
selection rule** — which is consistent with the mechanism, since `-c` and `-t` rank candidates
using information that HF changes the meaning of (see §6).

Caveats on this cell block: 8 episodes per env-cell, 4 seeds, one generator, one K. It is the
best-supported positive result in the study and it is not strong evidence.

---

## 5. Why K=5 must not be cited

AF K=5 is the standout — 9 dominating cells, **zero** dominated, time ratio 0.63×. It is also
the only K in the ladder where the two projectors did **not** run the same schedule.

The gate now reads (`hardflow_projection.py:510`, added in `924db516`):

```python
active = (k >= int((1.0 - self.activation_threshold) * K)) or (k == K - 1)
```

with the comment: *"Comparing against the raw float is CEIL, which made HardFlow do one FEWER
projection step than both references whenever (1−T)·K is not an integer. No-op at integer
boundaries."* Before that commit HF compared against the raw float.

At threshold 0.5, `(1−T)·K` is an integer at K=2, 10 and 20 — and **2.5 at K=5**:

| K | `(1−T)·K` | DPCC active steps | HF active steps (pre-fix) | affected? |
|---|---|---|---|---|
| 1 | 0.5 | 1 | 1 | no (forced final step) |
| 2 | 1.0 | 1 | 1 | no |
| **5** | **2.5** | **3** (k=2,3,4) | **2** (k=3,4) | **YES — HF ran 33% fewer solves** |
| 10 | 5.0 | 5 | 5 | no |
| 20 | 10.0 | 10 | 10 | no |

**`924db516` is dated Aug 3 16:28; this batch was generated Aug 2 12:34.** The data predates
the fix by a day, and K=5 is the single rung the fix touches.

So AF K=5's 0.63× time advantage is, at minimum, partly a 33% solve-count discount that HF is
no longer entitled to. **K=5 is excluded from every conclusion in this document and needs a
re-run on current code.** The §4 result at K=10 is unaffected — `(1−0.5)·10 = 5.0` is an
integer, so both projectors ran five active steps then and now.

---

## 6. The `-c` rescue — HF's largest single effect, and what it actually shows

Pooled over the three environments:

| generator | suffix | quality HF vs DPCC | steps HF vs DPCC |
|---|---|---|---|
| **AF K=2** | `-c-tightened` | **24/30 vs 6/30** | 68.1 vs 109.0 (**−37.5%**) |
| **AF K=2** | `-c` | **19/30 vs 5/30** | 70.8 vs 116.8 (**−39.4%**) |
| **MF K=2** | `-c-tightened` | **23/30 vs 3/30** | 71.4 vs 134.7 (**−47.0%**) |
| **MF K=2** | `-c` | **12/30 vs 2/30** | 82.2 vs 133.7 (**−38.5%**) |
| AF K=1 | `-c-tightened` | 23/24 vs 21/24 | 77.8 vs 143.2 (−45.7%) |
| AF K=5 † | `-c-tightened` | 24/24 vs 23/24 | 67.7 vs 65.2 (+3.7%) |
| AF K=10 | `-c-tightened` | 24/24 vs 24/24 | 68.7 vs 62.0 (+10.7%) |

† confounded, §5.

**At K=2 this is the claim in its strongest form: 4–8× the quality and ~40% fewer steps**, for
1.5–1.8× the time — the mildest time penalty anywhere at K=2. It is by a wide margin the
largest effect in the study, and it is at the operating point the MF/AF line targets.

**But it is a repair, not an advantage.** DPCC's `-c` is *broken* at K=2 — `minimum_projection_cost`
ranks candidates by how cheap they are to project, and on a half-integrated τ=0.5 iterate a
stalled trajectory is nearly free to project, so the ranking selects it (the 109–135 step counts
in the DPCC column are that: trajectories that do not go anywhere). HardFlow scores the same
ranking on the *predicted clean endpoint*, where standing still is not cheap, so the pathology
disappears.

Two consequences for how this should be written up:

1. **The evidence is for the endpoint idea, not for HardFlow's machinery.** Computing DPCC's
   `-c` ranking on an extrapolated endpoint would plausibly recover most of this at none of
   HF's cost. That is a ~10-line change and it has not been tried.
2. **The effect vanishes once `-c` is not broken.** At K=5/K=10 DPCC's `-c` is healthy and HF's
   step advantage inverts to **+4% and +11%** — HF's `-c` is then the one that wanders.

---

## 7. What this study establishes

**Established:**

1. **HF is not strictly worse in 72% of cells** (52 non-dominated + 15 dominating of 93).
   Under the stated rule — partial optimality passes — HardFlow passes broadly.
2. **The "fewer steps" half of the claim holds in 45/93 cells**, including 11/18 at AF K=2.
3. **The "equivalent time" half fails at K≤2**: median 2.1–3.3× DPCC's wall clock, because at
   K=2 both projectors run one active step, so HF pays a heavier NLP plus an extra network pass
   with no schedule saving to offset it (§3).
4. **At K=10 on the `-r` rule, HF strictly dominates DPCC in all 6 cells** — same-or-better
   quality, 7–34% fewer steps, 16–38% less time (§4). This is the only clean positive result.
5. **HF repairs the `-c` collapse at K=2** — 4–8× quality, ~40% fewer steps, at 1.5–1.8× time
   (§6). Largest effect in the study.

**Not established, and actively contradicted:**

6. **The claim as a whole at K=1–2.** No cell at K≤2 on either generator has HF winning all
   three axes; the time axis rules it out.
7. **Anything at K=5.** Confounded by the pre-fix gate rounding, which handed HF 33% fewer NLP
   solves at exactly that K and nowhere else (§5).
8. **That the `-c` rescue is an HF advantage.** It repairs a broken baseline, and the repair is
   probably available directly (§6). Where `-c` is healthy, HF's `-c` is 4–11% *worse* on steps.
9. **Anything about MeanFlow beyond K=2** — no MF K-ladder exists (§1).
10. **Anything from FM K=20** — CAND_42 is the `wrong_batch_parallel` tree (§1).

**Structural caveats:** 8–10 episodes per env-cell; `n_steps` conditioned on success, which
flatters the weaker arm in cells with large quality gaps (notably the DPCC column of §6 — though
there the gap is 40% and the direction is not in doubt); AF ladder on seeds 7–10 while AF K=2
adds seed 6; all HF arms here run at activation threshold 0.5, which the companion doc shows is
3× off HF's optimum at K=20.

---

## 8. Verdict

> **HardFlow does deliver fewer control steps at equal or better success+constraints — in 45 of
> 93 cells, and by 15–47% where it does. What it does not deliver at the project's operating
> point is equivalent time: at K=1–2 it costs 2.1–3.3× DPCC, because both projectors run a
> single active projection step there and HardFlow's is the more expensive one.**
>
> **The claim is fully satisfied in exactly one place: AlphaFlow at K=10 under the neutral
> `-r` rule, where HF strictly dominates DPCC in all six environment × tightening cells —
> same-or-better quality, 7–34% fewer steps, 16–38% less time.**
>
> The broadest-looking result, K=5, is unusable: the batch predates `924db516` and HF ran 33%
> fewer NLP solves than DPCC at that K and only that K. The largest single effect, HF's rescue
> of `-c` at K=2, is a repair of a broken baseline rather than a projector advantage, and it
> reverses where `-c` is healthy.

---

## 9. Run queue

1. **Re-run AF K=5 on current code (post-`924db516`), 4+ seeds, all six arms.** The study's
   headline number is currently unusable. This is one job and it either restores or kills §2's
   "never dominated at K=5".
2. **AF/MF K=2 and K=10 at activation threshold 0.1**, all six arms, full seeds. Every HF arm
   here ran at 0.5, which the threshold doc shows is 3× off the optimum at K=20 —
   §3's 2.1–3.3× time penalty at K=2 may be smaller than measured, though the gate analysis
   says the threshold is a no-op at K=2 and the penalty should stand.
3. **MeanFlow K-ladder (K=1, 5, 10) with HF arms.** §4's positive result is AlphaFlow-only and
   MF has no ladder at all.
4. **Fix DPCC `-c` to rank on an extrapolated endpoint, then re-run K=2.** Directly tests
   whether §6 is an HF result or an endpoint result. ~10 lines, and it decides how §6 is written.
5. **More trials per env-cell.** 8–10 episodes means a 5/8-vs-2/8 cell in §4 is three episodes;
   the dominance classification is sensitive to single-episode flips.
6. **Re-run FM K=20 HF outside the `wrong_batch_parallel` tree**, with all six arms, so the
   high-K end of §3 has a usable point.
