# DA — Gen14 · **K=10 / T=0.4 vs K=20 / T=0.2** · three stages: min dist → constraints → avg time

**Drop** `temp/0109/` · **Batch** `batch_va2_20260902_114841` · **Task** `aligning-d3il-visual`
**Protocol** seed 6 · **the same 10 paired contexts in every row** · train split · `mpc4` fan · `filmv1`
**Companion** `DA_20260902_Gen14_K10_T0.4_mf_and_fm_resubmit.md` (cost mechanism, NFE/NLP accounting)

**The question:** K=10/T=0.4 was proposed as a cheaper replacement for the K=20/T=0.2 flagship. Is it?

## Ground rules

1. **Three stages, in this order: minimum distance → constraint satisfaction → average time.**
   A stage is only consulted when the one above it ties.
2. **`context_final_xy_dist`**, metres, box→target. Mean initial distance **0.4530 m**.
   🪤 `mean_dist_per_rollout` is never used — it is `0.5*(pos_dist_3D + rot_err/π)`, not a distance.
3. **Constraint claims are made on the `combined_5-tightened` geometry only.** On untightened
   `combined_5` the executed-violation check is stricter than the set handed to the projector, so the
   arms rank incoherently and no projector claim is valid there.
4. **`n_success` = 0.000 in every cell.** Nothing here is a success claim.
5. **`untouched`** = box never moved (`final == init` to 1e-6). Reported beside every distance,
   because "did nothing" caps the error at the initial distance and can masquerade as a good mean.
6. ⚠️ **MIN is n = 1.** It is the best single rollout of ten — a *capability ceiling*, not a
   performance measure, and it has no error bar. It leads because it answers "can this cell get the
   box to the target at all", which is the prior question. Median is carried beside it throughout.

---

# Stage 1 · Minimum distance — unguided (`diffuser`) arm

No projector, so the threshold `T` is inert and this is a clean NFE axis: **K=10 vs K=20**.

| | **MIN** | p25 | median | mean | MAX | untouched | ms |
|---|---|---|---|---|---|---|---|
| **K=10 T0.4** | **0.0082** | 0.0549 | 0.1421 | 0.1992 | 0.4746 | 2/10 | **89.7** |
| K=20 T0.2 | 0.0278 | 0.0506 | **0.0902** | **0.1431** | **0.4524** | **1/10** | 172.5 |

*(tightened geo; untightened tells the same story — MIN 0.0074 vs 0.0278, median 0.1194 vs 0.0741)*

**Paired on the same 10 contexts:** K=10 worse by +0.0561 m, **5/4**, sign p = 1.000
(untightened: +0.0797 m, 5/5, sign p = 1.000, permutation p = 0.250).

### Stage 1 does not separate them

- **K=10 wins the minimum by 3.4×** — 0.0082 m against 0.0278 m. It is also the best MIN in the
  entire corpus, unguided (§4). The 2-step-cheaper sampler can land closer than the 20-step one.
- **K=20 wins every robust statistic** — median 0.0902 vs 0.1421, mean 0.1431 vs 0.1992, worst case
  0.4524 vs 0.4746, and it leaves the box untouched less often.
- **The paired test is a tie** (5/4, p = 1.000). At n = 10 there is no separation.

**Neither is eliminated. Stage 1 is a tie; go to Stage 2.**

---

# Stage 2 · Constraint satisfaction — and the minimum distance *inside* it

The question that matters is not "who has the best min dist" but **"who has the best min dist among
the arms that are actually legal"**. Every arm in the tightened corpus that reaches **1.000
zero-violation / 0.00 violations**, ranked by MIN:

| rank | cell · arm | **MIN** | median | mean | MAX | untouched | ms |
|---|---|---|---|---|---|---|---|
| 1 | *mf K=2 T0.5 · `dpcc-t`* | *0.0175* | *0.2207* | *0.2499* | *0.4713* | *3/10* | *43.0* |
| **2** | **mf K=20 T0.2 · `hardflow_sls-r`** | **0.0220** | **0.1967** | **0.2108** | **0.4524** | **2/10** | 275.3 |
| 3 | mf K=20 T0.2 · `hardflow_sls-c` | 0.0241 | 0.3227 | 0.2863 | 0.4842 | 4/10 | 282.3 |
| — | *mf K=2 T0.5 · `hardflow_new-t`* | *0.0344* | *0.2439* | *0.2602* | *0.4741* | *3/10* | *159.0* |
| **5** | **mf K=10 T0.4 · `hardflow_sls-r`** | **0.0455** | **0.2416** | **0.2365** | 0.4524 | 2/10 | 192.5 |

*Italic rows are K=2 — carried here only because they are legal; treated as a separate question in §4.*
⚠️ `hardflow_new-t` at K=2/T=0.5 is **DEGENERATE** (`n_active = 1`, `n_genuine = 0`): no HardFlow
arithmetic runs. It is sample-then-project under a HardFlow label and is excluded from ranking.

## 2.1 🔴 Stage 2 breaks the tie against K=10

**K=10's unguided minimum does not survive projection.** It goes into Stage 2 holding the best MIN in
the corpus (0.0082 m) and comes out with **the worst MIN of every legal arm (0.0455 m)** — a 5.5×
degradation — and the worst median of the legal set (0.2416 m). K=20's MIN degrades far less
(0.0278 → 0.0220; it actually improves) and it keeps the best median at 0.1967 m.

| | unguided MIN | legal-arm MIN | change |
|---|---|---|---|
| K=10 | **0.0082** | 0.0455 | **× 5.5 worse** |
| K=20 | 0.0278 | **0.0220** | × 0.8 — *better* |

Breadth, secondarily: **K=20 has two arms at 1.000 zero-violation** (`-r` and `-c`); K=10 has one.
Pooled over `r`/`c`/`t`: **0.967 (K=20) vs 0.867 (K=10)**.

Head-to-head on the arm they share, `hardflow_sls-r`, paired:

| axis | mean Δ (K10 − K20) | worse/better | ties | p |
|---|---|---|---|---|
| zero-violation | 0.0000 | 0/0 | **10** | 1.000 |
| violations/rollout | 0.0000 | 0/0 | **10** | 1.000 |
| final distance | +0.0257 | 6/2 | 2 | 0.289 |
| ms/replan | −82.75 | 0/9 | 1 | **0.0039** |

Constraints are bit-identical on `-r`. **But K=10 reaches that identical outcome from a worse
starting distance and lands at a worse minimum.**

## 2.2 Does projection ever give distance *back*? — per-context rescue count

Averages hide it, so this is counted per context: how often does a projected arm land **closer than
its own unguided run on the same context**, and does any arm beat the unguided **minimum**?
Tightened, n = 10.

| cell | unguided MIN | arms beat unguided on… | best rescuer | **any arm sets a new MIN?** |
|---|---|---|---|---|
| **mf K=10 T0.4** | **0.0082** | 2–4 / 10 | `geo_free-model_free` 6/10 | 🔴 **No — not one arm** |
| **mf K=20 T0.2** | 0.0278 | 2–5 / 10 | `geo_free` 5/10 | ✅ **five arms do** |
| *mf K=2 T0.5* | *0.0145* | *3–6 / 10* | *`dpcc-c`, `hardflow_new-c` 6/10* | ✅ *`dpcc-c` → **0.0024*** |

New minima set by projection:

| cell | arm | MIN | vs unguided | legal? |
|---|---|---|---|---|
| mf K=20 | `model_free` | **0.0075** | 0.0278 | ❌ dynamics stripped |
| mf K=20 | `gradient` | 0.0089 | 0.0278 | ❌ 0.300 zero-viol |
| mf K=20 | `geo_free` | 0.0147 | 0.0278 | ❌ geometry stripped |
| **mf K=20** | **`hardflow_sls-r`** | **0.0220** | 0.0278 | ✅ **1.000 zero-viol** |
| mf K=20 | `hardflow_sls-c` | 0.0241 | 0.0278 | ✅ 1.000 zero-viol |
| *mf K=2* | *`dpcc-c`* | ***0.0024*** — *lowest in the corpus* | *0.0145* | *❌ 0.800 zero-viol* |

**So yes — projection genuinely rescues individual rollouts, and it is not rare (2–6 of 10).** Two
things follow, and both cut the same way:

1. **The worse the sampler, the more the projector can add.** K=2 (weakest unguided, 0.812×) is
   rescued on 5–6/10 contexts and reaches the corpus-best 0.0024 m. K=20 is rescued on 2–5/10 and
   picks up a new legal minimum. **K=10 — which went in holding the best unguided minimum in the
   corpus — gets nothing back from any of its twelve arms.** There is nothing left to save there, so
   projection can only take. That is the same conclusion as §2.1, reached from the opposite direction.
2. ⚠️ **The very best distances come from arms with the constraints removed.** At K=20 the top two
   minima (`model_free` 0.0075, `geo_free` 0.0147) are constraint-ablated and illegal; the best
   *legal* minimum is `hardflow_sls-r` at 0.0220. Those ablations measure what the constraint set
   costs in accuracy — they are not configurations. Consistent with §4.3: the cost sits in dynamics
   and action bounds, not geometry.

**Stage 2 decides: K=20.**

---

# Stage 3 · Average time — consulted, but it cannot overturn Stage 2

| cell · arm | ms/replan | NFE/replan | NLP solves/replan | s per rollout (~400 steps) |
|---|---|---|---|---|
| mf K=10 · `hardflow_sls-r` | **192.5** | 13 | 16 | 77.0 |
| mf K=20 · `hardflow_sls-r` | 275.3 | 23 | 16 | 110.1 |

K=10 is **30 % cheaper** (p = 0.0039), saving 33 s per rollout. Fan `B = 4` confirmed from the `mpc4`
plan-folder tag; NFE for arm C is `K + n_active − 1`. **The NLP solve count is identical (16) — the
whole difference is the 10 dropped sampler evaluations.**

Stage 3 is the last tiebreaker and Stage 2 already resolved the comparison, so this is a cost note,
not a verdict.

---

# 🏆 Verdict — **keep K = 20 / T = 0.2 with `hardflow_sls-r`. Drop K = 10.**

| stage | result |
|---|---|
| **1 · min dist** | **tie** — K=10 wins MIN 3.4×, K=20 wins median/mean/MAX, paired 5/4, p = 1.000 |
| **2 · constraints** | **K=20** — legal-arm MIN 0.0220 vs 0.0455, legal-arm median 0.1967 vs 0.2416, two perfect arms vs one |
| **3 · time** | K=10 by 30 % — noted, not decisive |

**K=10's case rested entirely on a Stage-1 minimum that the projector destroys.** It is the only cell
whose MIN gets 5.5× worse once constraints are enforced. K=20 goes into projection with a worse
minimum and comes out with a better one. That is the whole argument, and it is the right way round:
the number that matters is the one measured on a legal trajectory.

Counted per context (§2.2) the same result appears from the other side: projection rescues individual
rollouts often — 2 to 6 of 10 — and at K=20 five arms beat the unguided minimum outright. **At K=10,
not one arm of twelve does.**

**K=10 keeps exactly one honest use:** as the low-latency point on `-r`, where constraints are
bit-identical and it is 30 % cheaper. Quote it as a cost/quality trade-off, never as an equal.

⚠️ **This supersedes the "best horse = K=10" call in the companion DA §5.1**, which ranked the arm-C
row without a distance stage. The **mechanism** results there (§5.2–§5.4: matched NLP solve counts,
HardFlow's τ-invariant solve cost, DPCC's 4.6× inflation) are independent of this and stand.

**Load-bearing weakness:** Stage 1 is 5/4 at n = 10 and Stage 2's distance separation is p = 0.289.
The verdict rests on the *direction* of two ties, not on significance. **Seeds, not analysis.**

---

# 4 · The rest of the field, buried where it belongs

## 4.1 Minimum distance, unguided, whole corpus (untightened, same 10 contexts)

| cell | MIN | median | mean | mean/init | untouched | ms |
|---|---|---|---|---|---|---|
| **mf K=10 T0.4** | **0.0074** | 0.1194 | 0.1730 | 0.382 | 1/10 | 99.2 |
| mf K=100 T0.1 | 0.0123 | 0.0952 | 0.1440 | 0.318 | 0/10 | 924.6 |
| mf K=20 T0.2 | 0.0278 | **0.0741** | **0.0933** | **0.206** | 0/10 | 190.5 |
| fm K=20 T0.2 | 0.0394 | 0.4085 | 0.3266 | 0.721 | 4/10 | 286.2 |
| diffusion K=100 | 0.0463 | 0.1849 | 0.2359 | 0.521 | 0/10 | 1521.9 |
| mf K=2 T0.5 | 0.0689 | 0.2779 | 0.3676 | 0.812 | 0/10 | 28.4 |
| fm K=20 T0.5 | 0.1089 | 0.4126 | 0.3570 | 0.788 | 4/10 | 288.1 |
| diffusion K=20 | 0.1174 | 0.4619 | 0.4677 | **1.032** | 5/10 | 298.3 |

- 🔴 **`fm` never gets close** — worst-but-one MIN, 4/10 untouched, and **no projected arm improves it**
  (all five are worse than its own unguided arm, Δ +0.078…+0.111 m). It has nothing to rescue.
- 🔴 **`diffusion` at K=20 ends further from the target than it started** (1.032×, 5/10 untouched).
  Its `dpcc-r` reaches the corpus-best MIN of 0.0066 m — but by declining to move the box on 7/10
  contexts, at **1877 ms**. At K=100 its projected arm costs **10 584 ms/replan**.
- ⚠️ **Neither `fm`, `diffusion`, nor `mf K=100` has tightened-geometry rows**, so none can be scored
  at Stage 2 at all. That is the single biggest gap in this corpus.

## 4.2 mf K=2 — the one cell that beats K=20 on two of three stages

`mf K=2 T0.5 + dpcc-t` is legal (1.000 / 0.00), has the best legal MIN (0.0175), and runs at
**43.0 ms — 6.4× cheaper than K=20** (p = 0.0039) on **2 NFE and 4 NLP solves** against 23 and 16.
Paired against K=20 `hardflow_sls-r`: constraints bit-identical (10/10 ties), distance +0.0391 m at
**4/4, p = 1.000**.

But it loses median (0.2207 vs 0.1967), mean, MAX and untouched (3/10 vs 2/10), and unguided it is
near the bottom of the field (0.812×) — **its quality comes from the projector and its candidate
selection, not from the model.** The tell: `post_processing` at K=2 uses the same 4 candidates
without `-t` selection and reaches only 0.900 / 8.40.

**This reproduces the 2026-08-29 funnel champion** (MeanFlow K=2 + `dpcc-t`, 1.00, 42 ms) on a
different batch. It is a real cell and it deserves its own head-to-head against K=20 on seeds — but
it is a *different question* from K=10 vs K=20 and is not mixed into that verdict here.

⚠️ **Arm choice and K choice are not separable.** At fixed K, HardFlow beats the DPCC projector.
Across K, DPCC at K=2 beats HardFlow at K=20 on cost. Quoting either comparison alone overstates it.

## 4.3 The geo-free reference — what does projection *itself* cost in distance?

`geo_free` strips the entire geometric group (workspace bounds + halfspaces + obstacles) from the
projector, leaving dynamics + action bounds. It is the clean control for "what does projecting cost,
before geometry is involved". Distance penalty against each cell's own unguided arm, tightened:

| cell | unguided | `geo_free` | `hardflow_sls-r` / `-t` (full geometry) |
|---|---|---|---|
| mf K=10 | 0.1992 | **+0.0427** | **+0.0373** |
| mf K=20 | 0.1431 | **+0.0642** | **+0.0677** |
| mf K=2 | 0.3230 | +0.0578 | −0.0627 |

**The geometry is not what costs distance.** Stripping every obstacle and halfspace leaves the
penalty essentially unchanged (K=10: 0.0427 vs 0.0373; K=20: 0.0642 vs 0.0677). The ~0.04–0.07 m
cost is the **dynamics + bounds projection itself**. Any future attempt to buy accuracy back by
loosening obstacle margins is therefore aimed at the wrong term.

Two side observations from the same ablation family, both at K=10, tightened:
`geo_free-model_free` (action bounds alone) **improves** distance to 0.1415 and lifts zero-violation
0.100 → 0.500 for +22 ms; `bounds_free` (no action bounds) reaches 0.700 zero-violation but costs
**557.5 ms**. The action-bound rows are cheap and carry more of the constraint benefit than their
share of the cost — worth a dedicated ablation.

---

# Next actions

1. **Seeds on the unguided arm at K=10 and K=20.** Stage 1 is 5/4 and Stage 2's distance gap is
   p = 0.289. The whole verdict rests on the direction of two ties. Arm A is cheap (90 / 173 ms).
2. **Tightened geometry for `fm` K=20 and `diffusion` K=20/K=100.** Three cells cannot enter Stage 2
   at all without it. This is the binding gap, not more engines.
3. **`mf K=2 + dpcc-t` vs `mf K=20 + hardflow_sls-r` on seeds**, as its own question. It is the
   cheapest experiment on the board (43 ms/replan).
4. **Sweep the `-t` selection across K.** The K=2 result came from candidate selection, not from
   projection alone. `-t` is the variable that has never been swept.
5. **Ablate the action-bound constraint family** (§4.3) — it looks like the cheap half of the
   constraint benefit.
6. **Never quote a distance without its untouched count.** `dpcc-t` at K=10 reaches a moved-only mean
   of 0.0210 m on the 4/10 contexts where it acted; as a headline that number is a lie.

---

## Provenance

| item | value |
|---|---|
| Data | `temp/0109/batch_va2_20260902_114841/per_rollout_detail.csv` (12 222 rows) |
| Candidates | 18 (mf K10 T0.4) · 19 (mf K20 T0.2) · 20 (mf K2) · 16 (mf K100 T0.1) · 12 (fm K20 T0.2) · 13 (fm K20 T0.5) · 10 (diffusion K20) · 9 (diffusion K100) — all `filmv1`, all `mpc4` |
| Contexts | 10, shared by every cell; every n=30 cell restricted to them. Fingerprint `(box_init_xy, target_xy)` @ 1e-4 (`context_index` is blank in the CSV) |
| Geo | Stage 2 and all constraint claims on `combined_5-tightened`; untightened shown only in §4.1 |
| Constraint families | `geo_free` = no geo_bounds/halfspace/obstacles · `bounds_free` = no action bounds · `model_free` = no dynamics (`eval_mix_visual_aligning.py:232-292`) |
| Budget | arm C NFE = `K + n_active − 1`; NLP solves = `n_active` × 4 for arms B and C alike |
| Tests | exact two-sided sign test · exact paired sign-flip permutation test (2ⁿ enumeration, stdlib only) |
| Not used | `mean_dist_per_rollout`, `n_success` (0.000 everywhere) |
