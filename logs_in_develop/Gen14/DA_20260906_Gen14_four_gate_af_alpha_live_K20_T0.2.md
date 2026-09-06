# DA — Gen14 · **α-Flow with α actually on, at K=20 / T=0.2** · the same four gates

**Drop** `temp/0609/I/` · **Batch** `batch_va2_20260906_125603` · **Task** `aligning-d3il-visual`
**New jobs** `25416` (α_end=0.05) and `25417` (α_end=0.2) — `af`, K=20, T=0.2, arms A+B — **both complete**, `GIT REV 8648c41`
**Protocol** seed 6 · **the same 10 paired contexts in every row** · train split · `mpc4` fan · `filmv1` · `unet` · checkpoint `latest`
**Companions** `DA_20260903_Gen14_four_gate_fm_vs_mf_K20_T0.2.md` (same ladder, `fm`) ·
`DA_20260902_Gen14_three_stage_funnel_K10_vs_K20.md` (the K verdict) ·
`../Gen3v7_AlphaFlow/Study/REPORT_20260830_af_unet_vs_sit_avoiding_root_cause.md` (why α was never on)

**The question:** every `af` row published before this drop was trained with `af_alpha_end = 0.0`,
which snaps α to exactly zero and routes the loss into the MeanFlow body — *MeanFlow wearing an
α-Flow folder name*. These two jobs are the first V_A evaluations where α is provably non-zero at
the deployed checkpoint. **With α actually on, does `af` earn a place in the V_A comparison?**

> 📌 **This closes the question this whole line of work opened with** — *"why does the SiT bone of
> alphaflow work but U-Net not, when MF with U-Net works very well?"* The answer turned out to be
> that no α-Flow had ever run. Now one has, on the U-Net, at the flagship operating point. Unlike
> `fm`, it is **not** eliminated — but it does not displace `mf` either.

## Ground rules

1. **Four gates, in this order: does it run → does the box move → does more NFE help → does it
   survive projection.** A gate is only consulted when the one above it passes. A single 🔴 is fatal.
2. **`context_final_xy_dist`**, metres, box→target. Mean initial distance **0.4530 m**, identical
   across all four cells (verified, not assumed).
   🪤 `mean_dist_per_rollout` is never used — it is `0.5*(pos_dist_3D + rot_err/π)`, not a distance.
3. **Constraint claims are made on the `combined_5-tightened` geometry only.** On untightened
   `combined_5` the executed-violation check is stricter than the set handed to the projector, so the
   arms rank incoherently and no projector claim is valid there. Untightened numbers appear only as
   a robustness check.
4. ⚠️ **Arm C (HardFlow) was OFF in both new jobs** — `[ eval ] arm C (HardFlow) OFF — set
   HFFM_VARIANTS to enable`. So `af` carries **16 variants**, `mf` and `fm` carry **19**. Gate 4 is
   therefore **incomplete for `af` by construction**, not lost on the merits. This is the single
   biggest gap in the drop and it is Next-action #1.
5. ⚠️ **`n_success` is quoted per cell, never pooled across K.** Pooling `af/afon02` across K=2 and
   K=20 gives a flattering 2.19 % that belongs to neither cell. Per-cell counts are 2–11 events out
   of 320–380 — usable for engine-level direction, never for a per-variant claim.
6. **`untouched`** = box never moved (`final == init` to 1e-6). Reported beside every distance,
   because "did nothing" caps the error at the initial distance and can masquerade as a good mean.
7. ⚠️ **MIN is n = 1.** Best single rollout of ten — a *capability ceiling*, not a performance
   measure. Median carried beside it throughout.
8. **Constraint-ablation families** (`geo_free`, `bounds_free`, `model_free` and their pairs) are
   measurements of what the constraint set costs. They are marked 🚫 and **may never be quoted as a
   result**.

---

# Gate 1 · Does it run, and is α actually on? — ✅ **PASS**

Both jobs completed. The decisive lines are not the exit status but the loader's α report:

```
job 25417                                        job 25416
[ eval ] engine=af | alpha-Flow (Gen3v7) | two_time=True | wraps_unet=True
[ eval loading ] ml_bone = unet (VisualUNetTwoTime / VisualUNet — FiLM)
[ eval loading ] checkpoint = state_100000.pt  (trained to step 100000)
[ eval loading ] alpha(step 100000) = 0.2000    [ eval loading ] alpha(step 100000) = 0.0500
   [schedule sigmoid 1.0 -> 0.2, clamp 0.005]      [schedule sigmoid 1.0 -> 0.05, clamp 0.005]
```

**α = 0.2000 and α = 0.0500 at the evaluated step, both far above the 0.005 clamp.** This is the
first time in the project that a V_A rollout has been produced by the α-Flow objective rather than
by MeanFlow under an α-Flow path. Corroborating provenance: the checkpoint is `state_100000.pt`
(the U6 final-save, not the periodic 80000 save), the backbone is the 4.0 M `unet` — *not* the
9.4 M SiT — so this is an **architecture-matched** row, and `film_mode = v1` is read back from the
train-time `model_config.pkl`, so the backbone cannot silently mismatch the weights.

Corpus shape: `af/afon005` and `af/afon02` each hold **320 rows** (16 variants × 10 contexts × 2
geometries); `mf` and `fm` at the same K/T hold **380** (19 × 10 × 2). Ten contexts, **10 shared with
`mf` and `fm`**, verified by geometry fingerprint. Fully paired apart from the three missing arm-C
variants.

---

# Gate 2 · Does the box move? — ✅ **PASS, with a defect `mf` does not have**

Tightened geometry, unguided arm (`diffuser`), n = 10 each:

| cell | MIN | p25 | med | mean | MAX | untouched | viol | sat | max_phys_err | ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `af`/α=0.2   | **0.0110** | 0.0756 | 0.4232 | 0.3435 | 0.6970 | 3/10 | 127.40 | 0.645 | 0.894 | 170.8 |
| `af`/α=0.05  | 0.0460 | 0.1707 | 0.4495 | 0.4357 | **0.8294** | 1/10 | 70.60 | 0.803 | 0.655 | 169.4 |
| `mf`         | 0.0278 | **0.0506** | **0.0902** | **0.1431** | **0.4524** | 1/10 | **31.60** | **0.921** | **0.400** | 172.5 |
| `fm` *(excluded)* | 0.0394 | 0.2271 | 0.4131 | 0.3588 | 0.4975 | 5/10 | 109.80 | 0.605 | 0.986 | 260.0 |

α-Flow moves the box — that is the gate, and it passes. It is not the frozen engine `fm` was.
But the shape of the distribution is the finding:

**Outcome partition** (tightened, unguided, 10 rollouts each):

| cell | ended closer | frozen | **pushed the box AWAY** | gap closed, when it engaged |
|---|---|---|---|---|
| `mf`         | **9** | 1 | **0** | 75.0 % mean · 94.7 % best |
| `af`/α=0.2   | 5 | 3 | **2** | 58.3 % mean · **97.7 % best** |
| `af`/α=0.05  | 5 | 1 | **4** | 51.6 % mean · 90.5 % best |
| `fm`         | 3 | 5 | 2 | 66.0 % mean · 92.5 % best |

> **MeanFlow never ends a rollout further from the target than it started. α-Flow does, on 2 of 10
> (α=0.2) and 4 of 10 (α=0.05).** `af`/α=0.05 reaches 0.8294 m from a 0.4149 m start — it drove the
> box roughly twice as far away as it began. That is a categorical difference in failure mode, not a
> difference of degree, and no summary statistic that averages over it should be trusted.

Paired on the 10 shared contexts (`af` − `mf`, tightened, unguided):

| pair | Δ distance | wins/losses | sign p | perm p |
|---|---|---|---|---|
| α=0.2 − `mf`   | **+0.2004** | 6/3 | 0.5078 | 0.0469 |
| α=0.05 − `mf`  | **+0.2926** | 7/2 | 0.1797 | 0.0430 |
| α=0.2 − α=0.05 | −0.0923 | 4/5 | 1.0000 | 0.3750 |
| α=0.2 − `fm`   | −0.0153 | 2/6 | 0.2891 | 0.8672 |

**This is materially weaker evidence than the case against `fm`, and it must be reported that way.**
`fm` lost to `mf` 9/0 with sign p = 0.0039. α-Flow loses on the mean but the sign test is
**not significant** for either tag; only the magnitude-weighted permutation test crosses 0.05, and
it does so because of two or three large losses, not a consistent deficit. **α-Flow is behind
MeanFlow. It is not beaten by MeanFlow in the way `fm` was.**

### 2.1 Per-context — the variance, not the mean, is the story

| ctx | init | α=0.2 | α=0.05 | `mf` | `fm` | closest |
|---|---|---|---|---|---|---|
| 0 | 0.4345 | 0.4345 ❄ | 0.8047 | **0.0648** | 0.4345 ❄ | `mf` |
| 1 | 0.4741 | **0.0110** | 0.4466 | 0.0506 | 0.2271 | α=0.2 |
| 2 | 0.4104 | 0.3703 | **0.0754** | 0.2211 | 0.4104 ❄ | α=0.05 |
| 3 | 0.4066 | **0.0756** | 0.3399 | 0.2617 | 0.4066 ❄ | α=0.2 |
| 4 | 0.5223 | 0.6970 ⬆ | 0.6921 ⬆ | **0.0278** | 0.0394 | `mf` |
| 5 | 0.4591 | **0.0384** | 0.5000 ⬆ | 0.0817 | 0.4975 ⬆ | α=0.2 |
| 6 | 0.4524 | 0.4524 ❄ | 0.4524 ❄ | 0.4524 ❄ | 0.4524 ❄ | — all frozen |
| 7 | 0.4842 | 0.4315 | **0.0460** | 0.0986 | 0.4842 ❄ | α=0.05 |
| 8 | 0.4149 | 0.4149 ❄ | 0.8294 ⬆ | **0.0288** | 0.4158 | `mf` |
| 9 | 0.4713 | 0.5091 ⬆ | 0.1707 | **0.1433** | 0.2200 | `mf` |

❄ frozen · ⬆ ended further away than it started

**On per-context wins α=0.2 ties `mf` 4–4** (ctx 6 is a null for everyone — the known hard context).
α-Flow produces the two closest approaches in the table (0.0110, 0.0384) *and* three of the four
push-aways. It is a **high-variance policy**: when it engages it engages well, and when it fails it
fails in a direction MeanFlow never fails in. The median hides both halves.

**Gate 2 verdict: PASS.** The box moves, on 5 of 10 with either α. Carried forward as a defect:
the push-away mode, which is unique to `af` among the U-Net engines at this operating point.

---

# Gate 3 · Does more NFE help? — 🔴 **FAIL**

Unguided, tightened, `filmv1` throughout (MeanFlow K=2 filtered to `filmv1` — the corpus holds four
K=2 MeanFlow checkpoints and pooling them is what makes MeanFlow look flat):

| cell | K | MIN | med | mean | untouched | sat | ms | success |
|---|---|---|---|---|---|---|---|---|
| `af`/α=0.2 | 2 | 0.0098 | 0.4247 | 0.3470 | 3/10 | 0.688 | 24.5 | **11/320 = 3.44 %** |
| `af`/α=0.2 | **20** | 0.0110 | 0.4232 | 0.3435 | 3/10 | 0.645 | **170.8** | 3/320 = 0.94 % |
| `af`/α=0.05 | 2 | 0.0391 | 0.3783 | 0.3052 | 1/10 | 0.911 | 24.5 | 2/320 = 0.62 % |
| `af`/α=0.05 | **20** | 0.0460 | 0.4495 | 0.4357 | 1/10 | 0.803 | **169.4** | 3/320 = 0.94 % |
| `mf` (filmv1) | 2 | 0.0095 | 0.2157 | 0.2667 | 7/30 | — | 23.7 | — |
| `mf` | 10 | 0.0082 | 0.1421 | 0.1992 | 2/10 | 0.887 | 89.7 | 7/380 = 1.84 % |
| `mf` | **20** | 0.0278 | **0.0902** | **0.1431** | 1/10 | 0.921 | 172.5 | 2/380 = 0.53 % |

**Nineteen extra function evaluations buy α-Flow nothing.** α=0.2 is flat (median 0.4247 → 0.4232,
MIN 0.0098 → 0.0110); α=0.05 is *worse* on every column (median +0.0712, mean +0.1305, sat −0.108).
Both pay **7.0×** the latency for it, and α=0.2's success rate falls from 3.44 % to 0.94 %.

Contrast MeanFlow over the same ladder: **0.2157 → 0.1421 → 0.0902**, monotone, with sat rising
0.887 → 0.921. `fm`, from the companion DA: 0.4085 → 0.3725 → 0.4004, flat-to-worse.

> **MeanFlow is the only engine in the V_A corpus whose accuracy improves with NFE.** Flow Matching
> does not. α-Flow, with α provably live, does not either. Whatever MeanFlow's two-time
> parameterisation is buying, α-Flow's bootstrapped target does not inherit it — and the α anneal
> does not recover it at either 0.05 or 0.2.

**The T caveat, and why it does not bite.** The K=2 cells were evaluated at T=0.5 and the K=20 cells
at T=0.2, so two knobs move between the columns. T is the projector's `diffusion_timestep_threshold`;
the `diffuser` arm invokes no projector, so T is inert on every row of this table. The ladder is
single-knob in K for the unguided arm. It would **not** be single-knob for any Gate 4 row, and no
cross-K projector comparison is made below.

### 3.1 The latency model — α-Flow is the cheapest engine per NFE

Two-point fits from the table above:

```
af (α=0.2)   ms ≈  8.24 + 8.13·K      ← 8.13 ms per NFE
af (α=0.05)  ms ≈  8.40 + 8.05·K      ← 8.05
mf           ms ≈  7.46 + 9.17·K      ← 9.17
fm           ms ≈ 10.85 + 14.15·K     ← 14.15
```

`af` and `mf` both implement `_encode_once` — the dual-cam window is encoded once per replan and the
latent reused across all K ODE steps — so both scale at ~8–9 ms/NFE. `fm` (`VisualFlowMatching`) has
no such method and re-runs the ResNet every ODE step, which is the whole of its **1.54×** per-NFE
penalty. α-Flow is marginally *cheaper* per NFE than MeanFlow (8.13 vs 9.17), and at K=20 it is
**1.7 ms faster per replan step, 0/9 paired, sign p = 0.0039** — a real but tiny win. α=0.2 costs
1.46 ms/step more than α=0.05 (9/0, p = 0.0039): the price of keeping α off the clamp.

**Gate 3 verdict: FAIL.** α-Flow does not scale with NFE, so its operating point is K=2, not K=20 —
which is where its 3.44 % success rate lives, and where it costs 24.5 ms.

---

# Gate 4 · Does projection rescue it? — 🟡 **INCOMPLETE — arm C was never run**

MIN-ranked, tightened geometry. 🚫 = constraint-ablated, illegal to quote.

**`af` / α=0.2 — 16 variants**

| variant | | MIN | med | untouched | zero-viol | viol | sat | ms |
|---|---|---|---|---|---|---|---|---|
| `gradient` | | **0.0052** | 0.4435 | 2/10 | 0.40 | 118.50 | 0.695 | 177.1 |
| `geo_free-model_free` | 🚫 | 0.0103 | 0.4247 | 3/10 | 0.30 | 126.00 | 0.661 | 197.3 |
| `diffuser` | | 0.0110 | 0.4232 | 3/10 | 0.20 | 127.40 | 0.645 | 170.8 |
| `dpcc-c-dt0p25` | | 0.0223 | 0.4247 | 3/10 | 0.60 | 75.20 | 0.812 | 442.7 |
| `dpcc-t` | | 0.0592 | 0.4216 | 3/10 | 0.60 | 23.20 | 0.942 | 320.5 |
| `dpcc-c-dt2p0` | | 0.0905 | 0.4435 | **8/10** | 0.30 | 10.90 | 0.972 | 310.1 |
| `dpcc-c-dt0p5` | | 0.0915 | 0.3351 | 3/10 | **0.70** | 33.10 | 0.917 | 440.2 |
| `post_processing` | | 0.1424 | 0.4617 | 4/10 | **0.80** | 10.80 | 0.973 | 201.2 |
| `dpcc-r` | | 0.1454 | 0.4287 | 2/10 | 0.40 | 44.60 | 0.889 | 415.8 |
| `dpcc-c` | | 0.1518 | 0.4435 | 5/10 | 0.50 | 10.70 | 0.973 | 311.4 |
| `dpcc-c-dt4p0` | | 0.4066 | 0.4557 | **10/10** | 0.90 | 7.20 | 0.982 | 253.7 |

**`af` / α=0.05 — 16 variants (top of the ranking)**

| variant | | MIN | med | untouched | zero-viol | viol | sat | ms |
|---|---|---|---|---|---|---|---|---|
| `dpcc-r` | | **0.0068** | 0.3444 | 2/10 | 0.70 | **1.40** | **0.995** | 303.4 |
| `dpcc-c-dt0p5` | | 0.0190 | 0.4840 | 1/10 | 0.70 | 33.40 | 0.916 | 470.6 |
| `dpcc-c-dt0p25` | | 0.0191 | 0.3630 | 2/10 | 0.70 | 20.80 | 0.945 | 578.6 |
| `dpcc-t` | | 0.0262 | 0.4557 | 5/10 | 0.50 | 9.00 | 0.977 | 359.5 |
| `diffuser` | | 0.0460 | 0.4495 | 1/10 | 0.50 | 70.60 | 0.803 | 169.4 |
| `dpcc-c-dt4p0` | | 0.4066 | 0.4557 | **10/10** | **1.00** | **0.00** | **1.000** | 236.2 |

Three things fall out.

**4.1 Projection helps α-Flow, unlike `fm`.** `af`/α=0.05 `dpcc-r` reaches **MIN 0.0068 at
zero-violation 0.70, 1.40 violations, sat 0.995** — it beats its own unguided MIN (0.0460) by 6.8×
while *also* improving constraints, and it is not frozen (2/10). `af`/α=0.2 `gradient` reaches
**MIN 0.0052**, the lowest legal MIN anywhere in the K=20/T=0.2 field, `mf` included. In the `fm`
DA the equivalent finding was the opposite: no `fm` arm beat its own unguided MIN and the ceiling
was zv 0.700. **α-Flow is projectable; Flow Matching was not.**

**4.2 But no `af` arm reaches zero-violation 1.000 without freezing.** The only 1.000 in either
table is `dpcc-c-dt4p0` at **10/10 untouched** — perfect safety by never moving, the degenerate
corner. Compare `mf` at the identical operating point, where **`hardflow_sls-r` reaches MIN 0.0220,
zv 1.000, 0.00 violations at 2/10 untouched** and `hardflow_sls-c` MIN 0.0241, zv 1.000 at 4/10 —
genuinely safe *and* genuinely moving.

**`af` has no arm-C row to put against those, because arm C was switched off.** This is the reason
Gate 4 cannot be closed. It is a coverage gap in the run configuration, not a result. Nothing in
this section may be read as "α-Flow cannot be made safe."

**4.3 `dpcc-c-dt4p0` is 10/10 untouched in all three cells** — `af`/α=0.2, `af`/α=0.05 and `mf`
alike, all at MIN = 0.4066, median 0.4557. The `dt4p0` variant freezes every rollout regardless of
engine. This reproduces the `fm` DA's finding on a third and fourth engine and settles it: **the
`dt2p0`/`dt4p0` arms are a projector pathology, not an engine one.** `dt2p0` is 8/10 frozen in all
three cells.

**Gate 4 verdict: INCOMPLETE.** Arms A+B say α-Flow responds to projection better than `fm` did.
Arm C — the one that decides the safety claim — was never executed on `af`.

---

# 🏆 Verdict — **`af` does not beat `mf` on any axis that survives scrutiny. Close the V_A α-Flow line.**

| gate | `fm` (09-03) | `af` α=0.05 | `af` α=0.2 |
|---|---|---|---|
| 1 · runs, α live | ✅ ran | ✅ **α = 0.0500** | ✅ **α = 0.2000** |
| 2 · box moves | 🔴 5/10 frozen, 9/0 to `mf` | ✅ 5/10 closer, **4/10 pushed away** | ✅ 5/10 closer, 2/10 pushed away, **ties `mf` 4–4 per context** |
| 3 · NFE helps | 🔴 worse at K=100 | 🔴 **worse** at K=20 | 🔴 **flat**, success 3.44 % → 0.94 % |
| 4 · survives projection | 🔴 no legal arm | 🟡 `dpcc-r` 0.0068 / zv 0.70 — **arm C never run** | 🟡 `gradient` 0.0052 / zv 0.40 — **arm C never run** |
| outcome | **excluded** | **secondary, K=2** | **secondary, K=2** |

α-Flow is not the engine `fm` was — it moves the box and it responds to projection. But it does not
beat MeanFlow anywhere. **Two axes looked like α-Flow wins on first pass, and neither survives §5.5:**

| axis | first read | after correction |
|---|---|---|
| strict success at K=2 | `af` 3.44 % > `mf` 2.54 % | 🔴 **`af`'s 11 successes come from 2 contexts; `mf`'s 29 from 7.** Fisher p = 0.4370 even on the inflated rollout denominator. On the honest unit — contexts — **`mf` wins 7 to 2.** |
| lowest legal MIN | `af` 0.0052 < `mf` 0.0089 | 🟡 MIN is n = 1 by Ground rule 7 — a capability ceiling, not a performance measure. A 0.0037 m edge on one rollout of ten. |

What is left is `ms`/NFE — 8.13 vs 9.17, an 11 % latency edge on an engine that is **4.7× worse on
median distance** (0.4232 vs 0.0902 at K=20). That is not a trade-off; it is a dominated arm with one
cheap column. **`af` is not Pareto non-dominated against `mf` in V_A.** `mf` wins on median, mean,
constraint satisfaction, violations, max physical error, NFE scaling, contexts-ever-solved, and the
absence of the push-away failure mode.

**This is a clean negative result, and it is worth exactly that.** With α provably live at two
settings, on the architecture-matched 4.0 M U-Net, at the flagship operating point, α-Flow does not
improve on MeanFlow in visual aligning. The question that opened this line of work
— *"is α-Flow better, or was the earlier SiT-vs-U-Net gap just architecture?"* — is now answered:
**it was architecture, and α buys nothing back.**

**Load-bearing weakness:** one seed, ten contexts, and *two* structural gaps rather than one. Arm C
was never run, so the safety half of Gate 4 is empty. And the α sweep has exactly two points, 0.05
and 0.2, which disagree with each other on almost every axis — α=0.05 pushes the box away twice as
often but projects far better; α=0.2 freezes more but wins more contexts. With n = 2 there is no way
to tell a trend from noise. **Coverage, not seeds.**

---

# 5 · The rest of the field, buried where it belongs

**5.1 The `af` K=20 cells did not change any standing V_A result.** The flagship claim (HardFlow ≻
DPCC on latency at K=10, 206.3 vs 491.8 ms, 26/1, p < 0.001) is untouched — it is an `mf` result and
`af` contributed no arm-C rows. The `K*` bracket `2 < K* ≤ 10` is untouched for `mf`; α-Flow's own
optimum now looks like **K = 2**, which is a *different* bracket and should not be pooled with it.
The funnel's binding gap is unchanged at 2 cells (`diffusion` K=20 and K=100 on tightened geometry).

**5.2 Engine ledger for V_A after this drop.**

| engine | status | operating point | basis |
|---|---|---|---|
| `mf` | **headline** | K=10–20 | monotone in NFE; only engine with zv 1.000 unfrozen |
| `af` | **closed — dominated by `mf`** | — | flat in NFE; 2 contexts ever solved vs `mf`'s 7; unique push-away mode |
| `fm` | **excluded** | — | 09-03 DA, four gates, three 🔴 |
| `diffusion` | **blocked** | — | tightened geometry still missing at K=20/K=100 |

**5.3 The α=0.05 vs α=0.2 disagreement, for the record.** α=0.05 is better on violations at K=20
(70.60 vs 127.40), better on sat (0.803 vs 0.645), better on max physical error (0.655 vs 0.894),
and far better under projection (`dpcc-r` 0.0068 at zv 0.70 vs α=0.2's best legal safe arm at zv
0.40). α=0.2 is better on distance (median 0.4232 vs 0.4495, mean 0.3435 vs 0.4357), on push-aways
(2 vs 4) and on success (3.44 % vs 0.62 % at K=2). Paired, the two are **statistically
indistinguishable on distance** (4/5, sign p = 1.0000, perm p = 0.3750) and differ only in latency
(+1.46 ms for α=0.2, 9/0, p = 0.0039). **No α recommendation is supportable from two points.**

**5.4 Untightened geometry agrees.** Every Gate 2 and Gate 3 ordering above reproduces on
untightened `combined_5` (α=0.2 median 0.3897, α=0.05 0.4488, `mf` 0.0741, `fm` 0.4085), with
`af` − `mf` on distance strengthening slightly for α=0.05 (9/1, sign p = 0.0215, perm p = 0.0078).
No projector claim is made there, per Ground rule 3.

**5.5 The success-rate edge is one lucky context.** At K=2 with the checkpoint held to `filmv1`,
`af`/α=0.2 scores 11/320 (3.44 %) against `mf`'s 29/1140 (2.54 %) — a 1.35× edge that does not
survive either correction:

- **Fisher exact on the rollout denominator: p = 0.4370.** Not significant even before the
  denominator is fixed.
- **The rollout denominator is pseudo-replicated ~32×.** 320 rows = 16 variants × 10 contexts × 2
  geometries; there are only **10 independent contexts**. Counting distinct contexts that were ever
  solved: **`af` = 2 of 10** (9 of its 11 successes are the *same* context re-solved across variants),
  **`mf` = 7 of 10**. The direction reverses, decisively.

This is the metric α-Flow looked strongest on, and it is the metric where it loses hardest once the
unit of analysis is right. No α-Flow success claim is supportable in V_A.

---

# Next actions

1. 🛑 **Do NOT run arm C (HF-SLSQP) on `af`.** It was Next-action #1 in the first draft of this DA;
   §5.5 and the low-K guard together remove the case for it. Two independent blockers:
   - α-Flow's only tolerable operating point is **K=2**, and at `A=0.5` the HardFlow guard reports
     `n_genuine=0 [DEGENERATE]` for K≤2 — the arm runs **no HardFlow arithmetic**, it is
     `Π_S(Euler sample)` ≡ DPCC modulo solver. Genuine needs K≥3, attributable K≥5.
   - At K≥5, where arm C *would* be genuine, `af` is already flat-to-worse than `mf` on distance.
     Proving a dominated arm can be made safe does not make it competitive.

   Arm C on `af` is only worth revisiting if some *other* result revives α-Flow first.
2. 🛑 **Do not extend the α sweep.** α ∈ {0.05, 0.2} disagree with each other on nearly every axis
   and agree on the one that matters: neither beats `mf`. A denser sweep buys resolution on a
   dominated arm.
3. **Write the negative result up as a result.** "α-Flow with α provably live, architecture-matched
   on the 4.0 M U-Net, does not improve on MeanFlow in visual aligning, and does not scale with NFE"
   is a closed question and belongs in the thesis as one. It also retires the standing worry that
   every prior `af` row was mislabelled MeanFlow — that worry is now resolved *and* moot.
4. **Diagnose the push-away mode.** 2/10 (α=0.2) and 4/10 (α=0.05) rollouts end further from the
   target than they started; `mf` never does. Inspect `diag_first_replan.txt` for these contexts —
   the sign of the velocity field at τ→1 is the first suspect, and it connects directly to the
   `DA_20260901_AF_UNet_alpha_clamp_T1_negative.md` finding.
5. **Seeds 7–10 for `af` at K=2.** Every number here is seed 6.
6. **Do not re-open `fm`.** Nothing in this drop bears on it.
7. *(carried)* Tightened geometry for `diffusion` K=20/K=100 — still the funnel's remaining 2 cells.

---

## Provenance

| item | value |
|---|---|
| drop | `temp/0609/I/` |
| batch | `batch_va2_20260906_125603` — 635 run-config rows, 13,722 rollout rows |
| new jobs | `25416` (α_end=0.05), `25417` (α_end=0.2) — both `Job completed successfully` |
| git rev | `8648c41` (Gen15 U7) — U6 `ca0eb314` and U12 `ba05cb7c` are ancestors |
| engine | `af` · alpha-Flow (Gen3v7) · `two_time=True` · `wraps_unet=True` |
| checkpoint | `state_100000.pt`, selector `latest` (U12 `MIX_EPOCH`) |
| α at eval | **0.0500** / **0.2000**, sigmoid `1.0 → α_end` over 100000 steps, clamp 0.005 |
| backbone | `unet` (VisualUNetTwoTime, 4.0 M) — architecture-matched to `mf`/`fm` |
| film | `v1`, read back from train-time `model_config.pkl` |
| K / T | 20 / 0.2 → 4 projector calls per replan |
| arm C | **OFF** — `set HFFM_VARIANTS to enable` |
| geometry | `combined_5` and `combined_5-tightened` |
| seed / contexts | 6 / 10, shared with `mf` and `fm` by geometry fingerprint |
| variants | 16 (`af`) vs 19 (`mf`, `fm`) |
| init distance | 0.4530 m mean, identical across all four cells |
| statistics | exact two-sided sign test; exact paired sign-flip permutation test (stdlib only) |
