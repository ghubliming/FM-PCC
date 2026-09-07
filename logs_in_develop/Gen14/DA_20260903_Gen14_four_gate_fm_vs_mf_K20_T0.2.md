# DA — Gen14 · **`fm` vs `mf` at K=20 / T=0.2** · four gates: runs → moves → scales → survives projection

**Drop** `temp/0309/` · **Batch** `batch_va2_20260903_154740` · **Task** `aligning-d3il-visual`
**New job** `25312` — `fm`, K=20, T=0.2, arms A+B+C — **complete 38/38**, `GIT REV c721f7d`
**Protocol** seed 6 · **the same 10 paired contexts in every row** · train split · `mpc4` fan · `filmv1`
**Companions** `DA_20260902_Gen14_three_stage_funnel_K10_vs_K20.md` (the K verdict) ·
`DA_20260902_Gen14_K10_T0.4_mf_and_fm_resubmit.md` (cost mechanism, the crash this run fixes)

**The question:** Fix_11 unblocked arm C on `engine=fm`, and this run is the first tightened-geometry
`fm` cell in the corpus. **Does `fm` now earn a place in the V_A comparison?**

> 📌 **This run delivers Next-action #2 of the funnel DA** — *"Tightened geometry for `fm` K=20 …
> Three cells cannot enter Stage 2 at all without it. This is the binding gap, not more engines."*
> `fm` can now be scored on constraints. It enters the funnel and fails it at every gate below the
> first.

## Ground rules

1. **Four gates, in this order: does it run → does the box move → does more NFE help → does it
   survive projection.** A gate is only consulted when the one above it passes. A single 🔴 is fatal.
2. **`context_final_xy_dist`**, metres, box→target. Mean initial distance **0.4530 m**.
   🪤 `mean_dist_per_rollout` is never used — it is `0.5*(pos_dist_3D + rot_err/π)`, not a distance.
3. **Constraint claims are made on the `combined_5-tightened` geometry only.** On untightened
   `combined_5` the executed-violation check is stricter than the set handed to the projector, so the
   arms rank incoherently and no projector claim is valid there.
4. ⚠️ **`n_success` is NOT 0.000 in every cell.** *This supersedes Ground rule 4 of the funnel DA and
   the same claim in the 09-01 DA.* It is 0.000 for `fm` and `diffusion`; `mf` runs 0.5–2.4 % and `af`
   0.5–4.9 %. Still far too underpowered for any per-variant success claim — but the **engine-level**
   count is now usable, and it is used at Gate 2. Full correction in §5.3.
5. **`untouched`** = box never moved (`final == init` to 1e-6). Reported beside every distance,
   because "did nothing" caps the error at the initial distance and can masquerade as a good mean.
6. ⚠️ **MIN is n = 1.** Best single rollout of ten — a *capability ceiling*, not a performance
   measure. It leads because it answers "can this cell reach the target at all", which is the prior
   question. Median carried beside it throughout.

---

# Gate 1 · Does it run? — ✅ **PASS**

```
JOB ID:    25312          NODE: i6-gpu-1
GIT REV:   c721f7d                       ← Fix_11 (159a0318) is an ancestor
[ eval ] engine=fm   seeds='6'   film_mode=v1   ml_bone=unet
[ eval ] NFE override: flow_steps_v3 = 20
[ eval ] projection threshold sweep: T = 0.2  ->  4 projector call(s)/replan
[ eval ] arm C ENABLED: HFFM_VARIANTS='hardflow_new-r hardflow_new-c hardflow_new-t'
...
[ eval ] >>> item 38/38: geo=combined_5-tightened  variant=hardflow_new-t  tightened=True
Job completed successfully.
```

All 38 items. Each of the six arm-C items (17–19 untightened, 36–38 tightened) printed
`[hardflow][NLP-BACKEND] slsqp … dof=66 reg_scale=1.0` and ran to completion.

**This is the first run in the project where the HardFlow in-loop NLP executes on `engine=fm`.** Job
25274 died at item 17/38 on `AttributeError: 'VisualFlowMatching' object has no attribute
'_encode_once'`; 25312 is the same command on a current checkout and item 17 now runs. The corpus
agrees: the `fm` K=20/T=0.2 cell holds **380 rows** (38 × 10), mirroring the `mf` K=20/T=0.2 cell
exactly — one checkpoint per side, 10 contexts each, **10 shared**. A fully paired A/B.

**Gate 1 is about the code, not the model.** The plumbing is fixed and verified in production.
**Proceed to Gate 2.**

---

# Gate 2 · Does the box move? — 🔴 **FAIL**

Unguided (`diffuser`) arm, so the threshold `T` is inert and this is the model alone.

| | **MIN** | p25 | median | mean | MAX | untouched | ms |
|---|---|---|---|---|---|---|---|
| **fm K=20 T0.2** | 0.0394 | 0.2720 | **0.4131** | 0.3588 | 0.4975 | **5/10** | 260.0 |
| mf K=20 T0.2 | **0.0278** | **0.0541** | **0.0902** | **0.1431** | **0.4524** | **1/10** | **172.5** |
| *mf K=10 T0.4* | *0.0082* | *0.0586* | *0.1421* | *0.1992* | *0.4746* | *2/10* | *89.7* |

*(tightened; untightened tells the same story — fm median 0.4085 vs mf 0.0741)*

🔴 **The `fm` median is 0.4131 m. The mean starting distance is 0.4530 m.** The median `fm` rollout
does not move the box in any useful direction. `mf` at the same K leaves 0.0902 m.

**Paired on the same 10 contexts** (positive = `fm` worse):

| metric | Δ mean | fm worse / better | sign p | perm p |
|---|---|---|---|---|
| distance | **+0.216 m** | **9 / 0** | **0.0039** | **0.0039** |
| violations | +78.20 | 8 / 1 | **0.0391** | **0.0195** |
| ms/replan | +87.52 | **9 / 0** | **0.0039** | **0.0039** |

**9/0 on distance and 9/0 on latency is the floor of what n=10 can report.** Untightened agrees and
is stronger on violations (`dpcc-r` +88.10, **10/0, p=0.0020**).

## 2.1 The mechanism is engagement, not precision

| ctx | start | **fm** | **mf** | fm left | mf left |
|---|---|---|---|---|---|
| 0 | 0.4345 | 0.4345 `U` | 0.0648 | 1.00× | 0.15× |
| 1 | 0.4741 | 0.2271 | 0.0506 | 0.48× | 0.11× |
| 2 | 0.4104 | 0.4104 `U` | 0.2211 | 1.00× | 0.54× |
| 3 | 0.4066 | 0.4066 `U` | 0.2617 | 1.00× | 0.64× |
| 4 | 0.5223 | **0.0394** | **0.0278** | 0.08× | 0.05× |
| 5 | 0.4591 | 0.4975 ✗ | 0.0817 | 1.08× | 0.18× |
| 6 | 0.4524 | 0.4524 `U` | 0.4524 `U` | 1.00× | 1.00× |
| 7 | 0.4842 | 0.4842 `U` | 0.0986 | 1.00× | 0.20× |
| 8 | 0.4149 | 0.4158 | 0.0288 | 1.00× | 0.07× |
| 9 | 0.4713 | 0.2200 | 0.1433 | 0.47× | 0.30× |

*`U` = untouched · ✗ = ended further away than it started · "left" = final ÷ start*

**`fm` closes the gap on two contexts. `mf` closes it on nine.** Context 6 is the one both fail
identically — a genuinely hard geometry, and the only honest tie on the board. This is not a
resolution deficit spread thinly across ten rollouts; it is a binary engage/freeze split.

Every other diagnostic points the same way: `sat_rate` **0.719 vs 0.913**, `max_phys_error`
**0.482 vs 0.252**, and by Ground rule 4 the engine-level success count — **`fm` 0 / 712 across all
its cells**, against `mf` 67 / 4365 (1.5 %) and the frozen d3il vision baseline 8 / 3884 (0.2 %).
🔴 **`fm` is the only learned engine in the corpus that fails to reach the frozen baseline.**

**Gate 2 is fatal on its own. Gates 3 and 4 are run anyway, because the interesting question is no
longer "does `fm` lose" but "is it fixable by anything on the eval side".**

---

# Gate 3 · Does more NFE fix it? — 🔴 **FAIL**

Unguided arm, every `fm` cell in the corpus:

| cell | n | init | final | mean/init | untouched | ms/replan |
|---|---|---|---|---|---|---|
| `fm` K=20 T=0.2 | 10 | 0.4530 | 0.3266 | 0.721 | 4/10 | 295.8 |
| `fm` K=20 T=0.5 | 30 | 0.4547 | 0.3373 | 0.742 | 10/30 | 293.9 |
| `fm` K=100 T=0.5 | 30 | 0.4547 | **0.3471** | **0.763** | **12/30** | 1426.0 |

🔴 **5× the NFE makes `fm` slightly worse and 40 % more likely to freeze, at 4.8× the cost.** A
velocity field that improved with integration resolution cannot do that. This is the signature of a
mis-trained field, not an under-resolved ODE.

Contrast `mf`, which behaves as a working sampler should: 0.3623 (K=2) → **0.1730** (K=10) →
**0.0933** (K=20) → 0.1440 (K=100); untouched 5/30 → 1/10 → 0/10 → 0/10.

**This reproduces the 08-26 K-sweep DA §5** — *"K changes nothing, because this arm is not doing the
task at either setting … This needs debugging, not tuning — exclude it from comparisons until it
moves the box."* That was arms A+B only, T=0.5, n=30. Gate 3 adds T=0.2 and the verdict is unchanged.

## 3.1 The latency column is a code artefact, not the flow formulation

Cost model per engine, `diffuser` arm, `combined_5`:

| engine | fit | K=10 | K=20 | K=100 | per-NFE |
|---|---|---|---|---|---|
| `mf` | `ms = 7.46 + 9.171·K` | 99.2 | 190.5 | 924.6 | **9.17 ms** |
| `fm` | `ms = 10.85 + 14.151·K` | — | 293.9 | 1426.0 | **14.15 ms** |

`fm` costs **1.54× more per sampler step**. A single-time engine ought to be *cheaper* than a
two-time one, so this is backwards — and the cause is in the code:

- `VisualMeanFlow` / `VisualAlphaFlow` expose `_encode_once` and pass `visual_latent` (B, 128) to the
  sampler. `visual_mf_diffusion.py:105-110` states the intent outright: *"encoded ONCE, reused across
  all K ODE steps … cuts K ResNet passes per replan down to 1."*
- `VisualFlowMatching.forward` (`visual_fm_diffusion.py:81-103`) has no such method and passes
  `'visual': (bp_imgs, inhand_imgs, obs_seq)` — the **raw images** — so the ResNet encoder re-runs on
  **every one of the K ODE steps**.

At K=20 that is 20 encoder passes per replan against `mf`'s 1. The same structural split is what
Fix_11 had to special-case in `encode_visual_cond`. ⚠️ **Do not report 14.15 ms/NFE as a property of
single-time flow matching.** Fixing it would land `fm` at ~190 ms at K=20 — and would not move a
single number in Gate 2.

---

# Gate 4 · Does projection rescue it? — 🔴 **FAIL**

The first tightened `fm` scoring in the corpus. All 19 variants, MIN-ranked. Unguided MIN = **0.0394**.

| variant | **MIN** | median | mean | untch | zero-viol | viol | ms | beats unguided |
|---|---|---|---|---|---|---|---|---|
| *geo_free-model_free* | *0.0239* ⭐ | *0.4126* | *0.3653* | *6/10* | *0.200* | *94.80* | *285.3* | *4/10* |
| **diffuser** | **0.0394** | 0.4131 | 0.3588 | 5/10 | 0.200 | 109.80 | 260.0 | — |
| *model_free-bounds_free* | *0.0560* | *0.4126* | *0.3703* | *6/10* | *0.200* | *105.60* | *299.6* | *2/10* |
| dpcc-c-dt0p5 | 0.0641 | 0.4247 | 0.3905 | 5/10 | 0.600 | 27.30 | 468.0 | 4/10 |
| gradient | 0.0874 | 0.4263 | 0.3865 | 4/10 | 0.100 | 143.40 | 265.3 | 2/10 |
| **hardflow_sls-r** | **0.1104** | 0.4435 | 0.3979 | 7/10 | 0.600 | 12.00 | 498.0 | 4/10 |
| dpcc-r | 0.1110 | 0.4247 | 0.4089 | 7/10 | 0.500 | 14.30 | 409.7 | 3/10 |
| dpcc-t | 0.1128 | 0.4247 | 0.4111 | 7/10 | 0.600 | 12.80 | 402.0 | 3/10 |
| dpcc-c-dt0p25 | 0.1138 | 0.4247 | 0.3858 | 5/10 | **0.700** | 16.00 | 477.0 | 4/10 |
| dpcc-c | 0.1144 | 0.4247 | 0.4094 | 8/10 | 0.400 | 14.90 | 406.8 | 3/10 |
| post_processing | 0.1149 | 0.4247 | 0.4062 | 7/10 | 0.400 | 14.70 | 292.1 | 3/10 |
| hardflow_sls-t | 0.1188 | 0.4247 | 0.4096 | 8/10 | 0.500 | 7.50 | 452.4 | 3/10 |
| hardflow_sls-c | 0.1306 | 0.4247 | 0.4069 | 7/10 | 0.500 | 10.40 | 502.4 | 3/10 |
| bounds_free | 0.1484 | 0.4247 | 0.4141 | 7/10 | 0.500 | 16.70 | 402.9 | 3/10 |
| *model_free* | *0.2246* | *0.4126* | *0.4032* | *6/10* | *0.200* | *100.60* | *310.4* | *3/10* |
| *geo_free* | *0.2644* | *0.4247* | *0.4192* | *6/10* | *0.100* | *152.70* | *347.7* | *3/10* |
| *geo_free-bounds_free* | *0.2679* | *0.4247* | *0.4259* | *8/10* | *0.100* | *148.90* | *320.1* | *2/10* |
| dpcc-c-dt2p0 | 0.4066 | 0.4557 | 0.4530 | **10/10** | 0.300 | 22.30 | 374.1 | 2/10 |
| dpcc-c-dt4p0 | 0.4066 | 0.4557 | 0.4530 | **10/10** | 0.500 | 10.80 | 339.2 | 2/10 |

*Italic = constraint-ablated (illegal as a configuration; they measure what the constraint set costs).*
⭐ = the only new minimum.

## 4.1 🔴 Not one legal arm exists

**No `fm` arm reaches 1.000 zero-violation. The ceiling is 0.700** (`dpcc-c-dt0p25`), and that arm is
5/10 untouched. For contrast, the same projectors on `mf` at the identical K/T:

| cell · arm | MIN | median | mean | untch | zero-viol | viol | ms | beats unguided |
|---|---|---|---|---|---|---|---|---|
| mf · diffuser | 0.0278 | 0.0902 | 0.1431 | 1/10 | 0.200 | 31.60 | 172.5 | — |
| mf · dpcc-r | 0.0329 | 0.1786 | 0.2001 | 2/10 | 0.900 | 2.70 | 265.9 | 3/10 |
| **mf · hardflow_sls-r** | **0.0220** ⭐ | 0.1967 | 0.2108 | 2/10 | **1.000** | **0.00** | 275.3 | 3/10 |
| **mf · hardflow_sls-c** | **0.0241** ⭐ | 0.3227 | 0.2863 | 4/10 | **1.000** | **0.00** | 282.3 | 2/10 |

**`mf` puts two arms at 1.000 / 0.00 and both set a new minimum below their own unguided run.**
`fm` puts zero arms at 1.000, and its best *legal-ish* MIN (0.1104, `hardflow_sls-r` at 0.600) is
**2.8× worse than its own unguided minimum**. This is Stage 2 of the funnel DA applied to `fm`, and
it is the same failure mode the funnel found in K=10 — *"went in holding a minimum, came out with a
worse one"* — except an order of magnitude more severe and with no legal arm at the end of it.

## 4.2 The one new minimum is illegal, and the two cheapest arms freeze completely

- ⭐ **`geo_free-model_free` sets `fm`'s only new MIN (0.0239)** — and it has **dynamics and geometry
  both stripped**. It is not a configuration; it is a measurement of what the constraint set costs.
  Exactly the pattern the funnel DA flagged at K=20 (`model_free` 0.0075, `geo_free` 0.0147).
- 🔴 **`dpcc-c-dt2p0` and `dpcc-c-dt4p0` are 10/10 untouched** — MIN = median = 0.4066 = the starting
  distance on the easiest context. The projector at a loose `dt` does not filter this plan; it
  cancels it. No other cell in the corpus contains a 10/10-untouched arm.
- ⚠️ **`beats unguided` reads 3–4/10 across the board and means nothing here.** Beating a frozen
  baseline by moving slightly is not a rescue. The funnel DA's rescue count is only interpretable
  when the unguided arm is actually trying.

**Arm C is nevertheless mechanically correct on `fm`.** On the rollouts where the box moves:
197.20 executed violations unguided → **0.00** under `hardflow_sls-r`, at the best engaged-subset
distance of any projected variant (0.2629). ✅ **That is the Fix_11 acceptance criterion, met** — and
the entire extent of what these rows support, at **n = 3**.

---

# 🏆 Verdict — **exclude `fm` from the V_A comparison. Keep the arm-C plumbing.**

| gate | result |
|---|---|
| **1 · runs** | ✅ **PASS** — first arm-C execution on `fm`; 38/38; Fix_11 verified in production |
| **2 · box moves** | 🔴 **FAIL** — median 0.4131 m from a 0.4530 m start; 5/10 untouched; 9/0 paired, p = 0.0039; 0/712 successes |
| **3 · NFE helps** | 🔴 **FAIL** — K=100 is *worse* and freezes 12/30, at 4.8× cost |
| **4 · survives projection** | 🔴 **FAIL** — zero legal arms (ceiling 0.700); best legal-ish MIN 2.8× worse than its own unguided; two arms 10/10 frozen |

**`mf` Pareto-dominates `fm` at matched K=20/T=0.2** — closer, safer *and* faster, simultaneously,
with the sign tests at their n=10 floor. By the project's definition this is not a trade-off and the
word "non-dominated" does not apply.

**What survives, and it is not nothing:** Fix_11 is closed, arm C runs on both engine families, and
the funnel DA's binding gap — *"`fm` K=20 has no tightened rows and cannot enter Stage 2"* — is now
closed. `fm` entered Stage 2 and failed it. **That is a real result: the gap was worth closing, and
closing it removed an engine from the table rather than adding one.**

⚠️ **This does not touch the flagship.** HardFlow ≻ DPCC on latency at K=10 (206.3 vs 491.8 ms,
26/1, p < 0.001) and the funnel DA's *"keep K=20 / T=0.2 with `hardflow_sls-r`"* are both `mf`
results and are unaffected by anything here.

**Load-bearing weakness:** one seed, ten contexts. Gate 2 is 9/0 and Gate 4 is categorical (zero
legal arms), so neither is fragile in the way the funnel's 5/4 ties were — but the *diagnosis* in
Gate 3 (mis-trained field vs bad checkpoint) rests on no training evidence at all. **Diagnostics,
not seeds.**

---

# 5 · The rest of the field, buried where it belongs

## 5.1 The arm-C NLP non-convergence is engine-independent — and that is new information

**6 non-converged solves / 6 arm-C items**, all at **τ = 0.850**:

```
[hardflow][NLP-FAILURE] first non-converged SLSQP solve at tau=0.850. Keeping scipy's last
iterate, which may be INFEASIBLE — the terminal-solve safety guarantee does not hold for this
plan. Further failures are silent; read `nlp_failures` in the run summary for the total.
```

| job | engine | K / T | failures | τ | which call |
|---|---|---|---|---|---|
| 25247 | `mf` | 20 / 0.2 | 6 / 6 items | 0.850 | **#2 of 4** |
| **25312** | **`fm`** | 20 / 0.2 | **6 / 6 items** | **0.850** | **#2 of 4** |
| 25273 | `mf` | 10 / 0.4 | 6 / 6 items | 0.700 | **#2 of 4** |

**Swapping the engine changes nothing.** The failure τ is set by the `(K, T)` window and lands on
call #2 whether the velocity field is single-time or two-time. The cause is in the **NLP conditioning
at that τ**, not in the generative model — which rules out the most obvious hypothesis and is worth
more than the failure itself. Executed violations on the converged `mf` arms are still 0.00, so every
result stands, but the terminal-solve feasibility guarantee is weaker than documented. **Belongs in
the limitations section of the write-up.**

⚠️ Only *first* failures are logged. Totals live in `nlp_failures` in the run summaries and are still
unextracted for all three jobs.

## 5.2 What this changes in the standing V_A ledger

| | before this drop | after |
|---|---|---|
| Engines | `mf` ✅ · `af` ✅ · `fm` ❓ never scored on constraints · `diffusion` ⚠ confounded | `fm` **❌ excluded** |
| Arm C coverage | `mf` K=10, K=20 · `fm` **blocked by a crash** | `fm` K=20 present, **unusable for projector claims** |
| Funnel binding gap | 3 cells cannot enter Stage 2 | **2 cells** (`diffusion` K=20/K=100, `mf` K=100) |
| Flagship claim | HardFlow ≻ DPCC on latency, K=10, p<0.001 | **unchanged** |
| `K*` bracket | `2 < K* ≤ 10` | **unchanged** — no K=5 cell |

## 5.3 ⚠️ Correction — `n_success` is not zero everywhere

*Supersedes Ground rule 4 of the funnel DA and the same sentence in the 09-01 and 09-02 DAs.* The
operative warning stands; the factual premise was wrong. It was read off the per-run
`Success rate: 0.0000` line for the variants then being quoted, and over-generalised.

| cell | strict | rate | | cell | strict | rate |
|---|---|---|---|---|---|---|
| `af` K=100 T=0.5 | 2 / 41 | 4.88 % | | `mf` K=2 T=0.5 | 53 / 3380 | 1.57 % |
| `mf` K=100 T=0.5 | 1 / 41 | 2.44 % | | `af` K=2 T=0.5 | 13 / 2420 | 0.54 % |
| `mf` K=100 T=0.1 | 4 / 184 | 2.17 % | | `mf` K=20 T=0.2 | 2 / 380 | 0.53 % |
| **`mf` K=10 T=0.4** | **7 / 380** | **1.84 %** | | *d3il baseline* | *8 / 3884* | *0.21 %* |
| | | | | **`fm` (all cells)** | **0 / 712** | **0.00 %** |
| | | | | `diffusion` | 0 / 208 | 0.00 % |

Verified against the raw log, job 25273 item 1/38 (`mf`, K=10, `combined_5`, `diffuser`):

```
  - Success status: True        ← rollout 6
Successrate 0.10000000149011612
Success rate:              0.1000
```

The same data is present in the previous batch (`batch_va2_20260902_114841`: 90 strict successes, 7
in `mf` K=10), so this is a reading error, not a pipeline change.

- ✅ **Usable:** the engine-level count. `mf` clears the frozen d3il baseline ~7×; `fm` is the only
  learned engine that does not. Used at Gate 2.
- ❌ **Not usable:** any per-variant success claim. The largest per-variant count anywhere is 2/20,
  on one seed. Every projector claim stays on distance / violations / latency.

---

# Next actions

1. **Seeds 7–10 for arm C on `mf` at K=10 and K=20.** Unchanged as the top priority — the flagship
   rests on n=1, and nothing in this drop touched it. Highest value per GPU-hour on the board.
2. **Stop scheduling `fm` runs on V_A.** Gate 1 is answered and Gates 2–4 are fatal. The GPU-hours
   are better spent on item 1.
3. **Diagnose `fm` before any further `fm` eval** — training curves for `filmv1_Efm`, and compare its
   `t` sampling (`t = 1 − Beta(1.5, 1.0)`, a hard skew toward t≈0) against `mf`'s logit-normal.
   This is the *only* open question that could revive the engine, and Gate 3 cannot answer it from
   eval data.
4. **Tightened geometry for `diffusion` K=20 / K=100.** Two cells still cannot enter Stage 2. This
   run shows the gap is worth closing even when the answer is negative.
5. **K=5 unguided-only `mf` cell** to close the `2 < K* ≤ 10` bracket. ~53 ms/replan by the fitted
   model; no `MIX_PROJ_T`, no `HFFM_VARIANTS`. Cheap.
6. **Extract `nlp_failures` totals** into the arm-C tables (§5.1). Cheap, and converts a caveat into
   a number.
7. *Only if `fm` is revived:* give `VisualFlowMatching` an encoder cache (§3.1). ~294 → ~190 ms at
   K=20. Pointless until Gate 2 passes.

---

## Provenance

| item | value |
|---|---|
| New log | `temp/0309/2026-09-02/14_05_18_eval_mix_visual_aligning_25312.log` — `fm` K=20 T=0.2, 38/38, `GIT REV c721f7d` |
| Cross-check logs | `temp/0109/2026-09-01/15_36_10_..._25273.log` (`mf` K=10) · `temp/3008/2026-08-31/16_32_53_..._25247.log` (`mf` K=20) |
| Data | `temp/0309/batch_va2_20260903_154740/per_rollout_detail.csv` (12 442 rows) |
| Cross-check data | `temp/0109/batch_va2_20260902_114841/per_rollout_detail.csv` (12 222 rows) — §5.3 only |
| Cells | `fm` filmv1 K20 T0.2 (380 rows, 19 variants × 2 geos × 10) vs `mf` filmv1 K20 T0.2 (380 rows), seed 6, `mpc4` |
| Contexts | 10, shared by both cells; every n=30 cell restricted to them. Fingerprint `(box_init_xy, target_xy)` @ 1e-4 (`context_index` is blank in the CSV) |
| Geo | Gates 2 and 4 on `combined_5-tightened`; untightened quoted only where it agrees |
| Constraint families | `geo_free` = no geo_bounds/halfspace/obstacles · `bounds_free` = no action bounds · `model_free` = no dynamics (`eval_mix_visual_aligning.py:232-292`) |
| Code cited | `models/visual_fm_diffusion.py:81-103` · `models/visual_mf_diffusion.py:42-60,105-124` · `sampling/hardflow_projection.py:917-957` (Fix_11, commit `159a0318`) |
| Tests | exact two-sided sign test · exact paired sign-flip permutation test (2ⁿ enumeration, stdlib only) |
| Not used | `mean_dist_per_rollout` · per-variant `n_success` (see §5.3) |
| Not covered | Gen3v7 α-Flow (25290–25293) → `Gen3v7_AlphaFlow/DA/DA_20260903_AF_UNet_alphaflow_ENABLED_seed6_diffuser.md` · UAV-MIX K-sweep (25316–25321) **incomplete**: K=5 missing on both `fix16` arms (25318 hit the wall clock, 25321 has no `JOB END`) |
