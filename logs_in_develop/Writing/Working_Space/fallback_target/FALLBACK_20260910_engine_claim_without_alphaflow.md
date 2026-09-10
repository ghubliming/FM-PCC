# FALLBACK — the engine claim when `af_unet` does not beat `mf`

**Created:** 2026-09-10 · **Type:** fallback target + storytelling plan · **Status:** 🟢 **this branch has fired — treat it as the active plan**
**Scope:** the engine half of the thesis (Goal A). Goals B and C are untouched by this document.
**Governs together with:** [`../TARGET_20260905_thesis_claim_ladder.md`](../TARGET_20260905_thesis_claim_ladder.md) — this file is the discharge of its **§8 kill criterion 1**, not a replacement for it.
**Current draft:** [`../v2/thesis_v2.tex`](../v2/thesis_v2.tex) ([README](../v2/README.md)) — v1/v2 already demote α-Flow; this file supplies the *reason, the mechanism and the narrative* that the draft still marks as deferred.

> 🔴 **What this file is.** A target/plan statement with the minimum evidence needed to justify it.
> The numbers quoted here are pointers into the DAs of record (§9), not new analysis. Never cite
> this file as a source for a number — cite what it points at.

---

## 0. The one-sentence change

> **Before.** `af_unet ≥ mf > fm > diffusion` — a four-rung ladder, α-Flow on top.
> **After.** **`mf` ≫ { `af`, `fm`, `diffusion` }** — one engine separates, three cluster.
> α-Flow stops being a rung and becomes **the auxiliary discussion the engine chapter is built around.**

The thesis headline is now simply: **the average-velocity objective (MeanFlow) is the engine**, on an
identical U-Net, an identical projector and an identical harness. α-Flow is not deleted, not hidden,
and not apologised for — it is the **road we drove down and mapped**, and the map is what turns a
one-engine result into an argument about *why* that engine wins.

Nothing else moves. **Title unchanged. RQ set unchanged. Chapter structure unchanged.** TARGET §8
pre-committed to exactly this, and the commitment is being honoured rather than renegotiated.

---

## 1. Why the fallback fired — four environments, no support anywhere

| environment | regime | what the scene can measure | `af` vs `mf` verdict | strength |
|---|---|---|---|---|
| `avoiding-d3il` | **all-pass** (every arm S&C = 1.00) | cost only | 🟡 **nominal win** — 57.20 vs 60.75 steps at the cheapest S&C = 1.00 cell | Fisher **p = 0.231**; a **wash** (8 vs 9 Pareto cells) once averaged over all six projection rules |
| `aligning-d3il-visual` | **all-fail** (S&C floored → funnel) | plan quality, 0.45 m of headroom | 🔴 **loses** — `≤ mf` everywhere: tie at K=2, worse at K=20, better nowhere | α=0.05: **1/10, p = 0.0215**; α=0.2: 3/10, sign p = 0.344 but **3.4× mean deficit**; 2/10 rollouts diverge on each α arm, 0/10 on `mf` |
| `uav-corridor` | **all-pass** | cost only | 🟡 **nominal win** — 270.9 vs 271.7 steps | spread **2.4 steps against σ ≈ 4.5**; paired split 5/10–5/10 |
| `uav-pillars` K=5 | **discriminating** (S&C separates) | the real thing | 🔴 **refuted — `af` finishes last, behind naive `fm`** | mean S&C **0.235** vs `fm` 0.359 vs `mf` 0.635; ceiling 0.700; **0 of 17 cells ≥ 0.8**; `mf` owns the only S&C = 1.000 cell in the family |
| `uav-s_curve` | all-fail | nothing clears funnel Stage 2 | ⬜ no verdict | retired as a ranking scene |

**Read the table by regime, not by row.** 🔑 The two environments where α-Flow "wins" are the two
where **every engine already passes**, so the only axis left is cost and the margins land inside the
noise. The two environments with the power to separate engines both go against it, and the one that
separates them on *constraint satisfaction* — the quantity the thesis actually claims — puts α-Flow
**below naive flow matching**.

> **The correct summary is not "α-Flow is worse."** It is: **α-Flow is never better, and it is worst
> exactly where the measurement is sharpest.** That asymmetry is the finding.

---

## 2. Why it is sometimes good and sometimes bad — the mechanism

This section is the substance of the fallback. It is what lets the thesis say *"we understand this
result"* rather than *"this did not work."* Full derivation:
[`CLOSURE_20260907` §M](../../../Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md).

### 2.1 One axis, three engines

All three engines train the *same* network against a target for the same quantity — the average
velocity `u(z_r, r, h)` — and differ **only in how the target is built.** With `α = dt/h`:

| engine | α | target | what it is |
|---|---|---|---|
| `fm` | **1** | `u_tgt = v` | the **instantaneous** field — eq. (1) never used |
| `af` | 0 < α < 1 | `u_tgt = α·v + (1−α)·u_next`, `u_next` a `no_grad` forward | a **finite-difference** composition closed with the network's **own frozen output** |
| `mf` | **0** | `u_tgt = v + h·∂u/∂r` (JVP) | the **analytic** α→0 limit — an exact derivative |

So `fm` and `mf` are the two endpoints of one axis and `af` is the interior. 🔑 **This is not three
methods; it is one axis with three samples on it.** The engine chapter can be written as a
single-axis study, which is a far better thesis than a shopping list.

### 2.2 The error decomposition — why the interior has no good limit

With `u*` the true average velocity and `ε = u_θ − u*`:

```
u_tgt − u*  =  (1−α)·ε_next  +  O(α²h²)
                  └────┬───┘     └───┬──┘
          bootstrap propagation   finite-difference bias
          of the model's OWN      from  ∫v ds ≈ dt·v
          error, undamped
```

Two terms, **opposite in α**:

- **`(1−α)·ε_next`** — the target inherits a fraction `(1−α)` of the model's own error, so training
  contracts error only at rate `(1−α)`. Any field with `ε = (1−α)·ε_next` is a **fixed point of the
  objective**: a learned field can be perfectly self-consistent **and wrong**. As α → 0⁺ the factor
  → 1 and there is *no contraction at all*.
- **`O(α²h²)`** — finite-difference bias, growing with α; at α = 1 the target degenerates to `v`,
  i.e. back to `fm`, which learns the wrong object.

⇒ **α-Flow has an interior optimum in α and no good limit at either end.** MeanFlow escapes the
trade entirely: the analytic branch carries **neither** term.

Two further code-level facts that matter and must be stated, because they pre-empt the two obvious
reviewer objections:

1. 🔴 **α → 0⁺ does not converge to MeanFlow.** The implementation *changes branch* at `α == 0`
   (`af_diffusion.py:552`), from a finite difference to an exact derivative. Small α is therefore
   the **worst-conditioned** regime — a finite difference over a vanishing step — which is precisely
   why the measured **α = 0.05 arm is worse than α = 0.2** on distance, on push-aways (4/10 vs 2/10)
   and on success.
2. 🔴 **Containment is not dominance.** "α-Flow contains MeanFlow at α = 0, so it must be at least as
   good" is false for a *trained checkpoint*: α is a **fixed schedule, not an optimised parameter**.
   The family contains MeanFlow; training picks one point in it, and the point it picks is chosen by
   a hand-set anneal. This is the single most likely question in the defence and it has a one-line
   answer.

### 2.3 The consequence that unifies every measurement: **flat in K**

Total inference error decomposes as

```
E(K)  ≈  discretisation O(1/K)  +  field bias b
```

`b` is a property of the **learned field**, so it does not depend on K.

| engine | target built from | `b` | E(K) as K grows | V_A median, K=2 → 20 |
|---|---|---|---|---|
| `mf` | analytic JVP (external) | small | 🟢 falls | 0.2354 → **0.0741** |
| `diffusion` | true DDPM posterior (external) | small | 🟢 falls | 0.4140 → 0.1849 (K=100) |
| `af` α=0.2 | **its own frozen output** | large | 🔴 **flat** | 0.3738 → 0.3918 |
| `af` α=0.05 | **its own frozen output** | larger | 🔴 flat / worse | 0.3978 → 0.4488 |

> **The split is exactly along target construction.** Engines whose target is computed from something
> *external* to the network get better as the sampler integrates more finely. The engine whose target
> is the network's own output does not. **More NFE integrates the wrong field more accurately.**

### 2.4 The headroom rule — why "sometimes good" is a property of the *task*

This is the sentence that makes the whole story cohere:

> ### α-Flow's plan quality is insensitive to NFE. MeanFlow's improves with NFE **wherever the task has headroom to show it.** The environments differ in **headroom**, not in which engine is better.

- `avoiding-d3il`: raw-plan success is **1.000 for every engine at every K**. The task is
  **saturated** — there is nothing left for a better field to buy. All that varies is a ~13 % band in
  `n_steps`, and α-Flow's nominal win lives there.
- `uav-corridor`: same shape — every engine reaches S&C = 1.000, the whole spread is 2.4 steps
  against σ ≈ 4.5, and the *cheapest* S&C = 1.000 cell is the **unprojected** plan for all three
  engines. A scene the policy already solves cannot rank engines.
- `aligning-d3il-visual`: 0.45 m of headroom → the engines separate, and α-Flow's flat-in-K field is
  exposed at K = 20.
- `uav-pillars` K=5: S&C itself separates → α-Flow last.

**Therefore "α-Flow sometimes beats MeanFlow" is more precisely: "some tasks cannot tell them
apart, and on those tasks the tie breaks arbitrarily on cost."** Two independent environments
(`avoiding`, `uav-corridor`) produced the *same nominal ordering with the same absence of power* —
which is a consistent **non-result**, not corroboration.

### 2.5 The predictive rule — when α-Flow *would* have won

α-Flow needs **all** of these simultaneously. State them as a checklist; it makes the negative
falsifiable, which is what makes it a contribution.

| condition | `avoiding` | `uav-corridor` | `V_A` | `uav-pillars` |
|---|---|---|---|---|
| operating point at **low K** | ✅ | ✅ | ✅ (K=2 available) | ✅ |
| MeanFlow's K-curve **flat or falling** there | ✅ (saturated) | ✅ (saturated) | 🔴 no — `mf` *improves* 0.2354 → 0.0741 | 🔴 no |
| task has **no headroom**, so plan quality cannot separate | ✅ | ✅ | 🔴 no — 0.45 m | 🔴 no |
| latency is the **binding** constraint | 🔴 no | 🔴 no | 🔴 no (8.13 vs 9.17 ms/NFE on a 4.7× worse plan) | 🔴 no |
| α set **high enough** to escape the ill-conditioned bootstrap | ❓ untested > 0.2 | ❓ | ❓ | ❓ |

> **The killer line, from the V_A gate:** *matching MeanFlow's setup requires K = 20; K = 20 is where
> α-Flow has no advantage to offer.* **The requirement to match and the requirement to win are not
> simultaneously satisfiable for this method on this task.**

---

## 3. The two "bad-result" environments, characterised honestly

The user framing is right and must survive into the text: **V_A is over-strict, and UAV is still
running.** Neither fact weakens the fallback — but each changes *what may be claimed*.

### 3.1 `aligning-d3il-visual` — over-strict, and why the α-Flow verdict survives it

- The scene is in the **all-fail regime**: S&C is floored, no arm satisfies the constraint set, and
  ranking needs the three-stage **funnel** (distance → constraints → time).
- There is a measured **~0.4 m reproducibility floor** on projected arms, which disqualifies most
  small margins on constrained rows.
- ✅ **Why the verdict still holds:** the α-Flow comparison is run on the **unguided** arm, on *plan
  quality* (distance to target), not on constraint satisfaction. The strictness of the constraint set
  is exactly the quantity the funnel's Stage 1 removes from the comparison. The over-strictness kills
  the *`fm > diffusion`* rung and the constraint claims on this scene; it does not touch the
  distance-ladder in §2.3.
- ⚠️ **Do not oversell the two halves:**
  - **K=2 is a genuine tie** in paired distance (4/4, perm p = 0.875) — *but* on contexts-ever-solved
    `mf` takes **7 of 10** and `af` **2 of 10**. Report both; the tie is metric-specific.
  - **K=20**: the α=0.05 arm is significantly worse (**1/10, p = 0.0215**); the α=0.2 arm is a 3.4×
    mean deficit that the sign test **does not** reject (3/10, p = 0.344). Quote the arm you mean.
    🔴 The Gen15 closure's summary table quotes `p = 0.0215` under a bare "`af`" label — that is the
    **α=0.05** arm. Do not let that shorthand into the thesis.

### 3.2 UAV — still running, and the verdict is one seed deep

- ✅ `uav-pillars` K=5 is the **only** scene in the study whose S&C separates engines, and it is
  unambiguous: `mf` 0.635 / `fm` 0.359 / `af` **0.235**, `af` ceiling 0.700, **0 of 17** cells ≥ 0.8.
  This is a **mean over 17 projection rules**, not a chosen cell — exactly the aggregation `avoiding`
  lacked.
- The budget axis does not rescue it: `af` mean S&C 0.060 → 0.280 → 0.320 at K = 1/2/5, ~85 % of the
  gain in by K = 2, and K = 5 is where four of ten variants **lose physical safety**.
- 🔴 **Open, and it gates the strength of the wording:** the UAV engine verdict rests on **seed 6,
  n = 10, one scene**. A 0.1 step in mean S&C is one rollout. Required run: **2–3 seeds × 3 engines ×
  `pillars` K=5 × 17 variants**.
- ⚠️ **`corridor` K = 1 must not be quoted.** `af`'s K = 1 arm (266.4 steps, 9.49 ms) would dominate
  everything, but `mf` and `fm` have **no K = 1 corridor arm** — a coverage gap, not a result.
- ⬜ **No UAV scene has a `diffusion` arm**, so no UAV row can speak to `fm > diffusion` or to
  "beats the DPCC baseline". The baseline claim rests on `avoiding-d3il` (−0.3744 m, 0/10,
  **p = 0.0020**) and on V_A.

### 3.3 Why the closure is robust even so

α-Flow is **not** closed because one test rejected it. It is closed because it is **never better
across four environments**, its K-response is **flat by construction**, and the mechanism that
predicts flatness was derived before the UAV numbers arrived and then held. Two of the four
environments produce a *tie inside noise*; two produce a loss. **A closure that does not depend on
any single p-value is the one that survives a defence** — and it is the reason the pending seeds
(§3.2) change the *wording strength*, not the *verdict*.

---

## 4. The fallback claim ladder (replaces TARGET §1)

> ### Goal A′ — one engine separates
> **`mf` ≫ { `af`, `fm`, `diffusion` }**, per environment, under an identical U-Net, identical
> projector, identical data/normalisation/horizon/seeds/harness.
> Demonstrated across **three environments spanning all three evaluation regimes**: all-pass
> (`avoiding-d3il`, `uav-corridor`), discriminating (`uav-pillars` K=5), all-fail
> (`aligning-d3il-visual`).

| sub-claim | status | where it is carried |
|---|---|---|
| `mf` ≻ diffusion-DPCC (**THE** baseline) | ✅ **holds decisively** — V_A −0.3744 m, 0/10, p = 0.0020, at 0.64× per-step cost; `avoiding` 31.6× cheaper at equal S&C = 1.00 | `sec:res:state`, `sec:res:visual` |
| `mf` ≻ `fm` | ✅ **holds on three environments** — V_A +0.216 m, 9/0, p = 0.0039; `uav-pillars` 0.635 vs 0.359; `corridor` `fm` alone drops to 0.800 | `sec:res:fewstep` |
| `fm` ≻ diffusion | 🔴 **does not hold** — 0.0055 m inside a 0.4 m band; no UAV arm exists to repair it | `sec:disc:negative` — reported as a withdrawn ordering |
| `af` ≻ `mf` | ⛔ **refuted / closed** — the subject of this document | `sec:res:fewstep` + `sec:disc:negative` |

**Definition of "better" is unchanged**: Pareto dominance — equal success *and* equal constraint
satisfaction, strictly fewer NFE *and* lower wall-clock. Everything else is named a trade-off.
**Goals B (projector ladder) and C (candidate selection) are unaffected** and keep their TARGET text.

---

## 5. Storytelling — how α-Flow earns its pages as an *auxiliary* strand

The instruction is: **main goal says MeanFlow is best; α-Flow is the further discussion.** Here is
the shape that does that without sounding defensive.

### 5.1 The frame: don't report four engines — report one axis

Chapter structure follows §2.1. The engine chapter poses **one question**: *how should the
average-velocity target be constructed?* Three answers are placed on one axis (`α = 1`, `0 < α < 1`,
`α = 0`), all on the same 4.0 M U-Net, and the thesis measures the axis. The result — **the endpoint
wins, and the interior is flat in NFE** — then explains `fm`, `af`, `mf` *and* why diffusion behaves
like `mf` in K, in a single picture. A reader who only remembers one figure remembers the K-ladder
of §2.3.

**This is strictly stronger than the ladder we set out to prove.** "Our newest engine is best" is a
leaderboard entry. "Targets built from something external to the network improve with NFE; targets
bootstrapped from the network's own output do not" is a **finding with a mechanism and a
prediction.**

### 5.2 The five beats

| # | beat | content |
|---|---|---|
| 1 | **The promise** | α-Flow is published as *"Understanding and **Improving** MeanFlow"*; it contains MeanFlow at α = 0; upstream reports FID 43.1 → 40.2 at NFE-1. Taking it seriously was correct. |
| 2 | **The containment fallacy** | Containing MeanFlow does not dominate it: α is a fixed schedule, not an optimised parameter — and the code *changes branch* at α = 0, so the family is not even continuous at that end (§2.2). |
| 3 | **The measurement** | Architecture-matched, α **verified live** (`discrete_frac ≈ 0.5`, α floored at 0.2 and 0.05, checked on every arm), two α floors, four environments, three regimes. |
| 4 | **The mechanism** | The `(1−α)·ε_next + O(α²h²)` decomposition → interior optimum, no good limit at either end → field bias `b` → **flat in K** → matches every measured ladder (§2.2–2.3). |
| 5 | **The rule that falls out** | The headroom rule (§2.4) and the win-conditions checklist (§2.5) — which also explains why the `avoiding` and `corridor` "wins" are non-results rather than contradictions. |

### 5.3 Three things this arm buys the thesis that a win would not have

1. **It pre-empts the strongest reviewer objection to the whole engine chapter.** "You just didn't
   turn α on" is answered by instrumentation, not by argument: every prior `ae0.0` checkpoint in this
   project *was* MeanFlow in an α-Flow folder (α ≤ 0 routes into the JVP branch), we found that with
   our own logging, retired those runs, and re-ran with a floored α.
2. **It supplies the negative-result contribution the thesis already promises.** Contribution 6 in
   `sec:intro:contributions` is "retained negative results". This is one of exactly two, and it is
   the one with a derivation behind it.
3. **It converts a benchmark weakness into a methodological result.** The reason α-Flow looked good
   on `avoiding` is the reason `avoiding` alone cannot rank engines — **task saturation**. That
   observation motivates the whole regime taxonomy (all-pass / discriminating / all-fail) which the
   Results chapter needs anyway, and which is itself novel in this line of work.

### 5.4 Tone — the one paragraph to reuse

> α-Flow keeps its place as a **negative result with a verified mechanism**: the bootstrapped target
> is genuinely live, the implementation is correct and architecture-matched, and it still does not
> improve on the analytic MeanFlow target on any of four environments — while the mechanism that
> predicts *why* (a target bootstrapped from the network's own output carries a K-independent field
> bias) also predicts the two environments where it appears to tie. That is a stronger contribution
> than a cherry-picked cell.

---

## 6. Where it lands in the draft

v1/v2 already demote α-Flow, so this is mostly **confirmation + sharpening**. Four concrete deltas:

| # | file / section | current state | fallback action |
|---|---|---|---|
| 1 | `sec:bg:fewstep` — *A curriculum variant on the same objective* | "*No mathematics is given for it in this draft — deliberately deferred.*" | 🔴 **Reverse the deferral.** Under the fallback the **mathematics is the contribution**. Write eq. (1)–(4) of §2.1–2.3 here or in `app:derivations`; without them the section is a null result. |
| 2 | `sec:method:engine` — *Engine 3, the curriculum variant* | "ties at low budget, loses at high budget, better nowhere **across two environments**" | Update to **four environments and three regimes**; add the α = 0 branch-change and the containment-fallacy sentence (§2.2), which is already half-written there as the "collapse argument". |
| 3 | `sec:res:fewstep` | bone | Lead with the **K-ladder table** (§2.3) — it is the figure the chapter turns on — then the per-environment rows of §1 with their regime labels. |
| 4 | Abstract + `sec:intro:contributions` | "an additional engine that does not improve on its predecessor" | Keep the wording, add the mechanism clause. The abstract's `\hole{}` headline numbers should be filled from Goal A′ (§4), not from the old ladder. |

Also worth a line in `sec:disc:threats`: the `avoiding`/`corridor` saturation finding is a **threat to
validity of single-benchmark engine comparisons in general**, and it is one this work found in its
own results.

---

## 7. Rules of engagement for the α-Flow write-up

Binding on every sentence about `af`, the same way TARGET §6 binds every table.

1. **Never write "α-Flow fails."** Write **"does not improve on MeanFlow."** It is accurate, it is
   what was tested, and it is the stronger claim because it is falsifiable.
2. **Name the α floor** in every quoted number. α = 0.2 and α = 0.05 are different arms with
   different verdicts (§3.1).
3. **Two α floors were run and the better one was quoted on `avoiding`** — that is a selection effect
   and must be stated where that number appears.
4. **Tag the nominal orderings.** `avoiding` (p = 0.231) and `corridor` (2.4 steps vs σ ≈ 4.5) are
   *inside noise* and must never be written as wins, not even as "slight wins".
5. **Never pool over projection rules** to make a claim, and never quote a best cell without its
   all-rules aggregate beside it.
6. **Never quote `corridor` K = 1 `af`** — `mf`/`fm` have no K = 1 arm there (§3.2).
7. **State the seed depth** wherever the UAV negative appears: seed 6, n = 10, one scene.
8. **Degenerate HardFlow rows stay tagged and excluded** — unchanged from TARGET §6.4; the α-Flow
   discussion does not get an exemption.

---

## 8. Kill criteria for the fallback itself — decided now

Symmetry with TARGET §8: this document must also be able to be wrong.

| if | then |
|---|---|
| a **higher α floor** (α ≥ 0.4, untested) beats `mf` on V_A at K = 20 with paired significance | α-Flow re-enters as a rung and this file is retired. This is the **only** live re-entry path, and it is a *single* pre-registered run, not a sweep. |
| the **seed replication** on `uav-pillars` K=5 reverses the ordering | soften the UAV row to "inconclusive"; the closure still stands on V_A + the mechanism, but the "worst where measurement is sharpest" line is withdrawn. |
| a **constant-α** arm (never run) on the same visual U-Net beats `mf` | the *anneal*, not the objective, was the problem — rewrite §2 accordingly and say so. |
| none of the above lands by the writing deadline | ship as written; α-Flow is closed on four environments and a derivation. |

🔴 **Anti-rule, carried from TARGET §8:** *stop sweeping for a winning cell.* Two α floors, four
environments and a K-sweep have been run. Further α-Flow compute is **closed** (Gen15 closure §6.1)
except for the single re-entry run above.

---

## 9. Provenance — the evidence this file points at

| claim area | source of record |
|---|---|
| UAV verdict, regime taxonomy, aggregate ladder | [`Gen15 · CLOSURE_20260910_uav_engine_ladder_final.md`](../../../Gen15/Campaign_20260907_five_missions/CLOSURE_20260910_uav_engine_ladder_final.md) — §2, §4, §6, §7 |
| V_A verdict, the α mechanism, eq. (1)–(4), the win-conditions checklist | [`Gen14 · CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md`](../../../Gen14/CLOSURE_20260907_Gen14_V_A_engine_comparison_final.md) — R.1–R.5, §4, §5, §M |
| the K = 20 flagship kill, per-α-floor paired tests | [`Gen14 · DA_20260907_Gen14_Gate1_AF_vs_MF_K20_flagship_KILL.md`](../../../Gen14/DA_20260907_Gen14_Gate1_AF_vs_MF_K20_flagship_KILL.md) |
| the `avoiding` nominal win, its wash, its single-seed limit | [`Report_20260903_AF_UNet/README.md`](../../../../Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) §1, §10 |
| α is training-only; α = 0 **is** MeanFlow branch-for-branch; the anneal schedule | [`ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md`](../../../../Data_Analysis/DA_Result_Curated_MD/ANALYSIS_20260829_alphaflow_vs_meanflow_visual_aligning_are_they_the_same.md) Q1–Q4 |
| UAV mission detail (`pillars` K-sweep, `corridor`, HardFlow, `s_curve`) | `Gen15/Campaign_20260907_five_missions/DA_20260910_T1…`, `DA_20260908_T2…`, `DA_20260910_T3…`, `DA_20260910_T4…`, `DA_20260909_T5…` |
| running index of headlines | [`NOTEBOOK_20260829_key_headlines.md`](../../../../Data_Analysis/DA_Result_Curated_MD/NOTEBOOK_20260829_key_headlines.md) |

**Not yet reflected in:** [`../../Auxiliary/NOTES_open_questions.md`](../../Auxiliary/NOTES_open_questions.md)
(item 2's scope list still carries Gen3v7 α-Flow as an "in" generation without the demotion note) and
[`../v2/README.md`](../v2/README.md) (still says "across two environments"). Both are one-line edits,
left for the next writing pass rather than made silently here.
