# CLOSURE — the UAV verdict on `af_unet > mf > fm > diffusion`

*Gen15 · campaign `Campaign_20260907_five_missions` · 2026-09-10 · **closes the campaign.**
All UAV numbers from `temp/0909/batch_uav_20260910_092309` (`DA_UAV_v1`, 1299 units, 0 failed),
seed 6, n = 10 except C94 (n = 3, by design). d3il figures cited from
`Data_Analysis/DA_Result_Curated_MD/`. Written to be reusable as paper material.*

---

## 0. Method — three regimes, three treatments

A scene where every arm passes S&C is **not unrankable**; it is the regime `avoiding-d3il` is in,
and it is ranked on cost at equal safety. A scene where every arm *fails* S&C is the regime
`aligning-d3il-visual` is in, and it is ranked with the **three-stage funnel**. Only the middle
regime is ranked on S&C directly. Each UAV scene falls in a different one:

| regime | criterion | treatment | precedent | UAV scene |
|---|---|---|---|---|
| **all-pass** | every arm reaches S&C = 1.00 | Pareto on **steps × time** at each engine's *cheapest* S&C = 1.00 point | `avoiding-d3il` (`Report_20260903_AF_UNet` §1) | **`corridor`** |
| **discriminating** | S&C separates arms | rank on **S&C**; only arms reaching 1.00 proceed to a cost comparison | — | **`pillars` K=5** |
| **all-fail** | S&C floored near 0 | **3-stage funnel**: distance → constraints → time, Stage 1 on the *unguided* arm only | `aligning-d3il-visual` (`outdated_Report_20260829_VA_funnel`) | **`s_curve`** |

> **Two corrections to the mission DAs, applied here.** T2 concluded `corridor` "cannot rank" —
> wrong: it is the all-pass regime and ranks on cost (§1). T4 reported "`fm > mf` on `s_curve`,
> reversing `pillars`" — that comparison is **tracker-confounded and is retracted** (§3.1).

---

## 1. `corridor` — the all-pass regime, ranked like `avoiding`

Every arm has S&C = 1.000 cells: af K=1 **10/10**, af K=2 **10/10**, mf K=2 **10/10**,
fm K=2 **7/10** (fm loses all three `-t` rows to 0.800 — the only S&C separation the scene offers).

**Cheapest S&C = 1.000 point per engine**, matched at K = 2:

| engine | variant | steps | `avg_ms` | `fm_ms` | `proj_ms` |
|---|---|---|---|---|---|
| **af** (α-Flow U-Net, 3.97 M) | `diffuser` | **270.9** | **18.22** | 18.22 | 0.00 |
| **mf** (MeanFlow U-Net) | `diffuser` | 271.7 | 18.25 | 18.25 | 0.00 |
| **fm** (naive Flow Matching) | `diffuser` | 273.3 | 18.25 | 18.25 | 0.00 |

**The ordering reproduces the target ladder — `af < mf < fm` on steps at identical cost — and it is
statistically empty.** The entire spread is **2.4 steps** against a per-cell σ of **4.5–5.0**, and the
paired per-variant analysis (T2 §3) splits 5/10–5/10 with mean |Δ| = 0.83 steps.

🔑 **This is precisely the `avoiding` failure mode, reproduced independently.** There the ladder also
appeared in the right order at the cheapest S&C = 1.00 point (af 57.20 < mf 60.75 < fm 65.65 steps)
and the same report withdrew it: *"averaged over all six projection rules rather than the best one,
the two engines are a wash"*, largest margin 3 episodes, Fisher **p = 0.231**. **Two environments now
show the same nominal ordering with the same absence of power.** That is a consistent *non-result*,
not corroboration.

### 1.1 ✅ Verified: on `corridor` the constraint half of S&C carries no information

The cheapest S&C = 1.000 cell is the **unprojected** `diffuser` plan on every engine, which invites
the obvious objection — *can a plan with no projection really satisfy every constraint?* Checked
against the raw constraint columns rather than the S&C aggregate:

| arm | S&C | `n_violations` | `total_violations` | `collision_free_completed` | `phys_min_z` |
|---|---|---|---|---|---|
| af K=1 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 | 1.127 |
| af K=2 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 | 1.127 |
| mf K=2 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 | 1.127 |
| fm K=2 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 | 1.127 |

**The columns are populated and the metric is sensitive** — the same unprojected arm reports
**197.70** violations with `collision_free_completed` = 0.000 on `pillars` (af) and **24.30** with
`cfree` = 0.000 on `s_curve` (fm, at `success` 0.700). Corridor really is constraint-trivial.

**And the projected rows are identical:** `dpcc-t` and `dpcc-t-tightened` also report `n_violations`
= 0.00 and `cfree` = 1.000 on all three engines.

> 🔑 **On `corridor` no arm violates a constraint, with or without the projector.** The constraint
> half of S&C is therefore uninformative on this scene: S&C ≡ `success`. Even fm's only separation —
> 0.800 on the three `-t` rows — is a **goal-reach** failure at **zero** violations.

This sharpens the overhead finding: the projector on `corridor` is not merely expensive, it is
**provably solving a problem that does not exist** — 155–202 ms per step to remove violations that
were never there. It is also why `corridor` cannot arbitrate the engine ladder: the scene exercises
the generator's goal-reaching and nothing of the constrained-control problem the thesis is about.

**Two further corridor findings that do survive:**

* **The cheapest S&C = 1.00 cell is the *unprojected* plan for all three engines** — 0 ms of
  projection vs 155–202 ms (§1.1).
* **The projector buys path length, at ~10× the step cost.** af: 270.9 → 249.2 steps
  (`dpcc-t-tightened`) for **9.4×** the time; mf 271.7 → 248.5 for 9.3×; fm 273.3 → 266.5 for 11.1×.
  Per the Pareto rule these are **trade-offs, not dominance**.

⚠️ af's K = 1 arm is cheaper still (266.4 steps, **9.49 ms**) and would dominate everything — but
**mf and fm have no `u7hg` K = 1 corridor arm**, so this is a coverage gap, not a result, and must
not be quoted.

---

## 2. `pillars` K=5 — the discriminating regime

This is the only UAV scene where S&C separates engines, and the elimination logic is unambiguous:

> ### Exactly **one** cell in the entire `u7hg` `pillars` family reaches S&C = 1.000.
> **`mf` · `dpcc-t-geo_free` · 411.1 steps · 78.74 ms · 0 violations · `phys_safe` 1.000**

**`af` and `fm` never reach S&C = 1.000 at any K, on any of the 17 variants** — they are eliminated
before a cost comparison can be run. Ceilings: **mf 1.000 · fm 0.900 · af 0.700**.

Across all 17 matched variants (architecture-matched: `mf` and `af` are the same U-Net class):

| engine | mean S&C | ceiling | cells ≥ 0.8 |
|---|---|---|---|
| **mf** | **0.635** | **1.000** | **7 / 17** |
| fm | 0.359 | 0.900 | 2 / 17 |
| **af** | **0.235** | 0.700 | **0 / 17** |

**α-Flow does not merely fail to beat MeanFlow — it finishes behind naive Flow Matching, and is the
only engine with zero cells above S&C 0.8.** This is a mean over 17 projection rules, not a chosen
cell, which is exactly the aggregation `avoiding` lacked.

The budget axis does not rescue it (T1): af's mean S&C is 0.060 → 0.280 → 0.320 at K = 1/2/5, ~85 %
of the gain arriving by K = 2, and K = 5 is where four of ten variants lose physical safety.

---

## 3. `s_curve` — the all-fail regime, run through the funnel

S&C ceiling across all **40** `u7hg` cells is **0.100**. Under the funnel — Stage 1 on the *unguided*
`diffuser` arm only, no projection, controller held fixed:

### Stage 1 · can the generative model get near the goal?

| engine | K | controller | `goal_reached` | `goal_dist` | |
|---|---|---|---|---|---|
| **mf** | 10 | **`mjpc`** | **1.000** | **0.297** | ✅ **passes — best on the board** |
| fm | 2 | `pid_stopgo` | 0.700 | 1.110 | ✅ passes |
| fm | 20 | `pid_stopgo` | 0.600 | 1.193 | ✅ passes |
| mf | 2 | `pid_stopgo` | 0.100 | 2.515 | ❌ fails |
| mf | 10 | `pid_stopgo` | 0.000 | 2.809 | ❌ fails |

### 3.1 🔴 Stage 1 exposes a tracker artefact — and retracts a T4 claim

**Same model, same K = 10, same scene, same constraint set: `goal_reached` 0.000 under
`pid_stopgo`, 1.000 under `mjpc`.** MeanFlow's Stage-1 "failure" on `s_curve` is **not** a generative
failure — it is the tracker. T5 measured this paired on three shared initial conditions
(goal distance 2.86/2.72/2.89 → 0.299/0.298/0.294, Fisher **p = 0.0035**).

**Consequence: T4 §4's "`fm` beats `mf` on `s_curve`, reversing `pillars`" is withdrawn.** Both arms
in that comparison ran `pid_stopgo`, and the only controller-controlled measurement available shows
the tracker dominating the Stage-1 outcome on this scene. There is no engine reversal in evidence —
there is a tracker effect.

### Stage 2 · of the arms that got near, which did it legally?

| entrant (Stage-1 survivor) | best S&C | cells with S&C > 0 | |
|---|---|---|---|
| mf K=10 `mjpc` | **0.000** | 0 / 3 | ❌ eliminated |
| fm K=2 | 0.100 | 2 / 10 | ❌ eliminated |
| fm K=20 | 0.000 | 0 / 9 | ❌ eliminated |

### Stage 3 · time — **never reached. No arm survives Stage 2.**

### 3.2 What the funnel decides that "unrankable" did not

**`s_curve` fails at Stage 2, not Stage 1.** The generators *can* fly it — MeanFlow under `mjpc`
reaches the goal on 3 of 3 unguided rollouts at 0.297 m. What no configuration can do is fly it
*legally*: 40 cells, ceiling S&C 0.100.

Three rescue hypotheses have now been tested and all fail at Stage 2:

| hypothesis | tested by | result |
|---|---|---|
| more sampling budget | **T4** | ❌ strictly worse — fm K2→K20 drops `goal_reached` on 5/5 shared variants at 9.9× cost |
| a stronger tracker | **T5** | ⚠️ clears Stage 1 outright, **Stage 2 unchanged at 0.000** |
| a different engine | T4 §4 | ⛔ withdrawn — tracker-confounded (§3.1) |

**The defect is the constraint set, and the funnel localises it precisely.** `s_curve` is retired as
a ranking scene; if it is kept for any purpose, `phys_min_z` must be reported beside `goal_reached`
(T4 §5: mf K=10 HardFlow rows reach the goal 0.900 of the time while flying at `min_z` ≤ 0.183).

---

## 4. The aggregate verdict

| environment | regime | `af_unet ≥ mf` | `mf > fm` | `fm > diffusion` |
|---|---|---|---|---|
| `avoiding-d3il` | all-pass | 🔴 nominal only — a wash over all rules, p = 0.231 | 🟢 Pareto-dominant, 30× | 🟡 21×, never NFE-matched |
| `aligning-d3il-visual` | all-fail (funnel) | ⛔ **KILLed** — worse, p = 0.0215, 1/10 | 🟢 9/0, p = 0.0039 | 🔴 fails — 0.0055 m in a 0.4 m band |
| **`uav-corridor`** | **all-pass** | 🟡 **nominal only** — 2.4 steps inside σ ≈ 4.5 | 🟢 fm alone drops to 0.800 | ⬜ no `diffusion` arm |
| **`uav-pillars` K=5** | **discriminating** | 🔴 **refuted — af is last, behind naive FM** | 🟢 **0.635 vs 0.359** | ⬜ no `diffusion` arm |
| **`uav-s_curve`** | **all-fail (funnel)** | ⬜ nothing clears Stage 2 | ⬜ tracker-confounded | ⬜ no `diffusion` arm |

**Top rung — dead.** Where the scene has power to separate (`uav-pillars`), α-Flow is **last**. Where
it does not (`corridor`, `avoiding`), the ladder ordering appears but is inside noise on both
environments. `aligning-visual` killed it outright. **Four environments, no support anywhere.**

**Bottom rung — unsupportable.** Already 🔴 on `aligning-visual`, and **no UAV scene has a
`diffusion` arm**, so UAV cannot repair it.

**Middle rung — holds, and UAV strengthens it.** `mf > fm` is now supported on three environments,
and on `pillars` MeanFlow owns the only S&C = 1.000 cell in the family.

> ## The ladder the data supports
> ### `mf` ≫ { `af`, `fm`, `diffusion` } — one winner, one cluster.
> Demonstrated across **three environments and all three regimes**: all-pass (`avoiding`,
> `uav-corridor`), discriminating (`uav-pillars`), all-fail (`aligning-visual`).

---

## 5. Goal B — the projector rung holds, and UAV adds a safety result

| environment | HardFlow vs the DPCC projector |
|---|---|
| `avoiding-d3il` (A=1.0, K ∈ {3,5}) | 🟢 `-t` Pareto-dominates — 16 % fewer steps, 7.4× cheaper |
| `aligning-d3il-visual` (K=10) | 🟢 −325 ms/step, 0/9, **p = 0.0039** at equal safety |
| **`uav-pillars` K=5** *(new)* | 🟢 **8.2–21.3× cheaper projection**, 5.6–12.4× end-to-end; S&C parity (8–9–1) |

**New from UAV — not previously measured anywhere:**

| arm | rollouts | unsafe |
|---|---|---|
| **HardFlow** (21 cells × 3 engines) | 210 | **0** |
| DPCC (27 cells) | 270 | **43** |

Every unsafe rollout in the study is a DPCC row, and **seven of the eight failing cells are `-r`**
(random candidate selection) — simultaneously the most expensive projection row on every engine.
T1 sharpens it: af is `phys_safe` = 1.000 on all 200 rollouts at K = 1 and K = 2; the `-r`/`-c`
failures appear only at K = 5.

⚠️ **Honest limit.** HardFlow beats the *full-geometry* projector on cost, but `dpcc-*-geo_free`
projects for the same price (≈ 33.7 vs 34.8 ms) and, because HardFlow's generator is ~45 % dearer, is
**cheaper end-to-end in all 9 pairs**. The cheapest projected plan on `pillars` is `dpcc-t-geo_free`
— which is also the only S&C = 1.000 cell in the study (§2).

---

## 6. Closed

| # | closed | why |
|---|---|---|
| 1 | **`af_unet` on UAV** — no further af compute | Last on the one scene with power; nominal-only elsewhere. Fourth environment to refuse it. |
| 2 | **`s_curve` as a ranking scene** | Fails at funnel Stage 2; ceiling 0.100 over 40 cells; budget and tracker both tested. |
| 3 | **UAV budget sweeps above K = 5** | Non-monotone on `pillars` (T1 §5) and `s_curve` (T4 §3); K = 5 introduces the safety failures. |
| 4 | **`corridor` as an arbiter** — keep as a sanity scene | All-pass; 2.4-step spread inside σ. |
| 5 | **`dpcc-r` / `dpcc-r-tightened`** — drop or keep as a control | Most expensive, only unsafe rows, never the best cell. |
| 6 | **The four-rung ladder as a thesis claim** | Two rungs dead across four environments. |

---

## 7. What still needs a run — two items

**7.1 🔴 Seeds on `uav-pillars` K=5 — required.**
The UAV engine verdict rests on **seed 6, n = 10, one scene**. `mf 0.635 > fm 0.359 > af 0.235` is a
wide ordering, but a 0.1 step is one rollout, and this single measurement carries a thesis-level
negative claim about α-Flow.
→ **2–3 seeds × 3 engines × `pillars` K=5 × 17 variants.**

**7.2 🟡 A `diffusion` arm on `uav-pillars` K=5 under `u7hg` — conditional.**
No UAV scene has one, so `fm > diffusion` and "beats the DPCC baseline" are untestable on UAV.
**Run only if the thesis wants a UAV baseline claim.** If that claim rests on `avoiding-d3il`
(already 🟢 decisive: −0.3744 m, 0/10, p = 0.0020), drop this.

**Explicitly not needed:** more af runs · any `s_curve` compute · K > 5 on UAV · a `corridor`
K-sweep · re-running HardFlow at K = 5 (`hf_n_genuine = 2`, gate passed).

---

## 8. Proposed thesis text

> **Engine.** MeanFlow with a U-Net backbone separates categorically from α-Flow, naive Flow
> Matching and diffusion-DPCC, which form one indistinguishable cluster. Shown across three
> environments spanning all three evaluation regimes: where every arm satisfies the constraints
> (`avoiding-d3il`, `uav-corridor`) the four engines are separated only nominally, within noise;
> where the constraint set discriminates (`uav-pillars`, K = 5) MeanFlow holds the sole
> constraint-clean cell and α-Flow ranks last, behind naive Flow Matching; and where no arm
> satisfies the constraints (`aligning-d3il-visual`, `uav-s_curve`) a staged distance→constraint→time
> funnel shows the failure is located in the constraint set, not the generative model.
>
> **Projector.** HardFlow-SLSQP matches the DPCC projector's success-and-constraints rate at an
> order of magnitude lower projection cost (8.2–21.3× on `uav-pillars`), and produced no physically
> unsafe rollout in 210 where the DPCC projector produced 43 in 270.

α-Flow keeps its place as a **negative result with a verified mechanism**: the bootstrapped target is
genuinely live (`discrete_frac ≈ 0.5`, α floored at 0.2, checked on every arm), the implementation is
correct, and it still does not beat the analytic MeanFlow target on any of four environments. That is
a stronger contribution than a cherry-picked cell.

---

### Provenance

Candidates C29/C30 (corridor af), C38/C44 (corridor fm/mf), C47/C49/C50 (pillars af), C62/C74
(pillars fm/mf), C92/C93/C95/C96 (s_curve), C94 (s_curve `mjpc`). Jobs 25494–25503, 25554, 25588,
25589. Mission DAs: [T1](DA_20260910_T1_af_unet_pillars_K_sweep.md) ·
[T2](DA_20260908_T2_af_unet_corridor_K1_K2.md) · [T3](DA_20260910_T3_hardflow_vs_dpcc_pillars_K5.md) ·
[T4](DA_20260910_T4_s_curve_budget_explore.md) · [T5](DA_20260909_T5_mjpc_vs_pid_s_curve.md).
d3il figures: `Report_20260903_AF_UNet/README.md`, `outdated_Report_20260829_VA_funnel/README.md`,
`NOTEBOOK_20260829_key_headlines.md` §10.2–10.3 and its 2026-09-07 evidence board.

---

## 9. Runs in flight after this closure (2026-09-10)

Two waves submitted after the closure was written. Neither changes §4's verdict; they close
coverage gaps §7 named, plus a new scene from U11.

### 9.1 `corridor_ball` — U11, the constraint-trivial fix (jobs 25599–25604)

`corridor_hg` cannot arbitrate anything (§1.1: zero violations, projected and unprojected alike), so
U11 adds a virtual centre ball that forces a vertical detour. Driver:
`Slurm_Codes/temp_bash/eval_20260910_corridor_ball.sh`.

| tier | K | HardFlow | af | mf | fm |
|---|---|---|---|---|---|
| **A** PCC only | 2 | degenerate (`n_genuine = 0`) | **25599** | **25600** | **25601** |
| **B** PCC + HF-SLSQP | 5 | genuine (`n_genuine = 2`) | **25602** | **25603** | **25604** |

🔴 **Gate before any reading:** the unprojected `diffuser` row must report `n_violations > 0`. If it
is still 0.00 the ball is not binding and the geometry needs revisiting, not analysing.

### 9.2 Benchmark-matrix completion (jobs 25611–25614)

Driver: `Slurm_Codes/temp_bash/submit_20260910_diffusion_baseline_and_scurve_mirror.sh`.

| job | arm | type | note |
|---|---|---|---|
| **25611** | `pillars` · **diffusion** | **train + eval** | §7.2, resolved as a **reference row** |
| **25612** | `s_curve` · diffusion | eval only | matrix completeness |
| **25613** | `s_curve` · af K=5 part A (DPCC) | eval only | matrix completeness |
| **25614** | `s_curve` · af K=5 part B (HardFlow) | eval only | split against the 24 h wall |

Only 25611 trains — `u7hg` was a constraint-set change, not a model change, so the existing
`AlphaFlowODE_9D_as1_ae0.2_bbunet` and `GaussianDiffusion_9D_K20` s_curve checkpoints stay valid.

**§7.2 is resolved, not upgraded.** The user's decision (2026-09-10) is that the diffusion arm is a
**reference row for the reader's scale, not a proof**: single seed, and the budgets are deliberately
unmatched (diffusion K=20 — a *training* parameter for DPCC — against the flow family at K=5).
Label it as a reference in the table caption. *"mf beats the DPCC baseline"* remains an
`avoiding-d3il` claim, where it is already decisive.

**The two `s_curve` rows are for completeness, not ranking.** Max S&C across the 40 existing `u7hg`
s_curve cells is 0.100 (§3); these two will not change that. They exist so the matrix reads
"all four engines, identical geometry" and no reviewer can ask why an engine is missing.

### 9.3 Still outstanding

**§7.1 — seeds on `pillars` K=5 — remains the one required run.** Nothing above addresses it, and
every UAV headline still rests on seed 6, n = 10.
