# TARGET — what the thesis has to prove

**Created:** 2026-09-05 · **Status:** target statement, not a results document
**Scope:** the whole thesis (`Bone/thesis_bone.tex`), all chapters
**Companions:** [`../Auxiliary/NOTES_open_questions.md`](../Auxiliary/NOTES_open_questions.md) (decisions still open) ·
[`../Auxiliary/NOTES_dpcc_lineage.md`](../Auxiliary/NOTES_dpcc_lineage.md) (what is inherited vs. own) ·
[`../Auxiliary/NOTES_paper_map.md`](../Auxiliary/NOTES_paper_map.md) (which reference goes where)

> **This file states the goal. It also states, per cell, whether the goal is currently
> supported, unsupported, or contradicted by our own data.** A target document that only
> lists wishes is useless for planning cluster time. §5 is the part to read on any day
> when deciding what to run.

---

## 0. The claim, in one sentence

> **Under an identical constraint projector and an identical U-Net backbone, the deterministic
> ODE-transport family beats the stochastic diffusion engine of DPCC, and it beats it in a
> strict order — α-Flow (U-Net) ≥ MeanFlow > naive Flow Matching > diffusion — across two
> observation modalities (state, visual) and two embodiments (manipulator, UAV); and where the
> task forces a high step budget, the trajectory-optimization constraint arm (HardFlow-SLSQP)
> in turn beats the per-step DPCC projection arm.**

Two ladders, deliberately kept apart, because they are answered by different experiments and
they can fail independently.

---

## 1. Ladder A — the engine ladder

$$\texttt{af\_unet} \;\succeq\; \texttt{mf} \;\succ\; \texttt{fm} \;\succ\; \texttt{diffusion (DPCC)}$$

**Held fixed across the whole ladder** — this is what makes it a ladder and not four separate
papers:

| held fixed | value | why it is load-bearing |
| :-- | :-- | :-- |
| backbone | temporal U-Net, matched parameter count (~4.0 M state / ~26.4 M visual) | the *architecture-matched* claim is the only defensible one; SiT/DiT wins are confounded by capacity and are reported as **secondary** with a params column |
| projector | the **same** projector arm on both sides of every comparison | otherwise the engine comparison silently becomes a projector comparison |
| data, normalisation, horizon, seed set, eval harness | inherited from DPCC | this is the reason the substitution is close to a single-variable change — say so once, in `sec:setup:baselines` |
| definition of "better" | **Pareto dominance**: at equal success **and** equal constraint satisfaction, strictly fewer NFE **and** lower wall-clock | anything else is a *trade-off* and must be written as one, never as "best" |

**The mechanism the ladder rests on** (this is the thesis's actual argument, the numbers only
confirm it): `K` is an *inference-time* knob for the flow family but a *training-time*
commitment for diffusion. DPCC cannot walk down the K-ladder — at K=1 it collapses. MeanFlow's
average-velocity target and α-Flow's bootstrapped/annealed target make the whole ODE traversable
in one or two evaluations. **The win is therefore structural, not a tuning artefact**, and the
thesis must argue it that way in `sec:bg:fewstep` before any table appears.

**Where the ladder must hold:** `avoiding-d3il` (state) → `aligning-d3il` (state, 3-D) →
`aligning-d3il-visual` → `uav-pillars` (+ `s_curve`, `corridor`).

---

## 2. Ladder B — the projector ladder, and why it is regime-split

$$\text{low-}K \Rightarrow \text{DPCC per-step projection wins} \qquad
\text{high-}K \Rightarrow \text{HardFlow-SLSQP wins}$$

This is **not** a hedge. It follows from HardFlow's own step accounting:

```
n_active  = max( K − int(A·K), 1 )        # steps above the activation threshold
n_genuine = n_active − 1                  # steps that actually solve an NLP
```

At `K∈{1,2}, A=0.5` → `n_genuine = 0`: **HardFlow runs no HardFlow math at all.** Those rows are
not "HardFlow losing", they are "HardFlow absent", and the degeneracy guard now tags them
(`HF_DEGENERATE`). The thesis must present this as a *structural domain of validity*
(`sec:res:constraints:degenerate`), never average over it.

**Consequence — and this is the paper's central tension, state it explicitly in
`sec:disc:interpretation`:** Ladder A wins by driving `K` down; Ladder B needs `K` (jointly with
the activation threshold `A`/`T`) high enough to leave genuine NLP steps. The reconciliation is
that the binding quantity is `n_genuine`, not `K`: at K=20 with T=0.2 the flagship still yields
`n_active=4, n_genuine=3 → HF_OK`. So the two ladders are compatible **at moderate K with a low
threshold**, and the thesis should say which tasks force that regime:

| regime | tasks | expected projector winner |
| :-- | :-- | :-- |
| **low-K feasible** — few-step engines suffice | `avoiding-d3il` (state) | **DPCC projection** (HardFlow is degenerate or not worth its 3–4× cost) |
| **high-K forced** — task needs many steps and/or a low threshold | `aligning-d3il-visual`, `uav-*` (tentative) | **HardFlow-SLSQP** |

---

## 3. The benchmark matrix — what each environment is for

| # | environment | modality / embodiment | origin | its job in the thesis |
| :-- | :-- | :-- | :-- | :-- |
| 1 | `avoiding-d3il` | state, manipulator | **DPCC's own benchmark** | the home turf of the baseline; the low-K regime; Ladder A's primary proof |
| 2 | `aligning-d3il` (3-D) | state, manipulator | D3IL, **imported by us** | shows the ladder is not an avoiding-specific artefact; harder, contact-rich |
| 3 | `aligning-d3il-visual` | **visual**, manipulator | **built by us** on D3IL | modality transfer (RQ4) + the high-K regime where Ladder B is expected to switch on |
| 4 | `uav-pillars` (primary), `uav-s_curve`, `uav-corridor` | state, **aerial** | **built by us** | embodiment transfer (RQ4); second high-K candidate |

**Rule:** the ladder is claimed **per environment**, never pooled. A pooled "our method wins"
number across four environments with different metrics and different `n` is not a claim, it is
an average of incomparable things.

---

## 4. Methodology the thesis owes the reader

Three of the four environments are not off-the-shelf. Each needs a **method** subsection, not a
sentence in the setup chapter, because each is a contribution and each is a threat to validity if
under-described.

### 4.1 Visual aligning — environment construction (`sec:method:` + `sec:setup:tasks`)

Must document: the rendering/observation pipeline added to D3IL aligning; the vision encoder and
its provenance (`diffusion_policy` is the true upstream of D3IL's encoder — see the Gen14 U8
work); FiLM conditioning (v1) and the `freq_dim`/token-conditioning choices; how the visual
backbone keeps the parameter comparison honest (26.4 M on both sides); and the **evaluation
caveat that `n_success = 0` for every arm at these checkpoints** — every visual-aligning claim in
this thesis is about *distance / progress / constraint satisfaction*, never success rate. Say
that once, prominently, or the Results chapter reads as if it is hiding it.

### 4.2 UAV — environment construction and the low-level controller (`sec:method:` + `sec:setup:tasks`)

Must document: scene generation for `empty / corridor / s_curve / pillars`; the expert data
source and its recording rate; the **PID low-level tracking controller** and the plan→setpoint→
thrust chain (the arm-vs-UAV control-chain difference is already written up in Gen11 Epoch7/8 —
port it, do not re-derive); and, non-negotiably, the **honest-geometry finding**: the original
scenes leave less free channel around their own expert routes (0.000 m on `corridor`, 0.060 m on
`pillars` outer) than the policy's measured tracking error (0.30–0.49 m), so success+constraints
was bounded near zero *before any engine ran*. Any UAV result predating the `*_hg` geometries
must be labelled accordingly. Reporting this is a strength; omitting it and then reporting near-
zero S&C is not survivable at defence.

### 4.3 Timing on UAV — what not to claim

`budget_ms ≈ 30.3 ms` (= `1000/DATASET_HZ`) is a **data-rate artefact**, and the wall-clock is
cluster GPU latency. It is not a real-time target and must never appear as a pass/fail criterion.

---

## 5. 🔴 Evidence board — where the target stands today

Legend: 🟢 supported · 🟡 partial / caveated · 🔴 **contradicted by our own data** · ⬜ not measured

### 5.1 Ladder A

| environment | `mf > fm` | `fm > diffusion` | `af_unet ≥ mf` | overall |
| :-- | :-- | :-- | :-- | :-- |
| `avoiding-d3il` | 🟢 MF-UNet K1 Pareto-dominant vs DPCC K20 (30× `avg_time`, 12 fewer steps, S&C 0.97 vs 1.00 at resolution 1/30) | 🟡 cross-family n=20 DA exists; needs restating as an NFE-matched claim | ⬜ AlphaFlow ran (job 24515) but has never been ranked on this ladder | 🟡 |
| `aligning-d3il` (3-D state) | ⬜ | ⬜ | ⬜ | ⬜ **no data at all** |
| `aligning-d3il-visual` | 🟢 `mf` K2 beats `fm`/`diffusion` K20 on distance, 0-viol and cost (20–41×) | 🟢 same corpus | 🔴 **α-Flow has never beaten MeanFlow here.** AF has only ever been run at K=2 and K=100 — never at the MF flagship K=20 | 🔴 the ladder's top rung is missing |
| `uav-pillars` / `s_curve` / `corridor` | 🔴 **regime-split, not a win:** `mf` beats `fm` at K1–K2 (W13) but *loses* at K5–K20 (L25) | 🟡 | ⬜ `af` on UAV was **SiT, 10.0 M — not architecture-matched**; `af_unet` only landed with Gen15 U6 (jobs 25434/25439, resubmitted 2026-09-05) | 🔴/⬜ |

### 5.2 Ladder B

| environment | claim | status |
| :-- | :-- | :-- |
| `avoiding-d3il` (low-K) | DPCC projection wins | 🟢 consistent with the target — but the supporting sweep has a **candidate-fan mismatch** (HF at fan 1, DPCC at fan 4); re-run at B4 parity before it is cited |
| `aligning-d3il-visual` (high-K) | HardFlow-SLSQP beats DPCC | 🔴 **contradicted.** Best-vs-best: HF within ±0.04 m on distance, never higher 0-viol, at a flat 3.3–3.7× cost. `dpcc-t`+tightening reaches 0-viol = 1.00 at 42 ms; HF's best is 0.97 at 146 ms |
| `uav-corridor` (high-K) | HardFlow-SLSQP beats DPCC | 🟡 **the only positive evidence:** the win switches on cleanly at K=5 on the fan-matched corridor sweep — but 1 seed, n=10/cell, PILOT, and pre-honest-geometry |

### 5.3 The three gaps, named

1. **`af_unet > mf` is currently unproven everywhere.** This is the top rung of the headline
   ladder. The Gen14 attack plan is the right instrument: match MF's flagship exactly (unet,
   FiLM v1, K=20, T=0.2, seed 6, 26.4 M) and beat `ratio ≤ 0.267 mean / 0.178 median` **and**
   `0-viol ≥ 0.150` on the unprojected `diffuser` arm. **Stage 1 (unprojected) first — an arm
   that loses unprojected is never ranked projected.**
2. **`HardFlow > DPCC` at high K is contradicted on visual aligning and supported only on a
   pilot-grade UAV sweep.** Either the claim narrows to "on UAV, at K≥5, fan-matched", or a
   mechanism is found for why visual aligning defeats it (candidate: the DPCC tightening margin
   is doing the work that HF's terminal constraint would).
3. **`aligning-d3il` 3-D state has no data.** It is currently a hole in the modality-transfer
   argument: the thesis claims state→visual transfer on *aligning*, but only ever measured
   aligning in the visual variant.

---

## 6. What has to be run to close each gap

Ordered by claim value per GPU-hour. Nothing here is scheduled; it is the queue the target implies.

| # | run | closes | note |
| :-- | :-- | :-- | :-- |
| 1 | α-Flow U-Net at the **MF flagship** (K=20, T=0.2, seed 6, all U9 vision knobs default) on visual aligning, `diffuser` arm | gap 1, Stage 1 | the single highest-value run in the queue; no K sweep |
| 2 | if #1 wins: the same α-Flow config through DPCC and HardFlow-SLSQP arms | gap 1, Stage 2 | also feeds Ladder B |
| 3 | `af_unet` UAV results (25434 / 25439) → rank against `mf`/`fm` at matched K on `pillars_hg` | Ladder A row 4 | first architecture-matched α-Flow on UAV |
| 4 | HardFlow vs DPCC on `avoiding-d3il` at **B4 parity** | Ladder B row 1 | removes the fan confound from the low-K half of the projector claim |
| 5 | UAV `*_hg` geometry re-run of the K-sweep, ≥3 seeds | Ladder B row 3 + all UAV S&C numbers | pilot → paper grade |
| 6 | `aligning-d3il` 3-D state, full four-engine ladder | gap 3 | new environment plumbing; cheapest per-run but highest setup cost |

---

## 7. Rules of engagement — binding on every table in Chapter 6

1. **The Target baseline is pinned once:** diffusion DPCC at `K=20`, `aw=10`, `GaussianDiffusion`,
   at *its own best* projection variant. Other `K` are additional/conservative checks. Pinned in
   `sec:setup:baselines`, never renegotiated per table.
2. **"Beats" = S&C held equal, plus improvement on steps or wall-clock. Both = Pareto dominance.**
   Non-dominated results are written as *trade-offs*, never as "best".
3. **Every table carries backbone + parameter count.** The `unet` row leads; SiT/DiT rows are
   secondary and explicitly labelled confounded.
4. **No aggregation across projectors.** The unit is the cell
   `(engine × projector × geometry × split)`, each with its own `n`. Model-vs-model comparisons
   use each model's own best projector and always name it.
5. **Degenerate HardFlow rows are tagged and excluded from claims**, never silently averaged.
6. **`budget_ms` / 33 Hz never appears as a pass/fail criterion.**
7. **Negative results are kept.** The iMF refutation, the α-cliff, the `mf`-loses-above-K5 split
   on UAV, and the visual-aligning HardFlow loss all belong in `sec:disc:negative`. A thesis whose
   every hypothesis was confirmed is less credible, not more.

---

## 8. Mapping to the bone

| target element | thesis home |
| :-- | :-- |
| Ladder A, mechanism | `sec:bg:fewstep`, `sec:method:engine`; RQ1 + RQ2 |
| Ladder A, results | `sec:res:state`, `sec:res:fewstep`, `sec:res:visual`, `sec:res:uav` |
| Ladder B, mechanism + degeneracy | `sec:bg:mpc:trajopt`, `sec:method:constraints`, `sec:res:constraints:degenerate`; RQ3 |
| Regime split, the central tension | `sec:disc:interpretation` |
| Visual-aligning env methodology (§4.1) | `sec:method:conditioning` + `sec:setup:tasks` |
| UAV env + PID methodology (§4.2) | `sec:method:deployment` + `sec:setup:tasks` |
| Honest-geometry finding | `sec:setup:tasks` **and** `sec:disc:threats` |
| Definition of "better" | `sec:setup:metrics:pareto` |
| Provenance / inherited-vs-own | `sec:method:dpcc` + `app:repro` |

---

## 9. Kill criteria — when the target must be downgraded

Write these now, honour them later; deciding after seeing the numbers is how a thesis becomes
unfalsifiable.

- **α-Flow fails run #1 on the matched flagship** → the headline ladder drops to
  `mf > fm > diffusion`, and α-Flow moves to an ablation on curriculum/annealing
  (`sec:res:ablations`) plus a negative-result subsection. **The title and the RQ set do not
  change.**
- **HardFlow does not win on the honest-geometry UAV re-run either** → RQ3's answer becomes
  *"no — the per-step projector dominates in every regime we could measure, and here is the
  mechanism"*. That is a publishable answer; it is not a failure. Do **not** keep sweeping for a
  configuration where it wins.
- **The UAV scenes remain infeasible after `*_hg`** → UAV is demoted from a claim-bearing
  environment to a feasibility/methodology chapter, and RQ4's embodiment half is answered on
  what the geometry actually permits.
- **`mf`'s UAV loss above K=5 replicates at ≥3 seeds** → Ladder A is restated as
  regime-conditional (`mf > fm` **in the few-step regime**), everywhere, including the abstract.
