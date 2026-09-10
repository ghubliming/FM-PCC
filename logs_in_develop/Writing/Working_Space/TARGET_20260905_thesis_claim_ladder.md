# TARGET — what the thesis has to prove

**Created:** 2026-09-05 · **Revised:** 2026-09-10 (topic framing, benchmark roles, protocol) · **Type:** target statement only
**Scope:** whole thesis (`Bone/thesis_bone.tex`)

> 🔴 **This file states goals, not progress.** No results, no numbers, no DA.
> Evidence and status live in [`Data_Analysis/DA_Result_Curated_MD/`](../../../Data_Analysis/DA_Result_Curated_MD/),
> indexed by `NOTEBOOK_20260829_key_headlines.md`. Never record a finding here.

**Companions:** [`../Auxiliary/NOTES_open_questions.md`](../Auxiliary/NOTES_open_questions.md) ·
[`../Auxiliary/NOTES_dpcc_lineage.md`](../Auxiliary/NOTES_dpcc_lineage.md) ·
[`../Auxiliary/NOTES_paper_map.md`](../Auxiliary/NOTES_paper_map.md) ·
[`../Auxiliary/NOTES_method_naming.md`](../Auxiliary/NOTES_method_naming.md)
**Discharged branch:** [`fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md`](fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md) — §8 criterion 1 has **fired**; that file is the operative engine plan.
**Data status (never in this file):** [`data_status/DATASTATUS_20260910_v3_entry_readiness.md`](data_status/DATASTATUS_20260910_v3_entry_readiness.md)

---

## 0. The claim

**As pre-registered, 2026-09-05** — kept verbatim, because a target that is quietly rewritten
after the data arrives is not a target:

> Under an identical projector and an identical U-Net, the deterministic ODE-transport family beats
> DPCC's diffusion engine in a strict order, across two modalities and two embodiments; where the
> task forces a high step budget, HardFlow-SLSQP in turn beats the DPCC projection arm; and the
> projector's candidate-selection machinery is shown to be dead weight and removed.

**As it stands, 2026-09-10** — after §8 criterion 1 fired (the strict order did not survive; the
ordering *within* the flow family is not measurable on these tasks):

> Under an identical projector and an identical U-Net, **the average-velocity engine (MeanFlow)
> separates from every other engine — diffusion, naive flow matching and the bootstrapped-target
> variant — and Pareto-dominates the DPCC baseline**, across a task ladder of increasing difficulty
> (2-D planar state → 3-D state → 3-D visual) and a second embodiment whose dynamics are
> underactuated and open-loop unstable; where the task forces genuine intermediate steps, **endpoint
> projection** in turn beats the per-step DPCC projection arm; and the projector's
> candidate-selection machinery is shown to be dead weight and removed.

🏷️ **Naming.** *"HardFlow-SLSQP"* is retired from thesis prose — arm C is **in-ODE endpoint
projection**, arm B **per-step iterate projection**. See [`../Auxiliary/NOTES_method_naming.md`](../Auxiliary/NOTES_method_naming.md).

---

## 0.5 The three topics, and what each is for

The thesis has **three experimental topics**, not four environments. Each answers a different
question, and they are ordered by difficulty of the control problem:

| # | topic | environments | the question it answers | role |
| :-- | :-- | :-- | :-- | :-- |
| **1** | **Avoiding — the foundation** | `avoiding-d3il` (2-D planar state) | *Does the engine swap pay, on the baseline's own benchmark, at its own protocol?* | **carries the paper.** Multi-seed, powered, Pareto claim |
| **2** | **Aligning — the harder task** | `aligning-d3il` (**3-D state**) → `aligning-d3il-visual` (**3-D visual**) | *Does the result survive a harder control problem, and then a harder observation problem?* | **two legs.** State first, then visual — the difficulty is added **one axis at a time** |
| **3** | **UAV — the other embodiment** | `uav-corridor` · `uav-pillars` · `uav-s_curve` | *Does it survive a plant that is **underactuated and open-loop unstable**, where a plan must be dynamically feasible, not merely kinematically feasible?* | **embodiment transfer** + the controller study |

**Why topic 2 needs both legs.** Avoiding is 2-D and planar; aligning is **3-D** (position *and*
orientation of a pushed box, `action_dim = 3`) *and* adds a target pose. Running only the visual leg
confounds two increments — *harder control* and *harder perception* — in one step. The state leg
isolates the first; the visual leg then adds the second on top of a known answer.

**Why topic 3 is a different kind of hard.** The manipulator is **fully actuated and quasi-static**:
it holds its configuration under gravity compensation, and a plan is tracked as a *kinematic*
sequence of setpoints (IK → joint PD). The quadrotor is **underactuated** (4 rotor inputs for 6
pose DOF; thrust acts along body-*z* only, so it must tilt to translate) and its hover is an
**open-loop-unstable equilibrium** requiring continuous closed-loop stabilisation at a rate far
above the planner's. That is the transfer being claimed — *from a statically stable, fully-actuated,
kinematically-controlled plant to an underactuated, open-loop-unstable, second-order one* — and it
is why the UAV stack needs a cascaded geometric controller and a multi-rate loop at all.

---

## 1. Goal A — the engine ladder

> 🔴 **§8 criterion 1 has fired (2026-09-10).** The four-rung ladder below is the *pre-registered*
> goal; the operative one is **`mf` ≫ { `af`, `fm`, `diffusion` }** — one engine separates, three
> cluster — with the bootstrapped-target arm carried as a **negative result with a derived
> mechanism**. The reasoning, the mechanism and the narrative are in
> [`fallback_target/FALLBACK_20260910…`](fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md).
> The rest of this section stands unchanged: it is what the goal *was*, and the rules it sets are
> still binding.

**`af_unet` ≥ `mf` > `fm` > `diffusion` (DPCC)** — same projector, same U-Net, matched params, per environment.

- Held fixed: backbone + param count, projector arm, data, normalisation, horizon, seeds, harness.
- "Better" = **Pareto dominance**: equal success *and* equal constraint satisfaction, strictly fewer NFE *and* lower wall-clock. Anything else is a trade-off.
- The argument is **structural**: `K` is inference-time for the flow family, training-time for diffusion. Argue the mechanism in `sec:bg:fewstep` before any table.
- Architecture-matched (`unet`) rows lead; SiT/DiT rows are secondary and labelled confounded.

## 2. Goal B — the projector ladder, regime-split

**low-`n_genuine` → per-step iterate projection (DPCC) · high-`n_genuine` → in-ODE endpoint projection (arm C)**

- The binding quantity is `n_genuine = max(K − int(A·K), 1) − 1`, **not `K`**. At `n_genuine = 0` HardFlow runs no HardFlow math; those rows are tagged and excluded from claims.
- Both arms must run at **matched activation threshold** (arm C's `activation_threshold` ≡ DPCC's `diffusion_timestep_threshold`) and **matched candidate fan**. An unmatched comparison is not a comparison.
- **Select one citable cell per environment** rather than reporting the whole sweep as if every cell were a claim. The two arms solve the *same* program on a *different point*, so where they tie, say they tie.
- Prove the split is a **measured domain of validity**, not a search for a favourable cell.
- Central tension to resolve in `sec:disc:interpretation`: Goal A pushes `K` down, Goal B needs genuine steps. Show they meet at moderate `K` with a low threshold.

## 3. Goal C — remove the candidate-selection machinery

**Prove `dpcc-{r,c,t}` does not earn its compute, and ship the final system with one rule (or none).**

- Strong form: at fan 1 the three rules are the same computation — running all three is pure waste.
- Weak form: at fan > 1 they diverge, but no rule is reliably best across engines and tasks.
- Deliverable: one default, the others disabled; the fan's cost handled by **parallelising the projector**, not by shrinking it.
- This is a *simplification* claim. It must survive being wrong: if a rule proves load-bearing, keep it and say so.

## 4. Benchmark matrix

Ordered by **what each one adds**, not by when it was run. Roles only — status lives in
[`data_status/`](data_status/DATASTATUS_20260910_v3_entry_readiness.md).

| topic | environment | what it adds over the row above | origin | role |
| :-- | :-- | :-- | :-- | :-- |
| **1** | `avoiding-d3il` | — (2-D planar state, halfspace geometry) | DPCC's own benchmark | **the foundation.** Two protocol tiers: theirs (5 seeds × 2 trials) then ours (5 seeds × 20) |
| **2a** | `aligning-d3il` (**3-D state**) | **harder control**: 3-D actions, box pose + target pose | D3IL, imported by us | isolates *harder control* from *harder perception* |
| **2b** | `aligning-d3il-visual` | **harder perception**: pixels instead of state | built by us | modality transfer, on top of a known state answer |
| **3** | `uav-corridor` | **underactuated, open-loop-unstable plant** | built by us | the intended UAV headline scene |
| **3** | `uav-pillars` | a constraint set that actually discriminates | built by us | the UAV ranking scene |
| **3** | `uav-s_curve` | an aggressive reference the tracker cannot hold | built by us | **controller-limit case**, not a ranking scene |

**Claimed per environment, never pooled.** Each scene is also labelled by **evaluation regime**
(all-pass / discriminating / all-fail), because a scene where every arm passes cannot rank engines
and must not be presented as if it could.

## 5. Methodology the thesis owes

| # | must document |
| :-- | :-- |
| 5.1 | **Visual-aligning env construction** — rendering/observation pipeline on D3IL, vision encoder provenance, FiLM conditioning, how the param match is kept honest, and the success-criterion caveat |
| 5.2 | **UAV env construction** — scene generation for `empty`/`corridor`/`s_curve`/`pillars`, expert data source, the **PID low-level controller** and the plan→setpoint→thrust chain, and the honest-geometry finding (scene feasibility vs. tracking error) |
| 5.3 | **Provenance** — inherited vs. own, file-level, with licences (`app:repro`) |
| 5.4 | **The evaluation protocol, per topic, and why it differs.** Avoiding: seeds × `n_trials`, reported at **both** DPCC's protocol and ours, with the measured cost of the difference. Aligning/visual: **`n_contexts` × `n_trajectories_per_context` — there is no `n_trials`**; contexts are enumerated and held fixed, which is what makes every comparison *paired*. UAV: seeds × rollouts, one scene at a time. A table that does not say which protocol produced it is not readable |
| 5.5 | **Dynamic vs. kinematic feasibility.** Why a manipulator plan needs only kinematic feasibility while a quadrotor plan must be dynamically feasible (thrust/attitude limits, second-order dynamics) — and what that does to the constraint set |

## 6. Rules of engagement — binding on every table

1. Target baseline pinned once: diffusion DPCC `K=20`, `aw=10`, `GaussianDiffusion`, at its own best projection variant. Never renegotiated per table.
2. Every table carries **backbone + parameter count**.
3. **No aggregation across projectors.** Unit = cell `(engine × projector × geometry × split)`. Model-vs-model uses each model's own best projector, always named.
4. Degenerate HardFlow rows tagged, never averaged in.
5. UAV `budget_ms` / 33 Hz is a data-rate artefact — never a pass/fail criterion.
6. Negative results are kept and written up (`sec:disc:negative`).
7. **Every table names its protocol** — seeds × trials, or contexts × trajectories. Never a bare *n*.
8. **Mechanism names, not artefact tags** (`../Auxiliary/NOTES_method_naming.md`): *iterate
   projection* / *endpoint projection*, never "HF-SLSQP"; the selection rules are named, never
   lettered; never bare "diffuser".

## 7. Mapping to the bone

| topic | results home |
| :-- | :-- |
| 1 — avoiding, the foundation | `sec:res:state` (+ `sec:res:fewstep` for the K-ladder) |
| 2a — aligning, 3-D state | `sec:res:state` (second subsection) or its own, per RQ4 |
| 2b — aligning, visual | `sec:res:visual` |
| 3 — UAV | `sec:res:uav` (+ the controller study in `sec:method:deployment` / `sec:setup:tasks`) |

| goal | thesis home |
| :-- | :-- |
| A — mechanism / results | `sec:bg:fewstep`, `sec:method:engine` · `sec:res:{state,fewstep,visual,uav}` · RQ1, RQ2 |
| B — mechanism / degeneracy | `sec:bg:mpc:trajopt`, `sec:method:constraints`, `sec:res:constraints:degenerate` · RQ3 |
| C — selection machinery | `sec:method:constraints` + `sec:res:ablations` |
| Regime split | `sec:disc:interpretation` |
| Methodology 5.1 / 5.2 | `sec:method:*` + `sec:setup:tasks`; honest geometry also in `sec:disc:threats` |
| Definition of "better" | `sec:setup:metrics:pareto` |
| Transfer across modality + embodiment | RQ4 |

## 8. Kill criteria — decided now, honoured later

| if | then |
| :-- | :-- |
| ~~α-Flow never beats MeanFlow at a matched flagship~~ **🔴 FIRED 2026-09-10** | ~~ladder drops to `mf > fm > diffusion`~~ → the measured shape is **`mf` ≫ {`af`, `fm`, `diffusion`}**; the bootstrapped-target arm becomes a negative result **with a derived mechanism**. **Title and RQ set unchanged.** Discharged in [`fallback_target/`](fallback_target/FALLBACK_20260910_engine_claim_without_alphaflow.md) |
| the **3-D state** aligning leg (2a) does not land in time | topic 2 ships on the visual leg alone, and the thesis says plainly that *harder control* and *harder perception* were added in one step rather than two. **Do not claim the state→visual transfer** on a leg that was never run |
| a V_A **retrain** does not move the scene out of the all-fail regime | the over-strictness is a property of the **constraint set**, not of training — say so, keep the funnel, and stop treating it as a defect to be fixed |
| `corridor_ball` does not bind (unprojected arm still reports 0 violations) | the UAV headline scene stays constraint-trivial; `pillars` carries the UAV ranking alone and the chapter says why |
| HardFlow stays non-dominated but not superior | RQ3 answers "regime-dependent, not superior", with the mechanism. **Stop sweeping for a winning cell.** |
| UAV scenes stay infeasible after honest geometry | UAV demoted to a feasibility/methodology chapter; RQ4's embodiment half answered on what the geometry permits |
| a selection rule proves load-bearing | Goal C narrows to "remove the redundant ones"; do not force the deletion |
| an engine's win is regime-conditional | say so **everywhere**, including the abstract |
