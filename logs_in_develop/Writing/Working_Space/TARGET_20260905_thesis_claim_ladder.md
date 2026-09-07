# TARGET — what the thesis has to prove

**Created:** 2026-09-05 · **Type:** target statement only
**Scope:** whole thesis (`Bone/thesis_bone.tex`)

> 🔴 **This file states goals, not progress.** No results, no numbers, no DA.
> Evidence and status live in [`Data_Analysis/DA_Result_Curated_MD/`](../../../Data_Analysis/DA_Result_Curated_MD/),
> indexed by `NOTEBOOK_20260829_key_headlines.md`. Never record a finding here.

**Companions:** [`../Auxiliary/NOTES_open_questions.md`](../Auxiliary/NOTES_open_questions.md) ·
[`../Auxiliary/NOTES_dpcc_lineage.md`](../Auxiliary/NOTES_dpcc_lineage.md) ·
[`../Auxiliary/NOTES_paper_map.md`](../Auxiliary/NOTES_paper_map.md)

---

## 0. The claim

> Under an identical projector and an identical U-Net, the deterministic ODE-transport family beats
> DPCC's diffusion engine in a strict order, across two modalities and two embodiments; where the
> task forces a high step budget, HardFlow-SLSQP in turn beats the DPCC projection arm; and the
> projector's candidate-selection machinery is shown to be dead weight and removed.

---

## 1. Goal A — the engine ladder

**`af_unet` ≥ `mf` > `fm` > `diffusion` (DPCC)** — same projector, same U-Net, matched params, per environment.

- Held fixed: backbone + param count, projector arm, data, normalisation, horizon, seeds, harness.
- "Better" = **Pareto dominance**: equal success *and* equal constraint satisfaction, strictly fewer NFE *and* lower wall-clock. Anything else is a trade-off.
- The argument is **structural**: `K` is inference-time for the flow family, training-time for diffusion. Argue the mechanism in `sec:bg:fewstep` before any table.
- Architecture-matched (`unet`) rows lead; SiT/DiT rows are secondary and labelled confounded.

## 2. Goal B — the projector ladder, regime-split

**low-`n_genuine` → DPCC per-step projection · high-`n_genuine` → HardFlow-SLSQP**

- The binding quantity is `n_genuine = max(K − int(A·K), 1) − 1`, **not `K`**. At `n_genuine = 0` HardFlow runs no HardFlow math; those rows are tagged and excluded from claims.
- Both arms must run at **matched activation threshold** (HF `activation_threshold` ≡ DPCC `diffusion_timestep_threshold`) and **matched candidate fan**. An unmatched comparison is not a comparison.
- Prove the split is a **measured domain of validity**, not a search for a favourable cell.
- Central tension to resolve in `sec:disc:interpretation`: Goal A pushes `K` down, Goal B needs genuine steps. Show they meet at moderate `K` with a low threshold.

## 3. Goal C — remove the candidate-selection machinery

**Prove `dpcc-{r,c,t}` does not earn its compute, and ship the final system with one rule (or none).**

- Strong form: at fan 1 the three rules are the same computation — running all three is pure waste.
- Weak form: at fan > 1 they diverge, but no rule is reliably best across engines and tasks.
- Deliverable: one default, the others disabled; the fan's cost handled by **parallelising the projector**, not by shrinking it.
- This is a *simplification* claim. It must survive being wrong: if a rule proves load-bearing, keep it and say so.

## 4. Benchmark matrix

| environment | modality / embodiment | origin | role |
| :-- | :-- | :-- | :-- |
| `avoiding-d3il` | state, manipulator | DPCC's own benchmark | baseline home turf; low-`K` regime |
| `aligning-d3il` (3-D) | state, manipulator | D3IL, imported by us | ladder is not avoiding-specific |
| `aligning-d3il-visual` | **visual**, manipulator | built by us | modality transfer; high-`K` regime |
| `uav-pillars` (+ `s_curve`, `corridor`) | state, **aerial** | built by us | embodiment transfer |

**Claimed per environment, never pooled.**

## 5. Methodology the thesis owes

| # | must document |
| :-- | :-- |
| 5.1 | **Visual-aligning env construction** — rendering/observation pipeline on D3IL, vision encoder provenance, FiLM conditioning, how the param match is kept honest, and the success-criterion caveat |
| 5.2 | **UAV env construction** — scene generation for `empty`/`corridor`/`s_curve`/`pillars`, expert data source, the **PID low-level controller** and the plan→setpoint→thrust chain, and the honest-geometry finding (scene feasibility vs. tracking error) |
| 5.3 | **Provenance** — inherited vs. own, file-level, with licences (`app:repro`) |

## 6. Rules of engagement — binding on every table

1. Target baseline pinned once: diffusion DPCC `K=20`, `aw=10`, `GaussianDiffusion`, at its own best projection variant. Never renegotiated per table.
2. Every table carries **backbone + parameter count**.
3. **No aggregation across projectors.** Unit = cell `(engine × projector × geometry × split)`. Model-vs-model uses each model's own best projector, always named.
4. Degenerate HardFlow rows tagged, never averaged in.
5. UAV `budget_ms` / 33 Hz is a data-rate artefact — never a pass/fail criterion.
6. Negative results are kept and written up (`sec:disc:negative`).

## 7. Mapping to the bone

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
| α-Flow never beats MeanFlow at a matched flagship | ladder drops to `mf > fm > diffusion`; α-Flow becomes a curriculum ablation + negative result. **Title and RQ set unchanged.** |
| HardFlow stays non-dominated but not superior | RQ3 answers "regime-dependent, not superior", with the mechanism. **Stop sweeping for a winning cell.** |
| UAV scenes stay infeasible after honest geometry | UAV demoted to a feasibility/methodology chapter; RQ4's embodiment half answered on what the geometry permits |
| a selection rule proves load-bearing | Goal C narrows to "remove the redundant ones"; do not force the deletion |
| an engine's win is regime-conditional | say so **everywhere**, including the abstract |
