# Methodology Sources — the auxiliary work the thesis owes

**Created:** 2026-09-07 · **Type:** source map, not results
**Serves:** [`TARGET_20260905_thesis_claim_ladder.md`](../../Working_Space/TARGET_20260905_thesis_claim_ladder.md) §5
("Methodology the thesis owes") and the `sec:method:*` / `sec:setup:*` blocks of the bone.

---

## Why this folder exists

The thesis's *claims* live in `Data_Analysis/DA_Result_Curated_MD/`. Its *apparatus* does not live
anywhere writable — it is spread across ~1000 dev-log MDs in `logs_in_develop/`, mostly as
`CHANGELOG`/`Fix_N` documents written while the code was moving. None of it is in a form a chapter
can be written from.

**These files are not new findings.** Each one maps a body of auxiliary work to the section of the
thesis that has to describe it, states what is already documented and where, and — the part that
matters — **names what is missing**. A hole named here is a writing task, not a research task.

> 🔴 **Nothing in this folder is a result.** No numbers that constitute a claim. If a number appears
> it is an apparatus parameter (a scene dimension, a control rate, a mesh source), and it is cited.

---

## The files

| file | covers | thesis home |
|---|---|---|
| [`AUX_uav_model_and_scenes.md`](AUX_uav_model_and_scenes.md) | the Skydio X2 asset, its provenance and parameters; the four MuJoCo scenes and their geometry | `sec:setup:tasks`, `sec:method:uav` · TARGET §5.2 |
| [`AUX_uav_control_stack.md`](AUX_uav_control_stack.md) | the cascaded PID, the MJPC thrust-control line, and the plan → setpoint → thrust chain | `sec:method:uav` · TARGET §5.2 |
| [`AUX_uav_expert_data.md`](AUX_uav_expert_data.md) | the two-layer expert-data pipeline, the D3IL-shaped pkl schema, rejection criteria | `sec:setup:data` · TARGET §5.2 |
| [`AUX_visual_aligning_env.md`](AUX_visual_aligning_env.md) | how `aligning-d3il-visual` was built on D3IL; rendering, encoder provenance, FiLM | `sec:method:visual`, `sec:setup:tasks` · TARGET §5.1 |
| [`AUX_rendering_and_gif_pipeline.md`](AUX_rendering_and_gif_pipeline.md) | offscreen rendering, **state injection** ("replay without re-flying"), GIF/overlay artefacts | `sec:setup:tasks`, `app:repro` · TARGET §5.1/5.2 |
| [`AUX_constraint_geometry.md`](AUX_constraint_geometry.md) | the constraint sets beside the main env — 3-D bounds, halfspaces, obstacles, tightened/ablated variants, feasibility and honest geometry | `sec:method:constraints`, `sec:disc:threats` · TARGET §5.2 |

## How to use them when writing

1. Write the section from the **"What the thesis must say"** block — it is already ordered as prose.
2. Pull apparatus numbers from the **"Parameters of record"** table; every row cites a file.
3. Check the **"Holes"** list before claiming completeness. A hole is either filled by reading one
   more dev log, or it is declared as a limitation in `sec:disc:threats`.

## Standing rules inherited from the TARGET

- **Provenance is a deliverable** (§5.3): every asset is inherited or ours, file-level, with licence.
  "Copy, don't generate" is the repo's explicit policy for physics XML and is worth stating.
- **UAV `budget_ms` / 33 Hz is a data-rate artefact**, never a real-time pass/fail criterion (§6.5).
- **The honest-geometry finding is a strength, not an embarrassment** — a measured, self-diagnosed
  benchmark defect belongs in methodology and threats-to-validity.
