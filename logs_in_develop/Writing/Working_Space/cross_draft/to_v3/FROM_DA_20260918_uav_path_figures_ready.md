# TO v3 — three flown-path figures are built and in the store

**2026-09-18 · from the DA side (figure pipeline), not from another draft.**
Closes the figure half of **D10** in
[`PENDING_20260916_missing_data_and_analyses.md`](../../data_status/PENDING_20260916_missing_data_and_analyses.md).
Nothing in `v3/` has been touched — the `.tex` edits below are yours.

## What is ready

The rollout artefacts were staged off the cluster on 2026-09-18 and three figures now exist in the
official store, `Data_Analysis/DA_in_Paper/figures/da/`:

| figure file | the `\hole` / `\todofigure` it fills | panels |
| :-- | :-- | :-- |
| `fig_uav_scurve_paths` | `fig:uav-scurve-paths` (`\todofigure` + `\hole`), companion of `tab:uav-controller` | 2 — sampling-based MPC, proportional stop-go |
| `fig_uav_pillars_paths` | `sec:res:uav:pillars` — "figure of the flown paths" | 4 — one per model, each under its best projection |
| `fig_uav_corridor_paths` | `sec:res:uav:corridor` — "figure of the flown paths" | 10 — three models × 3 budgets, baseline last |

To get them into the draft:

```bash
python3 plotting/export_to_draft.py v3     # AFTER your \includegraphics lines exist
```

The export copies only what the `.tex` actually includes, so it will not pick these up until you have
replaced the holes. Everything upstream of that is done.

## What each figure shows, so the caption matches the drawing

All three are the same picture as `fig:constraints-uav` — the scene from above, its obstacles inflated
by the vehicle radius and the tightened twin dashed — with the executed paths drawn over it. That is
literally the same code (`scenes._uav_constraint_panel`), so the constraint set under a path is the one
the projection saw.

A flight carries **two independent outcomes**, and the figure never collapses them:

- **colour** — green: collision-free; red: entered an inflated obstacle
- **stroke** — solid: reached the goal; dashed: did not
- **end mark** — filled: reached the goal; hollow: did not

They are independent because a flight can reach the goal *through* a pillar. Each panel's subtitle
carries its own counts (`9/10 reached · 9/10 clean`), so a caption does not need to repeat them.

## Three things in the drawings that the prose should not contradict

1. **The s-curve figure is the controller story, and it is stark.** Sampling-based MPC flies the full
   S and 2 of 3 flights reach the goal; proportional stop-go stalls at the second passage and **0 of 10**
   reach it. Both panels are the *unprojected* plan (variant `diffuser`) on purpose — the figure is about
   the controller, so the plan must not be filtered first. Neither controller is collision-free here
   (0/3 and 0/10), which is expected without projection and should not be read as a projection result.
2. **The pillars panels do not share a projection variant.** Each model is drawn under the configuration
   `tab:uav-pillars-best` reports as its best: endpoint-with-random-selection for average-velocity
   matching, plain endpoint for instantaneous-velocity, per-step tightened for consistency-interpolated
   and for the diffusion baseline. The subtitle of each panel names it. A caption saying "under the same
   projection" would be wrong.
3. **The corridor figure is model-major**, one row per model across budgets 1/3/5, with the diffusion
   baseline alone in the last row at the only budget it has (20). The baseline's row is 12/12
   collision-free and 0/12 reached, which is the same trade-off `tab:uav-corridor` reports.

## What is NOT ready

- **The D3IL-aligning box paths** (`sec:res:aligning:projection`, "figure of the ten contexts") are still
  a `\hole`. That cell was staged before a naming bug in the fetch script was fixed, so it is absent from
  the download; it needs one more cluster run of the stager. Keep that hole.
- **`fig:raw-plans`** still has its five empty panels (ledger R5): those need a cluster *evaluation* that
  has not been submitted, not a download.

## Provenance, if a reviewer asks

Paths are `obs_all[:, 3:5]` — the executed position, not the reference — of each run's `<variant>.npz`,
with per-episode outcomes from the matching `results.json`. 173 flights across 16 evaluated cells.
The extract is committed at `Data_Analysis/DA_in_Paper/data/uav_paths.json`; the run folder and variant
behind every panel are declared in `plotting/extract/uav_paths.py::PANELS`, and the download itself is
recorded in
[`LEDGER_20260918_v3_figure_artefact_fetch.md`](../../../../../Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md).
The raw drop lives under the gitignored `temp/18-09-2026/` and will not survive a clean checkout — the
JSON extract is the record, and the ledger says how to recreate the drop.
