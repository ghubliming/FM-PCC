# TO v3 — `fig:raw-plans` is complete: four panels landed, the eighth is provably impossible

**2026-09-19 · from the DA side (figure pipeline), not from another draft.**
Closes ledger row **R5**. Nothing in `v3/` has been touched — the `.tex` edits below are yours.

## What is ready

Four panels of the `fig:raw-plans` matrix were **fetched, not run**. `REQUEST_20260916`'s own
"CORRECTION 2026-09-18 (v3.39)" established that the August 20-trials evaluations had already written
these dashboards; the earlier claim that they needed a fresh cluster evaluation was wrong. They were
staged off the cluster this morning and are in the official store, `DA_in_Paper/figures/demo/`:

| figure file | matrix cell | model |
| :-- | :-- | :-- |
| `fig_raw_plans_fm_K1` | instantaneous-velocity, $\nfe=1$ | naive flow matching (`aw10`) |
| `fig_raw_plans_fm_K2` | instantaneous-velocity, $\nfe=2$ | same |
| `fig_raw_plans_af_K1` | consistency-interpolated, $\nfe=1$ | CI-MeanFM (U-Net, $\alpha_\text{end}=0.2$) |
| `fig_raw_plans_af_K2` | consistency-interpolated, $\nfe=2$ | same |

With the three that already existed — average-velocity $\nfe=1,2$ and diffusion $\nfe=1$ — the matrix
is **seven of eight**.

```bash
python3 plotting/export_to_draft.py v3     # AFTER your \includegraphics lines exist
```

The export copies only what the `.tex` actually includes, so it will not pick these up until you have
replaced the `\todofigure` boxes. Everything upstream of that is done.

## The eighth cell: say it is impossible, not missing

**Diffusion at $\nfe=2$ cannot be fetched and cannot be run.** Every `GaussianDiffusion` folder in the
batch is `K20`, because a diffusion model's step count is fixed when its noise schedule is discretised
**at training time**. The $\nfe=1$ diffusion panel exists only because a model was separately *trained*
at $K=1$. Filling this cell needs a $K{=}2$-trained checkpoint that does not exist.

This is worth a sentence rather than a silent gap: it is the same property of diffusion the chapter
argues about elsewhere — the budget is not a free parameter at sampling time the way it is for a flow
model. An empty cell that the caption explains is a result. An empty cell that looks like an oversight
is not. `sources.PLANNED` now carries exactly one entry, and it says this.

## What the four new panels show

Read off the rendered panels, not inferred:

- **Naive flow matching** throws a wide, jagged plan fan that leans hard into the **left forbidden
  wedge** at both budgets — the fan is outside the feasible corridor for much of its length.
- **CI-MeanFM** is noisy at $\nfe=1$ and markedly tighter at $\nfe=2$.
- Both are looser than **average-velocity matching** at the same budgets, whose fan stays inside the
  corridor and threads the obstacle field.
- The **diffusion baseline at $\nfe=1$** is the outlier: near-scribble, filling the panel.

That ordering — average-velocity tightest, then consistency-interpolated, then naive FM, with the
one-step baseline collapsing — is the figure's whole argument, and it now has every panel it needs to
make it.

## Two facts that belong in the caption

1. **The panels are one episode, not an aggregate.** Each is the last column of episode 1 of its run:
   every plan the planner proposed during that episode, overlaid on the scene, with no projection. A
   caption saying "typical" or "representative" is a claim the figure does not support; "one episode"
   is what it shows.
2. **The seven panels come from two evaluation campaigns.** The three older ones are episode 1 of a
   two-episode run (`n_trials: 2`); the four new ones are episode 1 of a ten-episode run
   (20 trials). Same scene, same seed 6, same `both-hard` geometry, same axes and data range — but I
   **cannot** assert from the artefacts that episode 1 is the identical initial context in both
   campaigns, because the saved `.npz` carries only scalars. So do not write that the panels share a
   task instance. "Same scene and seed" is safe; "the same episode" is not.

## What I had to do to make them sit in one matrix, in case a reviewer asks

The two campaigns render their dashboards at different sizes — $3000\times1000$ for two episodes,
$3000\times5000$ for ten. The six columns are at the *same* x, but the ten-episode layout has a title
band that squeezes row 1: its plot frame is $332\times326$ px against the older $332\times350$, over
an identical data range. Untreated, the four new panels would sit 7 % flatter than the three old ones
and their obstacles would read as **ellipses beside circles** in the same figure.

So the crop box for the new panels is chosen such that scaling the cut to $403\times400$ lands the plot
frame on exactly $(56,28)$–$(388,378)$, where the older panels already have it. All seven now share one
canvas and one data aspect — verified, not assumed. The stretch is declared in
`sources.VENDORED_CROP` and applied by `prep/crop_vendored.py`, so it is reproducible and is recorded in
`figures/MANIFEST.md` provenance for every affected panel.

## Provenance, if a reviewer asks

Each panel is the last column of episode 1 of the seed-6 `both-hard` `diffuser` dashboard — the
unprojected arm — written by the evaluation as it ran. Run folders:
`…FlowMatchingODE_a1.5_b1.0_aw10/H8_K{1,2}_…_msg20trials` and
`…AlphaFlowODE_aw10_bbunet_tslogit_normal_…/H8_K{1,2}_…_msgafon02_s6`. The full dashboards are committed
at `DA_Result_Curated_MD/Report_20260903_AF_UNet/fig8{a,b,e,f}_*.png` (that report's §8 reserved those
names), so the build path holds no gitignored file. The download is recorded in
[`LEDGER_20260918_v3_figure_artefact_fetch.md`](../../../../../Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md),
2026-09-19 section.

> **A trap that was avoided, recorded because it would silently produce a wrong figure.** A stub run
> folder with the *same* name sits under `…FlowMatchingODE_a1.5_b1.0_aw1` — 6 files, 153 KiB — beside
> the real `aw10` run of 4706 files. The fetch cell pins `aw10` literally instead of globbing the model
> folder. A glob would have matched the stub and reported success on nothing.

## Also fetched today, not yet a figure

The **D3IL-aligning box paths** (`sec:res:aligning:projection`, "figure of the ten contexts") came down
in the same archive — the cell that the `_train_set` naming bug cost us on 09-18 now resolves, and the
three variants are on disk with their rollout `.npz`. The builder for it is not written yet, so **keep
that `\hole` for now**; it is no longer blocked on the cluster.

## Unchanged from yesterday's note, and still true

[`FROM_DA_20260918_uav_path_figures_ready.md`](FROM_DA_20260918_uav_path_figures_ready.md) carries a
correction block at its top that matters more than this note:

- `fig_uav_pillars_paths` — **do not include**; the scene is withheld.
- `fig_uav_scurve_paths` — figure fine, but its caption must say the drone **flips** (all ten flights
  aborted `inverted`), not that it stalls.
- `fig_uav_corridor_paths` — unchanged, use it.
