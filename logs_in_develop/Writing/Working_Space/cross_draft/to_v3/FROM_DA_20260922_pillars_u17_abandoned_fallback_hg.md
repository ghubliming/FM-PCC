# TO v3 — UAV-pillars: the enlarged-geometry campaign (U17) is ABANDONED; fall back to `pillars_hg`

**2026-09-22 · from the DA side, on the author's decision.** Nothing in `v3/` has been touched. This
changes what the pillars section says, what its tables are, and removes the v3.56 templates.

## 1 · What happened

The `pillars_xl` campaign (R23, Gen15 U17) ran to completion: 94 cells × 10 flights, seed 6, all four
models, per-step and endpoint projection, jobs 25970–25995. **Under the thesis metric it is an all-zero
grid**: S&C = 0.00 and collision-free = 0.00 in every cell, *unprojected rows included*. No flight of the
940 was collision-free.

The per-flight rows show why, and it is not noise. The enlarged keep-out (0.35 + 0.31 m rotor reach =
0.66 m) closes the corridor between the two pillar rows by only 6 cm per side, while that corridor is
physically 0.96 m wide against a 0.62 m vehicle. **Every projector in the study — DPCC per-step and
HardFlow endpoint — routes the plan into that forbidden centre corridor** instead of 0.15 m outward into
the legal detour: contact-free, across the finish line (relaxed success 1.00), ~1 m to the side of the
goal point, 0.4 m inside the keep-out at every column. Per-step at $\nfe=5$ under $r$/$c$ then weaves
between lanes and hits pillars on 8–10 flights of 10. HardFlow reports non-converged SLSQP on every
variant. The scene is degenerate in the opposite direction to `pillars_hg`: there the projector had
nothing to repair, here it cannot repair anything. Fixing it is a third geometry plus ~5 GPU-days with no
guarantee — not two weeks out.

Separately, the diffusion baseline's projected block cannot be evaluated inside the 24 h job limit at
$\nfe=20$ (10.4 s per control step, 18.6 h per ten-flight variant); only its unprojected row and `dpcc-r`
exist on `pillars_xl`, and they are not used.

Post-mortem: `Data_Analysis/DA_in_Paper/analysis/DA_20260922_pillars_xl_wave.md`.
Closure: `logs_in_develop/Gen15/U17/CLOSURE_20260922_U17_abandoned.md`.

## 2 · The decision — restore `pillars_hg`, with its caveat

The author falls back to the U7 geometry and the numbers you already had.

**Restore** `v3/withheld/20260918_uav_pillars_section.tex` between UAV-corridor and UAV-s-curve: the
prose, `tab:uav-pillars` (per-step, every evaluated budget), `tab:uav-pillars-endpoint` ($\nfe=5$),
`tab:uav-pillars-best`, and the flown-path figure. The numbers are unchanged — they were computed from
the 15-09 batch and `pillars_grid.py` reproduces them today (regression-checked: MeanFM 5 = 0.90 / 0.20
/ 0.80 / 0.40 / 0.10 / 0.90 / 0.80; FM 5 = 1.00 / 0.00 / 1.00 / 0.80 / 0.00 / 0.90 / 0.40; endpoint FM
0.95 mean). `uav_results.py` has `PILLARS_EXCLUDED = False` again.

**Keep the caveat, in the prose, as the section's first move — it is true and it is the story.** The
withheld text already opens with it: on this scene the unprojected flows are already feasible (FM 10/10,
MeanFM 9/10 clean without projection), so the projected rows measure *how much of an already feasible
plan each method preserves*, not whether it can repair an infeasible one — that is UAV-corridor's
question. Present pillars as the preservation case beside corridor's repair case. Do not soften it.

**Remove** the v3.56 `pillars_xl` scaffolding: the `\hole` in `sec:res:uav:pillars`, the templates
`tab:uav-pillars` (xl version), `tab:uav-pillars-projection`, and the `\todofigure` for
`fig_uav_pillars_xl_paths`. No `pillars_xl` number may appear anywhere.

**Downstream**, everything that read *withheld* for this scene since v3.40 reads the `pillars_hg` values
again: the UAV conclusion, both cross-environment summary tables, the selected-combination table, and the
projection comparison for the scene.

One sentence you may want for the scene's own caveat paragraph, if you choose to mention the attempt at
all (optional — the thesis does not owe the reader a failed campaign): *an evaluation with the keep-out
enlarged so that the demonstrated lane became infeasible was carried out and is not reported; every
projector routed into the physically open but forbidden corridor between the pillar rows, so the scene
ranked nothing.* Chapter 7 material, if anywhere.

## 3 · What the DA side has done

- `DA_in_Paper/analysis/INDEX.md`: the `pillars_xl` template row is struck; the `pillars_hg` row is
  back as the analysis of record with the caveat text.
- `pillars_grid.py` repointed to `pillars_hg` / 15-09 (xl settings kept as a comment for the
  post-mortem); `uav_results.py` un-gated.
- `config/uav_projection.yaml`: `pillars_xl` / `pillars_xxl` carry an ABANDONED banner and are never
  active.
- `PENDING_20260922_all_lacking_runs.md` R23 → abandoned/fallback; its "no `pillars_hg` number" rule
  is superseded.
