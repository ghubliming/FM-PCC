# UF-16.4–16.7 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-28
**Branch**: `update_into_FM`
**Source MD**: [u_f_16_debug_PCC/CHANGELOG_UF16_4.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16_debug_PCC/CHANGELOG_UF16_4.md)
**Scope**: `d3il/simulation/aligning_sim.py`,
           `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`,
           `fm_visual_aligning_test/eval_fm_visual_aligning.py`,
           `config/visual_aligning_eval.yaml`

---

## Summary

See source MD for full detail.  All changes are shared infrastructure — no DPCC-specific
divergence from the FM version.  Both eval scripts receive identical additions.

---

## UF-16.4 — Final box position + angle logging

`d3il/simulation/aligning_sim.py`: after the rollout loop, reads live MuJoCo box
pos+quat and passes them as `final_box_pos` / `final_box_quat` into `update_rollout_info`.

`eval_visual_aligning_dpcc.py` — `update_rollout_info()`: extracts final XY, converts
quat to Z-angle (exact formula for pure Z-rotation), computes 2D dist to target, stores
three new keys in `curr_context_info`:

- `final_box_xy`
- `final_box_angle_deg`
- `final_xy_dist`

These flow automatically into per-rollout console print, `rollout_N_stats.json`, and
`history_context_info`.

New console line:
```
  - Box  final XY=(0.501,  0.330)  angle=-57.1°  (dist_to_target: 0.0043 m)
```

---

## UF-16.5 — Success / Mode / Angle explainer

New document added at:
`logs_in_develop/…/u_f_16_debug_PCC/SUCCESS_MODE_ANGLE_EXPLAINER.md`

Documents success criteria (pos ≤ 1.8 cm AND rot ≤ 8.6°), mode semantics (0 = robot in
contact at last step, 1 = moved away), `mean_distance` formula, angle verification.
No bugs found — logic is correct as-is.

---

## UF-16.6 — Dual-boundary visualization for tightened variants

`eval_visual_aligning_dpcc.py` (and FM counterpart):

- `_hs_xy_draw(ax, hs, enlarge, xlim, ylim, dashed=False)` — new `dashed` param;
  `dashed=True` draws only the boundary line (no fill polygon, no arrow/annotation).
- `_c_layers` variable: `[(0.0, False)]` for non-tightened; `[(0.0, False), (δ, True)]`
  for tightened runs.  Both XY and 3D overlay blocks loop over `_c_layers`.
- XY panel: bounds Rectangle + halfspace line + obstacle Circle all drawn twice when
  tightened — solid for nominal (eval boundary), dashed for planning boundary.
- 3D panel: bounds wireframe, halfspace quad, obstacle sphere — solid nominal,
  dashed planning layer.
- Legend addition (tightened only): solid = `nominal constraint (eval boundary)`;
  dashed = `planning constraint (δ=X.XXXm inside)`.

Non-tightened runs: single layer, unchanged appearance.

---

## UF-16.7 — YAML comment updates

`config/visual_aligning_eval.yaml`: outdated comments corrected; avoiding comments
preserved untouched.  Key corrections: `n_contexts` reference de-hardcoded;
`combined_5` added to available entries list; active/inactive tags updated to reflect
`active_geo_variants: [combined_5]`; `combined_4` annotated with UF-16.1 relaxed-
constraints note; `combined_5` given full `[CURRENTLY ACTIVE]` description.

---

## No changes to

- `diffuser_visual_aligning/sampling/projection.py` — projector internals untouched
- Existing `results.pkl` / `{variant}.npz` format — fully backwards compatible
- Constraint metrics infrastructure from UF-16.3
