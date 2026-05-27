# UF-15 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Source MD**: [u_f_15_constrainst_visual/CHANGELOG_UF15.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_15_constrainst_visual/CHANGELOG_UF15.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

---

## Summary

Identical to FM UF-15 (see source MD for full detail). Adds automatic constraint geometry visualisation to the DPCC eval script.

**New output per geo entry:**
```
results/{geo_name}/constraint_overview.png
results/{geo_name}-tightened/constraint_overview.png
```

**3-panel figure**: 3D box wireframe | XY top-down | XZ side view. Generated once per geo entry before any trajectory runs — serves as pre-run sanity check for bounds, halfspace line orientation, and obstacle placement.

**Functions added** (same code as FM eval):
- `_hs_xy_draw` — halfspace line + feasible-side arrow for XY panel
- `plot_geo_constraints` — 3-panel figure generator, called at geo entry start

**Call site**: inside `if geo_variant == projection_variants[0]:` guard in the geo loop — 4 lines added, no structural change to the loop body.

**Idempotent**: skips generation if `constraint_overview.png` already exists.

---

## UF-15.2: Constraint overlay on foresight SVG

See [CHANGELOG_UF15.md UF-15.2](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_15_constrainst_visual/CHANGELOG_UF15.md) for full detail. DPCC-side changes are identical to FM.

**XY panel overlay**: bounds rectangle (steelblue, zorder=1), halfspace line+arrow (darkorange), obstacle circle (tomato) — all drawn behind trajectories.

**3D panel overlay**: workspace box wireframe (steelblue, 12 edges).

**Wiring**: `geo_config` and `is_tightened` added as `VisualAgentWrapper` constructor params and stored as instance attributes. Passed from the geo loop instantiation. Empty `geo_config` → all overlay guards False → no drawing, no error.

---

## UF-15.3: `_hs_xy_draw` clipping fix + moderate halfspace value (2026-05-27)

See [CHANGELOG_UF15.md UF-15.3](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_15_constrainst_visual/CHANGELOG_UF15.md) for full detail. DPCC-side changes are identical to FM.

**`_hs_xy_draw` parametric clipping fix**: old outer-extreme t selection drew halfspace lines with endpoints outside the display viewport, causing auto-scaling axes in the foresight SVG to expand incorrectly. Fixed with Cohen-Sutherland slab intersection (`t_lo = max(tx[0], ty[0])`, `t_hi = min(tx[1], ty[1])`).

**Halfspace moderated** in `config/visual_aligning_eval.yaml`: `[[0.35,-0.35],[0.65,0.35],'above']` (45° diagonal, too aggressive) → `[[0.30,-0.05],[0.70,0.05],'above']` (nearly horizontal, y ≈ -0.05→0.05).

---

## UF-15.4: Halfspace shown as filled plane, not just a line (2026-05-27)

See [CHANGELOG_UF15.md UF-15.4](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_15_constrainst_visual/CHANGELOG_UF15.md) for full detail. DPCC-side changes are identical to FM.

**2D XY panel** (`_hs_xy_draw`): infeasible region now filled with darkorange polygon (alpha=0.15) via CCW viewport corner walk. No `BIG` coordinates — all vertices within `xlim×ylim`.

**3D panels** (overview + foresight SVG): halfspace boundary shown as a darkorange vertical `Poly3DCollection` rectangle from z_lo to z_hi.

---

## UF-15.5: Missing obstacle in SVG 3D panel + XZ overview panel (2026-05-27)

See [CHANGELOG_UF15.md UF-15.5](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_15_constrainst_visual/CHANGELOG_UF15.md) for full detail. DPCC-side changes are identical to FM.

**Foresight SVG 3D `ax_3d`**: obstacle `plot_surface` sphere was missing entirely — added after halfspace block. 2D obstacles placed at workspace z midpoint.

**Overview XZ panel**: 2D obstacles (xy-only) now rendered as a shaded vertical band `axvspan(cx-r, cx+r)` with dashed edge lines. Halfspace presence noted via italic text annotation `"N× halfspace (XY plane only)"`.
