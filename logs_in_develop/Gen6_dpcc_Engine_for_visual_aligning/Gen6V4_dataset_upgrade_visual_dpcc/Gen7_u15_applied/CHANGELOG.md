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
