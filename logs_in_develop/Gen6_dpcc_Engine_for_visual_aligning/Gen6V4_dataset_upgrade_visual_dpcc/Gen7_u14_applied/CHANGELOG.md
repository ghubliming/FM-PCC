# UF-14 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Source MD**: [u_f_14/CHANGELOG_UF14.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/CHANGELOG_UF14.md)  
**Guide**: [u_f_14/CONSTRAINTS_GUIDE.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/CONSTRAINTS_GUIDE.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, `config/visual_aligning_eval.yaml`

---

## Summary

Identical to FM UF-14 (see source MD for full detail). Key parts:

**1. Geo constraint outer loop** — `for variant` replaced by `for geo_name, geo_config, geo_variant, is_tightened in _run_items`. Output paths become `results/{geo_name}/{variant}/`. Backward-compatible fallback when yaml key absent.

**2. Obstacle support in `setup_dpcc_projector`** — `_DIM` named-dimension map; `'obstacles'` block reads `obstacle_constraints` from `geo_config`. `ws_lb`/`ws_ub` read inside `if 'bounds'` block only (no crash for no-bounds entries).

**3. Geo-level tightened auto-generation** — replaces per-projection-variant `-tightened` suffix.
- `setup_dpcc_projector(... is_tightened=False)` — explicit flag, no longer inferred from variant name.
- Geo loop auto-appends `{geo_name}-tightened` runs for entries with `'bounds'` or `'obstacles'` when `enlarge_constraints` is non-null.
- `no_constraint` and `dynamics_only` never get tightened twins.
- Set `enlarge_constraints: null` to disable all tightening.
- Obstacle tightening: `radius + enlarge_constraints` (matches original DPCC).

**4. Yaml redesign** — `geo_constraint_variants` in 3 tiers:
- Baseline: `no_constraint`
- Single-type ablation: `dynamics_only`, `bounds_only_1` (active); `bounds_only_2`, `obstacle_only_1/2` (disabled)
- Combinations: `combined_2` = dynamics + bounds (active, DPCC-equivalent); `combined_1/3` with obstacles (disabled)
- `projection_variants`: all `-tightened` entries removed; `dpcc-c-tightened-dt*` → `dpcc-c-dt*`
- `enlarge_constraints: 0.01` at top-level global (null = disabled)

---

## Revision B — Halfspace, 2D/3D bounds, constraint loading analysis

See [CHANGELOG_UF14.md Revision B](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/CHANGELOG_UF14.md) for full detail. DPCC-side changes:

**1. Halfspace support** — `setup_dpcc_projector` now handles `'halfspace'` in `constraint_types`. Iterates all `halfspace_constraints` list items (no integer-index picking). Tightening shifts each halfspace boundary inward. `_has_geo` updated to include `'halfspace'`, so `halfspace_only_1-tightened` is auto-generated.

**2. 2D/3D bounds scheme** — `bounds_only_1` changed to 2D (`z=±inf`). `bounds_only_2` is the 3D variant. `combined_2` intentionally keeps 3D bounds (full physical model). PyYAML parses `-.inf`/`.inf` correctly.

**3. New yaml entries** — `halfspace_only_1` (2D, active for debugging), `halfspace_only_2` (3D, pending), `combined_4` (`['dynamics','bounds','halfspace','obstacles']` — full DPCC match once obstacle geometry measured, currently active with placeholder values only).

**4. Constraint loading design** — Our design avoids the original DPCC's brittle integer-index selection and redundant bounds definitions. See [UF14_investigation_constraint_loading.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/UF14_investigation_constraint_loading.md) for analysis.
