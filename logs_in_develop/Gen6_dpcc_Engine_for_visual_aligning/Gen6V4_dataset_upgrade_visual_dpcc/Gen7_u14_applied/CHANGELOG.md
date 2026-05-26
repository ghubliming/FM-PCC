# UF-14 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Source MD**: [u_f_14/CHANGELOG_UF14.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/CHANGELOG_UF14.md)  
**Guide**: [u_f_14/GEO_CONSTRAINTS_GUIDE.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/GEO_CONSTRAINTS_GUIDE.md)  
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
