# UF-14 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Source MD**: [u_f_14/CHANGELOG_UF14.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/CHANGELOG_UF14.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, `config/visual_aligning_eval.yaml`

---

## Summary

Identical change to FM UF-14. Three parts:

**1. Geo constraint outer loop** — `for variant` replaced by `for geo_name, geo_config, geo_variant in _run_items`. Output paths become `results/{geo_name}/{variant}/`. Backward-compatible fallback when yaml key absent.

**2. Obstacle support in `setup_dpcc_projector`** — added `_DIM` named-dimension map and `'obstacles'` block. `obstacle_constraints` list from `geo_config` is parsed and appended to the projector constraint list as `(type, dims, center, radius)` tuples.

**3. Yaml redesign** — `geo_constraint_variants` restructured into 3 tiers:
- Baseline: `no_constraint`
- Single-type ablation: `dynamics_only`, `bounds_only_1` (active); `obstacle_only_1/2` (disabled, geometry not measured)
- Combinations: `combined_2` = dynamics + bounds (active, DPCC-equivalent); `combined_1/3` with obstacles (disabled)
