# UF-14 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Source MD**: [u_f_14/CHANGELOG_UF14.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_14/CHANGELOG_UF14.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, `config/visual_aligning_eval.yaml`

---

## Summary

Identical change to FM UF-14. The single `constraint_types` key is replaced by a `geo_constraint_variants` outer loop, producing separate `results/{geo_name}/{variant}/` output subtrees for `bounds_only_1` and `bounds_dynamics_1`. Commented-out `_2` placeholders provided for parameter tuning. Old output paths are backward-compatible (fallback fires when yaml key is absent).
