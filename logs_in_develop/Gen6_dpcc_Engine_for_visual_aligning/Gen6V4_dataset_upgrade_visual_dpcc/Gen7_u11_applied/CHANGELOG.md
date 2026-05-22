# Gen7 Upgrade 11 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-22  
**Branch**: `update_into_FM`  
**Source MD**: [u_f11/CHANGELOG_U11_FM.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f11/CHANGELOG_U11_FM.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

---

## Summary

Identical overhaul to FM Upgrade 11. Only difference: uses `c_pos_h` (DPCC variable name) instead of `c_pos_hist` (FM variable name) for `c_arr` construction — kept consistent with existing DPCC code.

All rendering changes, anchor dot logic, legend, stride value (`_STRIDE=6`), and dual PNG+SVG output are identical to FM.

---

## U11.2 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-22  
**Source MD**: [u_f11_mpc_plot/CHANGELOG_U11_FM.md §U11.2](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f11_mpc_plot/CHANGELOG_U11_FM.md)

Identical changes to FM U11.2. All four touchpoints applied symmetrically to `VisualAgentWrapper` in `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`.
