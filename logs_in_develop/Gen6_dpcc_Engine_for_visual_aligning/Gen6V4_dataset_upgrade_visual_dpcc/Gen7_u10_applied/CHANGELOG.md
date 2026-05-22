# Gen7 Upgrade 10 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-22  
**Branch**: `update_into_FM`  
**Source MD**: [u_f10/CHANGELOG_U10_FM.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f10/CHANGELOG_U10_FM.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` · `d3il/simulation/aligning_sim.py`

---

## Summary

Identical to FM Upgrade 10. `aligning_sim.py` is shared — the `hasattr` hook fires for both agents. All five touchpoints applied symmetrically to `VisualAgentWrapper` in the DPCC eval.

## Changes

- `__init__`: `curr_context_info = {}`, `history_context_info = []`
- `reset()`: `curr_context_info = {}`
- New `record_context_info(context, context_idx)`: same logic as FM — extracts box/target XY, angles, init XY dist
- `update_rollout_info()`: `context_info` in rollout dict + `history_context_info` append + console print block
- `_export_rollout_realtime()` JSON: `'context_info'` field added to `rollout_N_stats.json`

`aligning_sim.py` change is shared (one edit covers both FM and DPCC).
