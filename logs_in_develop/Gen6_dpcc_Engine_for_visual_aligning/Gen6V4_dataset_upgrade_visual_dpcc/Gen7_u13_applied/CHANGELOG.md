# UF-13 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-26  
**Branch**: `update_into_FM`  
**Source MD**: [u_f13/CHANGELOG_UF13.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_13/CHANGELOG_UF13.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`, `Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh`

---

## Summary

Identical fix to FM UF-13. When `record_mode != 'none'`, `if_vision` is auto-promoted to `True` at the `Aligning_Sim` call site so GIFs/videos are always captured regardless of config `if_vision=False`. Warning printed when auto-promote fires. No new CLI flags added.
