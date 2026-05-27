# UF-16 Applied to Gen6V4 (DPCC)

**Date**: 2026-05-27  
**Branch**: `update_into_FM`  
**Source MD**: [u_f_16_debug_PCC/CHANGELOG_UF16.md](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/u_f_16_debug_PCC/CHANGELOG_UF16.md)  
**Scope**: `config/visual_aligning_eval.yaml`, `diffuser_visual_aligning/sampling/projection.py`, `fm_visual_aligning/sampling/projection.py`

---

## Summary

See source MD for full detail.  Changes are shared config/infrastructure — no DPCC-specific divergence.

---

## UF-16.1: Relaxed constraints in `visual_aligning_eval.yaml`

Nominal `combined_4` constraints were too tight for initial debug runs: SLSQP projector was active on every sample, making it impossible to assess raw policy behaviour.

**Workspace bounds** widened by ±0.10 m in x/y, z floor −0.03 m, z ceiling +0.10 m:
- Original: `lb=[0.30,-0.35,0.05]` / `ub=[0.70,0.35,0.40]`
- Relaxed:  `lb=[0.20,-0.45,0.02]` / `ub=[0.80,0.45,0.50]`

**Halfspace** pushed to near the lower-y workspace edge (only ~0.08 m strip forbidden):
- Original: `[[0.30,-0.05],[0.70,0.05],'above']` — cuts ~50% of y range
- Relaxed:  `[[0.20,-0.38],[0.80,-0.30],'above']` — minimal restriction

Original values preserved as comments directly below each relaxed line.

---

## UF-16.2: `active_geo_variants` selector

See source MD for full detail.  Summary:

- Added `active_geo_variants` list key in `config/visual_aligning_eval.yaml` (geo constraint section, after `enlarge_constraints`).  One-line change to select which named geo entries run; `null` = run all.
- All ready geo entries uncommented in yaml so any can be activated directly via the list.
- Dead top-level `constraint_types` fallback removed.
- Both eval scripts updated: filter `_geo_specs` by active names; fix `_has_geo` to use effective `_gc['constraint_types']`.
- `halfspace_only_2` kept commented — 3D normal/offset format not implemented in code.

---

## UF-16.3: Suppressed SLSQP per-sample console print

`diffuser_visual_aligning/sampling/projection.py` and `fm_visual_aligning/sampling/projection.py` — the Fix 9.3 diagnostic block that printed one line per sample per replanning step was commented out.  It produced O(B × T / stride) lines per rollout, flooding the console and hiding all other eval output.  Lines kept as comments for easy re-enable.
