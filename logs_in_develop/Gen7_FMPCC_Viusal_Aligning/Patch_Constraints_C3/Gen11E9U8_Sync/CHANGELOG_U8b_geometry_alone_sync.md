# Gen11E9U8b_Sync — add the "geometry alone" (`model_free-bounds_free`) variant into visual-aligning

**Date:** 2026-07-09. Companion to
`logs_in_develop/Gen11/Epoch9_PCC_Constraints/U_8_new_projection_var_upgrade/CHANGELOG_U8b_add_geometry_alone_variant.md`
(the UAV-side origin of this change) and follow-up to this folder's
`CHANGELOG_U8_sync_from_UAV.md` (the original U8 toggle sync).

## Why this applies here too
The U8 toggle machinery (`geo_free` / `bounds_free` / `model_free`, substring-composable) was
already synced into visual-aligning and is present in **both** eval scripts:
- `fm_visual_aligning_test/eval_fm_visual_aligning.py:133-190` (Gen7, FM)
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py:127-183` (Gen6V4, diffuser)

Both read the **same** `config/visual_aligning_eval.yaml`, so its `projection_variants` list
drives the sweep for both generations. That list already had two of the three
single-family-active ("X alone") probes — bounds-alone (`geo_free-model_free`) and
dynamics-alone (`geo_free-bounds_free`) — but was **missing geometry-alone**, exactly like the
UAV config was. The U_13 corridor ordering investigation
(`../../../Gen11/Epoch9_PCC_Constraints/U_13/INVESTIGATION_geo_free_model_free_worse_than_diffuser.md`)
needs that cell to isolate "geometry projected without dynamics."

## The change
- `config/visual_aligning_eval.yaml` — added `'model_free-bounds_free'` to
  `projection_variants` (after `geo_free-model_free`). **Zero code change** — the U8 substring
  toggles already compose it: `model_free` → dynamics OFF, `bounds_free` → action-bound OFF,
  no `geo_free` → geometry ON.
- Covers **both** Gen7 (FM) and Gen6V4 (diffuser) at once, since they share this yaml — no
  per-generation config edit needed.

## Not touched
- No `-tightened` sibling added (kept parallel with the UAV decision), though unlike the other
  `-free` composites a `model_free-bounds_free-tightened` WOULD be meaningful here (geometry is
  retained). Flagged for later if the tightened geometry-alone case is wanted.
- No Python change in either eval script — the gates already handle the new name.

## Verification
- Confirmed the gates exist and are identical in both scripts (grep above).
- Confirmed the shared-yaml dispatch resolves `model_free-bounds_free` to geometry-only.
- Execution untested here (Docker has no torch/MuJoCo — cluster-only).
