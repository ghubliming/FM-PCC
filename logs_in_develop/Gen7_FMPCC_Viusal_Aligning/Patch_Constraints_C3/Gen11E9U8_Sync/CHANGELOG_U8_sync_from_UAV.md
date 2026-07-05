# Gen11E9U8_Sync — sync U8's projection-variant ablation toggles back into visual-aligining

**Date:** 2026-07-04. Companion to
`logs_in_develop/Gen11/Epoch9_PCC_Constraints/U_8_new_projection_var_upgrade/`. This session's
`dynamics_bounds_only` and `action_bounds_only` geo entries (added earlier in
Patch_Constraints_C3, for visual-aligining) — plus the pre-existing `dynamics_only` — are all
now redundant given two new `projection_variants`-level toggles, and are removed.

## Why this applies here too

`model_free` (dynamics-off) has always been a **variant-level** toggle in this codebase, not a
geo-level one — it was the pattern the UAV work generalized. Once generalized, the fix belongs
back here: visual-aligining is where `model_free` originates, and where this session's earlier
work (Patch_Constraints_C3) added the now-superseded `dynamics_bounds_only`/
`action_bounds_only` geo entries in the first place.

## The fix — `bounds_free` + `geo_free`, identical to the UAV change

Applied to **both** live consumers of `config/visual_aligning_eval.yaml`'s
`setup_dpcc_projector` (per the original Patch_Constraints_C3 dual-file pattern):
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7, FM engine)
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4, DDPM engine)

```python
if 'geo_bounds' in ctypes and 'geo_free'    not in variant: ...   # was: no variant gate
if 'bounds'     in ctypes and 'bounds_free' not in variant: ...   # was: no variant gate
if 'dynamics'   in ctypes and 'model_free'  not in variant: ...   # unchanged
if 'halfspace'  in ctypes and 'geo_free'    not in variant: ...   # was: no variant gate
if 'obstacles'  in ctypes and 'geo_free'    not in variant: ...   # was: no variant gate
```

Same truth table as UAV (full detail there):

| Variant | dynamics | bounds | geo_bounds+halfspace+obstacles | = |
|---|:---:|:---:|:---:|---|
| `model_free` | ❌ | ✅ | ✅ | (existing) |
| `bounds_free` | ✅ | ❌ | ✅ | new |
| `geo_free` | ✅ | ✅ | ❌ | new |
| `geo_free-bounds_free` | ✅ | ❌ | ❌ | dynamics alone |
| `geo_free-model_free` | ❌ | ✅ | ❌ | bounds alone |

## `config/visual_aligning_eval.yaml`

- **`projection_variants`** gained `bounds_free`, `geo_free`, `geo_free-bounds_free`,
  `geo_free-model_free` (no `-tightened` siblings — same reasoning as UAV: tightening only
  affects families `geo_free` may already remove).
- **Removed** geo entries: `dynamics_only` (pre-existing, predates this session),
  `dynamics_bounds_only` and `action_bounds_only` (both added earlier in
  Patch_Constraints_C3/this session). All three are now exactly reproducible on `combined_5`
  (the currently-active full-stack entry) via a variant toggle:
  - `dynamics_only` → `combined_5` + `geo_free-bounds_free`
  - `dynamics_bounds_only` → `combined_5` + `geo_free`
  - `action_bounds_only` → `combined_5` + `geo_free-model_free`
- **Kept** `geo_bounds_only_1`/`geo_bounds_only_2`, `obstacle_only_1`/`obstacle_only_2`,
  `halfspace_only_1`, and `combined_1`/`combined_2`/`combined_3` — these are **not** redundant:
  they either test alternate *geometry values* (2D vs 3D box, 2D vs 3D obstacle — a different
  axis than "which family is on"), or combinations the three-toggle scheme structurally can't
  reach (e.g. `combined_2` = dynamics+geo_bounds with halfspace/obstacles/bounds all off —
  `geo_free` removes geo_bounds/halfspace/obstacles as one indivisible group, so it cannot
  selectively keep geo_bounds while dropping halfspace+obstacles). Removing these would lose
  real, non-redundant experimental capability — left untouched.
- Comments updated throughout (enlarge_constraints doc, "all available entries" list, naming
  scheme note) to drop stale references to the removed entries.

**Note on `dynamics_only`:** this one predates this session (it was part of the original TIER 2
ablation grid). Removed anyway, once confirmed genuinely redundant, for consistency with the
"ablate via variant, not via geo entry" principle now applied throughout — flagging here in
case anything external referenced that name.

## Verification
- `py_compile` clean on both eval scripts.
- `config/visual_aligning_eval.yaml` parses; confirmed via `yaml.safe_load` that
  `dynamics_only`/`dynamics_bounds_only`/`action_bounds_only` are gone from
  `geo_constraint_variants`, and `combined_4`/`combined_5` still declare both `geo_bounds` and
  `bounds` (so the new toggles have something to act on).
- Truth table verified identically to the UAV changelog's simulation (same dispatch logic,
  same three independent gates) — not re-run here since the code is byte-for-byte the same
  pattern, just copied across two files per the existing Patch_Constraints_C3 precedent.
- Full SLSQP/rollout execution untested here (cluster-only).

## Files touched
- `fm_visual_aligning_test/eval_fm_visual_aligning.py` — `setup_dpcc_projector` dispatch.
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — identical dispatch change
  (Gen6V4's live copy of the same function).
- `config/visual_aligning_eval.yaml` — `projection_variants` additions; 3 redundant geo
  entries removed; comments updated.
