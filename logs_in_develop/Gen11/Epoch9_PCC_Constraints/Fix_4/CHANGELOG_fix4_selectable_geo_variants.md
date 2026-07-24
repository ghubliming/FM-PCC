# Epoch 9 Fix_4 — per-scene geometry made selectable (visual-aligning `geo_constraint_variants` style)

**Date:** 2026-07-04. Restructures HOW the per-scene geometry is selected in
`config/uav_projection.yaml`, without changing WHAT runs by default. All current geometry
values/content are unchanged — only the selection mechanism was upgraded.

## Before

Each scene had exactly **one** `geo_constraint_variants` entry, resolved by exact
`name == scene` match (`empty`, `corridor`, `pillars`, `s_curve`). There was no way to define
an alternate/ablation configuration for a scene (e.g. bounds-only, obstacles-only, mirroring
visual-aligning's `bounds_only_1`/`obstacle_only_1`) without renaming the scene's one entry
and losing the "current" one.

## After — selectable, like `config/visual_aligning_eval.yaml`

Same mechanism visual-aligning already uses: a **flat list** of named variants
(`geo_constraint_variants`) + an **active selector** (`active_geo_variants`) that picks which
named entries actually run. The difference from visual-aligning (which has one task/scene, so
`name` alone identifies a variant): UAV has four physically-distinct scenes, so each entry now
also carries an explicit `scene:` field, decoupling "which named variant" from "which scene's
geometry it uses" — the way visual-aligning's `combined_4`/`combined_5` are both variants of
the *same* task, UAV's `corridor_combined_1` and a future `corridor_bounds_only` would both be
variants of the *same* scene.

### Naming (mirrors visual-aligning's own convention)
- `<scene>_combined_1` — the full-stack entry (dynamics+bounds+halfspace+obstacles), naming
  modeled on visual-aligning's own `combined_4`/`combined_5`.
- `empty_no_constraint` — the unconstrained baseline, naming modeled on visual-aligning's own
  `no_constraint`.

### Current default selection = exactly the prior behavior
```yaml
active_geo_variants: ['empty_no_constraint', 'corridor_combined_1', 'pillars_combined_1', 's_curve_combined_1']
```
Every scene's geometry (workspace_bounds, halfspace_constraints, obstacle_constraints,
constraint_types) is **byte-identical** to before this fix — only `name` gained a
`_combined_1`/`_no_constraint` suffix and each entry gained a `scene:` field. Nothing about
what runs by default changed.

### How to add a future ablation (now possible, wasn't before)
Copy a `*_combined_1` block, set `scene: <that scene>`, narrow `constraint_types` to the one
family under test (e.g. `['bounds']` for a `corridor_bounds_only` entry), give it a new `name`,
and swap it into `active_geo_variants` in place of its `*_combined_1` sibling.

## `FM_v3_uav_test/eval_fm_uav.py::load_pcc_config` — resolution logic updated

Old: dict keyed by `name`, looked up by `scene in _geo_variants`. New: scans the flat list,
matches each entry's `scene` field (falling back to `name`, for backward compat with any
hand-written entry that omits `scene:`) against the current scene, filtered to entries listed
in `active_geo_variants`:
```python
_matches = [g for g in _all_geo
            if g.get('scene', g['name']) == scene and (_active is None or g['name'] in _active)]
```

**New safety guard:** since exactly one geo config runs per scene per eval invocation, having
**two** active entries resolve to the same scene is a config mistake (ambiguous — which one
should run?), not a valid multi-select. `load_pcc_config` now raises a clear `ValueError`
naming the scene and the conflicting entry names, rather than silently picking one (the old
dict-keyed-by-name approach couldn't even represent this case, since `name` and `scene` were
the same string — this failure mode is new *because* the feature is new, and is guarded from
its first commit).

## Verification
Simulated the exact resolution snippet against the real `config/uav_projection.yaml`
(`yaml.safe_load`, no torch/MuJoCo needed):
- All four scenes resolve to their `*_combined_1`/`_no_constraint` entry with the correct
  `constraint_types`, matching the pre-fix behavior exactly.
- Ambiguity guard fires correctly when a synthetic second active entry
  (`corridor_bounds_only`) is added for `corridor` alongside `corridor_combined_1`.
- Narrowing `active_geo_variants` to select only the ablation entry resolves cleanly to
  `constraint_types=['bounds']`, confirming the ablation path works once the conflict is
  resolved by the user (as the error message instructs).
- `py_compile` clean on `eval_fm_uav.py`; yaml parses.

## Files touched
- `config/uav_projection.yaml` — `geo_constraint_variants` entries renamed + `scene:` field
  added; `active_geo_variants` updated to the new names. No geometry values changed.
- `FM_v3_uav_test/eval_fm_uav.py` — `load_pcc_config`'s geo-resolution block rewritten to
  match by `scene` field across a flat list, with an ambiguity guard.
