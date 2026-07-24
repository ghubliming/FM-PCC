# Epoch 9 U8 — ablate via `projection_variants`, not via per-scene geo entries

**Date:** 2026-07-04. Undoes most of Fix_5's per-scene ablation-entry design and replaces it
with something simpler, triggered by the user's observation: `model_free` was already a
**variant-level** toggle ("skip dynamics"), so the natural, symmetric fix for "skip bounds" (or
"skip the geometric families") is another variant-level toggle — not a whole new
`geo_constraint_variants` entry per ablation, per scene.

## What was wrong with Fix_5's design

Fix_5 added `<scene>_dynamics_only` / `<scene>_dynamics_bounds_only` as separate
`geo_constraint_variants` entries, one pair per scene (corridor/pillars/s_curve) — 6 extra
entries, each requiring its own `scene:` tag, competing for `active_geo_variants` slots,
subject to Fix_4's ambiguity guard, and needing Fix_6's multi-geo-variant loop just to run
them alongside the full-stack entry in one job. All of that machinery existed to express
something `model_free` already proved didn't need it: **"run the SAME geo entry, but skip one
constraint family"** is a one-line variant-name check, not a new geo entry.

## The fix — two new variant-level toggles, orthogonal to `model_free`

```python
if 'dynamics'   in ctypes and 'model_free'  not in variant: ...   # existing
if 'bounds'     in ctypes and 'bounds_free' not in variant: ...   # NEW
if 'geo_bounds' in ctypes and 'geo_free'    not in variant: ...   # NEW
if 'halfspace'  in ctypes and 'geo_free'    not in variant: ...   # NEW
if 'obstacles'  in ctypes and 'geo_free'    not in variant: ...   # NEW
```
`geo_free` gates all three geometric families (`geo_bounds`, `halfspace`, `obstacles`)
**together**, as one group — they're naturally one thing (spatial constraints), distinct from
the two "trajectory-space" constraints (`dynamics`, `bounds`). All three toggles are
independent and compose by substring in the variant name:

| Variant | dynamics | bounds | geo_bounds+halfspace+obstacles | = |
|---|:---:|:---:|:---:|---|
| `model_free` | ❌ | ✅ | ✅ | (existing) |
| `bounds_free` | ✅ | ❌ | ✅ | new |
| `geo_free` | ✅ | ✅ | ❌ | new |
| `geo_free-bounds_free` | ✅ | ❌ | ❌ | **dynamics alone** — replaces `<scene>_dynamics_only` |
| `geo_free-model_free` | ❌ | ✅ | ❌ | **bounds alone** — replaces `<scene>_dynamics_bounds_only`'s complement |

Verified by simulating the exact dispatch logic against the real `pillars` geo entry for all
six named variants (`dpcc-c`, `model_free`, `bounds_free`, `geo_free`,
`geo_free-bounds_free`, `geo_free-model_free`) — the family-active table above matches exactly.

**One caveat, not fixed:** `geo_free-bounds_free-model_free` (all three combined) is *not*
byte-identical to `diffuser` — it still runs the SLSQP/gradient projector machinery with an
empty constraint list, whereas `diffuser` skips the projector object entirely
(`projector=None`). Functionally near-identical result, different code path.

## `FM_v3_uav_test/eval_fm_uav.py::setup_dpcc_projector`
- Added `'geo_free' not in variant` to the `geo_bounds`/`halfspace`/`obstacles` gates.
- Added `'bounds_free' not in variant` to the `bounds` gate.
- Docstring updated to document the three-toggle truth table and point here.

## `config/uav_projection.yaml`
- **Removed**: `<scene>_dynamics_only` / `<scene>_dynamics_bounds_only` for
  corridor/pillars/s_curve (6 entries) — Fix_5's per-scene ablation pairs, now redundant.
- **Renamed**: `<scene>_combined_1` → plain `<scene>` (`corridor`, `pillars`, `s_curve`) —
  there's no longer a sibling to disambiguate from, so the suffix is dead weight. Content
  (geometry, constraint_types) is otherwise byte-identical to the old `*_combined_1` entries.
- **`active_geo_variants`** simplified to
  `['empty_no_constraint', 'corridor', 'pillars', 's_curve']` — exactly one entry per scene,
  every time, permanently (not just by convention).
- **`projection_variants`** gained `bounds_free`, `geo_free`, `geo_free-bounds_free`,
  `geo_free-model_free`. No `-tightened` siblings added for these — tightening only affects
  `geo_bounds`/`halfspace`/`obstacles`, which `geo_free` may already remove, so a tightened
  sibling would frequently be a wasted no-op run; simpler to omit than to special-case.
- Fix_4's underlying mechanism (named entries + `scene:` field) and Fix_6's multi-geo-variant
  loop are **unchanged in code** — still available if a scene ever needs a genuinely different
  *geometry* (not just a different family subset), just not exercised by the default config
  anymore.

## Known minor inefficiency (not fixed, flagged)
`eval_scene`'s `_has_spatial` check (which decides whether to skip `-tightened`
`projection_variants` for a scene) operates on the geo entry's **declared** `constraint_types`,
not on what a `geo_free`-style variant-name toggle actually leaves active. This means a
hypothetical `geo_free-tightened` variant would not be skipped by that check (the geo entry
does declare spatial families) even though tightening is a no-op once `geo_free` has removed
them. Not a correctness bug — the tightened run would simply be identical to the untightened
one — and avoided in practice here since no `-tightened` sibling was added for `geo_free`/
`bounds_free` in `projection_variants` above. Would need fixing if someone later adds one.

## Verification
- `py_compile` clean on `eval_fm_uav.py`.
- `config/uav_projection.yaml` parses; `active_geo_variants` / `geo_constraint_variants`
  resolve to exactly one entry per scene, confirmed via the same resolution snippet used in
  Fix_4/Fix_6.
- Simulated `setup_dpcc_projector`'s family-dispatch logic (without invoking the real
  `Projector`, which needs torch) for all 6 variants against the real `pillars` entry — matches
  the truth table above exactly, including both target combinations
  (`geo_free-bounds_free` = dynamics alone, `geo_free-model_free` = bounds alone).
- Full SLSQP/rollout execution untested here (cluster-only).

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `setup_dpcc_projector`'s constraint dispatch + docstring.
- `config/uav_projection.yaml` — `projection_variants` additions; `geo_constraint_variants`
  simplified to one full-stack entry per scene; `active_geo_variants` updated; comments
  rewritten throughout to describe the new mechanism.

## Companion fix
The identical projector-code change was applied to visual-aligining's two eval scripts
(`fm_visual_aligning_test/eval_fm_visual_aligning.py`,
`diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`) and
`config/visual_aligning_eval.yaml`, since that's where `model_free` (the pattern this whole
change generalizes) originates and where the equivalent Fix_5-style entries
(`dynamics_only`/`dynamics_bounds_only`/`action_bounds_only`) were also added/renamed this
session. See
`logs_in_develop/Gen7_FMPCC_Viusal_Aligning/Patch_Constraints_C3/Gen11E9U8_Sync/`.
