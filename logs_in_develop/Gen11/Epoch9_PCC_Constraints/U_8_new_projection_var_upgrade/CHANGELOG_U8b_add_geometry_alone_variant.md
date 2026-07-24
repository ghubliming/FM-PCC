# Epoch 9 U8b — add the missing `model_free-bounds_free` ("geometry alone") ablation variant

**Date:** 2026-07-09. Follow-up to `CHANGELOG_U8_projection_variant_ablation.md`. Triggered by
the U_13 corridor ordering investigation
(`../U_13/INVESTIGATION_geo_free_model_free_worse_than_diffuser.md`), which needed a clean
"geometry projected WITHOUT dynamics" probe and revealed the ablation set was missing one cell.

## What prompted this
While explaining the corridor variant ordering
(`model_free < geo_free-model_free < geo_free-bounds_free < diffuser < post_processing < dpcc-*`),
the user asked to add a variant that is **"geometry + action-bounds, no dynamics."**

**Clarification first (important, to avoid a duplicate):** that exact combination *already
exists* — it is **`model_free`**. The `model_free` name is "free of the dynamics **model**
only"; it keeps BOTH the action bound and all geometry. It does **not** strip everything (a
common misread of the name). Verified against the live gates in `setup_dpcc_projector`
(`FM_v3_uav_test/eval_fm_uav.py:674-745`): for `variant='model_free'`, `geo_bounds`/`halfspace`/
`obstacles` are active (`'geo_free' not in variant`), `bounds` is active
(`'bounds_free' not in variant`), and only `dynamics` is skipped (`'model_free' in variant`).

## The real gap: the "X alone" trilogy was missing its third member
U8 introduced three composable, substring-matched toggles (`model_free`, `bounds_free`,
`geo_free`). The eight (dynamics, action-bound, geometry) on/off combinations were all
*expressible*, but the projection-variants list only *named* two of the three
single-family-active ("X alone") probes:

| combo (dyn / bounds / geo) | meaning | variant name | before U8b |
|:--:|---|---|:--:|
| ❌ / ✅ / ✅ | dynamics off, rest on | `model_free` | present |
| ✅ / ❌ / ✅ | bounds off, rest on | `bounds_free` | present |
| ✅ / ✅ / ❌ | geometry off, rest on | `geo_free` | present |
| ❌ / ✅ / ❌ | **bounds alone** | `geo_free-model_free` | present |
| ✅ / ❌ / ❌ | **dynamics alone** | `geo_free-bounds_free` | present |
| ❌ / ❌ / ✅ | **geometry alone** | `model_free-bounds_free` | **MISSING → added** |
| ✅ / ✅ / ✅ | full stack | `dpcc-*` / `post_processing` / `gradient` | present |
| ❌ / ❌ / ❌ | empty (≈ diffuser) | `geo_free-bounds_free-model_free` | not listed (≈ `diffuser`) |

**`model_free-bounds_free` = geometry alone** (no dynamics anchoring, no action-magnitude
bound; box + halfspace + obstacle families active). It is the natural complement of the two
existing "alone" probes and the cleanest test of the U_13 hypothesis — *"geometry projected
without the dynamics coupling corrupts the executed action, and more geometry = worse"* —
because it removes the action-bound confound that `model_free` still carries.

## Why `model_free-bounds_free` resolves correctly with NO code change
Variants are pure yaml + substring toggles; there is no allowlist. `eval_scene` reads
`config['projection_variants']` (`eval_fm_uav.py:1267,1488`) and dispatches by name. For
`variant='model_free-bounds_free'`, the gates evaluate:
- `'dynamics' … 'model_free' not in variant` → **skipped** (dynamics OFF) ✓
- `'bounds' … 'bounds_free' not in variant` → **skipped** (action bound OFF) ✓
- `'geo_bounds'/'halfspace'/'obstacles' … 'geo_free' not in variant` → **active** (geometry ON) ✓
- `_selection_for('model_free-bounds_free')` → `'random'` (no `dpcc-c/-t`) ✓

So the single yaml line is sufficient — the U8 toggle machinery already composes it.

## Change
### `config/uav_projection.yaml`
- Added `'model_free-bounds_free'` to `projection_variants`, directly after
  `geo_free-model_free`, with a comment marking it the 3rd "X alone" probe and noting a
  `-tightened` sibling *would* be meaningful here (geometry is kept — unlike the other `-free`
  composites where `geo_free` removes the only family tightening affects), but is omitted for now.

### `config/visual_aligning_eval.yaml` (Gen7 sibling sync)
- Gen7 visual-aligning already implements the full U8 toggle machinery — the `geo_free`/
  `bounds_free`/`model_free` gates are present and identical in BOTH eval scripts
  (`fm_visual_aligning_test/eval_fm_visual_aligning.py:133-190`,
  `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py:127-183`) — and its
  `projection_variants` list already had bounds-alone (`geo_free-model_free`) and dynamics-alone
  (`geo_free-bounds_free`) but was **missing the same geometry-alone cell**. Added
  `'model_free-bounds_free'` there too (after `geo_free-model_free`), keeping the sibling
  generations in sync per the repo copy-modify convention. No Gen7 code change — same
  zero-code substring-toggle dispatch as UAV.

No Python change in either generation.

## Interaction checks (no regressions)
- **`-tightened` skip logic** (`eval_fm_uav.py:1472-1478`): only strips variants whose name
  contains `'tightened'` when a scene has no spatial families. `model_free-bounds_free` has no
  `'tightened'`, so it is never skipped on the geometry scenes (corridor/pillars/s_curve) and,
  on `empty` (no geometry), it simply projects an empty set ≈ no-op — consistent with every
  other geometry-bearing variant there.
- **Exec-time collision metrics** (`_exec_constraint_violations`): keyed off the geo entry's
  declared `constraint_types`, independent of the variant toggle, so `model_free-bounds_free`
  still gets correct flown-path collision numbers.
- **Cost:** adds one more variant to each scene's sweep — bumps eval wall-time by ~1 variant's
  worth of rollouts (jobs already brush the 24h SLURM limit; bump `--time` if adding alongside
  a full multi-seed run).

## Expected reading in the U_13 study (hypothesis, cluster-only to confirm)
Placed in the ordering, `model_free-bounds_free` (geometry alone, no dynamics) should land
**near `model_free`** — i.e., well below `diffuser` — since both project full geometry without
the dynamics coupling. If `model_free-bounds_free ≈ model_free`, that isolates **geometry-
without-dynamics** as the corruption source (the action bound in `model_free` is a minor
add-on). If instead `model_free ≪ model_free-bounds_free`, the action bound is contributing too.
Either outcome directly tests U_13 §2.

## Verification done here
- Confirmed `model_free` already = (geo + action-bound, no dynamics) by reading the gates — the
  requested variant was not actually missing; the missing one is geometry-alone.
- Confirmed the new name composes correctly through the substring toggles (dispatch trace above).
- yaml edit is syntactically a plain list-item addition; `projection_variants` still a flat list.
- Full SLSQP/rollout execution untested here (Docker has no torch/MuJoCo — cluster-only).

## Files
- `config/uav_projection.yaml` — one variant added to `projection_variants` (UAV).
- `config/visual_aligning_eval.yaml` — same variant added (Gen7 sibling sync).
- (companion, no change needed) `FM_v3_uav_test/eval_fm_uav.py`,
  `fm_visual_aligning_test/eval_fm_visual_aligning.py`,
  `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — toggle gates already handle it.
