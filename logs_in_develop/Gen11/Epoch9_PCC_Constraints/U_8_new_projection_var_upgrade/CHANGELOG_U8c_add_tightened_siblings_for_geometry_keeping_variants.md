# Epoch 9 U8c — add the missing `-tightened` siblings for the geometry-KEEPING new variants

**Date:** 2026-07-10. Follow-up to `CHANGELOG_U8b_add_geometry_alone_variant.md` (which
**flagged** this gap) and `CHANGELOG_U8_projection_variant_ablation.md`. Triggered by the user
noticing `model_free` has a `-tightened` sibling but the newly-added `bounds_free` /
`model_free-bounds_free` do not — and asking whether they should.

## The rule (what a `-tightened` sibling actually does)
`enlarge_constraints` shifts **only** the geometric boundaries — `geo_bounds` (workspace box),
`halfspace` (walls), `obstacles` (balls) — inward by δ (`eval_fm_uav.py:664-670`,
`is_tightened = 'tightened' in variant`; `margin = inflation_base + enlarge`). The action-bound
family (`bounds`) is **never** tightened (`eval_fm_uav.py:1467-1470`), and dynamics has no
spatial margin. Therefore:

> A `-tightened` sibling is **meaningful iff the variant keeps geometry ON**, and a pure
> **no-op** whenever `geo_free` removed it.

Mapping that over the new U8/U8b toggles:

| variant | dyn | action-bound | **geometry** | `-tightened` meaningful? | had one before U8c |
|---|:--:|:--:|:--:|:--:|:--:|
| `model_free` | ❌ | ✅ | ✅ | **yes** | ✅ `model_free-tightened` |
| `bounds_free` | ✅ | ❌ | ✅ | **yes** | ❌ **missing → added** |
| `model_free-bounds_free` (geometry alone) | ❌ | ❌ | ✅ | **yes** | ❌ **missing → added** |
| `geo_free` | ✅ | ✅ | ❌ | no (nothing to tighten) | ✅ correctly omitted |
| `geo_free-bounds_free` (dynamics alone) | ✅ | ❌ | ❌ | no | ✅ correctly omitted |
| `geo_free-model_free` (bounds alone) | ❌ | ✅ | ❌ | no | ✅ correctly omitted |

So the bug was an **inconsistency**, not a missing capability: three variants keep geometry
(`model_free`, `bounds_free`, `model_free-bounds_free`) but only one of them (`model_free`) had
its explicit `-tightened` twin listed. The `geo_free*` variants were already (correctly)
tightened-free.

## Change — `config/uav_projection.yaml` (UAV, Gen11)
Added two entries to `projection_variants`:
- `'bounds_free-tightened'` — right after `bounds_free`. Geometry kept (only the action bound is
  dropped) → tightening the box/walls/balls is a real, comparable run, exactly parallel to
  `model_free-tightened`.
- `'model_free-bounds_free-tightened'` — right after `model_free-bounds_free`. This is the
  sibling the U8b changelog explicitly said "WOULD be meaningful here (geometry is kept) —
  omitted for now." U8c stops omitting it.

Also rewrote the docstring rule above `projection_variants` (was: "`-tightened` siblings are
omitted for these three") to the precise **"tightened iff geometry kept; `geo_free*` get none"**
statement, and noted that UAV enumerates each `-tightened` explicitly (unlike visual-aligning,
which auto-generates — see below).

**Zero Python change.** `is_tightened = 'tightened' in variant` already dispatches both new
names; the geometry gates (`'geo_free' not in variant`) already keep geometry ON for both, so
`margin` picks up `enlarge` on exactly the spatial families.

## Gen7 / Gen6V4 (visual-aligning): **no config change needed — and here's why**
The user asked to mirror the patch into the Gen7/Gen6V4 synced configs. Checked, and there is
**nothing to add** there — the tightened dimension works differently:

- **UAV** enumerates each `-tightened` variant *explicitly* in `projection_variants` (so an
  omitted one genuinely doesn't run — the U8c bug).
- **Visual-aligning** (both `fm_visual_aligning_test/eval_fm_visual_aligning.py:2184-2190` and
  `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py:2179-2185`) **auto-generates** the
  tightened twin inside the geo loop: for every geo entry with a spatial family and non-null
  `enlarge_constraints`, it appends `(geo_name + '-tightened', variant, is_tightened=True)` for
  **every** `variant in projection_variants`. So `bounds_free` and `model_free-bounds_free`
  **already get their tightened twins** automatically — as a `<geo>-tightened` folder — with no
  list entry. Adding `*-tightened` names to that yaml's `projection_variants` would be **wrong**
  (it would try to run `bounds_free-tightened` as a *base* variant under BOTH the nominal and
  the auto-generated tightened geo, i.e. a double-tighten / duplicate).

The visual-aligning yaml already documents that `geo_free-tightened` is a harmless auto-generated
no-op (`config/visual_aligning_eval.yaml:147-151`) — consistent with the U8c rule. So the two
generations reach the **same** end state (every geometry-keeping variant has a tightened
counterpart) through two different mechanisms; only UAV needed an edit.

## Interaction checks (no regressions)
- **Per-scene tightened skip** (`eval_fm_uav.py:1467-1478`): strips `*tightened` variants for
  scenes whose `constraint_types` carry no spatial family. `empty_no_constraint`
  (`constraint_types: []`) → both new tightened variants auto-skipped (correct, nothing to
  tighten). corridor / pillars / s_curve declare spatial families → both run and enlarge the
  geometry as intended.
- **Tightened-plot dedup** (`eval_fm_uav.py:1268-1275`): picks one representative tightened
  variant per geo_dir for the `constraint_overview_tightened.png`; extra tightened variants just
  reuse it — no new files, no clobber.
- **Exec-time collision metrics** (`_exec_constraint_violations`): keyed off the geo entry's
  `constraint_types`, independent of the variant/tightened toggle → correct flown-path numbers
  for both new variants.
- **Cost:** +2 variants × (corridor, pillars, s_curve) = up to 6 extra rollout groups per seed.
  UAV eval jobs already brush the 24 h SLURM wall — bump `--time` if sweeping these alongside a
  full multi-seed run.

## Expected reading (hypothesis, cluster-only to confirm)
Both new tightened variants only *narrow* the geometry, so they inherit their base variant's
regime and should move in the direction tightening always moves a geometry-keeping variant:
fewer planned violations, but on the **s_curve** scene (non-convex, ~24 cm bands — see
`../U_13/INVESTIGATION_s_curve_geometry_destabilizes_ordering_flip.md`) tightening the already-
tight corridor should make the geometry-driven crashes **no better / slightly worse**, matching
the user's "the tightened variants are no better" observation and the U8c/U_13 story that on
s_curve geometry is the bottleneck.

## Verification done here
- Confirmed the tightened mechanism is substring-only (`'tightened' in variant`) and applies
  `enlarge` to spatial families exclusively — read `eval_fm_uav.py:664-670, 1467-1478`.
- Confirmed visual-aligning auto-generates the twin per-variant — read the geo loop in both eval
  scripts (`:2184-2190` / `:2179-2185`), hence no yaml edit there.
- yaml edits are plain list-item additions; `projection_variants` stays a flat list.
- No local execution (Docker has no torch/MuJoCo — cluster-only).

## Files
- `config/uav_projection.yaml` — added `bounds_free-tightened`, `model_free-bounds_free-tightened`;
  rewrote the `-tightened` rule comment. (UAV only.)
- (companion, no change) `config/visual_aligning_eval.yaml` — tightened twins auto-generated;
  documented why in `../../../Gen7_FMPCC_Viusal_Aligning/Patch_Constraints_C3/Gen11E9U8_Sync/CHANGELOG_U8c_tightened_siblings_sync.md`.
- (companion, no change) `FM_v3_uav_test/eval_fm_uav.py` — toggle/tightened dispatch already handles it.
