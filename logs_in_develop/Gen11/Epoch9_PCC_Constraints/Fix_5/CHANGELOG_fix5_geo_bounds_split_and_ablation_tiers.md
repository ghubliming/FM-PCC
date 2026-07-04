# Epoch 9 Fix_5 — UAV gets the same `geo_bounds`/`bounds` split + selectable ablation tiers

**Date:** 2026-07-04. Two things, both mirroring what
`Gen7_FMPCC_Viusal_Aligning/Patch_Constraints_C3/` just did for visual-aligning:

1. **Split UAV's conflated `'bounds'`** into `'geo_bounds'` (workspace box on `p`) and
   `'bounds'` (restored DPCC action-magnitude limit) — UAV's E9/Fix_3 code had built both
   under ONE flag (correct behavior, but same naming conflation the C3 fix eliminated
   elsewhere); now consistent across the whole repo.
2. **Added selectable ablation entries per scene** (`<scene>_dynamics_only`,
   `<scene>_dynamics_bounds_only`), mirroring visual-aligning's `geo_constraint_variants`
   tiers, so a scene's constraint combo can be swapped by editing `active_geo_variants` alone
   — same mechanism as Fix_4, just more entries to choose from.

**Default behavior is unchanged**: `active_geo_variants` still points at the four
`*_combined_1`/`empty_no_constraint` entries; their enforced geometry is byte-identical to
before, only now correctly split into two constraint_types flags instead of one, and a
`bounds` action-limit family is included (matching what `*_combined_1` already claimed to be:
the full per-scene stack).

## Code split — `FM_v3_uav_test/eval_fm_uav.py`

Five call sites renamed `'bounds'` → `'geo_bounds'` (geo-box gates), matching the exact set
fixed in visual-aligning:
- `_exec_constraint_violations`'s `spatial` set + workspace-box check.
- `plot_geo_constraints`'s `has_bounds`.
- `setup_dpcc_projector`'s geo-box block (split OFF into its own `if 'geo_bounds' in ctypes:`).
- `eval_scene`'s `_spatial` set (tightened-variant skip logic — `'bounds'` is deliberately
  excluded here: it's a dataset-range cap, not a spatial surface, so tightening doesn't
  apply to it, same as visual-aligning).

The action-bound logic itself (self-derive via `act_normalizer.mins/.maxs` when
`action_bounds: 'auto'`, from Fix_3) is **unchanged** — only its gate is now the independent
`if 'bounds' in ctypes:` block, no longer sharing a flag with the geo box.

## `config/uav_projection.yaml` changes

- `*_combined_1` entries (`corridor`/`pillars`/`s_curve`): `constraint_types` renamed
  `'bounds'` → `'geo_bounds'`, **and** `'bounds'` added back
  (`['dynamics','geo_bounds','halfspace','obstacles','bounds']`) — these are the "full stack"
  entries so they must carry both bounds families, exactly the same reasoning as
  `combined_4`/`combined_5` in visual-aligning.
- **New per-scene ablation pairs** (not active by default):
  - `<scene>_dynamics_only` — `constraint_types: ['dynamics']`
  - `<scene>_dynamics_bounds_only` — `constraint_types: ['dynamics', 'bounds']`
  for `corridor`, `pillars`, `s_curve` — mirrors the `dynamics_only`/`dynamics_bounds_only`
  A/B pair just added to `config/visual_aligning_eval.yaml`. Neither needs geometry fields
  (dynamics/bounds don't read `workspace_bounds`/`halfspace_constraints`/`obstacle_constraints`).
- To run a comparison: swap a scene's `*_combined_1` name in `active_geo_variants` for its
  `*_dynamics_only`/`*_dynamics_bounds_only` pair (same mechanism documented in Fix_4).

## Verification
- `py_compile` clean on `eval_fm_uav.py`.
- `yaml.safe_load` on `config/uav_projection.yaml`: all 10 entries print with correct
  `constraint_types`; default `active_geo_variants` resolves to the four `*_combined_1`/
  `_no_constraint` entries with zero ambiguity (simulated the exact resolution code from
  `load_pcc_config`).
- Simulated switching `active_geo_variants` to `corridor_dynamics_only` in place of
  `corridor_combined_1` — resolves cleanly, no ambiguity error.
- Re-rendered the U2 constraint schematic (`plot_geo_constraints`) for `pillars_combined_1`
  after the rename — still produces `constraint_overview.png`+`.svg` with no error, confirming
  the `geo_bounds` rename didn't break the plotting path.
- Full SLSQP/rollout path untested here (cluster-only).

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `'bounds'`→`'geo_bounds'` rename at 4 call sites +
  `setup_dpcc_projector`'s geo-box block split into its own gate.
- `config/uav_projection.yaml` — `*_combined_1` constraint_types updated; 6 new ablation
  entries added (`<scene>_dynamics_only` / `<scene>_dynamics_bounds_only` × 3 scenes);
  comments updated to describe the split and the new selectable tiers.
