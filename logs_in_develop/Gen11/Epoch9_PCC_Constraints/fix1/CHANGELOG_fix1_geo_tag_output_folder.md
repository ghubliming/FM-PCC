# Epoch 9 fix1 — restore the "which constraint geometry" output-path level

**Date:** 2026-07-04. Fixes a gap found right after the initial E9 implementation
(`../CHANGELOG_E9_PCC_constraints.md`): the UAV eval had no path level for "which
geometry/constraint combo produced this run" — the old avoiding-task convention had one.

## The gap

Old avoiding-task path (`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`):
```python
save_path = f'{args.savepath}/results/halfspace_{halfspace_variant}'   # geometry axis
...
f'{save_path}/{variant}.npz'                                          # projection-variant axis
```
`halfspace_variant` (e.g. `both-hard`, `top-left-hard`) is a **second, swappable axis**:
multiple obstacle-course layouts can be run against the *same* dataset/exp, each kept in its
own `results/halfspace_<name>/` subfolder, with the projection variant (`diffuser`/`dpcc-c`/…)
as the file inside it.

The initial E9 UAV path had only one axis:
```
scene_root/plans/<model_dir>/<eval_params_dir>/<seed>/<variant>/
```
Scene name was doing double duty as "which scene" AND "which geometry" (since each scene
resolves to exactly one fixed `geo_constraint_variants` entry). Re-running the **same** scene
under a different `constraint_types` subset (e.g. `pillars` with obstacles-only vs. the full
stack, for an ablation) would silently **overwrite** the previous run — nothing in the path
distinguished them.

## The fix

Added the missing axis back as `<geo_tag>`, inserted between `<seed>` and `<variant>`:
```
scene_root/plans/<model_dir>/<eval_params_dir>/<seed>/<geo_tag>/<variant>/
```
mirroring `seed_dir/results/halfspace_<name>/<variant>.npz` from avoiding.

`geo_tag` is computed once in `load_pcc_config` (`FM_v3_uav_test/eval_fm_uav.py`), right after
the per-scene geometry resolution, from the **actually active** `constraint_types` (not just
the scene name), so it changes whenever the enforced constraint combo changes:
```python
_ctypes = cfg.get('constraint_types') or []
cfg['geo_tag'] = f'{scene}_unconstrained' if not _ctypes else f"{scene}_{'+'.join(sorted(_ctypes))}"
```
Examples: `empty_unconstrained`, `corridor_bounds+dynamics+halfspace+obstacles`,
`pillars_bounds+dynamics+obstacles` (if halfspace were disabled for an ablation).

`_run_variant` reads `config['geo_tag']` and inserts it into the path:
```python
geo_dir = os.path.join(seed_dir, config.get('geo_tag', scene))
out_dir = os.path.join(geo_dir, variant)
```
The config-snapshot write stays at `seed_dir` (above `geo_dir`) — it's a model/eval-param
snapshot, not geometry-specific, so it's written once per seed regardless of how many
`geo_tag`s run under it.

## Why this form of the tag (not just the scene name)

Using the scene name alone would still collide if someone edits `constraint_types` for the
same scene entry between runs (the exact ablation case above). Keying on the **resolved
constraint_types set** guarantees two runs only share a folder when they enforced the
identical constraint combo — matching the spirit of avoiding's `halfspace_variant` (a distinct
folder per distinct enforced geometry), without requiring a new CLI flag: editing the yaml's
`constraint_types` for a scene entry naturally produces a new `geo_tag` and a fresh folder.

## Verification
- `py_compile` clean on `eval_fm_uav.py`.
- No other path in this eval needs the same fix — `plans/` per-variant IS the terminal
  artifact directory; there is no separate `all_seeds`-style aggregation step in this file
  (unlike avoiding's `eval_flow_matching_v3_ode_selectable.py` L423) to update in parallel.

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `geo_tag` computed in `load_pcc_config`; `_run_variant`
  path construction inserts it between `<seed>` and `<variant>`.
