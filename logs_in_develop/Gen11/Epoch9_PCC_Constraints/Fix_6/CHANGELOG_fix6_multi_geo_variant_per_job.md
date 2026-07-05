# Epoch 9 Fix_6 — run multiple active geo variants for one scene in a single job

**Date:** 2026-07-04. Triggered by an actual cluster failure:
```
ValueError: E9: scene 's_curve' matches MULTIPLE active geo_constraint_variants
(['s_curve_dynamics_only', 's_curve_dynamics_bounds_only', 's_curve_combined_1']) —
only one may be active per scene at a time.
```
This was the Fix_4 ambiguity guard working exactly as designed — it caught a real config
state (three `s_curve`-tagged entries all listed in `active_geo_variants` at once) that the
code genuinely could not resolve to one. But the underlying INTENT — submit one job, run all
three geo variants for `s_curve` back-to-back — is legitimate and matches how
`visual_aligning_eval.yaml`'s eval script already behaves (it loops over every active geo
entry in one invocation). The UAV eval didn't have that loop; this fix adds it.

## Why the guard fired (recap, not a bug)

UAV's `eval_scene(scene, args)` previously called `load_pcc_config(scene, seed)` **once**,
which resolves **exactly one** geo entry per scene. Visual-aligning's `eval.py`, by contrast,
has an outer loop over `_geo_specs`/`_run_items` that can run **several** geo entries in one
script invocation, each into its own `results/<geo_name>/` folder. UAV never had that loop —
so when three entries were tagged `scene: s_curve` and all three ended up in
`active_geo_variants`, there was no way to pick one, and the code correctly refused to guess.

## The fix — give UAV the same multi-entry loop visual-aligning already has

### `FM_v3_uav_test/eval_fm_uav.py` — refactor, no behavior change for the common (single-match) case

- **`_load_base_cfg(scene, seed)`** (new): the yaml-load + defaults + eval-control-params part
  of the old `load_pcc_config`, with geo resolution removed.
- **`_resolve_active_geo_matches(scene, cfg)`** (new): returns the list of active
  `geo_constraint_variants` entries for a scene — 0, 1, or many (previously inlined in
  `load_pcc_config` with an immediate raise on >1; now reusable).
- **`_apply_geo_entry(cfg, scene, entry)`** (new): applies one entry's constraint_types/
  geometry to a **copy** of the base cfg and computes its `geo_tag` (Fix_1) — the part that
  makes each geo variant land in its own output folder automatically.
- **`load_pcc_config(scene, seed)`**: now a thin wrapper — `_load_base_cfg` +
  `_resolve_active_geo_matches` + raise-if->1 + `_apply_geo_entry`. Behavior for any caller
  wanting a single resolved config is **unchanged** (same raise, same return shape).
- **`eval_scene(scene, args)`**: now calls `_load_base_cfg` once (shared model/dataset/mj_model
  across geo variants — constraint geometry doesn't affect any of those), then
  `_resolve_active_geo_matches`, then **loops over every match**, running the full
  `projection_variants` sweep for each via `_apply_geo_entry`'s per-entry config (each with its
  own `geo_tag`, hence its own output folder — no collisions, confirmed below).
  - Returns the **same flat `{variant: summary}` shape as before** when exactly one geo
    variant is active (the overwhelming common case — zero behavior change for every existing
    single-entry config).
  - Returns `{geo_variant_name: {variant: summary}}` only when multiple entries are active —
    the new Fix_6 case.

## What this means for your job

Your original `active_geo_variants` (all three `s_curve_*` entries listed together) now
**works as originally intended** — no yaml edit needed, no separate job submissions. One
`eval_fm_uav.py --scene s_curve` run will execute:
1. `s_curve_dynamics_only` → `.../s_curve_dynamics/<variant>/...`
2. `s_curve_dynamics_bounds_only` → `.../s_curve_bounds+dynamics/<variant>/...`
3. `s_curve_combined_1` → `.../s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles/<variant>/...`

each in its own `geo_tag`-named subfolder (Fix_1), each running the full
`projection_variants` sweep (diffuser/dpcc-r/-c/-t/gradient/etc.) independently.

## Verification
- `py_compile` clean.
- Reproduced the exact failing scenario (`active_geo_variants` with all three `s_curve_*`
  entries) via `_resolve_active_geo_matches` + `_apply_geo_entry`, called directly against the
  real `config/uav_projection.yaml`: all three resolve correctly, each with a distinct
  `geo_tag` (`s_curve_dynamics`, `s_curve_bounds+dynamics`,
  `s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles`) — confirming no output-folder
  collision across the three.
- `load_pcc_config`'s single-match contract (raise on >1) is preserved for any caller that
  still wants it — `eval_scene` simply no longer routes through that raising path.
- Full rollout/SLSQP execution untested here (cluster-only) — this fix is resolution/wiring
  level, independently verified above.

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `_load_base_cfg`, `_resolve_active_geo_matches`,
  `_apply_geo_entry` extracted; `load_pcc_config` simplified to a single-match wrapper;
  `eval_scene` rewritten with the multi-geo-variant loop.

---

## Addendum — config-snapshot was saving the WRONG yaml (unrelated bug, found while auditing a run)

**Reported symptom:** a UAV run's saved config snapshot was
`.../6/config_snapshot_uav/projection_eval.yaml` — i.e. **avoiding-d3il's** yaml, not the UAV's
own `uav_projection.yaml`. This makes it impossible to use the snapshot to determine which
`constraint_types`/`action_bounds`/etc. actually produced a given run (exactly the provenance
question that came up while auditing a `pillars_bounds+dynamics+halfspace+obstacles` folder
earlier in this thread).

### Root cause

`flow_matcher_v3_uav/utils/setup.py::snapshot_configs` hardcoded:
```python
yaml_path = 'config/projection_eval.yaml'   # WRONG — this is avoiding-d3il's yaml
```
`config/projection_eval.yaml` exists **repo-wide** (it's avoiding's own yaml, sitting in the
shared `config/` directory used by every package), so `os.path.exists(yaml_path)` always
succeeded — the snapshot silently copied avoiding's yaml into every UAV run's
`config_snapshot_uav/` folder, every time, regardless of what the UAV eval actually used. This
is a copy-paste artifact: `flow_matcher_v3_uav/utils/setup.py` is a per-package copy of a
shared template, and this one line was never updated to point at `uav_projection.yaml`.

Checked the sibling per-package copies for the same mistake — both already correct:
`fm_visual_aligning/utils/setup.py` and `imf_visual_aligning/utils/setup.py` both have
`yaml_path = 'config/visual_aligning_eval.yaml'`. Only the UAV copy had the leftover
avoiding-task hardcoding.

### Fix

```python
yaml_path = 'config/uav_projection.yaml'   # the UAV's own real yaml
...
dest = os.path.join(snapshot_dir, 'uav_projection.yaml')   # matching destination filename
```
This file (`flow_matcher_v3_uav/utils/setup.py`) is UAV-specific — it should always snapshot
the UAV's own yaml, never any other package's. No generic/parameterized fix needed since each
package owns its own copy of this function.

### Practical consequence

Every past UAV run's `config_snapshot_uav/projection_eval.yaml` was never useful for
provenance (it's a copy of a file the UAV eval never even reads) — the actual config that
produced a run has to be reconstructed from the git commit hash printed in the SLURM job
header (`GIT REV: <hash>`) instead, for any run predating this fix. Going forward, the
snapshot will contain the real `uav_projection.yaml` as actually loaded at eval time.

### Verification
- `py_compile` clean on `flow_matcher_v3_uav/utils/setup.py`.
- Confirmed `config/projection_eval.yaml` and `config/uav_projection.yaml` are two distinct,
  independently-existing files (not aliases), so this was a genuine wrong-file bug, not a
  naming coincidence.
- Confirmed sibling packages' `setup.py` copies do NOT have this bug (already point to their
  own correct yaml).

### Files touched
- `flow_matcher_v3_uav/utils/setup.py` — `snapshot_configs`'s hardcoded yaml path + destination
  filename corrected from `projection_eval.yaml` to `uav_projection.yaml`.
