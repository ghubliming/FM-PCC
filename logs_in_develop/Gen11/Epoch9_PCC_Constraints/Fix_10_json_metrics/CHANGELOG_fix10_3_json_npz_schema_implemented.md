# Fix_10 (3/3) — implemented the grouped JSON/NPZ schema from `fix10_2`'s plan

**Date:** 2026-07-06. Implements `PLAN_fix10_2_json_schema_redesign.md`'s locked decision:
`results.json`/rollout dicts nest into `physical`/`constraint`/`goal`/`success`/`timing`
groups, and the UAV NPZ writer's array names are renamed/extended to match — same leaf names
in both artifacts, no cross-pipeline (avoiding/visual-aligining) renaming.

## What changed, per file

### `FM_v3_uav_test/eval_fm_uav.py`
- **`rollout_one`**'s return dict restructured:
  ```python
  {
    'scene': ..., 'homotopy': ...,
    'physical':   {'safe', 'contact_frac', 'min_z', 'final_z'},
    'constraint': {'collision_free', 'n_violations', 'total_violations'},
    'goal':       {'reached', 'dist', 'crossed_line'},
    'success':    {'strict', 'relaxed', 'strict_and_constraints', 'relaxed_and_constraints'},
    'timing':     {'fm_ms_mean', 'fm_ms_p95', 'proj_ms_mean', 'total_ms_mean',
                   'total_ms_p95', 'total_over_budget', 'budget_ms'},
    'track_err_mean': ..., 'n_fm_steps': ..., 'decim': ..., 'dt': ...,
    'obs_traj': ..., 'act_traj': ..., 'plans': ..., 'frames': ...,   # heavy, unchanged
  }
  ```
- **`_run_variant`**'s `summary` dict mirrors the same 5 groups with `_rate`/`_mean` suffixes
  inside each group (e.g. `summary['success']['strict_rate']`,
  `summary['physical']['safe_rate']`) instead of flat `success_rate`/`safe_rate`.
- The one `print(...)` statement referencing `summary["success_relaxed_rate"]` etc. updated to
  the new nested paths.

### `FM_v3_uav_test/eval_artifacts.py`
- **`save_npz`**: array names renamed/extended to group-prefixed names matching the JSON —
  `success_strict`, `success_relaxed`, `success_strict_and_constraints`,
  `success_relaxed_and_constraints`, `phys_safe`, `phys_contact_frac`, `phys_min_z`,
  `phys_final_z`, `constraint_collision_free`, `constraint_n_violations`,
  `constraint_total_violations`, `goal_reached`, `goal_dist`, `goal_crossed_line`. **The
  physical and goal arrays are new** — the old NPZ only ever persisted the constraint-axis
  metrics (`n_violations`/`total_violations`/`success_and_constraints`), never `safe`/
  `contact_frac`/`goal_reached`/`goal_dist` at all; per the user's explicit "share similar
  metrics" instruction, these are now saved too, not just renamed.
- **`write_eval_log`**: every `r.get(...)`/`summary.get(...)` flat access updated to read from
  the new nested groups (`phys = r.get('physical', {})`, etc.).
- **`plot_overview`**, **`save_rollout_stats`**, **`write_mpc_foresight`**: confirmed (grepped)
  they only ever read `obs_traj`/`homotopy`/`plans` — none of the renamed fields — **no change
  needed**, exactly as scoped in the plan.

### `FM_v3_uav_test/aggregate_scene_summaries.py`
- `METRICS` changed from a flat list of key-strings to a list of `(output_name, group, key)`
  triples; a new `_extract(summ, group, key)` helper reads the nested `summary` groups
  (`group=None` for the one field that stayed top-level, `track_err_mean`).
- Output metric names updated to match the new leaf-name convention (`success_rate` →
  `success_strict_rate`, `contact_frac_mean` → `phys_contact_frac_mean`, etc.) so this rollup
  script's own `SCENE_SUMMARY.json`/`fm_uav_ALL_SCENES_SUMMARY.json` output stays consistent
  with what produced it.
- The one hardcoded print-string key (`agg.get("success_rate_mean")`) updated to
  `agg.get("success_strict_rate_mean")`.

### `npz_analysis/analyze_npz.py`
- `HEADLINE_KEYS` (module-level, "printed first when present") got UAV's new prefixed names
  **appended** — `success_strict`, `success_relaxed`, `success_strict_and_constraints`,
  `success_relaxed_and_constraints`, `phys_safe`, `phys_contact_frac`,
  `constraint_collision_free`, `constraint_n_violations`, `constraint_total_violations`,
  `goal_reached`, `goal_dist`. The old flat names (`n_violations`, `collision_free_completed`,
  etc.) are **kept, unchanged** — avoiding/visual-aligining still produce those, unaffected by
  this UAV-only rename. Confirmed (per the plan's own audit) no other hardcoded reference to
  the renamed keys exists in this file — everything else is schema-generic by the file's own
  design.

## Verification
- `py_compile` clean on all 4 touched files.
- **`aggregate_scene_summaries.py` run end-to-end** (pure stdlib, no torch/numpy needed)
  against a synthetic nested `results.json` — correctly extracted `success_strict_rate=0.85`
  from `summary['success']['strict_rate']`, produced the expected `SCENE_SUMMARY.json` with
  the new output key names.
- **`save_npz` and `write_eval_log` run end-to-end** (numpy available in this environment)
  against a synthetic nested rollout dict — confirmed the NPZ contains exactly the expected
  18 keys (`success_strict`, `phys_safe`, `constraint_n_violations`, `goal_dist`, etc.) with
  correct values, and the eval log renders correctly from the nested `summary`/`rollout` dicts.
- Repo-wide grep for every old flat-key pattern (`r['success']`, `.get('safe')`,
  `.get('collision_free')`, `.get('crossed_line')`, `.get('goal_reached')`,
  `.get('goal_dist')`) across `FM_v3_uav_test/*.py` confirms zero remaining stale accesses —
  every match left is one of the newly-nested ones.
- Full SLSQP/rollout/cluster execution untested here (no torch/MuJoCo runtime) — this fix is
  entirely at the artifact-writing/reading layer, and every piece of that layer that could be
  exercised without torch/MuJoCo was exercised above.

## What did NOT change
- Avoiding's and visual-aligining's own npz writers/`results.json` shapes — untouched,
  per the explicit "no cross-pipeline goal" decision in `fix10_2`.
- No `schema_version` field added (decision recorded in `fix10_2`: git commit hash is the
  existing, sufficient provenance mechanism).
- The underlying *measurements* (Axis A physical truth vs. Axis B constraint-margin truth) —
  unchanged; this was a presentation/naming fix, not a re-derivation of what gets computed.

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `rollout_one`, `_run_variant`.
- `FM_v3_uav_test/eval_artifacts.py` — `save_npz`, `write_eval_log`.
- `FM_v3_uav_test/aggregate_scene_summaries.py` — `METRICS`, `_extract`, `aggregate_scene`.
- `npz_analysis/analyze_npz.py` — `HEADLINE_KEYS`.
