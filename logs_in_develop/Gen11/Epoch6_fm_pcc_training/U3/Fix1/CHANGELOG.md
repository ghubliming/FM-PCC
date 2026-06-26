# U3 Fix1 — eval outputs go under `plans/`, not nested in the train model folder

## Problem
Eval wrote to `<train_savepath>/eval/<projection>/`, i.e. **inside the trained
model's own folder** (next to the `.pt` weights). The other FM-PCC models keep
train weights and eval results separate: weights under the experiment folder, eval
under a sibling `plans/` tree (e.g. `logs/avoiding-d3il-visual/plans/...`). UAV-FM
should match.

(Bonus: the aggregate glob `…/uav-<scene>/*/*/eval/<proj>/results.json` was already
**mismatched** to the real path depth — our `exp_name` contains a `/`
(`flow_matching_v3_uav/H8_…`), so there are 3 dirs before the results, not 2 — so it
matched nothing.)

## Fix
Eval now writes to a sibling `plans/` tree; train weights are untouched.

| | Before | After |
|---|---|---|
| Train weights | `…/uav-<scene>/flow_matching_v3_uav/<exp>/<seed>/` | **unchanged** |
| Eval results | `…/<seed>/eval/<proj>/` (inside train folder) | `…/uav-<scene>/**plans**/flow_matching_v3_uav/<exp>/<seed>/<proj>/` |

### New layout
```
logs/UAV_FM/uav-<scene>/
  flow_matching_v3_uav/<exp>/<seed>/         ← TRAIN: weights (state_*.pt), configs
  plans/
    flow_matching_v3_uav/<exp>/<seed>/<proj>/   ← EVAL: results.json, <proj>.npz,
                                                  <proj>.png, all.png, eval_<proj>.log,
                                                  diagnostics/
    SCENE_SUMMARY.json                          ← per-scene roll-up
  (logs/UAV_FM/fm_uav_ALL_SCENES_SUMMARY.json   ← cross-scene roll-up)
```

## Files changed
- `FM_v3_uav_test/eval_fm_uav.py` — `eval_scene` builds `out_dir` from
  `<logbase>/<dataset>/plans/<rel-savepath>/<projection>` instead of
  `<savepath>/eval/<projection>`. `--scene all` rollup also moved under `plans/`.
- `FM_v3_uav_test/aggregate_scene_summaries.py` — read glob now
  `…/uav-<scene>/plans/**/<proj>/results.json` (recursive, depth-robust);
  `SCENE_SUMMARY.json` written under `plans/`; docstring updated.

## Verify
- `py_compile` clean on both files.
- Path simulation confirmed:
  - weights `…/uav-empty/flow_matching_v3_uav/H8_…/6/`
  - eval `…/uav-empty/plans/flow_matching_v3_uav/H8_…/6/fm_only/`

## Note
Existing eval output already produced under the old `…/<seed>/eval/` path (e.g. the
pillars run) is NOT migrated — re-run eval to populate the new `plans/` location.
Working-tree only; sync to cluster before running.
