# Fix 4 (part 1) — UAV-FM output path: move under `logs/UAV_FM/`

## Problem

UAV-FM training/eval dumped its run folders **directly at the top of `logs/`**
as `logs/uav-empty/`, `logs/uav-corridor/`, `logs/uav-s_curve/`,
`logs/uav-pillars/`, plus a top-level `logs/fm_uav_ALL_SCENES_SUMMARY.json`.
That clutters `logs/` alongside every other experiment's output. It should be
namespaced under `logs/UAV_FM/`.

## Root cause

`savepath` is assembled in `flow_matcher_v3_uav/utils/setup.py:mkdir()`:

```python
args.savepath = os.path.join(args.logbase, args.dataset, args.exp_name, str(args.seed))
```

`logbase` came from `config/uav.py` (`logbase = 'logs'`) and `dataset` is the
`uav-<scene>` string. So everything landed at `logs/uav-<scene>/...`.

**Key constraint:** the `dataset` string `uav-<scene>` is *load-bearing* — the
data loader selects the scene branch from it
(`flow_matcher_v3_uav/datasets/d4rl.py:41`, `env.startswith('uav')`). So the
folder name `uav-<scene>` must NOT be renamed. The fix changes only `logbase`.

## Fix

Changed `logbase` and the three places that independently hardcoded the old
`logs` base so they all agree:

| File | Change |
|------|--------|
| `config/uav.py` | `logbase = 'logs'` → `logbase = 'logs/UAV_FM'` |
| `FM_v3_uav_test/aggregate_scene_summaries.py` | `--logbase` default `'logs'` → `'logs/UAV_FM'` |
| `FM_v3_uav_test/eval_fm_uav.py` | `--scene all` rollup path `logs/uav-all/...` → `logs/UAV_FM/uav-all/...` |

`savepath` (train weights + eval `results.json`) is now derived automatically
from the new `logbase`, since both `train_fm_uav.py` and `eval_fm_uav.py`'s
`build_policy()` go through the same `Parser`/`config.uav`. No path is hardcoded
in those hot paths — they all flow from `logbase`.

### New layout

```
logs/UAV_FM/
  uav-empty/    flow_matching_v3_uav/<exp_name>/<seed>/{weights, eval/<projection>/results.json, ...}
  uav-corridor/ ...
  uav-s_curve/  ...
  uav-pillars/  ...
  uav-<scene>/SCENE_SUMMARY.json              (per-scene roll-up)
  fm_uav_ALL_SCENES_SUMMARY.json              (cross-scene roll-up)
  uav-all/<projection>/SUMMARY.json           (only the experimental --scene all path)
```

### Cosmetic / doc updates (no behaviour change)

Updated the now-stale `logs/uav-...` strings in:
- `config/uav.py` module docstring
- `FM_v3_uav_test/aggregate_scene_summaries.py` module docstring
- echo lines in `Slurm_Codes/sbatch/uav_fm/{fm_uav_pipeline,fm_uav_all_pipeline,train_all_scenes,eval_all_scenes}.sh`

## Not changed / why

- **`dataset` string `uav-<scene>`** — load-bearing for data-branch selection;
  renaming it would break the loader. Only `logbase` moved.
- **`.gitignore`** — already ignores `logs/*`, so `logs/UAV_FM/` is covered; no
  change needed.
- **Existing old runs** under `logs/uav-*` (if any on the cluster) are NOT
  migrated by this change. New runs go to `logs/UAV_FM/`; move or delete old
  ones manually if desired (none are deleted automatically).

## Verify after sync

- A fresh train writes to `logs/UAV_FM/uav-<scene>/flow_matching_v3_uav/.../<seed>/`.
- Eval loads from that same path and writes `eval/<projection>/results.json` beside it.
- `aggregate_summaries.sh` (calls `aggregate_scene_summaries.py` with no
  `--logbase`) reads/writes under `logs/UAV_FM/` by default.

---

*Part 1 of Fix 4 (output path). Next fix in this folder addresses the
remaining train/eval issue.*
