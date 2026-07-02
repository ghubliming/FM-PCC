# Fix 1 — `pid_const_v` crash: `SequenceDataset` has no `.actions`

**Date:** 2026-07-02
**Scope:** `FM_v3_uav_test/eval_fm_uav.py`
**Job:** 22980 (pillars, seed 6, `--projection fm_only`)

## Error

```
File "FM_v3_uav_test/eval_fm_uav.py", line 539, in _run_variant
    _all_acts = dataset.actions.reshape(-1, 3)
AttributeError: 'SequenceDataset' object has no attribute 'actions'
```

## Root cause

`controller='pid_const_v'` auto-calibrates its constant speed from the dataset's mean
action magnitude (`mean(|action|) × DATASET_HZ`), so it needs the raw action array.
The code read it as `dataset.actions` — that attribute doesn't exist.

`SequenceDataset` (`flow_matcher_v3_uav/datasets/sequence.py`) stores raw arrays inside
a `fields` container, not as top-level attributes:

```python
self.action_dim = fields.actions.shape[-1]   # scalar dim IS a direct attribute
self.fields = fields                         # raw arrays live here
...
actions = self.fields.normed_actions[path_ind, start:end]   # sequence.py:124
```

So `dataset.action_dim` (a scalar) works fine directly on the object, but the raw
per-step array is only reachable via `dataset.fields.actions`.

## Why only `pid_const_v` hit this

The buggy line sits behind `if controller == 'pid_const_v':`
(`eval_fm_uav.py:538`). Every other controller (`pid`, `pid_stopgo`, `mjpc`, …) falls
into the `else` branch and never touches the dataset's raw actions at all
(`v_des_magnitude = 0.0   # unused by other controllers`). The bug was dormant until
`pid_const_v` was actually run.

## Fix

```python
# before:
_all_acts = dataset.actions.reshape(-1, 3)
# after:
_all_acts = dataset.fields.actions.reshape(-1, 3)
```

Confirmed no other `dataset.actions` / `dataset.observations` direct-attribute
accesses exist elsewhere in `eval_fm_uav.py` (grepped clean).
