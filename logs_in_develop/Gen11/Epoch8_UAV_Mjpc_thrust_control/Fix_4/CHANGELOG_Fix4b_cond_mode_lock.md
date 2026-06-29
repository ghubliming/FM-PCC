# Fix_4b — `cond_mode` Lock: Eval Must Read from Checkpoint, Not Plan Block

**File changed:** `FM_v3_uav_test/eval_fm_uav.py` (`eval_scene`)

---

## Bug

```
ValueError: operands could not be broadcast together with shapes (9,) (6,)
```

Crash at `normalization.py:159` during `policy({0: obs}, ...)`.

**Root cause:** `cond_mode` was read in `_run_variant` from the plan block config:
```python
cond_mode = str(config.get('cond_mode', 'p_des'))   # ← plan block, user-editable
```

If the model was trained with `cond_mode='pos_only'` (6D obs) but the plan block still has
the default `cond_mode='p_des'`, the eval builds a 9D obs tensor. The normalizer (baked at
train time with 6D) gets a 9D input → shape mismatch → crash.

## Why This Happens

`cond_mode` is a **model property** — the obs layout (and normalizer shape) is fixed at
training time. The plan block value is user-editable and can silently drift.

`controller` is legitimately a plan-block runtime choice (same checkpoint, different v_des).
`cond_mode` is NOT — it must match the checkpoint.

## Fix

In `eval_scene`, after `build_experiment` loads the checkpoint, override `config['cond_mode']`
from `parsed.cond_mode` (the TRAIN block's resolved value):

```python
config['cond_mode'] = str(getattr(parsed, 'cond_mode', config.get('cond_mode', 'p_des')))
print(f'[ eval ] cond_mode={config["cond_mode"]}  (source: train checkpoint args)')
```

`parsed` comes from `build_experiment` → `p.parse_args(experiment='flow_matching_v3_uav')`
which reads the TRAIN block. So `parsed.cond_mode` is always the value the model was trained
with, regardless of what the plan block says.

## Evidence from E8 Run

```
[ utils/serialization ] Loading model from ...H8_D..._cmpos_only_ctrlmjpc/6
[ datasets/buffer ]  observations: (475, 560, 6)    ← 6D obs (pos_only)
...
ValueError: operands could not be broadcast together with shapes (9,) (6,)
```

Model loaded 6D normalizer; eval built 9D obs; crash.

## After Fix

`cond_mode` is auto-derived from the checkpoint. Plan block value is ignored for this field.
Log line `[ eval ] cond_mode=pos_only  (source: train checkpoint args)` confirms it.
