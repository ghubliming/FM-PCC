# U2 Fix_1 — `KeyError: 'plan_fm_visual_avoiding'`

**Date**: 2026-06-05  
**Status**: ✅ Fixed (uncommitted)  
**Triggered by**: `temp/Gen9E2U2/outputs` — Slurm job 21276  
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md)

---

## Symptom

```
File "fm_visual_avoiding_test/eval_fm_visual_avoiding.py", line 192, in <module>
    args = Parser().parse_args(experiment='plan_fm_visual_avoiding', seed=seed)
  File "fm_visual_avoiding/utils/setup.py", line 96, in read_config
    params = getattr(module, 'base')[experiment]
KeyError: 'plan_fm_visual_avoiding'
```

Crash at the very first seed, before any model loading or environment startup.

---

## Root cause

`projection_eval.yaml` has `exps: ['avoiding-d3il']`. The eval loop sets:

```python
for exp in exps:          # exp = 'avoiding-d3il'
    ...
    class Parser(utils.Parser):
        dataset: str = exp            # 'avoiding-d3il'
        config:  str = 'config.' + exp  # 'config.avoiding-d3il'
```

`setup.py:95–96`:
```python
module = importlib.import_module(args.config)   # loads config/avoiding-d3il.py
params = getattr(module, 'base')[experiment]    # base['plan_fm_visual_avoiding']
```

`config/avoiding-d3il.py` is the **non-visual** config. It does not contain `'plan_fm_visual_avoiding'`. That entry only exists in `config/avoiding-d3il-visual.py`.

The `projection_eval.yaml` `exps` key was never updated to `'avoiding-d3il-visual'` for the visual pipeline. All the YAML constraint/bounds/ax_limits entries remain keyed by `'avoiding-d3il'` and are still looked up via `exp` — only the `Parser` class definition was wrong.

---

## Fix — both eval scripts

Decouple the YAML lookup key (`exp`) from the Python config the `Parser` loads. Override `dataset` and `config` to always point at the visual config, regardless of what `exps` says in the YAML:

```python
# Before:
class Parser(utils.Parser):
    dataset: str = exp              # 'avoiding-d3il'  ← loads non-visual config
    config:  str = 'config.' + exp  # 'config.avoiding-d3il'

# After:
class Parser(utils.Parser):
    dataset: str = 'avoiding-d3il-visual'       # ← visual config has plan_* entries
    config:  str = 'config.avoiding-d3il-visual'
```

`exp` (`'avoiding-d3il'`) is still used for all YAML lookups:
- `config['halfspace_constraints'][exp]`
- `config['obstacle_constraints'][exp]`
- `config['bounds'][exp]`
- `config['ax_limits'][exp]`

These YAML entries are keyed by `'avoiding-d3il'` and don't need to change.

---

## Files touched

| File | Change |
|---|---|
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | `Parser.dataset/config` hardcoded to `'avoiding-d3il-visual'` |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Same fix |
