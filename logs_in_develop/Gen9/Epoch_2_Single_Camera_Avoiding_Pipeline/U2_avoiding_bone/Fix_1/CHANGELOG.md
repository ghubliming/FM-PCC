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

These YAML entries are keyed by `'avoiding-d3il'` and don't need to change.

---

## Fix_1.2 — DPCC eval: `ModuleNotFoundError: fm_visual_avoiding.diffuser_visual_avoiding`

**Triggered by**: `temp/Gen9E2U2/outputs` — Slurm job 21278 (DPCC eval)  
**Affected**: `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` only

### Symptom

```
ModuleNotFoundError: No module named 'fm_visual_avoiding.diffuser_visual_avoiding'
```

Crash in `load_diffusion_with_override` while resolving the target class
`'diffuser_visual_avoiding.models.visual_gaussian_diffusion.VisualGaussianDiffusion'`.

### Root cause

`fm_visual_avoiding/utils/config.py:import_class` line 8–16:

```python
repo_name   = __name__.split('.')[0]          # = 'fm_visual_avoiding' (hardcoded from pkg)
module_name = '.'.join(_class.split('.')[:-1])  # = 'diffuser_visual_avoiding.models...'
# strip prefix only if it STARTS with repo_name — 'diffuser_...' does not → no strip
module = importlib.import_module(f'{repo_name}.{module_name}')
# tries: fm_visual_avoiding.diffuser_visual_avoiding.models... → ModuleNotFoundError
```

`import_class` always prepends `'fm_visual_avoiding.'`. This works when the target class lives in `fm_visual_avoiding` (the FM eval's `VisualFlowMatching` — a strip guard handles it). It breaks for `VisualGaussianDiffusion` which lives in `diffuser_visual_avoiding`.

### Fix

In the DPCC eval's `load_diffusion_with_override`, bypass `utils.config.import_class` and use plain `importlib` instead:

```python
# Before:
target_cls = utils.config.import_class(target_class)

# After:
import importlib as _ilib
_parts = target_class.rsplit('.', 1)
target_cls = getattr(_ilib.import_module(_parts[0]), _parts[1])
```

This imports `diffuser_visual_avoiding.models.visual_gaussian_diffusion` directly and gets `VisualGaussianDiffusion` — no package prefix manipulation.

The FM eval is unaffected (its `import_class` path works correctly for `VisualFlowMatching`).

---

## Files touched

| File | Fix | Change |
|---|---|---|
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Fix_1.1 | `Parser.dataset/config` hardcoded to `'avoiding-d3il-visual'` |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Fix_1.1 | Same Parser fix |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Fix_1.2 | `import_class` → direct `importlib` in `load_diffusion_with_override` |
