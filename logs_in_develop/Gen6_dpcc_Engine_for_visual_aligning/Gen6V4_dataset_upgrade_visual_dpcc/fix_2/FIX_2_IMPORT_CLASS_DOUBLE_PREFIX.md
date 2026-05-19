# Fix 2 — `import_class` Double-Prefix Crash at Eval Load (2026-05-19)

## Crash

```
ModuleNotFoundError: No module named 'diffuser_visual_aligning.diffuser_visual_aligning'
  File "diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py", line 586
    diffusion_config._class = utils.config.import_class(target_class)
  File "diffuser_visual_aligning/utils/config.py", line 15
    module = importlib.import_module(f'{repo_name}.{module_name}')
```

Triggered on first eval run, immediately after model checkpoint is loaded.

---

## Root Cause

`import_class` in `diffuser_visual_aligning/utils/config.py` was designed for the original `diffuser` repo where class paths are **repo-relative** (e.g. `'diffusion.models.TemporalUnet'`). It derives `repo_name` from its own module path and prepends it:

```python
repo_name   = __name__.split('.')[0]          # → 'diffuser_visual_aligning'
module_name = '.'.join(_class.split('.')[:-1]) # → 'diffuser_visual_aligning.models.visual_gaussian_diffusion'
import_module(f'{repo_name}.{module_name}')   # → 'diffuser_visual_aligning.diffuser_visual_aligning.models...'  ← CRASH
```

The `'diffusion'` config key in `plan_visual_aligning_dpcc` stores the **fully-qualified** class path:
```
'diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion'
```

When `import_class` prepends `repo_name` again → double-prefix → module does not exist.

### Why training did not crash

During training, `Config` is called with the **class object** (not a string):
```python
from diffuser_visual_aligning.models.visual_gaussian_diffusion import VisualGaussianDiffusion
diffusion_config = utils.Config(VisualGaussianDiffusion, ...)
```

`import_class` hits `if type(_class) is not str: return _class` immediately and exits. The class object is stored directly into the saved pkl — no string import needed.

During eval, `load_diffusion_with_override` **explicitly re-imports** from the string `args.diffusion` to allow class override, which triggers the bug.

---

## Fix

**File**: `diffuser_visual_aligning/utils/config.py`

Strip the `repo_name.` prefix from `module_name` when the caller already passed a fully-qualified path:

**Before**:
```python
repo_name   = __name__.split('.')[0]
module_name = '.'.join(_class.split('.')[:-1])
class_name  = _class.split('.')[-1]
module = importlib.import_module(f'{repo_name}.{module_name}')
```

**After**:
```python
repo_name   = __name__.split('.')[0]
module_name = '.'.join(_class.split('.')[:-1])
class_name  = _class.split('.')[-1]
# Strip repo_name prefix if caller passed a fully-qualified path — avoids double-prefix.
if module_name == repo_name or module_name.startswith(repo_name + '.'):
    module_name = module_name[len(repo_name):].lstrip('.')
module = importlib.import_module(f'{repo_name}.{module_name}')
```

### Trace after fix

```
_class      = 'diffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion'
repo_name   = 'diffuser_visual_aligning'
module_name = 'diffuser_visual_aligning.models.visual_gaussian_diffusion'
  → starts with repo_name+'.' → strip → 'models.visual_gaussian_diffusion'
import_module('diffuser_visual_aligning.models.visual_gaussian_diffusion')  ✓
```

---

## Files Changed

| File | Change |
|------|--------|
| `diffuser_visual_aligning/utils/config.py` | `import_class`: strip `repo_name.` prefix from `module_name` before `import_module` call |
