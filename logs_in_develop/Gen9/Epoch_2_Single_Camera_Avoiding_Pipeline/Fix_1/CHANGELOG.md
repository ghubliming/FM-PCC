# Gen9 Epoch 2 — Fix-1: Stale dataset re-exports in `datasets/__init__.py`

**Date**: 2026-06-03
**Status**: ✅ Fixed (uncommitted)
**Triggered by**: `temp/debug_Gen9E2/outputs` — first cluster smoke run of `train_fm_visual_avoiding.sh` (Slurm job 21143, commit `d0c2a5c`) crashed at import time.
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md) (Phase 1-4 land + §10 revision)

---

## 1. Symptom (from cluster log)

```
File "/data/home/llim/.../fm_visual_avoiding_test/train_fm_visual_avoiding.py", line 172, in <module>
    from fm_visual_avoiding.datasets.sequence import ParityAvoidingDataset
  File "/u/home/llim/.../fm_visual_avoiding/datasets/__init__.py", line 1, in <module>
    from .sequence import Batch, ParityAligningDataset, StateOnlyAligningDataset
ImportError: cannot import name 'ParityAligningDataset' from 'fm_visual_avoiding.datasets.sequence'
```

The train script's *own* import line targets the correct new class name (`ParityAvoidingDataset`). But Python evaluates the package's `__init__.py` first before any submodule import. The `__init__.py` had the OLD class names baked in.

## 2. Root cause

During Phase 2 (Epoch 2 dataset rewrite), `sequence.py` in both `diffuser_visual_avoiding/datasets/` and `fm_visual_avoiding/datasets/` had:

| Before | After Phase 2 |
|---|---|
| `class ParityAligningDataset` | `class ParityAvoidingDataset` *(renamed)* |
| `class StateOnlyAligningDataset` | *(class removed — out of scope per plan §4.1)* |

But the sibling `__init__.py` files still carried the upstream re-export line copied verbatim from the aligning package:

```python
# (broken — old names)
from .sequence import Batch, ParityAligningDataset, StateOnlyAligningDataset
from .normalization import LimitsNormalizer
```

Result: `import fm_visual_avoiding.datasets.<anything>` runs `__init__.py` first → `ImportError` before the train script even gets to its own (correct) import line.

**Why the Docker AST check didn't catch it**: `ast.parse()` only validates syntax, not import resolution. The `__init__.py` line is syntactically valid Python; the import only fails at runtime when the referenced names aren't in `sequence.py`'s namespace. Resolving this in Docker would have required actually importing the package, which the no-Python-runtime constraint blocks.

## 3. Fix

Both `__init__.py` files rewritten to match the actual class surface of the rewritten `sequence.py`:

**`diffuser_visual_avoiding/datasets/__init__.py`** (and `fm_visual_avoiding/datasets/__init__.py` — identical):

```python
from .sequence import Batch, ParityAvoidingDataset
from .normalization import LimitsNormalizer
```

Changes vs the broken state:
- `ParityAligningDataset` → `ParityAvoidingDataset` (rename).
- `StateOnlyAligningDataset` dropped from the re-export list (class no longer exists in `sequence.py`).
- `Batch` and `LimitsNormalizer` re-exports unchanged.

## 4. Other stale refs cleaned up in the same fix

Two **comments** in the train scripts still said "9D" and referenced the old dataset class. Cosmetic but cleaned for consistency:

| File | Line | Before | After |
|---|---|---|---|
| `diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py` | 3 | `# Core: diffuser_visual_avoiding  | Dataset: ParityAligningDataset (9D)` | `# Core: diffuser_visual_avoiding  | Dataset: ParityAvoidingDataset (6D, single-cam)` |
| `fm_visual_avoiding_test/train_fm_visual_avoiding.py` | 3 | `# Core: fm_visual_avoiding ... | Dataset: ParityAligningDataset (9D)` | `# Core: fm_visual_avoiding ... | Dataset: ParityAvoidingDataset (6D, single-cam)` |

## 5. Files Touched

```
M  diffuser_visual_avoiding/datasets/__init__.py        (1 line changed)
M  fm_visual_avoiding/datasets/__init__.py              (1 line changed)
M  diffuser_visual_avoiding_test/train_visual_avoiding_dpcc.py  (1 comment line)
M  fm_visual_avoiding_test/train_fm_visual_avoiding.py          (1 comment line)
```

Total: **4 files, ~4 lines** of actual change.

## 6. Verification

Local checks (no Python runtime in Docker, so import-level verification is structural):

| Check | Result |
|---|---|
| Both `__init__.py` files re-export only names that exist in `sequence.py` | ✅ Confirmed — `class ParityAvoidingDataset` defined; `Batch` namedtuple defined; `LimitsNormalizer` re-export untouched |
| Grep for `ParityAligningDataset` / `StateOnlyAligningDataset` across both packages and both test folders | ✅ Zero remaining hits |
| `ast.parse` on both `__init__.py` files | ✅ |
| Comments in train scripts no longer reference 9-D / old dataset class | ✅ |

**Cluster-side rerun expectation**: the same Slurm job (`train_fm_visual_avoiding.sh`) should now get past `line 172` and reach the dataset construction step. Next plausible failure point if it crashes: actual data presence at `d3il/environments/dataset/data/avoiding/all_data/` (the §6 hard-blocker from the parent changelog). That's a separate issue.

## 7. Lesson for next time

When renaming a class in `<pkg>/datasets/sequence.py` (or any submodule), **always grep the sibling `__init__.py`** for the old name first. The Phase 2 audit caught the rename in callers but not in re-export shims, and AST-level Docker checks can't surface this class of bug. Add a TODO to the Phase 5 smoke recipe (parent CHANGELOG §7): "step 0 — `python -c \"import diffuser_visual_avoiding; import fm_visual_avoiding\"` to catch `__init__.py` import-time failures before launching the bigger smoke tests."

## 8. Cross-reference

- Parent: [`../CHANGELOG.md`](../CHANGELOG.md) §2.2 (the dataset rewrite that introduced the rename) and §7 (smoke recipe to update with the lesson from §7 above).
- Original failure log: `temp/debug_Gen9E2/outputs` (Slurm job 21143).
