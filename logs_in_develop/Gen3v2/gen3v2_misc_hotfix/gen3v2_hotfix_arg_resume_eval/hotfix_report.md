# Hotfix: Redundant Evaluation Metadata Logging

## Problem Description
During evaluation runs (e.g., using `scripts/eval.py`), the experiment directories were becoming cluttered with hundreds of `args_resume_X.json` files. 

### Key Issues:
1.  **Confusing Terminology**: The "resume" naming is designed for training (resuming a crashed job). In evaluation, there is no "resume" state, making these files misleading.
2.  **Redundancy**: In `eval.py`, the code often loops through multiple variants (e.g., `halfspace_variants`) for each seed. Since these variants share the same `savepath`, the `Parser` was saving identical configuration files multiple times per execution.
3.  **Metadata Clutter**: High-seed counts and multiple variants resulted in over 140+ files in some seed directories, making it difficult to interpret the actual experiment results.

## Root Cause
The `Parser` class in `utils/setup.py` (both in `diffuser` and `flow_matcher_v3_ode_selectable`) had a hardcoded call to `self.save(args)` inside the `mkdir()` method. 

```python
# Before fix in Parser.mkdir
def mkdir(self, args):
    ...
    if mkdir(args.savepath):
        print(f'[ utils/setup ] Made savepath: {args.savepath}')
    self.save(args) # <--- Always triggered save
```

This method was called by `parse_args()` regardless of whether the `experiment` type was `'train'` or `'plan'`.

## Solution
Modified the `Parser` architecture to distinguish between training and inference/evaluation phases.

### 1. Conditional Save Flag
Updated `Parser.parse_args` to calculate a `save` boolean. It is only `True` if the experiment is `'train'`.

```python
# After fix in Parser.parse_args
save = (experiment == 'train')
self.mkdir(args, save=save)
```

### 2. Updated Directory Creation
Updated `Parser.mkdir` to accept the `save` flag. It now only calls `self.save(args)` if explicitly instructed to do so.

```python
# After fix in Parser.mkdir
def mkdir(self, args, save=True):
    ...
    if save:
        self.save(args)
```

## Impact
- **Training**: Still saves `args.json` and resume files as expected.
- **Evaluation**: No longer generates any `args_resume_*.json` files, keeping the logs clean and focused on output data/plots.

## Files Modified
- `/workspaces/FM-PCC/flow_matcher_v3_ode_selectable/utils/setup.py`
- `/workspaces/FM-PCC/diffuser/utils/setup.py`
