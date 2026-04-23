# Gen3v2u2 Hotfix: Eval Loading Logic Override

## Issue
When evaluating a Flow Matching model using an existing checkpoint, the `eval_flow_matching_v3_ode_selectable.py` script was hard-loading the diffusion class specified inside the checkpoint's `diffusion_config.pkl`. Since checkpoints from previous iterations or copied folders memorize their original module path (e.g., `flow_matcher_v3` or `diffuser`), the evaluation script ignored updates made to `config/avoiding-d3il.py` and bypassed the intended `flow_matcher_v3_ode_selectable` solver logic entirely. 

## Resolution
The `eval_flow_matching_v3_ode_selectable.py` script was modified to dynamically intercept the unpickling process and enforce the "existing config is king" paradigm.

### Logic Changes
1. **Custom Loading Interceptor**: Replaced `utils.load_diffusion` in the evaluation loop with an inline `load_diffusion_with_override` function.
2. **Dynamic Comparison**: The script now parses `config/avoiding-d3il.py` to get the intended target class (`args.diffusion`), resolves its full import string, and compares it against the class string found inside `diffusion_config.pkl`.
3. **Override & Warning**: If a discrepancy is found, the script throws a bold warning into `sys.stderr` detailing the mismatch, and forcefully overwrites the `.pkl` config's `_class` attribute with the one from `avoiding-d3il.py` before instantiating the model.

### 🐛 TypeError Bug and Fix
When the `_class` override logic was initially applied, it caused a crash:
```
TypeError: GaussianDiffusion.__init__() got an unexpected keyword argument 'time_beta_alpha_v3'
```
**The Root Cause**: The `diffusion_config.pkl` file saved the state of the configuration from *during training*. Because the `_class` override logic instantiates the new target class by passing the parameters saved in the old pickle's dictionary (`_dict`), it inadvertently passed training parameters (like `time_beta_alpha_v3` from `flow_matcher_v3`) into the new class. If the new class configuration didn't expect that parameter (for example, if it was an older copied script), Python threw an `unexpected keyword argument` error. Note that `plan_fm_v3_ode_selectable` intentionally removes those parameters from the plan, confirming that the eval parameter block should override training state.

**The Fix**: A dynamic signature inspection filter was added. Before instantiating the overriden class, the script now introspects the `__init__` signature of the new target class and strictly drops any unexpected kwargs from the old pickle's dictionary.

### Key Snippet Introduced
```python
if pickled_class_str != target_class_str:
    print(f"\n=======================================================", file=sys.stderr)
    print(f"[WARNING] Pickled diffusion class does not match existing d3il.py config!", file=sys.stderr)
    print(f"Pickled config class: {pickled_class_str}", file=sys.stderr)
    print(f"Existing d3il.py class: {target_class_str}", file=sys.stderr)
    print(f"Overriding picked config with existing d3il.py config!", file=sys.stderr)
    print(f"=======================================================\n", file=sys.stderr)
    diffusion_config._class = target_class_resolved

    # Safely filter _dict to only include arguments the new class accepts
    import inspect
    sig = inspect.signature(target_class_resolved.__init__)
    valid_kwargs = set(sig.parameters.keys())
    keys_to_remove = [k for k in diffusion_config._dict if k not in valid_kwargs]
    for k in keys_to_remove:
        print(f"[WARNING] Dropping unexpected kwarg from pickle: '{k}'", file=sys.stderr)
        del diffusion_config._dict[k]
```

## Result
* **Training Integrity Maintained**: Resuming training via `avoiding-d3il.py` will still honor the original folder structure, meaning training continuity is completely safe.
* **Eval Independence**: The evaluation script is no longer trapped by legacy `.pkl` paths. It now correctly identifies when the evaluation environment is running a newer codebase and automatically points the model construction logic to the correct, updated directory (`flow_matcher_v3_ode_selectable`).
* **Crash Proof**: Evaluator instantiation safely sanitizes outdated keyword arguments that exist in training checkpoints but have since been removed or updated in the target source code.
