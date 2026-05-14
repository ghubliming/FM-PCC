# iMeanFlow & Visual Aligning Pipeline Standardization Audit

**Objective**: Achieve complete behavioral parity with the FMv3ODE baseline, resolving critical `AttributeError`, `ModuleNotFoundError`, and dimension shape mismatches during the evaluation of `iMeanFlow` (iMF) and `ddpm_encdec_vision`.

## 1. iMF Matrix Dimension Mismatch (Projector)
**Error**: `RuntimeError: mat1 and mat2 shapes cannot be multiplied (4x48 and 32x32)`
**Root Cause**: In `eval_flow_matching_v3_imeanflow.py`, the script identifies transition matrix sizes for the Projector. Because the model was named `iMFDiffusion` (and not exactly `GaussianDiffusion`), it defaulted to the `states` only branch (transition dimension 4). However, `iMFDiffusion` predicts both states and actions (`observation_dim` + `action_dim` = 6). The Projector built 32x32 matrices instead of 48x48.
**Fix**: Updated the model class check to explicitly include `iMFDiffusion` within the `states_actions` logic branch, ensuring the transition dimension accurately reflects both states and actions.

## 2. Duplicate Model Argument in Config Instantiation
**Error**: `TypeError: iMFDiffusion.__init__() got multiple values for argument 'model'` (Also affected `GaussianDiffusion`)
**Root Cause**: The pickled `diffusion_config.pkl` saved the original `model` configuration string into `self._dict` during training. During evaluation, the instantiated model object is passed positionally: `diffusion_config(model)`. Python raised a `TypeError` because `model` was provided twice (positionally and via the kwargs dict).
**Fix**: Implemented a surgical filter in `eval_flow_matching_v3_imeanflow.py` (and the Visual pipeline) to explicitly call `diffusion_config._dict.pop('model', None)` immediately before model instantiation, resolving the duplication. Additionally, ported the FMv3ODE "Signature Filtering" logic to safely remove unexpected kwargs (like `time_beta_alpha_v3`) from older pickled configurations.

## 3. Recursive Import Error
**Error**: `ModuleNotFoundError: No module named 'flow_matcher_v3_imeanflow.flow_matcher_v3_imeanflow'`
**Root Cause**: The `import_class` utility in `flow_matcher_v3_imeanflow/utils/config.py` blindly prepended the repository namespace to provided class strings. Because the `avoiding-d3il.py` plan configuration explicitly used the full path (`flow_matcher_v3_imeanflow.models.iMFDiffusion`) to maintain deterministic checkpoint folder names, the utility recursively duplicated the prefix.
**Fix**: Upgraded `import_class` to check if the class string already begins with the repository namespace (`module_name.startswith(repo_name)`), allowing the use of explicit package paths without triggering recursion.

## 4. FMv3ODE Workflow Parity (Seed Iteration)
**Error**: `FileNotFoundError: No such file or directory: '.../5/dataset_config.pkl'`
**Root Cause**: The evaluation pipeline was being driven by hardcoded bash loops (`for SEED in 5 6 7 8 9; do`) which failed immediately if a sequential seed was missing, whereas the FMv3ODE standard reads the available seeds dynamically.
**Fix**: 
- Removed all `for SEED` loops from `eval_imf.sh` and `eval_visual_aligning.sh`.
- Rewrote the evaluation python scripts to parse `config/projection_eval.yaml` for targets (`seeds: [6, 7, 8, 9, 10]`) and gracefully iterate through them within the script, maintaining 100% architectural parity with the `eval_flow_matching_v3_ode_selectable.py` baseline.
- Ensured uniform `Tee` logger implementations across all pipelines for dual console-to-disk outputs.
