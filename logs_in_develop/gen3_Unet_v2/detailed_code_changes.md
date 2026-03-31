# Flow Matching U-Net V2 (Gen3) Implementation Log

## Goal Approached
1. **Incremental Update & Code Isolation:** Implement a safe, independent pipeline (`flow_matcher_unet_v2`) to develop and test a new U-Net architecture without risking stability or modifying the established `diffusion` and `flow_matcher` codebases.
2. **Preventing Data Clashing:** Eliminate the issue where evaluation and plan outputs overwrite each other. By assigning unique folder prefixes, both the training weights and evaluation plans for the new U-Net model are saved into strictly isolated subfolders (`logs/avoiding-d3il/flow_matching_unet_v2/` and `logs/avoiding-d3il/plans/flow_matching_unet_v2/`).

## Detailed Code Changes

### 1. Package Duplication and Class Renaming
- **Duplicated Package:** Created `flow_matcher_unet_v2` as an exact copy of `flow_matcher`.
- **Class Update:** In `flow_matcher_unet_v2/models/unet1d_temporal_cond.py`, renamed the core architecture `UNet1DTemporalCondModel` to `Flow_matcher_U_Net_v2` and added `TODO` block comments indicating where structural modifications for the new U-Net will take place.
- **Init Update:** Modified `flow_matcher_unet_v2/models/__init__.py` to expose the new `Flow_matcher_U_Net_v2` class rather than the original class.

### 2. Configuration (`config/avoiding-d3il.py`)
Appended two independent config entries (without modifying any existing `diffusion`, `flow_matching`, or `plan` entries):
- **`flow_matching_unet_v2`:** The training configuration utilizing `'model': 'models.Flow_matcher_U_Net_v2'` and assigning `'prefix': 'flow_matching_unet_v2/'`.
- **`plan_fm_unet_v2`:** The evaluation configuration specifically directing the loading logic to `'diffusion_loadpath': 'f:flow_matching_unet_v2/...'` and saving evaluation logs to `'prefix': 'plans/flow_matching_unet_v2/'`.

### 3. Test Scripts (`FM_Unet_v2_test/`)
- Duplicated the testing folder `FM_test/` into `FM_Unet_v2_test/`.
- **`train_FM_Unet_v2.py`**: Changed the import path to `import flow_matcher_unet_v2.utils` and targeted the parsed experiment `experiment='flow_matching_unet_v2'`.
- **`eval_FM_Unet_v2.py`**: Updated imports directly linked to `flow_matcher_unet_v2.sampling...` and parsed experiment `experiment='plan_fm_unet_v2'`.
- **`load_results_FM_Unet_v2.py`**: Switched from `diffuser` baseline imports to `flow_matcher_unet_v2.utils` and parsed the results under `experiment='plan_fm_unet_v2'`.

---

## Pseudo-Test Walk-through

Below is a theoretical execution trace mapping out the robustness of this pipeline:

### 1. Training Initialization (`train_FM_Unet_v2.py`)
1. **Execution:** Running `python FM_Unet_v2_test/train_FM_Unet_v2.py --seed 0` begins the process.
2. **Config Hooking:** The script queries `utils.Parser` with `experiment='flow_matching_unet_v2'`.
3. **Dynamic Import Magic:** When the `utils.Config` class instantiates `'model': 'models.Flow_matcher_U_Net_v2'`, it leverages the dynamic `import_class()` mechanism (inside `flow_matcher_unet_v2/utils/config.py`). Because the tool evaluates the Python package context (`repo_name = __name__.split('.')[0]`), it natively requests `flow_matcher_unet_v2.models.Flow_matcher_U_Net_v2`.
4. **Result:** The exact, safe, isolated architecture is loaded.

### 2. Checkpointing and Logging Separation
1. **Data Separation:** As training finishes, `prefix` assigns the logging path to `logs/avoiding-d3il/flow_matching_unet_v2/H8_K20_Dmodels.diffusion.GaussianDiffusion/0`.
2. **Result:** No conflicts occur with the `logs/avoiding-d3il/flow_matching/` weights directory.

### 3. Evaluation Initialization (`eval_FM_Unet_v2.py`)
1. **Execution:** Running `python FM_Unet_v2_test/eval_FM_Unet_v2.py` begins policy simulation.
2. **Retrieval hook:** The `Parser` assigns `experiment='plan_fm_unet_v2'`.
3. **Precision Loading:** Due to `'diffusion_loadpath': 'f:flow_matching_unet_v2/H{horizon}_{n_steps}_D{diffusion}'`, the `eval` script knows strictly to retrieve the variant model housed in the new subfolder (trained in the previous step) rather than mistakenly loading a standard diffuser or old flow matching run.
4. **Isolated Planning Logs:** Any planning visuals, arrays (`.npz`), or plots are piped flawlessly into `logs/avoiding-d3il/plans/flow_matching_unet_v2/`, avoiding all prior data clash issues.
