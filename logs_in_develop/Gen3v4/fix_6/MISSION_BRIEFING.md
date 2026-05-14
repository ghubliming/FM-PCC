# Mission Briefing: iMeanFlow (iMF) Pipeline Standardization (Fix #6)

**Status**: Completed
**Target**: Align the iMF-PCC pipeline with the standardized `FMv3ODE` and `Drifting` behavior.

---

## 1. Problem Statement
The iMeanFlow (iMF) pipeline was behaving inconsistently compared to the other modular FM-PCC engines. Specifically:
- **Hallucinating Saves**: The evaluation script was saving results to incorrect locations (root directory) because it lacked the `plan` serialization config.
- **Missing Features**: The eval script was missing the `Projector` instantiation and YAML-based variant looping used in other baselines.
- **Non-Standard Slurm**: The Slurm scripts lacked the "PRO-LOGGING" setup and standard environment exports required for cluster consistency.

## 2. Technical Objectives
- **Standardize Eval**: Replace the ad-hoc iMF eval script with the robust `FMv3ODE` logic.
- **Restore Config Parity**: Add the missing serialization blocks for iMF in the main config file.
- **Standardize Slurm**: Update the `.sh` scripts to match the reference "Bone" implementation.

## 3. Implementation Details

### A. Evaluation Rebuild (`eval_flow_matching_v3_imeanflow.py`)
- **Action**: Completely replaced the script logic with the `FMv3ODE` version.
- **Projection Integration**: Restored the `Projector` class and `config/projection_eval.yaml` reading, enabling standard DPCC-R/T/C evaluations.
- **Path Resolution**: Fixed the saving logic to utilize `args.savepath/results` instead of relying on manually passed `--results-dir` flags.

### B. Results Loader Rebuild (`load_results_flow_matching_v3_imeanflow.py`)
- **Action**: Adapted the FMv3 results loader to iMF.
- **Aggregation**: Restored automatic plot generation (success rates and timesteps) for all seeds, saving them in the `plans/.../plots/` directory.

### C. Configuration Synchronization (`config/avoiding-d3il.py`)
- **Added `plan_fm_v3_imeanflow`**: Implemented the missing plan block to enable correct `f:plans/...` path generation for the iMF engine.
- **Class Mapping**: Ensured `iMFDiffusion` is correctly mapped for serialization.

### D. Slurm Standardization
- **Scripts Updated**: `train_imf.sh`, `eval_imf.sh`, and `load_results_imf.sh` now include:
  - **PRO-LOGGING**: Shortcut creation for `Slurm_Codes/logs/latest.log`.
  - **EXIT Traps**: Proper job-end time tracking.
  - **Environment Exports**: Standardized `PYTHONPATH`, `MUJOCO_GL`, and `MPLBACKEND` settings.

## 4. Verification Plan
1. **Serialization Check**: Run `train_imf.sh` and verify the `diffusion_config.pkl` is saved in `logs_in_develop/flow_matching_v3_imeanflow/`.
2. **Inference Check**: Run `eval_imf.sh` and verify results are saved into `plans/flow_matching_v3_imeanflow/`.
3. **Plotting Check**: Run `load_results_imf.sh` and verify aggregated PNG/PDF charts are generated.

---
**Auditor Signature**: Antigravity AI
**Date**: May 14, 2026
