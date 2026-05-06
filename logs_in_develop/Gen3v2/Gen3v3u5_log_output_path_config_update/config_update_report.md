# Config Update: FMv3-ODE Log Output Path Standardization

**Date**: 2026-05-04
**Category**: Hotfix / Configuration Update
**Status**: Implemented

## Overview
This update standardizes the folder naming logic for the `flow_matching_v3_ode_selectable` (FMv3-ODE) configuration. The goal was to make experiment folders more reflective of their specific hyperparameters for both training and evaluation.

## Changes

### 1. Training Naming Logic
Training folders now include the Beta sampling parameters and action weights to ensure every unique model configuration is stored separately.
- **New Watch List**: `args_to_watch_fmv3_ode_train`
- **Labels Added**:
    - `a`: `time_beta_alpha_v3`
    - `b`: `time_beta_beta_v3`
    - `aw`: `action_weight`
- **Folder Example**: `logs/avoiding-d3il/flow_matching_v3_ode_selectable/H8_D..._a1.5_b1.0_aw1/`

### 2. Evaluation (Plan) Naming Logic
Evaluation folders now prioritize planning-specific parameters (like the ODE solver method) while maintaining a clean path by omitting training-only metadata.
- **New Watch List**: `args_to_watch_fmv3_ode_plan`
- **Labels Added**:
    - `M`: `ode_solver_method_v3` (e.g., `Meuler`, `Mdopri5`)
    - `K`: `flow_steps_v3`
- **Folder Example**: `logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/H8_K10_Meuler_D.../`

### 3. Automated Model Loading
The `diffusion_loadpath` in the planning block was updated to include the training-time parameters (`a`, `b`, `aw`). This allows the evaluation script to automatically resolve the correct model path based on the shared configuration values. 

> [!IMPORTANT]
> **Bugfix**: Explicitly added `action_weight`, `time_beta_alpha_v3`, and `time_beta_beta_v3` to the `plan_fm_v3_ode_selectable` block. This prevents an `AttributeError` during f-string expansion when the evaluation script tries to resolve the model's load path.

### 4. Smart Config Snapshots
Every experiment now automatically archives its source configuration files (`.py` and `.yaml`) into a dedicated subfolder for full traceability.
- **Detailed Report**: [smart_config_snapshot_feature.md](smart_config_snapshot_feature.md)

## Expected Directory Structure

```text
logs/avoiding-d3il/
├── flow_matching_v3_ode_selectable/
│   └── H8_D..._a1.5_b1.0_aw1/ (Training Weights)
└── plans/
    └── flow_matching_v3_ode_selectable/
        └── H8_K10_Meuler_D.../ (Evaluation Plots & Results)
```

## Verification
- [x] Training path reflects Beta parameters.
- [x] Plan path reflects Solver method.
- [x] Evaluation correctly loads weights from the structured training folder.

---

## Update: Nested Evaluation Folders (6. May)

**Status**: Implemented
**Objective**: Eliminate directory clutter and prevent overwriting results when evaluating multiple models with identical planning parameters.

### Changes
1. **Nesting Logic**: Updated the `prefix` in `plan_fm_v3_ode_selectable` to dynamically include the training model's hyperparameter signature as a parent directory level.
2. **Implementation**: Used Python string concatenation in `config/avoiding-d3il.py` to keep the base prefix clean while injecting the "Train Path" level.
    - **Code**: `'prefix': 'f:plans/flow_matching_v3_ode_selectable/' + 'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}/'`

### Resulting Directory Structure
Evaluation results are now perfectly isolated by their parent training model:
```text
logs/avoiding-d3il/
└── plans/
    └── flow_matching_v3_ode_selectable/
        └── H8_D..._a1.5_b1.0_aw1/          <-- New Parent Level (Train Path)
            └── H8_K10_Meuler_D.../         <-- Evaluation Results
```

### Audit Findings
- **Dependency Safety**: Verified that the lazy f-string expansion succeeds because the hyperparameters it depends on are static values loaded onto the `args` object before evaluation.
- **Legacy Integrity**: Confirmed that this change is localized ONLY to the FMv3-ODE block. All legacy diffusion and baseline configurations remain untouched.
