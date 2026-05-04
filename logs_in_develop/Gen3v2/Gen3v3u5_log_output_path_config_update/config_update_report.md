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
