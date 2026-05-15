# Code Changelog: Visual Evaluation Output Upgrade (Fix 9) - FINAL PARITY

## Overview
This document records the finalized code modifications to the Gen5 visual evaluation script. This update achieves **100% parity** with the legacy FMv3ODE pipeline in both **Data Schema (.npz)** and **Visual Diagnostics (.png)**.

## Files Modified
1. `d3il/simulation/aligning_sim.py`
2. `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`

## Detailed Changes

### 1. Simulation Return Expansion (`aligning_sim.py`)
**Change:** Updated `test_agent()` return signature.
- **Old:** `return success_rate, mode_encoding`
- **New:** `return success_rate, mode_encoding, successes, mean_distance`
- **Reason:** Allows the evaluation script to access the per-rollout success/fail booleans and distance metrics required for the Analysis Matrix.

### 2. Matplotlib Engine (`eval_ddpm_encdec_vision.py`)
**Change:** Integrated `matplotlib` with the `Agg` backend for headless plotting.
**Code Added:**
- Rollout Grid Plot: Generates `{variant}.png` with 6 columns per rollout.
    - **Cols 0-2:** X, Y, Z Time-series.
    - **Col 4:** 2D XY Path.
    - **Col 5:** "Real vs. Desired" overlay (Robot path vs. U-Net plans).
- Aggregate Plot: Generates `all.png` with all trial paths overlaid.

### 3. Metric Injection (`eval_ddpm_encdec_vision.py`)
**Change:** Expanded `np.savez` to include all legacy keys used by the "Matrix Explorer" data analysis scripts.
- `n_success`: Flat success array.
- `n_steps`: Step counts.
- `avg_time`: Inference latency.
- `obs_all`: Step-by-step positions.
- `act_all`: Predicted actions.
- `sampled_trajectories_all`: Full U-Net plans.
- `pos_tracking_errors`: Precision/Drift metrics.
- `mean_distance`: Target proximity.
- `args`: Experiment hyperparameters.

### 4. Tracking Error Logic
**Change:** Implemented geometric drift calculation in `VisualAgentWrapper`.
- Logic: `error = norm(current_pos - previous_prediction)`.
- Captured in the `pos_tracking_errors` field in the `.npz`.

## Result
The Gen5 evaluation is now structurally identical to the FMv3ODE state-based evaluation. It is fully compatible with all existing data aggregation and plotting tools in the FM-PCC repository.

---

**Final Changelog generated for FM-PCC Diagnostic Phase 9.**
