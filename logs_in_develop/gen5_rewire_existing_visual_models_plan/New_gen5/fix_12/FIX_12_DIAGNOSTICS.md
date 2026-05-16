# FIX_12: Scientific Diagnostic Visibility & Real-Time Export

## 1. Problem Statement
Evaluation of the Gen5 Visual-Aligning pipeline was previously a "black box" until the entire process finished. 
1. **Data Loss Risk**: If the script crashed during a 50-context run, all trajectory data was lost.
2. **Missing Foresight**: Diagnostics only showed raw pixels, hiding what the U-Net was "predicting" (the MPC horizon).
3. **Missing Baseline**: No side-by-side comparison with the Expert Dataset (Golden Standard).

## 2. Solution: Real-Time Diagnostic Engine
We have overhauled `eval_ddpm_encdec_vision.py` to support immediate data export and scientific reporting.

### A. Real-Time Export (`_export_rollout_realtime`)
* **Trigger**: Immediately after each rollout finishes (within the `reset()` call).
* **Data Persistence**: Saves `rollout_{idx}_data.pkl` containing the full trajectory, MPC plans, and tracking errors.
* **Immediate PNG**: Generates `rollout_{idx}_report.png` instantly.

### B. Scientific Report Format (6-Panel Grid)
The new PNG reports provide the following insights:
1. **XY Projection**: Overlays the actual path (Black) with the MPC Horizons (Blue). This shows if the model's "intent" matches its "execution."
2. **Temporal X/Y**: Tracks position over time to identify planning lag or oscillation.
3. **Z-Stability**: Monitors the height of the end-effector to ensure it stays in contact with the box (Aligning requirement).
4. **Tracking Error**: Calculates the Euclidean distance between the MPC's target and the physical state, exposing IK calibration issues.
5. **Velocity**: Monitors for "jerky" motion or sudden jumps.

## 3. Implementation Details
* **File**: `eval_ddpm_encdec_vision.py`
* **Mechanism**: Added `save_path` attribute to `VisualAgentWrapper`.
* **Plotting**: Uses `matplotlib` for the scientific grid and `imageio` for the camera GIFs.

## 4. How to Use
1. Run evaluation as normal: `python ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`.
2. Open a second terminal or file explorer.
3. Navigate to `logs/aligning-d3il-visual/.../realtime_diagnostics/`.
4. Observe the PNGs appearing one-by-one as the robot completes each test.

---
**Status:** IMPLEMENTED and VERIFIED.
