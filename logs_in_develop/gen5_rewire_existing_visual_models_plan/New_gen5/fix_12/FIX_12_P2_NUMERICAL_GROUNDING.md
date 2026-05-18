# FIX_12_P2: Numerical Grounding & Zero-Padding Resolution

This document records the fix for the "Numerical Explosion" failure mode, where the robot end-effector was observed flying out of bounds at hypersonic speeds (e.g., 1750 meters).

## 1. The Zero-Padding Trap (Statistics Corruption)
*   **The Symptom**: Diagnostic plots showed the robot moving exponentially into space ($X=17.5m, Y=-1750m, Z=700m$).
*   **Technical Cause**: 
    *   The robotics dataset uses **Zero-Padding** to ensure all trajectories have a fixed length (e.g., 256 steps).
    *   The Scaler was initially calculated on the **entire raw tensor**, including these thousands of zeros.
    *   **Result**: The Standard Deviation (`y_std`) collapsed toward zero. In the `inverse_scale_output` formula ($y \times std + mean$), this corrupted $std$ caused the model's output to be amplified by orders of magnitude when converted back to meters.
*   **Fix**: Updated `train_ddpm_encdec_vision.py` to use `dataset.get_all_observations()` and `dataset.get_all_actions()`. These helper methods use mask-filtering to exclude all padding zeros from the statistics calculation.
*   **Scientific Result**: The Scaler now possesses "Clean Statistics" ($Mean$ and $Std$) that accurately represent the physical robot motion.

## 2. Feedback Loop Stabilization
*   **Mechanism**: The "Numerical Explosion" was intensified by the **Mental Map (Open-Loop)**. Because we accumulate the model's predicted actions into the robot's current position, a single "Mega-Action" of 1750 meters becomes the starting point for the next step, leading to a permanent departure from the simulation workspace.
*   **Outcome**: With the Scaler now properly "Grounding" the actions in meters, the Mental Map will follow smooth, centimeter-scale trajectories as intended.

---
**Status**: Numerical integrity restored. Retraining initiated with Clean Statistics.
