# FIX_12_P2: Runtime Stabilization & Tensor Alignment

This document records the critical runtime fixes required to bridge the Gen5 U-Net Brain into the D3IL evaluation environment. These fixes ensure that temporal and dimensional expectations between the model and the simulator are perfectly aligned.

## 1. The Symmetry Lock (Temporal Synchronization)
*   **Problem**: `RuntimeError: shape '[8, -1]' is invalid for input of size 15`
*   **Technical Cause**: The D3IL body was providing a history of **5 steps** for robot coordinates (`obs_seq_len`), while the Gen5 U-Net was configured for a **8-step** window (`window_size`).
*   **Fix**: Forced `self.obs_seq_len = window_size` in the `VisualAgentWrapper`. 
*   **Scientific Result**: All input modalities (Vision + Proprioception) now provide a symmetric 8-step history to the U-Net, allowing for correct temporal feature fusion.

## 2. The Dimension Bridge (6D to 3D Slicing)
*   **Problem**: `RuntimeError: The size of tensor a (6) must match the size of tensor b (3) at non-singleton dimension 2`
*   **Technical Cause**: 
    *   **Gen5 Output**: The Diffusion model predicts a 6D vector containing both the denoised **Robot State (3D)** and the **Action (3D)**.
    *   **Scaler Knowledge**: The Scaler was trained only on the **3D Actions**.
    *   **Conflict**: Attempting to inverse-scale the raw 6D output failed because the Scaler only had 3D statistics.
*   **Fix**: Implemented explicit slicing `action_trajectory = trajectory[:, :, 3:]` before calling the inverse scaler.
*   **Scientific Result**: We correctly isolate the action plan from the predicted state, ensuring numerical stability while respecting the model's multi-modality output.

---
**Status**: Runtime Environment Stabilized. Mathematical alignment confirmed.
