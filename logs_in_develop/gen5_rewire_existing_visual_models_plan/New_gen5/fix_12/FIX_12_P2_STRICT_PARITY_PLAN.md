# FIX_12_P2: Strict D3IL Parity & Scaling Bridge

## 1. Objective
Achieve scientific parity with the D3IL (DDPM-ACT) baseline by fixing the **Numerical Signal-to-Noise Ratio (SNR)** and the **Feedback Drift** problems. This plan stabilizes the Gen5 "Visual Transplant" by ensuring the U-Net sees the same "language" of data as the original baseline.

## 2. Core "No BS" Requirements
*   **Keep Gen5 Brain**: Maintain the U-Net backbone, Horizon (8/10), and current Hyperparameters (LR 1e-4, Batch 8).
*   **Keep MPC Logic**: Retain the multi-step trajectory prediction.
*   **Reuse D3IL Body**: Reuse the Data Normalization, Vision Encoder config, and Loop Strategy.

## 3. Technical Implementation Details

### A. The Scaling Bridge (`scaler.py`)
*   **Problem**: Raw meters (0.005m) are 200x smaller than Diffusion noise (1.0). The model is "Blind."
*   **Fix**: Implement the D3IL `Scaler`. It will calculate Mean/Std from the expert dataset and normalize all $(X,Y,Z)$ inputs/outputs to the same scale as the noise.

### B. Mixed-Loop Control (The "Mental Map")
*   **Problem**: Feeding noisy simulator coordinates (Closed-loop) causes the model to "panic" when drift occurs (OOD error).
*   **Fix**: Switch to **Open-Loop State Conditioning**. 
    *   The model will look at **Images** (Closed-Loop Vision) to find the box.
    *   The model will trust its own **Planned Positions** (Open-Loop State) for proprioception.
    *   This prevents "Jitter Feedthrough" and stabilizes the trajectory.

### C. EMA (Exponential Moving Average)
*   **Mechanism**: Maintain a shadow copy of model weights that are updated slowly.
*   **Goal**: Use EMA weights during evaluation to provide smoother, more consistent trajectories (D3IL standard).

## 4. Success Metrics
1.  **SNR Correction**: Training loss should decrease more sharply once data is scaled.
2.  **Trajectory Smoothness**: Diagnostic PNGs from FIX_12 should show a "Blue Fan" (MPC Foresight) that is perfectly aligned and noise-free.
3.  **Success Rate**: Move from "Catastrophic" (0%) to "Stable" results.

---
**Plan authorized by USER for immediate execution.**
