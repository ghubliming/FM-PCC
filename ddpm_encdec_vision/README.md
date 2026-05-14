# Vision Pipeline: FMv3-ODE "PCC Bone" Integration

This directory contains the visual alignment pipeline, now fully refactored to match the **FM-PCC (Flow Matching)** infrastructure standard.

## 🏗️ Architecture Overview

The vision pipeline is decoupled into three modular layers, mirroring the project's state-based models:

1.  **Vision Encoder (`MultiImageObsEncoder`)**: ResNet-based encoder that converts raw images (Agentview + In-hand) into a 128D latent vector.
2.  **Backbone (`VisualUNet`)**: A temporal 1D-UNet that takes the visual latents and robot state to predict trajectory velocities.
3.  **Generative Engine (`VisualGaussianDiffusion`)**: A Flow Matching wrapper that manages the continuous-time training objective (Velocity Matching) and ODE-selectable inference.

---

## 🛠️ Critical Integration Fixes

To achieve parity with the `FMv3ODE` project, several critical "bridging" fixes were implemented. Understanding these is essential for future development:

### 1. Conditioning Dictionary Protocol
*   **Problem**: Standard FM-PCC `apply_conditioning` utility expects a simple state vector. Passing high-dimensional image tensors directly would cause shape mismatches or crashes.
*   **Fix**: `VisualGaussianDiffusion` implements a **dictionary-based conditioning pass**. 
    *   `cond[0]` is used for robot-state "snapping" (clamping the first frame).
    *   `cond['visual']` contains the raw images passed to the encoder.
*   **Why**: This allows the model to maintain physical constraints (snapping) while benefiting from high-dimensional visual features.

### 2. Normalization Bypass (`IdentityNormalizer`)
*   **Problem**: The project's unified `Policy` class automatically applies a normalizer to all observations. Visual data (pixels) is already pre-scaled to `[0, 1]`. Re-normalizing them would destroy the feature space.
*   **Fix**: During evaluation, we use an `IdentityNormalizer`. This acts as a pass-through, satisfying the `Policy` interface without modifying the pixel data.

### 3. Responsive Training Feedback
*   **Problem**: Vision training is computationally heavy. The default `log_freq` of 1,000 steps resulted in long periods of "CLI silence," making the process look hung.
*   **Fix**: Default `log_freq` is set to 100 in `train_ddpm_encdec_vision.py`. This provides a visible "heartbeat" in the console.

---

## 🚀 Running Experiments

### Training
```bash
python ddpm_encdec_vision_test/train_ddpm_encdec_vision.py \
    --seeds 6 \
    --use-wandb \
    --wandb-project fm-pcc-flow-matching
```

### Evaluation (ODE-Selectable)
```bash
python ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py \
    --seeds 6 \
    --ode_solver_method_v3 euler \
    --flow_steps_v3 10
```

---

## 📂 Directory Structure
- `models/`: Contains `visual_unet.py` and `visual_gaussian_diffusion.py`.
- `utils/`: Replicated FM-PCC training and configuration scaffolding.
- `ddpm_encdec_vision_test/`: Standardized entry points for train/eval.
