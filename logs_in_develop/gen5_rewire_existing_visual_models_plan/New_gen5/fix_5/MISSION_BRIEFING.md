# Mission Briefing: Gen5 Vision Pipeline (Fix #5) - Reverting to Strict DDPM Baseline

**Status**: Completed
**Target**: Restore the generative core to the original D3IL DDPM baseline while preserving the FM-PCC infrastructure ("PCC Bone").

---

## 1. Problem Statement
During the initial port of the vision pipeline, the generative engine was prematurely upgraded from **DDPM (Noise Prediction)** to **FMv3 (Flow Matching/Velocity Prediction)**. While this aligned the math with the state-based models, it deviated from the primary objective: **Step 1 is to perfectly replicate the existing D3IL visual aligning DDPM baseline.** 

## 2. Technical Objectives
- **Restore DDPM Objective**: The model must predict noise (`predict_epsilon=True`) rather than velocity.
- **Restore Stochastic Sampling**: Inference must use the standard DDPM denoising loop instead of deterministic ODE solvers.
- **Preserve the "Bone"**: The training scaffolding (W&B logging, `Trainer`, `utils.Config`) must remain identical to the FMv3ODE standard for consistency.

## 3. Implementation Details

### A. Engine Reversion (`VisualGaussianDiffusion`)
- **Inheritance Swapped**: Changed the base class from the FMv3 engine back to `diffuser.models.diffusion.GaussianDiffusion`.
- **Loss Function**: Restored `p_losses` mapping to calculate standard MSE on predicted noise instead of vector fields.
- **Forward Pass**: Ensure `forward()` triggers the stochastic `conditional_sample` -> `p_sample_loop` required by the `Policy` class.

### B. Backbone Reversion (`VisualUNet`)
- **Temporal UNet Swapped**: Replaced `Flow_matcher_U_Net_v2` with `UNet1DTemporalCondModel`.
- **Reasoning**: The DDPM UNet is designed for discrete integer timesteps (0–1000) and standard timestep embeddings, whereas the FMv3 UNet expected continuous float time.

### C. Training Defaults (`train_ddpm_encdec_vision.py`)
- Removed FMv3 specific parameters (`flow_steps_v3`, `time_beta_alpha_v3`).
- Hardcoded `predict_epsilon=True`.
- Set default `n_diffusion_steps` to 100 to match standard visual DDPM baselines.

### D. Vision Tuple Safety
- Improved the `forward` method in `VisualGaussianDiffusion` to safely unpack the `(bp_imgs, inhand_imgs, state)` tuple passed by the simulator, ensuring that batching and `einops.repeat` operations in the `Policy` class do not crash during evaluation.

## 4. Verification Plan
1. **Training Curve**: The `diffusion_loss` should now reflect standard DDPM epsilon-matching magnitude rather than velocity-matching.
2. **Evaluation**: Running `eval_ddpm_encdec_vision.py` will automatically utilize the stochastic `p_sample_loop` without requiring ODE solver arguments.

---
**Auditor Signature**: Antigravity AI
**Date**: May 14, 2026
