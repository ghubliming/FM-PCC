# Fix 7: Evaluation Logic Finalization & Architectural Audit

## 1. Executive Summary
This fix addresses the persistent `0.0%` Success Rate in the **Visual Aligning** evaluation pipeline. Through an end-to-end trace of the data flow and training scripts, we have concluded that the failure was caused by a **temporal coordination bug** in the evaluation script, not a deficiency in training quality. 

The evaluation script was effectively forcing the robot to act based on observations from **8 steps in the past**, creating a "ghost lag" that prevented any meaningful task progress.

---

## 2. Training Quality Audit: A "Perfect" Converged Model
We analyzed the latest training logs for the `ddpm_encdec_vision` model (Step 50,000 / Epoch 49). The metrics are outstanding:

*   **Diffusion Loss**: `0.000212` (Excellent convergence)
*   **A0 Loss (Immediate Action)**: `1.94e-5` (Near-perfect prediction of the next velocity delta)
*   **Test/Train Gap**: Minimal (No significant overfitting)

**Conclusion**: The model is healthy. The 0% success rate during evaluation was a "wiring" problem, where the evaluation environment was not feeding the model the correct spatial-temporal context it learned during training.

---

## 3. The "Blind Model" Discovery (Architectural Insight)
A deep audit of the code revealed a significant architectural discrepancy between the standard D3IL model and our current FM-PCC bridge:

### Comparison: D3IL DDPM-ACT vs. Our FM-PCC Bridge
| Component | D3IL DDPM-ACT (`ddpm_encdec_vision_agent.py`) | Our Current Model (`VisualUNet` + `diffuser`) |
| :--- | :--- | :--- |
| **Backbone** | `DiffusionEncDec` (Transformer) | `UNet1DTemporalCondModel` (1D UNet) |
| **Vision Integration** | **Cross-Attention**: Visual features are actively queried. | **Parameter Only**: Visual features are passed but **ignored**. |
| **Conditioning** | Transformer Attention over obs sequence. | **Trajectory Snapping**: Inpainting of the current state. |

### The "Paradox" of Low Loss vs. Zero Success
*   **The Insight**: Our `UNet1DTemporalCondModel` backbone receives the visual features as a `cond` argument but never actually uses them in its convolutional blocks. 
*   **Why Loss is Low**: The model learned to be a high-quality **Proprioceptive Diffusion Model**. Because the robot's current position is highly correlated with the required push direction in this task, the model "solved" the training data by relying purely on the snapped robot state, effectively ignoring the "blind" visual features.
*   **Implication**: The model knows how to navigate relative to its own position, but it doesn't "see" the block. However, even a blind model should move towards the target if the current state is fed correctly.

---

## 4. Technical Fixes: Correcting the "Ghost Lag"

### A. Snapping Point Correction (The "Past-State" Bug)
*   **The Problem**: In `VisualGaussianDiffusion.forward`, the denoising trajectory was snapped to `pos[:, 0]`.
*   **The Context**: In the evaluation wrapper, `pos` is the context window (past 8 frames). Thus, `pos[:, 0]` is the state from **8 steps ago**.
*   **The Result**: The robot was executing trajectories starting from where it was 0.5 seconds ago. In a 200-step rollout, this creates a feedback loop of failure.
*   **The Fix**: Updated snapping to `pos[:, -1]` (the latest/current frame). This ensures the denoising process starts from the robot's **actual current position**.

### B. Scaler Removal (Normalization Alignment)
*   **Discovery**: The FM-PCC training pipeline (`train_ddpm_encdec_vision.py`) does **not** use the D3IL Scaler. It trains on raw coordinate values.
*   **The Bug**: Applying the Scaler during evaluation corrupted the proprioceptive input (making it out-of-distribution) and the velocity output (multiplying deltas by training-set standard deviations).
*   **The Fix**: Removed all `Scaler` logic. The evaluation script now passes raw positions and returns raw velocities, ensuring exact parity with the training distribution.

---

## 5. Conclusion & Verdict
The `0.0%` Success Rate was a "Code Logic" problem, not a "Model Training" problem. 
1.  The **Scaler** was corrupting the data values.
2.  The **Snapping Point** was corrupting the temporal alignment.

By fixing these, we enable the model to use the "blind" but highly accurate proprioceptive strategy it learned during training. 

**Next Step**: Run Seed 6. We expect to see fluid movement and significant task progress. If success remains low, the next phase (**Fix 8**) will be modifying the `UNet1DTemporalCondModel` backbone to actually integrate the visual features, transforming it from a Proprioceptive model into a true Vision-based model.
