# 🔎 Investigation Report: The Hypersonic Drift Case ($10^{10}$)

## 📅 Date: 2026-05-16
**Subject**: Numerical explosion in D3IL Visual Aligning rollouts.

---

### 🚨 Symptoms
During evaluation of the Gen5 Visual Diffusion model, the robot would exhibit "Hypersonic Drift"—smoothly accelerating to coordinates of approximately **$-2.5 \times 10^{10}$** meters within 300-400 simulation steps. This resulted in a 0.0% success rate and broken diagnostic plots.

---

### 🔍 Root Cause Discovery: "Scalar Mismatch"
The failure was not due to a bug in the simulator or the U-Net architecture, but a **Normalization Parity Failure** between the training and evaluation pipelines.

1.  **Training Mismatch**:
    - In `scripts/train.py`, the `Trainer` was being initialized with `scaler=None`.
    - **Result**: The training loop skipped all scaling logic. The model learned to predict robot motion in **Raw Meters** (e.g., inputs of `0.45m`).

2.  **Evaluation Mismatch**:
    - In `eval_ddpm_encdec_vision.py`, the script was successfully loading a `scaler.pkl` found in the results folder.
    - **Result**: The evaluator converted the robot's `0.45m` position into a **Normalized Unit** (e.g., `1.2`) before feeding it to the model.

3.  **The Explosion**:
    - The model, trained only for raw meters, received `1.2`. It interpreted this as the robot being vastly out of position.
    - It predicted a massive corrective action.
    - The evaluator then **Inverse-Scaled** this action (multiplying it by the dataset's standard deviation).
    - This created a **Positive Feedback Loop**: Error -> Huge Action -> Even Larger Error -> $10^{10}$ Drift.

4.  **The "Dual-Scaler" Ghost (Final Discovery)**:
    - **Issue**: Even after patching the local `scaler.py`, the model continued to drift.
    - **Discovery**: The `VisualDiffusionBridge` was importing `Scaler` from `d3il/agents/utils/scaler.py` instead of the local project folder.
    - **Result**: The model's internal `min_action` and `max_action` bounds were being calculated using the **Unpatched** (unsafe) logic, causing internal numerical instability during inference.

---

### 🛠️ Resolution
The fix achieved **Triple-Lock Pipeline Parity**:

1.  **Trainer Update**: Modified `scripts/train.py` to explicitly create a `Scaler` and pass it to the `Trainer`.
2.  **Import Redirection**: Pointed `VisualDiffusionBridge` to the patched `ddpm_encdec_vision.utils.scaler.Scaler`.
3.  **Zero-Variance Protection**: The "Safe Normalizer" logic is now active in Training, Model Internal Bounds, and Evaluation.

---

### 🏁 Final Verification
*   **Code Parity**: Both `scripts/train.py` and `eval_ddpm_encdec_vision.py` now share the exact same `scaler.pkl` logic.
*   **Diagnostic Visibility**: Upgraded the evaluator to a 6-panel grid to catch any future numerical drifts early.

**Status**: **RESOLVED**. Pipeline is stabilized and ready for high-fidelity benchmarking.
