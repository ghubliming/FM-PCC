# 🪵 Visual Evaluation Fix Log (Gen5 Stabilization)

## 🚨 Status: STABILIZED (2026-05-16)
**Primary Objective**: Resolve numerical instability and diagnostic failures in the D3IL visual-aligning pipeline.

---

### ✅ Fix 25: The "Anti-Drift" Normalizer Lock
*   **File**: `ddpm_encdec_vision/datasets/normalization.py`
*   **Issue**: Hypersonic Drift ($10^{10}$). Caused by `GaussianNormalizer` dividing by zero standard deviation for constant dimensions (e.g., Z-height or gripper orientation).
*   **Solution**: Implemented a **Zero-Variance Safety Check**. Standard deviations smaller than `1e-4` are now forced to `1.0`. 
*   **Impact**: Robot coordinates remain grounded; division-by-zero errors are eliminated.

### ✅ Fix 26: 6-Panel Scientific Diagnostic Suite
*   **File**: `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`
*   **Issue**: Evaluation plots were incomplete/blank and didn't show the relationship between MPC Foresight and Real Paths.
*   **Solution**: Fully populated a 6-panel grid:
    1.  **X-Position** (Time-series)
    2.  **Y-Position** (Time-series)
    3.  **Z-Position** (Height stability check)
    4.  **Step Magnitude** (Drift velocity check)
    5.  **XY Trajectory** (Top-down view)
    6.  **MPC Foresight** (Blue "Ghost" plans vs Black Real path)
*   **Impact**: Immediate visual feedback for model maturity and drift detection.

### ✅ Fix 27: The "Symmetry Lock" (16-Step Window)
*   **File**: `ddpm_encdec_vision/models/visual_unet.py` & `eval_ddpm_encdec_vision.py`
*   **Issue**: Training/Eval mismatch when `obs_seq_len != window_size`.
*   **Solution**: 
    - Updated `VisualUNet` to auto-repeat state observations if sequence lengths differ.
    - Updated `VisualAgentWrapper` to respect trained `obs_seq_len` dynamically.
    - Configured standard at **`obs_seq_len: 16`** for perfect hand-eye sync.
*   **Impact**: Model now sees the full context of robot motion matching the visual window.

### ✅ Fix 28: First-Rollout Recovery (Context 0)
*   **File**: `eval_ddpm_encdec_vision.py`
*   **Issue**: `rollout -1` error caused Context 0 data to be lost and diagnostic exports to crash.
*   **Solution**: Shifted `rollout_counter` logic. Added safety checks to ensure `master_rollout_history` only exports when data actually exists.
*   **Impact**: Full scientific report generated from trial #1.

---

### ⚠️ IMPORTANT: RETRAIN REQUIRED
Since the **Scaler Logic** changed in `normalization.py`, all previous models and `scaler.pkl` files are obsolete. 
*   **Action**: Delete `scaler.pkl` and start a fresh training run with the "Safe Normalizer."

---
**Lead AI Assistant**: Antigravity
**Final State**: Ready for Benchmark Seed 6-10.
