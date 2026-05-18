# FIX_12: Phase 2 - Runtime Stabilization & Scientific Parity

## 1. The "Asymmetric Temporal" Solution
**Problem**: The Gen5 U-Net (Temporal Brain) expects Symmetric inputs (e.g., 16 images + 16 robot positions). However, the training config used `obs_seq_len: 1`. This caused a shape mismatch in the `encode_visual` flattening logic (`B * T`), leading to "Total Wrong" predictions or runtime crashes.

**The Fix**:
- **Temporal Auto-Sync**: Updated `VisualUNet.py` to handle `T_state != T_images`. If the state history is shorter than the image window, the latest state is **automatically repeated** to fill the temporal tensor.
- **Architectural Parity**: Updated `eval_ddpm_encdec_vision.py` to strictly respect the model's `obs_seq_len` from its saved `diffusion_config.pkl`, preventing "information leakage" during testing.

## 2. The "Rollout 0" Recovery
**Problem**: `realtime_diagnostics` folders were empty because the `rollout_counter` initialized at `-1` and the reset logic was failing to capture the very first attempt (Seed 6, Trial 0).

**The Fix**:
- **Counter Shift**: Shifted the increment logic in `VisualAgentWrapper.reset()`. The counter now correctly transitions from `-1` to `0` at the start of the evaluation, and `_save_diagnostics` is triggered at the end of every valid rollout.
- **Persistent Persistence**: Ensured `master_rollout_history` is only flushed if `step_counter > 0`, preventing ghost/empty files from cluttering the logs.

## 3. Scientific Parity (FMv3ODE Standard)
**Problem**: Evaluation reports were in an old format, making side-by-side comparison with the thesis ODE benchmarks difficult.

**The Fix**:
- **Replicated 7-Metric Report**: Standardized the output header and metrics:
  - `------------------------Running [Task] - [Variant] ([Seed])----------------------------`
  - Success Rate / Constraints / Steps / Violations / Time.
- **Tracking Error**: Added the **Max MPC Tracking Error** calculation to the report to validate "Mental Map" fidelity.

## 4. Final Numerical Grounding
- **Verified**: The user has cleared the disk and **retrained**.
- **Status**: The `scaler.pkl` is now clean (generated via `get_all_actions()` masked fix).
- **Result**: "Speed of Light" errors ($10^{10}m/step$) are eliminated. The robot is now physically "grounded" in the workspace.

---
**Status**: STABILIZED
**Action**: Proceed to full evaluation suite (Seed 6-10).
