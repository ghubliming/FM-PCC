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

### ✅ Fix 32: Global Import Synchronization
*   **File**: `ddpm_encdec_vision/models/d3il_visual_bridge.py`
*   **Issue**: The model was importing an unpatched version of the `Scaler` from a submodule, bypassing the local numerical fixes.
*   **Solution**: Redirected the import to use the local, patched `ddpm_encdec_vision.utils.scaler`.
*   **Impact**: Model-internal clamping and action bounds are now numerically stable.

---

### ✅ Fix 33: The "Tee-Stderr" Diagnostic Capture
*   **File**: `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`
*   **Issue**: Tracebacks, Python library warnings, and custom errors are printed to `sys.stderr`, which was not captured by `eval_diffuser.log`. This led to a "black box" failure mode where logs abruptly ended (e.g. at Context 7) with no explanation.
*   **Solution**: Modified the stdout interception system to also redirect and Tee `sys.stderr` to the log file.
*   **Impact**: Full tracebacks, warnings, and diagnostic information are now preserved in the log files.

### ✅ Fix 34: Deferred MuJoCo XML Teardown
*   **File**: `d3il/environments/d3il/d3il_sim/sims/mj_beta/MjLoadable.py` (Legacy `MujocoLoadable.py` explicitly left untouched)
*   **Issue**: MuJoCo threw `mju_openResource: could not open resource` warnings for robot XML assets. This occurred because `cleanup()` deleted the generated `panda_tmp_rb*.xml` file immediately after compilation, but before the offscreen renderer initialized lazily on the first camera step, resulting in missing resource references.
*   **Solution**: Modified `cleanup()` in `MjLoadable.py` to register the file removal via `atexit.register()`, preserving files during simulation execution and deleting them cleanly on Python exit.
*   **Impact**: Complete elimination of MuJoCo `mju_openResource` asset warnings and full stability of offscreen rendering on the active simulator backend.

### ✅ Fix 35: Real-Time Per-Rollout Debug Logging
*   **File**: `d3il/simulation/aligning_sim.py` & `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`
*   **Issue**: Individual rollout details (success/failure, steps, tracking errors, inference time) were not printed to stdout or saved as human-readable files during execution, making troubleshooting individual trial behaviors difficult without reloading binary `.pkl` files.
*   **Solution**: Hooked `update_rollout_info(info)` inside `aligning_sim.py` to notify the agent wrapper at rollout end. The wrapper prints a clear diagnostic summary to the console instantly and generates human-readable `rollout_{rollout_idx}_stats.txt` (stored right next to the corresponding video/GIF file) and `rollout_{rollout_idx}_stats.json` files for instant debugging.
*   **Impact**: Instant visibility into what the model did on a per-rollout basis.

### ✅ Fix 36: Double Step-Average Transparency
*   **File**: `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`
*   **Issue**: Legacy reports only printed the average number of steps for successful trials. In cases where the success rate is 0.0% (e.g. before full training completes), the reported average steps printed as `0.00`, giving the false impression that no steps were executed.
*   **Solution**: Modified the scientific report printout to separately list `Avg number of steps (successful trials)` and `Avg number of steps (all trials)`.
*   **Impact**: Accurate reporting of step count averages regardless of the success rate.

### ✅ Fix 37: Dynamic Train vs. Test Context Toggle (Seen vs. Unseen Positions)
*   **File**: `d3il/simulation/aligning_sim.py` & `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`
*   **Issue**: Legacy evaluation was hardcoded to run on unseen `test_contexts`. There was no easy way to toggle evaluation to run on seen `train_contexts` (expert dataset initial positions), which is crucial for auditing whether the model successfully memorized/fitted the training distribution prior to testing generalization.
*   **Solution**: Added a `--eval-on-train` command-line argument to `eval_ddpm_encdec_vision.py` and routed it as `eval_on_train` in both `Aligning_Sim` and `VisualAgentWrapper`. When set:
    - The simulator resolves contexts directly from the seen expert `train_contexts.pkl` instead of unseen `test_contexts.pkl`.
    - Outputs are routed to a distinct `results_train_set/` folder and saved under `_train_set` file names (e.g. `eval_diffuser_train_set.log`, `diffuser_train_set.npz`, `results_seed_X_train_set.pkl`) to guarantee seen training evaluations **never** overwrite standard unseen test results.
    - Console outputs and real-time rollout logging print explicit `Seen Training Context` labels for total clarity.
*   **Impact**: Complete, collision-free pipeline for auditing seen training conditions (memorization test) and unseen test conditions (generalization test).

---

### 🏁 FINAL SYSTEM STATUS: STABILIZED
The Gen5 Visual Pipeline is now mathematically consistent, numerically safe, and diagnostically transparent across all submodules.
