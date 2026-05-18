# Mission Briefing: Embedded Visual Diagnostics (Fix #7.1)

**Date**: 2026-05-14  
**Status**: ACTIVE  
**Component**: Visual-Aligning Evaluation Pipeline  

---

## 1. Executive Summary
During the initial testing of the re-wired Visual-Aligning model (Fix #7), we observed a **0.0% success rate** and a "blind" mean distance of **0.417**. To determine if this is caused by **model immaturity** or **simulation rendering failure** (black frames), we have integrated a permanent diagnostic protocol into the evaluation wrapper.

## 2. Technical Implementation
The `VisualAgentWrapper` in `eval_ddpm_encdec_vision.py` has been upgraded with a non-intrusive logging mechanism.

### 2.1 The Diagnostic Trigger
*   **Rollout Sampling**: To prevent disk-space bloat, diagnostics only trigger every **10th rollout** (0, 10, 20, etc.).
*   **Temporal Key-Frames**: Frames are captured at critical intervals:
    *   **Step 0**: Initial state (Did the block spawn? Is the robot visible?)
    *   **Step 50 / 100**: Mid-trajectory (Is the robot moving toward the target?)
    *   **Step 150**: Near-terminal state (Did contact occur?)

### 2.2 Data Integrity Check
The diagnostic saves a side-by-side grid containing:
1.  **Agentview Camera**: Global perspective of the table.
2.  **In-hand Camera**: Local perspective of the end-effector.

**Storage Path**:  
`logs/aligning-d3il-visual/plans/ddpm_encdec_vision/H8/<seed>/results/diagnostics/rollout_X_step_Y.png`

---

## 3. Interpretation Guide (Troubleshooting)

| Observed Image | Likely Root Cause | Action Required |
| :--- | :--- | :--- |
| **Pure Black Frames** | MuJoCo EGL/OpenGL rendering failure on the cluster. | Verify `MUJOCO_GL="egl"` and GPU visibility in SLURM. |
| **Table Visible, No Block** | `mju_openResource` error; robot/block assets failed to load. | Check `D3IL_ENV_ROOT` and asset paths in the sim. |
| **Sharp Images, No Movement** | **Model Immaturity**: Model is outputting zero/average velocity. | Continue training until Epoch 50+. |
| **Sharp Images, Erratic Movement** | **Solver/Scaling Error**: Architecture is working but uncalibrated. | Audit normalizers and action scales. |

---

## 4. Conclusion
This diagnostic protocol transforms the "Black Box" evaluation into a transparent "Film Strip" of the agent's behavior. We no longer need to guess why the success rate is 0%; we can now **see** the failure mode directly in the logs.
