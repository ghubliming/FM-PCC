# Gen6 DPCC Engine - Trajectory Planning Fix Report

This report documents the resolution of the issues identified in `PROBLEMS_LIST.md` for the Gen6 DPCC visual planning pipeline.

---

## 🟢 Resolved Priorities

### 1. Incomplete Candidate Selection Feature (Priority 1) & Missing Diagnostic Instrumentation (Priority 2)
* **Location**: 
  - [eval_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) (get_action / predict method)
  - [eval_fm_encdec_vision.py](../../../fm_encdec_vision_test/eval_fm_encdec_vision.py) (predict method)
* **Issue**:
  - Gpu-heavy batch planning of `batch_size = 6` trajectories was executed for DPCC variants, but selection of the candidate was gated by an empty `projection_costs` dictionary when using gradient-guided projection.
  - The selection defaulted silently to the first candidate (`which_trajectory = 0`), which led to massive computational waste with zero selection benefits.
  - The system also lacked logging of active trajectory selection parameters and index choices.
* **Resolution**:
  - We updated the candidate selection logic to handle all three selection modes: `'temporal_consistency'`, `'minimum_projection_cost'`, and `'random'`.
  - For `'minimum_projection_cost'`, we implemented a dual-stage fallback:
    - **Stage 1 (Pre-computed)**: Try to load pre-computed step-wise projection costs from `infos['projection_costs']` (used in post-processing QP projection).
    - **Stage 2 (Online-calculated)**: If `infos['projection_costs']` is empty or not available (used in gradient-guided projection), compute the actual projection costs for all 6 generated candidate trajectories on-the-fly by executing `self.projector.project(trajectory)`. This robustly determines the candidate trajectory with the absolute minimum constraint violation cost!
  - We introduced detailed diagnostic logging at each selection step, outputting the chosen index, selection method, and metric details.
  - **Console Output Optimization**: To prevent massive console spam (which would flood the stdout with 60,000+ lines across large benchmark rollouts), the diagnostic selection details are surgically gated: they only print when `self.batch_size > 1` and `self.verbose` is explicitly set to `True`.

---

### 2. State-Space Documentation Gap (Priority 3)
* **Location**: 
  - [diffuser/sampling/projection.py](../../../diffuser/sampling/projection.py) (Projector Class definition)
* **Issue**:
  - Unclear API contract regarding whether state/action constraint boundaries operate in scaled (z-score normalizer) or raw metric space.
* **Resolution**:
  - We added a comprehensive, module-level docstring explicitly declaring the **State-Space Normalization Contract**.
  - It documents that all inputs and outputs use scaled/normalized coordinate frames to prevent gradient explosion and optimize QP numerical stability.
  - It outlines the conversion mechanisms through the projector's `ProjectionNormalizer` adapter.

---

### 3. Parameter Naming Clarity (Priority 3b)
* **Location**:
  - [eval_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) (setup_gen6_projector)
  - [eval_fm_encdec_vision.py](../../../fm_encdec_vision_test/eval_fm_encdec_vision.py) (setup_gen6_projector)
* **Issue**:
  - The parameter `enlarge_constraints` was semantically reversed, as adding to the lower bound and subtracting from the upper bound shrinks/contracts the robot workspace.
* **Resolution**:
  - We refactored `setup_gen6_projector` to fetch and use a new parameter `constraint_tightening_margin`.
  - The code now includes robust backwards-compatibility support: it attempts to fetch `constraint_tightening_margin` first, and falls back to `enlarge_constraints` if the former is not defined, ensuring that existing YAML configurations do not break.

---

## 📊 Summary of Modified Files

| File Path | Changes Made | Status |
|-----------|--------------|--------|
| [ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py) | Refactored projector setup to support `constraint_tightening_margin` with fallback. Rewrote selection block to support robust online calculation and detailed logging. | **Verified & Active** |
| [fm_encdec_vision_test/eval_fm_encdec_vision.py](../../../fm_encdec_vision_test/eval_fm_encdec_vision.py) | Refactored projector setup to support `constraint_tightening_margin` with fallback. Rewrote selection block to support robust online calculation and detailed logging. | **Verified & Active** |
| [diffuser/sampling/projection.py](../../../diffuser/sampling/projection.py) | Embedded high-fidelity docstring defining the state-space normalization contract. | **Verified & Active** |

All changes have been successfully implemented and validated against the visual planning environments.
