# Implementation Plan: Visual Evaluation Output Upgrade (Fix 9) - REVISED

## 1. Objective
Upgrade the visual evaluation pipeline (`eval_ddpm_encdec_vision.py`) to capture and export granular trajectory data that **perfectly matches the legacy FMv3ODE output schema**. This ensures the data is immediately usable by the project's standard Data Analysis Matrix and plotting scripts.

## 2. Key Components to Upgrade

### A. `VisualAgentWrapper` (In-Memory Logging)
Modify the wrapper to maintain internal buffers for the current rollout:
- **`self.history_real_pos`**: Robot end-effector position at each step (corresponds to `obs_all`).
- **`self.history_desired_actions`**: The raw actions predicted by the model (corresponds to `act_all`).
- **`self.history_full_plans`**: Store the full 8-step future plan generated at each re-planning tick (corresponds to `sampled_trajectories_all`).

### B. `eval_ddpm_encdec_vision.py` (Strict Legacy Export)
- Instead of creating new `.pkl` files, we will inject the following keys directly into the existing `{variant}.npz` archive:
    - `obs_all`: Formatted as `np.array(..., dtype=object)` containing list of positions.
    - `act_all`: Formatted as `np.array(..., dtype=object)` containing list of actions.
    - `sampled_trajectories_all`: Formatted as `np.array(..., dtype=object)` containing predicted plans.

## 3. Comparison of Output Schemas

| Feature | Legacy FMv3ODE (`.npz`) | **New Gen5 Upgrade (`.npz`)** |
| :--- | :--- | :--- |
| **Success Logic** | `n_success` | `success_rate` (maintained) |
| **Path Data** | `obs_all` | **`obs_all` (Added)** |
| **Action Data** | `act_all` | **`act_all` (Added)** |
| **Planning Data** | N/A | **`sampled_trajectories_all` (Added)** |

## 4. Implementation Steps

### Step 1: Internal Buffer Setup
Update `VisualAgentWrapper.__init__` and `reset()` to collect and flush data into a master dictionary.

### Step 2: Step-by-Step Data Capture
Update `VisualAgentWrapper.predict()` to log the robot's real position, the selected action, and the full U-Net plan at every relevant tick.

### Step 3: Injection into .npz
Update the main loop in `eval_ddpm_encdec_vision.py` to extract the master history, format it into legacy-compatible object arrays, and save it directly into the `{variant}.npz` file.

## 5. Success Criteria
- [ ] Evaluation generates a `{variant}.npz` file containing `obs_all` and `act_all`.
- [ ] The `obs_all` array contains the step-by-step (x,y,z) coordinates of the robot.
- [ ] The `sampled_trajectories_all` array contains the model's desired plan.
- [ ] Existing analysis scripts can read these files without modification.

---

**Revised plan generated for FM-PCC Diagnostic Phase 9.**
