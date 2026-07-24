# Training Checkpoint Selection: A Cross-Repository Comparison

**Date:** 2026-06-29
**Objective:** Compare how different repositories in the lab's ecosystem treat model evaluation and checkpoint selection during the training phase. Specifically, whether they use offline metrics (like Validation Loss) or closed-loop evaluation (Real Simulation Success).

As established in the Zero-Success Root-Cause Audit, using **offline validation loss** instead of **real simulation success** causes severe covariate shift in imitation learning models, leading to brittle checkpoints. This document surveys how each project currently handles this critical step.

---

## 1. D3IL (`/workspaces/d3il`)
**Method: Real Simulation Success (The Gold Standard)**
- **How it works:** Periodically during training, the script pauses and spawns a multiprocessing pool of live MuJoCo environments (`train_sim.test_agent(agent)`). It runs full, closed-loop rollouts.
- **Selection Criterion:** It records the actual success rate of the agent in the simulator and overwrites `eval_best_ddpm.pth` **only** if the new success rate exceeds the previous best.
- **Verdict:** Highly robust to covariate shift. It guarantees the selected checkpoint actually works in the environment, not just on the training dataset.

## 2. DPCC (`/workspaces/dpcc`)
**Method: Offline Validation Loss**
- **How it works:** DPCC splits the offline trajectory dataset into `train` and `test` dataloaders. During training (`diffuser/utils/training.py`), it periodically evaluates the diffusion denoising loss (MSE) on the static `test_dataloader`.
- **Selection Criterion:** It saves `state_best.pt` whenever the offline `test_loss` reaches a new minimum. It **never** runs the simulation during training.
- **Verdict:** Vulnerable to covariate shift. The "best" model is simply the one that memorized the offline dataset the best, which does not guarantee robustness when rolled out in MuJoCo.

## 3. HardFlow (`/workspaces/HardFlow`)
**Method: Fixed Interval Saving (No active selection)**
- **How it works:** The training script (`run/train.py`) iterates through the dataset for a predefined number of steps. It does not have a validation dataloader, nor does it spin up any MuJoCo environments.
- **Selection Criterion:** It blindly saves checkpoints every `cfg.save_freq` steps (e.g., `model_0.pth`, `model_1.pth`). 
- **Verdict:** Leaves the burden of selection to the user. The user must manually run inference scripts post-training on all saved checkpoints to figure out which one actually works.

## 4. SafeFlowMPC (`/workspaces/SafeFlowMPC`)
**Method: Final Epoch Only (No active selection)**
- **How it works:** The imitation learning scripts (`train_imitation_learning.py`) run a simple `for` loop over the training dataset for a fixed number of iterations (e.g., `50001`). There is no validation set and no simulation testing.
- **Selection Criterion:** It saves exactly **one** checkpoint at the very end of the training script (`checkpoints/model_unsafe_vpsto.pth`).
- **Verdict:** Extremely risky. If the model overfits or collapses at epoch 40,000, the user is stuck with a broken final model at epoch 50,000 with no prior checkpoints to fall back on.

## 5. FM-PCC D3IL Baseline Wrapper (`/workspaces/FM-PCC/d3il_visual_aligning_baseline_test`)
**Method: Offline Validation Loss (The Deviation)**
- **How it works:** The custom wrapper explicitly overrides D3IL's native simulation evaluation to save compute time. It calculates `_eval_vision_loss(agent)` on a static dataset.
- **Selection Criterion:** Saves `eval_best_ddpm.pth` based on the lowest validation loss.
- **Verdict:** This deviation from D3IL's original method caused the 0% success rate issue. It selected an overfitted epoch that collapsed under covariate shift in the live environment.

---

## Final Ranking & Commentary (Best to Worst)

### 🥇 #1. D3IL (Real Simulation Success)
**Commentary:** This is the absolute gold standard for imitation learning and robotics. Because imitation learning suffers heavily from compounding errors (covariate shift), the only true measure of a policy's robustness is rolling it out in the simulator. By paying the compute cost upfront to test in MuJoCo, D3IL guarantees that the final `eval_best_ddpm.pth` checkpoint will actually perform well in deployment. 

### 🥈 #2. HardFlow (Fixed Interval Saving)
**Commentary:** While it lacks an automated selection mechanism, it is fundamentally safe. By saving a dense history of checkpoints (`model_0.pth`, `model_100.pth`, etc.), it never throws away a good model. The user has to do a bit of manual work post-training to run evaluations across the checkpoints, but they are guaranteed to find the true best epoch.

### 🥉 #3. DPCC & FM-PCC Wrapper (Offline Validation Loss)
**Commentary:** This approach is actively harmful in generative trajectory modeling. By using offline Validation Loss (MSE on a static dataset) to overwrite the "best" model, the pipeline actively selects for the most overfitted checkpoint. A model with the lowest validation loss has often merely memorized the training data and will instantly collapse when faced with slightly out-of-distribution states in the live simulator. This method throws away robust checkpoints in favor of brittle ones.

### 🚩 #4. SafeFlowMPC (Final Epoch Only)
**Commentary:** The worst and most dangerous approach. Saving exactly one checkpoint at the very last iteration leaves a single point of failure. If the model overfits, diverges, or collapses midway through the 50,000 iterations, the final model is completely useless, and the entire training run's compute is wasted because no prior history was preserved.

---

### Summary Table

| Rank | Repository | Evaluation Method | Selection Criterion | Robustness to Covariate Shift |
| :--- | :--- | :--- | :--- | :--- |
| **#1** | **D3IL** | Live MuJoCo Rollouts | Highest Success Rate | **High** (Tested in-distribution) |
| **#2** | **HardFlow** | None during training | Fixed Intervals (All saved) | **N/A** (Requires post-hoc testing) |
| **#3** | **DPCC / FM-PCC** | Static Test Dataloader | Lowest Validation Loss | **Low** (Actively selects for overfitting) |
| **#4** | **SafeFlowMPC** | None during training | Last Epoch Only | **Very Low** (Single point of failure) |
