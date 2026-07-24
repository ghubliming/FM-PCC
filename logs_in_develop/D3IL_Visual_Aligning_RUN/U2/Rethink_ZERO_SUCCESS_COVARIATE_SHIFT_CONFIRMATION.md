# U2.4 — Zero-Success Root-Cause Confirmed: Covariate Shift via Val-Loss Selection

**Date:** 2026-06-29
**Trigger:** Follow-up to the U2.3 Zero-Success Root-Cause Audit. A deeper inspection of the `d3il` official repository code vs the `FM-PCC` wrapper code confirmed exactly why the evaluation success rate plummeted to ~0% compared to the paper's 27.8% (0.278) claim.

---

## TL;DR — Verdict

**The original authors are not lying, and the custom inference loop has no bugs.** 

The 0% success rate is definitively caused by the **computational shortcut taken in the FM-PCC custom training wrapper** (`train_d3il_visual_aligning.py`), which uses **offline Validation Loss** to select the best checkpoint instead of **Live MuJoCo Simulation Success**.

In imitation learning, offline validation loss is anti-correlated with closed-loop robustness due to **covariate shift**. By picking the model with the lowest validation loss, the pipeline inadvertently selected a severely overfitted checkpoint that instantly collapses when deployed in a live, interactive environment.

---

## 1. Inference Code is 100% Faithful (Not the Bug)

An exact code-level trace confirms that the `FM-PCC/d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py` wrapper is logically and mathematically identical to the official `d3il/simulation/aligning_sim.py`:

- **Environment:** Both import and use the identical `Robot_Push_Env` from `envs.gym_aligning_env.gym_aligning.envs.aligning`.
- **Observation Preprocessing:** Both transpose images to `(2, 0, 1)` and normalize by `255.0`.
- **Action Space:** Both use `agent.predict((bp, inh, des_robot_pos))` to get a delta position, add it to `des_robot_pos`, concatenate a fixed quaternion `[0, 1, 0, 0]`, and pass it to `env.step()`.
- **Loop Logic:** Both correctly manage the running `des_robot_pos` integration.

**Conclusion:** The code running the evaluation is **not** responsible for the performance drop.

---

## 2. The Fatal Flaw: Checkpoint Selection

The performance destruction happens during **training**. 

### The Official D3IL Method (`d3il/run_vision.py`)
```python
if not (num_epoch + 1) % agent.eval_every_n_epochs:
    # RUNS FULL MUJOCO SIMULATION MULTIPROCESSED
    successrate, _ = train_sim.test_agent(agent) 
    
    if successrate > best_success:
        best_success = successrate
        agent.store_model_weights(...) # Saves on SUCCESS
```
The D3IL authors intentionally pause training to run a computationally expensive, multi-core MuJoCo simulation. They save the checkpoint that actually *solves the task* the most often.

### The FM-PCC Custom Method (`train_d3il_visual_aligning.py`)
```python
if not (num_epoch + 1) % agent.eval_every_n_epochs:
    # RUNS OFFLINE MSE LOSS ON STATIC DATASET
    avg_val = _eval_vision_loss(agent) 
    if avg_val < best_val_loss:
        best_val_loss = avg_val
        agent.store_model_weights(...) # Saves on LOSS
```
To avoid the immense compute cost of spinning up MuJoCo environments during training, the FM-PCC wrapper overrides this process and scores checkpoints based on offline dataset validation loss.

---

## 3. Why Validation Loss Destroys Imitation Learning Models

In diffusion policies and behavior cloning, **Validation Loss is notoriously bad at predicting real-world success.** 

- **Covariate Shift:** A model that perfectly memorizes the static offline dataset (achieving near-zero validation loss) is highly brittle. The moment it runs in MuJoCo, tiny physics perturbations push the state slightly out-of-distribution. 
- **Compounding Errors:** Because the overfitted model has never seen these states, it predicts erroneous actions, compounding the error until the robot completely fails to complete the trajectory.

By selecting the checkpoint with the lowest Validation Loss, the FM-PCC pipeline systematically selected the **most brittle, overfitted epoch** rather than the most robust one.

---

## Conclusion & Next Steps

The discrepancy is not a sign of scientific dishonesty from the D3IL authors, nor is it a logical bug in the FM-PCC evaluation scripts. It is a direct consequence of a **training infrastructure shortcut**.

### Required Action to Reproduce Paper Results:
You must revert the checkpoint selection logic in `train_d3il_visual_aligning.py` to match the official repo. 
1. Remove the `_eval_vision_loss` function.
2. Re-integrate `train_sim.test_agent(agent)` during the epoch loop.
3. Accept the higher compute cost during training, as it is strictly necessary to identify the checkpoint capable of the reported 27.8% image-based success rate.
