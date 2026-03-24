# Log Output Explanations

This document breaks down the meaning of various logging outputs from the training, evaluation, and data visualization phases of the Flow Matching / Diffusion Predictive Control (FM-PCC) models.

## 1. Training Log (Diffusion and Flow Matching)

**Log Entry:**
```text
Epoch 2: 100% 1000/1000 [01:19<00:00, 12.62it/s, a0_loss=0.086, a0_loss_test=0.0749, diffusion_loss=0.906, loss=0.453, loss_test=0.469, lr=9.99e-5, step=2999]
```

**Breakdown:**
- **`Epoch 2: 100% 1000/1000`**: The model has completed 100% of the 2nd epoch (a full pass over the dataset), consisting of exactly 1000 batches.
- **`[01:19<00:00, 12.62it/s`**: 
  - **`01:19`**: Elapsed time for this epoch is 1 minute and 19 seconds.
  - **`<00:00`**: Estimated time remaining for this epoch is 0 seconds.
  - **`12.62it/s`**: Training speed is 12.62 iterations (batches) processed per second.
- **`a0_loss=0.086` & `a0_loss_test=0.0749`**: The training and testing (validation) loss specifically for predicting the first action (`a0`) in a trajectory. This is often an auxiliary loss used in planning models to ensure the immediate action to take is accurate.
- **`diffusion_loss=0.906`**: The main objective loss of the diffusion or flow-matching model (likely the Mean Squared Error of the velocity field or noise prediction).
- **`loss=0.453` & `loss_test=0.469`**: The aggregate total loss for the training and test sets, respectively. This is a weighted combination of `diffusion_loss`, `a0_loss`, and potentially other loss terms used for gradient descent.
- **`lr=9.99e-5`**: The current learning rate is $9.99 \times 10^{-5}$ (roughly $1 \times 10^{-4}$). This value can fluctuate if a learning rate scheduler is being used.
- **`step=2999`**: The global training step across all epochs.

---

## 2. Evaluation / Testing Log

**Log Entry:**
```text
------------------ Variant: dpcc-c-tightened-dt4p0 ------------------
Success rate (goal): 0.67
Success rate (goal + constraints): 0.67
Success rate (constraints): 0.67
Average steps: 83.50 +- 7.16
Average violations: 0.00 +- 0.00
Average total violations: 0.000 +- 0.000
Average time: 0.44 +- 0.01
$83.5 \pm 7.2$ & $0.67$ & $0.67$ & $0.0 \pm 0.0$ \\
... [Lists of raw data arrays]
```

**Breakdown:**
- **`Variant: dpcc-c-tightened-dt4p0`**: Identifies the specific task, constraint formulation, or planner variant being evaluated. Here, it likely means Diffusion/Flow Predictive Control with continuous and tightened constraints, running at a specific step size or horizon (`dt4p0`).
- **Success Metrics:**
  - **`Success rate (goal): 0.67`**: $67\%$ of the evaluation rollouts successfully reached the target goal state.
  - **`Success rate (constraints): 0.67`**: $67\%$ of the rollouts completed without violating any safety constraints.
  - **`Success rate (goal + constraints): 0.67`**: $67\%$ of the rollouts were perfectly successful—they reached the goal *without* violating safety constraints. In this specific log, all trajectories that reached the goal also satisfied all constraints perfectly.
- **Performance Averages:**
  - **`Average steps: 83.50 +- 7.16`**: On average, a successful rollout required $83.5$ timesteps to reach the goal, with a standard deviation of $7.16$ steps.
  - **`Average violations: 0.00 +- 0.00` & `Average total violations: 0.000`**: The average model incurs 0 collisions or constraint breaches. The planner perfectly adhered to the feasible space in the scenarios it passed.
  - **`Average time: 0.44 +- 0.01`**: The average computational time required per planning step or per trajectory is $0.44$ seconds.
- **`$83.5 \pm 7.2$ & $0.67$ & $0.67$ & $0.0 \pm 0.0$ \\`**: This line is pre-formatted as a LaTeX table row for paper writing. It lists: 
  $$ \text{Steps} \pm \text{Std} \quad \& \quad \text{Goal Success} \quad \& \quad \text{Total Success} \quad \& \quad \text{Violations} \pm \text{Std} $$
- **`[Arrays below the LaTeX line]`**: These raw nested array printouts represent the breakdown of the aggregated metrics across different evaluation seeds, tasks, or sub-environments. For instance, `[1.0, 0.833, 0.833]` shows the success rates for 3 independent test seeds. The respective arrays contain standard deviations, step counts, and timing data before being averaged into the top summary.

---

## 3. Data Constraint Visualization (`visualize_data_constraint`)

**Log Entry:**
```text
Halfspace variant: top-right-hard
Number of feasible trajectories: 1/96, 1.04%
Halfspace variant: top-left-hard
Number of feasible trajectories: 0/96, 0.00%
Halfspace variant: both-hard
Number of feasible trajectories: 2/96, 2.08%
```

**Breakdown:**
This section evaluates how many completely "safe" trajectories exist generated randomly or without guided constraint optimization over a batch of 96 samples conditioned under geometric constraints (Halfspaces).
- **`Halfspace variant: [name]`**: Specifies the localized boundary constraint applied to the environment (e.g., restricted fly zones or obstacles at the top-right / top-left).
- **`Number of feasible trajectories: X/96, Y%`**: Out of a batch of 96 sampled trajectories, only `X` trajectories (`Y%`) naturally satisfied the geometry constraint without violating it at any time step.
  - **`top-right-hard`**: Only 1 out of 96 ($1.04\%$) trajectories avoided the top-right obstacle.
  - **`top-left-hard`**: None of the 96 generated trajectories avoided the top-left constraints ($0.0\%$).
  - **`both-hard`**: Only 2 out of 96 trajectories ($2.08\%$) successfully navigated through both constraints.

**Conclusion:** This implies that generating feasible trajectory purely unguided or naively (without the tightened predictive control constraints) results in a near $100\%$ failure rate due to collisions. The planner (from section 2) is highly necessary as it raises success from roughly $\sim 1\%$ to $67\%$.
