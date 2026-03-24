# Log Output Explanations (Deterministic Code Definitions)

This document provides a rigorous, mathematically deterministic breakdown of the logging outputs from the training, evaluation, and data visualization phases of the Flow Matching / Diffusion Predictive Control (FM-PCC) models, strictly based on the definitions in the codebase.

## 1. Training Log (Diffusion and Flow Matching)

**Log Entry:**
```text
Epoch 2: 100% 1000/1000 [01:19<00:00, 12.62it/s, a0_loss=0.086, a0_loss_test=0.0749, diffusion_loss=0.906, loss=0.453, loss_test=0.469, lr=9.99e-5, step=2999]
```

**Exact Mathematical Breakdown:**
- **`diffusion_loss`**: The weighted base loss (e.g., MSE for `WeightedL2` or L1 for `WeightedL1`) evaluated over the entire trajectory formulation (horizon $H$ and feature dimensions $D$). Features include both states and actions depending on the formulation.
  $$\mathcal{L}_\text{diff} = \frac{1}{B \cdot H \cdot D} \sum_{b=1}^B \sum_{h=1}^H \sum_{d=1}^D w_{h,d} \cdot \mathcal{L}_\text{base}(p_{b,h,d}, t_{b,h,d})$$
  Here, $p$ is the prediction, $t$ is the target, and $w_{h,d}$ is a set of defined weighting factors (often exponentially discounted over the horizon $H$). This loss is computed inside `WeightedLoss.forward` or `WeightedStateLoss.forward` (in `flow_matcher/models/helpers.py`).

- **`a0_loss` & `a0_loss_test`**: Specifically isolates the error on the *first predicted action* of the trajectory. It divides out the pre-applied $w_{0,d}$ weight so that it represents a clear, unscaled error metric for the immediate action dimension $D_a$:
  $$\mathcal{L}_{a_0} = \frac{1}{B \cdot D_a} \sum_{b=1}^B \sum_{d=1}^{D_a} \frac{w_{0,d} \cdot \mathcal{L}_\text{base}(p_{b,0,d}, t_{b,0,d})}{w_{0,d}} = \frac{1}{B \cdot D_a} \sum_{b=1}^B \sum_{d=1}^{D_a} \mathcal{L}_\text{base}(p_{b,0,d}, t_{b,0,d})$$

- **`loss` & `loss_test`**: The total gradient objective passed to `loss.backward()`. In standard diffusion/FM without auxiliary dynamic penalties, this exactly mirrors `diffusion_loss`. If dynamic losses are active, `loss` = `diffusion_loss` + `dyn_loss`.

---

## 2. Evaluation / Testing Log (`eval_FM.py`)

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
... [Raw seed array dumps]
```

**Exact Mathematical Breakdown:**
Consider $N$ total trial trajectories, where trajectory $i$ terminates at step $T_i$.
Let $\mathcal{S}_i \in \{0, 1\}$ signify if trajectory $i$ reached the terminal goal (`info['success'] == True`).
Let $V_{i,t} \in \{0, 1\}$ signify if trajectory $i$ violated *any* geometric/dynamic constraint explicitly during step $t$.
Let $P_{i,t} \ge 0$ denote the continuous numerical magnitude of penetration into a constrained region at step $t$ (e.g., $R - \|x - c\|$ for obstacles or $c^\top x - d$ for halfspaces).

- **`Success rate (goal)`**: The empirical probability that a rollout terminates at the desired target region, irrespective of constraints.
  $$\text{Success}_\text{goal} = \frac{1}{N} \sum_{i=1}^N \mathcal{S}_i$$

- **`Success rate (constraints)`**: The empirical probability that a rollout is executed completely without a single discrete timestep violating local constraints.
  $$\text{Success}_\text{constr} = \frac{1}{N} \sum_{i=1}^N \prod_{t=1}^{T_i} (1 - V_{i,t})$$

- **`Success rate (goal + constraints)`**: The strict subset of trials that both reach the goal *and* never violate constraints.
  $$\text{Success}_\text{total} = \frac{1}{N} \sum_{i=1}^N \left( \mathcal{S}_i \cdot \prod_{t=1}^{T_i} (1 - V_{i,t}) \right)$$

- **`Average steps`**: The average sequence length $T_i$, *conditioned strictly on the subset of successful runs* ($\mathcal{S}_i = 1$). A failure trail $T_i$ value never influences this number.
  $$\mu_\text{steps} = \frac{\sum_{i=1}^N \mathcal{S}_i \cdot T_i}{\sum_{i=1}^N \mathcal{S}_i}$$

- **`Average violations`**: The average number of discrete time ticks $t$ where a violation occurs in a single rollout. It counts timesteps, not distinct collisions.
  $$\mu_\text{v-tick} = \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{T_i} V_{i,t}$$

- **`Average total violations`**: Computes the expected depth/magnitude of the violations natively across the rollout (e.g. geometric overlap sum).
  $$\mu_\text{v-pen} = \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{T_i} P_{i,t}$$

- **`Average time`**: For trajectory $i$, let $c_{i,t}$ be `time.time() - start` for evaluating policy step $t$. This averages inference time per control step.
  $$\mu_\text{time} = \frac{1}{N} \sum_{i=1}^N \left( \frac{1}{T_i} \sum_{t=1}^{T_i} c_{i,t} \right)$$

---

## 3. Data Constraint Visualization (`visualize_data_constraints.py`)

**Log Entry:**
```text
Halfspace variant: top-right-hard
Number of feasible trajectories: 1/96, 1.04%
```

**Exact Mathematical Breakdown:**
This directly processes the $K=96$ completely pre-recorded, offline demonstration sequences (where no active predictive control projection is applied).
It iterates over every timestep $t$ in offline sequences $k \in \{1, \dots, K\}$.
- **`Number of feasible trajectories`**: The absolute count of statically collected trajectories that never violate the strict geometric constraint geometries. Let $V_{k,t}$ trigger if offline state $x_{k,t}$ breaks halfspace margins:
  $$ \text{Feasible Count} = \sum_{k=1}^K \prod_{t=1}^{\text{len}_k} (1 - V_{k,t}) $$
**Conclusion**: Determines the inherent safety (or lack thereof) in raw/natural data generation distributions compared to the controlled $\text{Success}_\text{constr}$ of the planner.
