# SafeFlowMPC vs FMPCC/DPCC: Action Weighting Mechanism Comparison

Date: 2026-04-04

## 1. Research Question

Does the SafeFlowMPC codebase implement an idea similar to FMPCC/DPCC action-priority weighting (for example, using action weight on the first control step), and can the same action weight analysis be transferred directly?

## 2. Executive Conclusion

Short answer: partially similar at the flow-matching objective level, but not similar in the specific action-weighting mechanism.

- Similar:
  - Both systems train a flow-matching velocity field using supervised regression of a velocity target.
- Different:
  - FMPCC/DPCC uses an explicit loss-weight matrix with first-step action override.
  - SafeFlowMPC does not implement this matrix-based first-action emphasis in the same way.

Therefore, the thesis discussion about action_weight calibration in FMPCC/DPCC cannot be transferred to SafeFlowMPC one-to-one.

## 3. Verified Code Evidence

### 3.1 FMPCC/DPCC mechanism (explicit weighted loss matrix)

In FMPCC/DPCC-style code:

- Config defines action weight values:
  - FM-PCC/config/avoiding-d3il.py
- Loss weights are constructed in GaussianDiffusion.get_loss_weights:
  - FM-PCC/flow_matcher/models/diffusion.py
- First action cell block is explicitly overridden:

$$
W[0, 0:action\_dim] = action\_weight
$$

- Weighted loss and unweighted first-action metric are computed in helper loss:
  - FM-PCC/flow_matcher/models/helpers.py

Mathematically:

$$
\mathcal{L}_{\text{DPCC/FM-PCC}} = \frac{1}{BHD} \sum_{b=1}^{B}\sum_{h=0}^{H-1}\sum_{d=0}^{D-1} W_{h,d}\, e_{b,h,d}^2
$$

with

$$
e_{b,h,d} = \hat{y}_{b,h,d} - y_{b,h,d}
$$

and special first-action emphasis:

$$
W_{0,d} = action\_weight \quad \text{for } d \in \{0,\dots,action\_dim-1\}
$$

The reported first-action metric divides out the weight (raw error view):

$$
a0\_loss \propto \mathbb{E}\left[\frac{e_{:,0,0:action\_dim}^2}{W_{0,0:action\_dim}}\right]
$$

### 3.2 SafeFlowMPC mechanism (flow-matching MSE, optional alternative scaling)

In SafeFlowMPC pretraining:

- Training script:
  - SafeFlowMPC/train_imitation_learning.py
- Core FM sampling pattern:

$$
x_0 \sim \mathcal{N}(0, I),\quad x_1 \sim p_{\text{data}},\quad x_t = \text{path}(t; x_0, x_1)
$$

- Velocity regression target (from path object) is trained by plain MSE:

$$
\mathcal{L}_{\text{safe-pretrain}} = \mathbb{E}\left[\|\hat{v}(x_t,t,c)-v_t\|_2^2\right]
$$

In SafeFlowMPC safe finetuning:

- Script:
  - SafeFlowMPC/train_imitation_learning_safe.py
- There is an optional use_weights flag (default false) and a horizon-dependent vector repeated across dimensions.
- If enabled, code uses:

$$
\mathcal{L}_{\text{safe-ft,weighted}} = \mathbb{E}\left[\|\hat{v} - (w \odot v_t)\|_2^2\right]
$$

This is not the same as multiplying squared error by a matrix:

$$
\mathbb{E}[\,W \odot (\hat{v}-v_t)^2\,]
$$

So SafeFlowMPC weighting (when enabled) is a different design from DPCC-style action loss weighting.

## 4. Why This Difference Matters

FMPCC/DPCC action_weight affects gradient allocation directly at the loss-element level:

$$
\nabla_\theta \mathcal{L} = \frac{1}{BHD}\sum_{b,h,d} W_{h,d}\,2e_{b,h,d}\,\nabla_\theta e_{b,h,d}
$$

A large first-step action weight forces optimization pressure toward those cells.

SafeFlowMPC default pretraining has no analogous first-step action override, and its optional safe finetuning weighting does not replicate the same gradient structure. Hence, expected behavior under weight sweeps can differ.

## 5. Transferability of the Action-Weight Thesis Discussion

### 5.1 What transfers

- High-level FM principle: model predicts velocity field between noise and data.
- The warning that loss scaling changes optimization priorities.

### 5.2 What does not transfer directly

- Exact interpretation of action_weight equal to 10 in first-action cells.
- Direct comparison of weighted-loss curve magnitudes across systems.
- Assumption that SafeFlowMPC has the same a0-centric metric pipeline.

## 6. Practical Recommendation for Your Experiments

If the goal is fair FM-vs-DDPM comparison inside FMPCC/DPCC, keep identical action_weight settings between DDPM and FM for baseline fairness, then run FM-specific ablations separately.

Suggested FM ablation set inside FMPCC/DPCC:

- action_weight = 1
- action_weight = 5
- action_weight = 10
- action_weight = 20

Primary comparison metrics:

$$
\text{Collision rate},\; \text{Success rate},\; \text{Projection cost},\; \text{unweighted } a0\_loss
$$

Do not treat weighted total loss as a cross-setting quality metric when weight matrices differ.

## 7. Final Answer to the Original Question

Is SafeFlowMPC similar in idea?

- Yes, in the broad sense of using flow matching and velocity-field regression.
- No, in the specific DPCC action_weight mechanism with first-step action loss-matrix emphasis.

So the action_weight-10 calibration argument is a direct issue for FMPCC/DPCC, not a direct mirror of SafeFlowMPC implementation.

## 8. Source Files Reviewed

- FM-PCC/config/avoiding-d3il.py
- FM-PCC/flow_matcher/models/diffusion.py
- FM-PCC/flow_matcher/models/helpers.py
- FM-PCC/diffuser/models/diffusion.py
- FM-PCC/diffuser/models/helpers.py
- SafeFlowMPC/train_imitation_learning.py
- SafeFlowMPC/train_imitation_learning_safe.py
- SafeFlowMPC/README.md

## 9. Direct Clarification (Your Question)

Question: "So no extended action-weighted adjustment for initial actions like in FMPCC?"

Answer: Correct. SafeFlowMPC does not implement the same first-action weighting extension used in FMPCC/DPCC.

- FMPCC/DPCC style:

$$
\mathcal{L} = \mathbb{E}\left[W \odot (\hat{y}-y)^2\right],\quad
W[0,0:action\_dim]=action\_weight
$$

- SafeFlowMPC default pretrain:

$$
\mathcal{L}_{\text{safe-pretrain}} = \mathbb{E}\left[\|\hat{v}-v_t\|_2^2\right]
$$

- SafeFlowMPC safe finetune optional branch (off by default):

$$
\mathcal{L}_{\text{safe-ft,weighted}} = \mathbb{E}\left[\|\hat{v}-(w\odot v_t)\|_2^2\right]
$$

This optional branch is not equivalent to FMPCC's first-action loss-matrix emphasis.
