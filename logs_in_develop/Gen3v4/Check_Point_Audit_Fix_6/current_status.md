# Gen3v4 Midterm Checkpoint Audit: iMF-PCC Fix #6 (Current Code Snapshot)

**Status**: Updated to current code paths in `flow_matcher_v3_imeanflow` (document refresh)
**Verification**: Not executed in this document pass
**Scope**: iMF-PCC training/inference/checkpoint behavior and math in the current implementation
**Primary references**: FMv3ODE-style training contract + iMF compatibility wrapper

## Reading Order

1. [flow_matcher_v3_imeanflow/models/imf_diffusion.py](../../../flow_matcher_v3_imeanflow/models/imf_diffusion.py)
2. [flow_matcher_v3_imeanflow/models/imf_trajectory_model.py](../../../flow_matcher_v3_imeanflow/models/imf_trajectory_model.py)
3. [flow_matcher_v3_imeanflow/models/imf_engine.py](../../../flow_matcher_v3_imeanflow/models/imf_engine.py)
4. [flow_matcher_v3_imeanflow/models/helpers.py](../../../flow_matcher_v3_imeanflow/models/helpers.py)
5. [FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py)
6. [FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py)
7. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py)

This report now matches what those files currently do.

## 1. What The Current Rebuild Is

This is a compatibility rebuild, not a new algorithm branch.

The implemented behavior is:

- FM-style velocity supervision is the dominant training signal
- iMF naming/API surface is preserved (`iMeanFlowEngine`, dual outputs, iMF scripts)
- auxiliary branch remains but is explicitly down-weighted
- checkpoint loading is tolerant to legacy inner-engine key layouts

Practical result: training and sampling semantics are FMv3-like, with iMF-compatible wrappers.

## 2. Current Code Structure (As Implemented)

### 2.1 `iMFTrajectoryModel`
File: [flow_matcher_v3_imeanflow/models/imf_trajectory_model.py](../../../flow_matcher_v3_imeanflow/models/imf_trajectory_model.py)

Key points:

- `velocity_net = Flow_matcher_U_Net_v2(...)` is the main field predictor
- `aux_head` is a small MLP over `velocity`
- final aux layer is zero-initialized (starts near no-op)
- returns `(velocity, aux)` in both `forward` and `forward_train`

### 2.2 `iMeanFlowEngine`
File: [flow_matcher_v3_imeanflow/models/imf_engine.py](../../../flow_matcher_v3_imeanflow/models/imf_engine.py)

Key points:

- wraps `iMFTrajectoryModel` and preserves iMF-style API (`u_fn`, `sample`, `forward_train`)
- keeps explicit Euler-style integration in `sample`
- combines outputs as `u_weight * velocity + 0.1 * v_weight * aux` for engine sampling

### 2.3 `iMFDiffusion` (Training + Runtime Contract)
File: [flow_matcher_v3_imeanflow/models/imf_diffusion.py](../../../flow_matcher_v3_imeanflow/models/imf_diffusion.py)

This is the main implementation contract used by trainer/eval.

Key points:

- builds weighted loss via `Losses[loss_type](loss_weights, action_dim)`
- samples training time with Beta law (`t = 1 - Beta(alpha, beta)`)
- uses FM linear interpolation path
- predicts velocity and applies auxiliary regularization
- runs Euler rollout in `p_sample_loop`
- aligns wrapper device to backbone device at init (`self.to(model_device)`)
- supports legacy checkpoint remap in `load_state_dict`

### 2.4 `apply_conditioning`
File: [flow_matcher_v3_imeanflow/models/helpers.py](../../../flow_matcher_v3_imeanflow/models/helpers.py)

Conditioning is hard overwrite, not soft guidance:

- for each condition time `t`: `x[:, t, action_dim:] = val` (or zero in noise mode)
- if `goal_dim > 0`: goal channels are copied across the full horizon from `conditions[0]`

This is the core constrained-planning behavior.

### 2.5 Train / Eval Wrappers
Files:

- [FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py)
- [FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py)

Current behavior:

- train script builds dataset/model/diffusion/trainer via `utils.Config`
- persists config pickles and standard artifacts in save folder
- eval script can override pickled diffusion class with current config class
- eval script applies plan-time solver/step settings after checkpoint load

## 3. Fundamental Math (Mapped To Current Code)

### 3.1 State Space

Trajectories are represented as

$$x \in \mathbb{R}^{B \times H \times D}, \quad D = d_{obs} + d_{act}.$$

`observation_dim`, `action_dim`, and optional `goal_dim` define channel layout.

### 3.2 Time Sampling In Training

In `loss()`:

$$u \sim \mathrm{Beta}(\alpha, \beta), \qquad t = 1 - u.$$

With defaults in config, this is Beta$(1.5, 1.0)$ reflected by `1 - u`.

### 3.3 Flow-Matching Interpolation Path

In `q_sample(...)`:

$$x_t = (1 - t)\,x_{base} + t\,x_{start},$$

where $x_{base}$ is Gaussian base sample (then conditioned in noise mode), and $x_{start}$ is data trajectory.

### 3.4 Supervised Velocity Target

In `p_losses(...)`:

$$v^\star = x_{start} - x_{base},$$

then conditioning is re-applied in noise mode so conditioned slots are structurally consistent.

### 3.5 Predicted Fields

Model output is

$$(u_\theta, a_\theta) = \text{model}(x_t, t, c).$$

Training main loss uses $u_\theta$.

Sampling path uses mixed field:

$$\tilde{u}_\theta = u_\theta + \lambda_{sample}\,a_\theta,$$

with

$$\lambda_{sample} = 0.1\,v_{mix}, \quad v_{mix} = \frac{v\_loss\_weight}{u\_loss\_weight + v\_loss\_weight}.$$

### 3.6 Objective

Weighted FM loss:

$$\mathcal{L}_{main} = \|u_\theta - v^\star\|_W^2,$$

where $W$ is built from discounting and per-dimension weights.

Auxiliary penalty:

$$\mathcal{L}_{aux} = \|a_\theta\|_2^2.$$

Total objective in code:

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \lambda_{aux}\,\mathcal{L}_{aux},$$

with

$$\lambda_{aux} = \max(0.01,\;0.1\cdot v\_loss\_weight).$$

So the aux branch is explicitly bounded to remain secondary.

### 3.7 Sampling / ODE Discretization

In `p_sample_loop(...)`, initialization is:

$$x_0 \sim 0.5\,\mathcal{N}(0, I),$$

then hard conditioning is applied.

For each rollout step:

$$x \leftarrow x + \Delta t\,\tilde{u}_\theta(x,t,c), \qquad \Delta t = \frac{1}{N_{flow}}.$$

Conditioning is re-applied after every step, and optionally after projection adjustments near the end of rollout.

## 4. What Is Different From Older iMF Behavior

Current implementation characteristics:

- no explicit dual-target curriculum in the main loss path
- main field is directly supervised by FM velocity target
- auxiliary field is regularized to zero and only lightly mixed in sampling
- wrapper saves full state and remaps legacy inner-engine checkpoints

This narrows instability surface compared to older dual-objective iMF variants.

## 5. What Is Added Relative To Plain FMv3ODE

- dual-output model interface (`velocity`, `aux`)
- auxiliary regularizer and small sampling-time aux mix
- iMF-compatible API/entrypoints and naming
- compatibility remapping for legacy checkpoint keys

Core FM contract remains: interpolation path + weighted velocity loss + Euler integration + hard conditioning.

## 6. Config And Entry Points

In [config/avoiding-d3il.py](../../../config/avoiding-d3il.py), current iMF blocks are:

- `flow_matching_v3_imeanflow` (train)
- `plan_fm_v3_imeanflow` (plan/eval)

They point diffusion class to:

- `flow_matcher_v3_imeanflow.models.iMFDiffusion`

## 7. Checkpoint Compatibility Notes

`iMFDiffusion.load_state_dict()` remaps legacy keys where old checkpoints were saved from inner engine scope (for example `model.velocity_net.*`) into wrapper-compatible names (for example `model.model.velocity_net.*`).

This enables loading older artifacts without forcing strict key identity.

## 8. Practical Debug Order

1. Inspect `iMFDiffusion.p_losses()` for train math mismatches.
2. Inspect `helpers.apply_conditioning()` for conditioning conflicts.
3. Inspect `iMFDiffusion.p_sample_loop()` for rollout/projection behavior.
4. Inspect train/eval scripts for config load/override issues.

If behavior looks wrong, start at diffusion wrapper first.

## 9. Deliberately Not Done In This Document Pass

- no runtime verification run
- no metric claims
- no architectural expansion beyond current code

## 10. Bottom Line

Current iMF-PCC code is an FMv3-style trajectory flow matcher with an iMF compatibility wrapper.

The right mental model is:

- FM interpolation path in trajectory space
- direct velocity supervision
- weighted trajectory loss
- explicit Euler rollout
- hard conditioning overwrite
- small, regularized auxiliary residual branch
