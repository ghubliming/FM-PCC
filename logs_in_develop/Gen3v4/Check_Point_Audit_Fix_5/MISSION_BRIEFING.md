# Gen3v4 Midterm Checkpoint Audit: iMF-PCC Fix #5

**Status**: Completed in code, verification intentionally not run
**Scope**: iMeanFlow rebuild for FM-PCC training, evaluation, checkpoint compatibility, and config alignment
**Primary comparison targets**: FMv3ODE and the original iMF / trajectory-flow implementation

## Reading Order

If you only want the minimum path to understand this checkpoint, read in this order:

1. [flow_matcher_v3_imeanflow/models/imf_diffusion.py](../../../flow_matcher_v3_imeanflow/models/imf_diffusion.py)
2. [flow_matcher_v3_imeanflow/models/imf_trajectory_model.py](../../../flow_matcher_v3_imeanflow/models/imf_trajectory_model.py)
3. [flow_matcher_v3_imeanflow/models/imf_engine.py](../../../flow_matcher_v3_imeanflow/models/imf_engine.py)
4. [flow_matcher_v3_imeanflow/models/helpers.py](../../../flow_matcher_v3_imeanflow/models/helpers.py)
5. [FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py)
6. [FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py)
7. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py)

The rest of this document explains what those files mean, what changed, and how the rebuilt math differs from both FMv3ODE and the earlier iMF design.

## 1. What This Rebuild Is

This checkpoint is not a new research direction. It is a repair pass that turns the unstable iMF glue layer into a stable FMv3ODE-compatible trajectory engine.

The key result is simple:

- the live training path now behaves like FMv3ODE in its loss shape, time sampling, conditioning, and checkpoint discipline
- the iMF naming surface remains, so existing scripts and saved artifacts still make sense
- the auxiliary iMF branch is retained only as a small residual regularizer, not as a second dominant learning objective

In other words, the code still says iMF, but the learning dynamics are now FM-style and the runtime is checkpoint-safe.

## 2. Code Structure

### 2.1 `iMFTrajectoryModel`
File: [flow_matcher_v3_imeanflow/models/imf_trajectory_model.py](../../../flow_matcher_v3_imeanflow/models/imf_trajectory_model.py)

This is the smallest meaningful model unit.

What it does:
- wraps the FMv3-style `Flow_matcher_U_Net_v2`
- keeps an `aux_head` that predicts a residual from the main velocity
- exposes both `forward()` and `forward_train()` so the wrapper code can stay close to the older iMF interface

What matters structurally:
- `velocity_net` is the real model
- `aux_head` is initialized to zero at the last layer, so it starts as a near-no-op
- the model returns two outputs: the main velocity and the auxiliary residual

### 2.2 `iMeanFlowEngine`
File: [flow_matcher_v3_imeanflow/models/imf_engine.py](../../../flow_matcher_v3_imeanflow/models/imf_engine.py)

This is the compatibility shell around the trajectory model.

What it does:
- preserves an iMF-flavored API surface
- exposes `u_fn`, `forward`, and `sample`
- keeps the ODE stepping logic in one place

What matters structurally:
- this class exists to keep the code readable as iMF while delegating the actual learning to the FMv3-style U-Net
- the sampling loop builds a time grid and integrates the velocity field explicitly

### 2.3 `iMFDiffusion`
File: [flow_matcher_v3_imeanflow/models/imf_diffusion.py](../../../flow_matcher_v3_imeanflow/models/imf_diffusion.py)

This is the real training and inference wrapper used by the FM-PCC trainer.

What it does:
- builds the loss weights
- samples a time `t` from a Beta distribution
- constructs the linear interpolation path between noise and data
- computes the velocity target
- applies the conditioning logic during sampling
- handles legacy checkpoint loading and wrapper-level `state_dict()` output

What matters structurally:
- this file is where the math and the checkpoint compatibility live
- this file is also where the runtime device alignment fix happens
- if you understand this file, you understand the checkpoint

### 2.4 `apply_conditioning`
File: [flow_matcher_v3_imeanflow/models/helpers.py](../../../flow_matcher_v3_imeanflow/models/helpers.py)

This helper is the conditioning gate.

It performs hard assignment on the trajectory tensor at the conditioned time index.
That means the model does not merely “suggest” the state at that index; the code overwrites it.

That is the key reason these trajectory models behave like constrained planners instead of generic sequence predictors.

### 2.5 Train / Eval wrappers
Files:
- [FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py)
- [FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py](../../../FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py)

These scripts are the outer shell around the model.

What they do:
- resolve seeds
- instantiate dataset, model, diffusion, and trainer through `utils.Config`
- save config pickles into the checkpoint folder
- load checkpoints with legacy-compatible config handling
- evaluate on validation samples and write `eval_results.json`

## 3. The Math

### 3.1 Data representation

The rebuilt iMF code works on trajectory tensors of the form

$$x \in \mathbb{R}^{B \times H \times D}$$

where:
- $B$ is batch size
- $H$ is horizon / sequence length
- $D = d_{obs} + d_{act}$ is the per-step transition dimension

In this repo, the FMv3-style state-action trajectory is built from observed state and action channels concatenated along the last dimension.

### 3.2 Interpolation path

The current training code does not use a DDPM-style forward noising schedule. It uses a linear interpolation path between random base noise and the dataset sample:

$$x_t = (1 - t) \epsilon + t x_0$$

where:
- $x_0$ is the data trajectory
- $\epsilon \sim \mathcal{N}(0, I)$ is the base noise
- $t \in [0,1]$

This is the core FM-style idea in the rebuild.

### 3.3 Target velocity

The supervised target is the velocity field that would move the base noise toward the data sample:

$$v^\star = x_0 - \epsilon$$

The model predicts this velocity, not an epsilon residual in the DDPM sense.

### 3.4 Main loss

The code uses the weighted FM loss family from the FM-PCC stack. Conceptually, the objective is:

$$\mathcal{L}_{main} = \mathbb{E}[\|v_\theta(x_t, t, c) - v^\star\|^2_W]$$

where:
- $c$ is the conditioning information
- $W$ is the per-timestep, per-dimension weight tensor from `Losses[loss_type]`

The loss weights emphasize the first action step and allow extra weighting on selected observation channels.

### 3.5 Auxiliary residual

The rebuilt iMF design keeps a second output head, but it is intentionally weak:

$$\mathcal{L}_{aux} = \lambda_{aux} \|a_\theta\|^2$$

with `aux_head` initialized to zero at the final layer.

The total loss is:

$$\mathcal{L}_{total} = \mathcal{L}_{main} + \lambda_{aux} \mathcal{L}_{aux}$$

The practical meaning is important:
- the main field learns the actual trajectory velocity
- the auxiliary branch stays small and cannot dominate training
- this avoids the old dual-target instability pattern

### 3.6 Sampling

Sampling is explicit Euler integration over the learned velocity field.

At each step:

$$x_{k+1} = x_k + \Delta t \; v_\theta(x_k, t_k, c)$$

The exact sign convention in the code depends on the wrapper layer and the chosen time grid, but the conceptual direction is always the same: integrate the flow field from noise toward data.

### 3.7 Conditioning

Conditioning is implemented as hard overwriting of selected trajectory entries.

If the condition dictionary says that a state value must be fixed at time index `t`, then the helper writes that value directly into the trajectory tensor at that slice.

That is why the model is a constrained trajectory generator, not just a predictor.

## 4. What Changed Compared with the Original iMF

The earlier iMF design used a more visible dual-target / dual-weight story:
- a main branch
- a deviation branch
- a curriculum or schedule between them
- more surface area for instability

The rebuild keeps the module names but changes the behavior:

- the main velocity field is now the dominant signal
- the auxiliary branch is reduced to a small regularizer
- the loss shape matches FMv3ODE much more closely
- checkpoint saving/loading is wrapper-aware instead of inner-module-only
- the runtime device is aligned at wrapper initialization so CPU/CUDA mismatch no longer crashes the first loss step

So the practical difference is not just “cleaner code”. The model now learns like an FMv3ODE model with an iMF wrapper, instead of learning like a separate experimental branch.

## 5. What Changed Compared with FMv3ODE

FMv3ODE is the reference implementation for the training scaffold.

The rebuild keeps that scaffold but adds the iMF wrapper behavior:

- same general `Config -> dataset -> model -> diffusion -> trainer` structure
- same save/load discipline
- same weighted trajectory loss shape
- same conditioning semantics via `apply_conditioning`
- same ODE-style rollout concept

What the iMF rebuild adds on top of FMv3ODE:

- an auxiliary residual head
- backward-compatible checkpoint key remapping
- wrapper-level `state_dict()` output
- explicit legacy checkpoint tolerance for inner-engine saves
- iMF-specific naming and compatibility surfaces

In short: FMv3ODE is the baseline contract; iMF is now a thin extension of that contract, not a competing pipeline.

## 6. What Changed Compared with the Original D3IL Visual Aligning Work

This checkpoint is not the same thing as D3IL visual aligning, but the comparison is useful because the D3IL work shows how a visually conditioned controller is structured.

D3IL visual aligning does this:
- loads bird’s-eye and in-hand images
- encodes them with `MultiImageObsEncoder`
- feeds the embedding into a DDPM transformer policy
- runs evaluation through `Aligning_Sim(if_vision=True)`

The Gen3v4 iMF rebuild does something else:
- it is not image-conditioned
- it is not a DDPM policy
- it is a state-action flow-matching trajectory engine
- it uses FM-style interpolation and ODE integration

So the relationship is architectural, not literal:
- D3IL visual aligning teaches how a conditioning pipeline is wrapped around an environment
- the iMF rebuild teaches how the FMv3ODE trajectory backbone is made stable enough to serve as the runtime core

## 7. Checkpoint Anatomy

A successful run writes the usual FM-PCC artifacts into the seed folder:

- `dataset_config.pkl`
- `model_config.pkl`
- `diffusion_config.pkl`
- `trainer_config.pkl`
- `args.json`
- `losses.pkl`
- `state_*.pt`
- `eval_results.json` for evaluation runs

The important point is that the checkpoint is now self-describing from the wrapper outward, not just from the inner velocity network.

That is what the state-dict repair was for.

## 8. How To Use This Report

If you are reading this as a checkpoint audit, use it in this order:

1. Read the code structure sections above.
2. Open `imf_diffusion.py` and check the loss path.
3. Open `imf_trajectory_model.py` and check how the auxiliary head is attached.
4. Open `helpers.py` and verify how conditioning is applied.
5. Open the train script and confirm how config objects are built and saved.
6. Open the eval script and confirm how checkpoint resolution and validation work.

If you need to debug a run, start from the diffusion wrapper, not the trainer.

## 9. What Was Deliberately Not Done

- No verification run was executed as part of this checkpoint write-up.
- No new experimental branch was introduced.
- No D3IL source files were modified.
- No unrelated FM-PCC cleanup was attempted.

## 10. Bottom Line

This rebuild makes iMF behave like a stable FMv3ODE-compatible trajectory learner instead of a brittle two-branch experiment.

The math is now centered on:
- linear interpolation between noise and data
- velocity prediction
- weighted trajectory loss
- explicit Euler rollout
- hard conditioning by tensor overwrite

That is the right mental model for the current code.
