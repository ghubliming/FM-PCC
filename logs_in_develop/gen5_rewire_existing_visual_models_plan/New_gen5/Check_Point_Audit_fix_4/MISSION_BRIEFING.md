# Gen5 Midterm Checkpoint Audit: Visual Aligning Rewire, Fix #4

**Status**: Completed in code, verification intentionally not run
**Scope**: visual aligning integration inside FM-PCC, including the bridge layer, modular vision backbone, FMv3ODE-style diffusion wrapper, and train/eval config parity
**Primary comparison targets**: FMv3ODE and original D3IL visual aligning

## Reading Order

If you only want the shortest path to understand the whole visual checkpoint, read in this order:

1. [config/aligning-d3il-visual.py](../../../config/aligning-d3il-visual.py)
2. [ddpm_encdec_vision_test/train_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/train_ddpm_encdec_vision.py)
3. [ddpm_encdec_vision/models/visual_unet.py](../../../ddpm_encdec_vision/models/visual_unet.py)
4. [ddpm_encdec_vision/models/visual_gaussian_diffusion.py](../../../ddpm_encdec_vision/models/visual_gaussian_diffusion.py)
5. [ddpm_encdec_vision/models/d3il_visual_bridge.py](../../../ddpm_encdec_vision/models/d3il_visual_bridge.py)
6. [d3il/environments/dataset/aligning_dataset.py](../../../d3il/environments/dataset/aligning_dataset.py)
7. [d3il/simulation/aligning_sim.py](../../../d3il/simulation/aligning_sim.py)
8. [d3il/agents/ddpm_encdec_vision_agent.py](../../../d3il/agents/ddpm_encdec_vision_agent.py)
9. [d3il/agents/models/diffusion/diffusion_policy.py](../../../d3il/agents/models/diffusion/diffusion_policy.py)
10. [d3il/agents/models/diffusion/diffusion_models.py](../../../d3il/agents/models/diffusion/diffusion_models.py)
11. [d3il/agents/models/vision/multi_image_obs_encoder.py](../../../d3il/agents/models/vision/multi_image_obs_encoder.py)
12. [ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py)

The report below explains the actual code, the math it implements, the comparison against FMv3ODE, and the delta from the original D3IL visual-aligning stack.

## 1. What This Checkpoint Is

This checkpoint is the visual-aligning branch of the FM-PCC rewrite.

The important idea is:
- the original D3IL visual pipeline already knew how to solve the aligning task with images
- FMv3ODE already knew how to train and sample trajectories with a flow-matching ODE backbone
- this checkpoint combines the two ideas inside FM-PCC

The result is a visual controller stack that is no longer a D3IL island and no longer a pure FMv3ODE state-only model.
It is a visual-conditioned FM-PCC experiment with ODE-style trajectory sampling.

## 2. Code Structure

### 2.1 The task config
File: [config/aligning-d3il-visual.py](../../../config/aligning-d3il-visual.py)

This file defines the experiment blocks.

The key pieces are:
- `ddpm_encdec_vision` for training
- `plan_ddpm_encdec_vision` for evaluation / planning
- `train_data_path` and `eval_data_path` pointing at the aligning dataset
- `obs_dim = 3`, `action_dim = 3`, `window_size = 8`
- `obs_seq_len = 5`, `action_seq_size = 4`

This config is important because it shows the intended surface of the experiment:
- image-conditioned aligning
- 3D robot pose / action channels
- FM-PCC experiment naming and checkpoint layout

### 2.2 The active training path
File: [ddpm_encdec_vision_test/train_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/train_ddpm_encdec_vision.py)

This is the outer training entrypoint.

What it does:
- resolves seeds and W&B settings
- builds dataset, model, diffusion, and trainer via `utils.Config`
- writes `dataset_config.pkl`, `model_config.pkl`, `diffusion_config.pkl`, and `trainer_config.pkl`
- loads `Aligning_Img_Dataset`
- instantiates `VisualUNet`
- instantiates `VisualGaussianDiffusion`
- trains through the FM-PCC trainer

This is the strongest evidence that the current implementation is not a one-off script anymore. It is using the project’s normal config-driven training scaffold.

### 2.3 The modular vision backbone
File: [ddpm_encdec_vision/models/visual_unet.py](../../../ddpm_encdec_vision/models/visual_unet.py)

This is the active model wrapper used by the training script.

What it does:
- creates a `MultiImageObsEncoder` for the two aligning cameras
- encodes bird’s-eye and in-hand images into a 128-dimensional latent
- sends that latent into the FMv3-style temporal backbone
- keeps the robot pose channels as the low-dimensional state part of the transition tensor

So the backbone is really two pieces:
- image encoder
- temporal trajectory model

That separation is the main architectural improvement over the original D3IL monolith.

### 2.4 The visual diffusion wrapper
File: [ddpm_encdec_vision/models/visual_gaussian_diffusion.py](../../../ddpm_encdec_vision/models/visual_gaussian_diffusion.py)

This file subclasses the FMv3ODE `GaussianDiffusion` implementation.

What it does:
- converts a visual batch `(bp_imgs, inhand_imgs, obs, act, mask)` into a trajectory tensor `x`
- builds a conditioning dictionary containing the visual tuple and the first robot state
- reuses the FMv3ODE loss and sampling machinery

This is the key bridge from D3IL-style data to FMv3ODE-style sampling.

### 2.5 The legacy bridge layer
File: [ddpm_encdec_vision/models/d3il_visual_bridge.py](../../../ddpm_encdec_vision/models/d3il_visual_bridge.py)

This file exists as a compatibility-oriented bridge layer.

What it does:
- instantiates the D3IL `MultiImageObsEncoder`
- wraps a D3IL-style transformer diffusion core
- derives action bounds from the aligning dataset when possible
- exposes `encode_visual()`, `loss()`, and `predict()` methods

Why it matters:
- it preserves a D3IL-native integration surface
- it documents the original visual-aligning coupling in a compact place
- it shows how the old DDPM visual path was expected to look

### 2.6 The original D3IL visual aligning stack
Files:
- [d3il/environments/dataset/aligning_dataset.py](../../../d3il/environments/dataset/aligning_dataset.py)
- [d3il/simulation/aligning_sim.py](../../../d3il/simulation/aligning_sim.py)
- [d3il/agents/ddpm_encdec_vision_agent.py](../../../d3il/agents/ddpm_encdec_vision_agent.py)
- [d3il/agents/models/diffusion/diffusion_policy.py](../../../d3il/agents/models/diffusion/diffusion_policy.py)
- [d3il/agents/models/diffusion/diffusion_models.py](../../../d3il/agents/models/diffusion/diffusion_models.py)
- [d3il/agents/models/vision/multi_image_obs_encoder.py](../../../d3il/agents/models/vision/multi_image_obs_encoder.py)

These are the baseline implementation files for the original D3IL visual-aligning task.

They show the original stack:
- `Aligning_Img_Dataset` produces image batches and low-dimensional robot state/action sequences
- `Aligning_Sim(if_vision=True)` returns image observations during rollout
- `DiffusionPolicy` encodes the images and routes them into a DDPM policy
- `DiffusionEncDec` is the original transformer-based generative core
- `MultiImageObsEncoder` is the shared two-camera image encoder

### 2.7 The evaluation wrapper
File: [ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py)

This is the rollout side.

What it does:
- loads the trained checkpoint and config pickles
- wraps the diffusion model in a policy interface
- runs `Aligning_Sim(if_vision=True)`
- writes results under the experiment tree

That means the evaluation side still uses the real D3IL environment and task metric, not a synthetic proxy.

## 3. The Math

### 3.1 Original D3IL visual-aligning math

The D3IL visual pipeline is a DDPM policy.

The core idea is the standard diffusion forward process:

$$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$$

where:
- $x_0$ is the clean action or action-state target
- $\epsilon \sim \mathcal{N}(0, I)$
- the model learns to predict the noise or denoising residual

The original visual-aligning code uses:
- image encoding to a latent embedding
- a transformer diffusion model
- reverse denoising to sample actions

That is different from FM-style flow matching.

### 3.2 FMv3ODE math

FMv3ODE uses a flow-matching ODE view, not a DDPM reverse chain.

The rebuilt FMv3ODE-style core uses a linear interpolation path:

$$x_t = (1 - t)\epsilon + t x_0$$

and the target velocity:

$$v^\star = x_0 - \epsilon$$

Sampling integrates the learned velocity field with an ODE step:

$$x_{k+1} = x_k + \Delta t\, v_\theta(x_k, t_k, c)$$

### 3.3 What the visual checkpoint does mathematically

The visual checkpoint keeps the FMv3ODE math but changes the conditioning variable.

Instead of conditioning on only low-dimensional state, it conditions on visual embeddings derived from two cameras:
- bird’s-eye image
- in-hand image

The visual encoder maps images to a latent feature:

$$z_{vis} = f_{enc}(I_{bp}, I_{hand}, s)$$

where $s$ is the optional low-dimensional robot state.

That latent is then fed into the temporal backbone as the conditioning signal.

So the actual learning target is still flow velocity over trajectories, but now the velocity field depends on image-derived context.

### 3.4 What the loss means here

The current visual diffusion wrapper uses the FMv3ODE loss shape.

In words:
- create a noisy base trajectory
- interpolate between noise and data
- predict the velocity that moves the noisy state toward the data trajectory
- keep a small auxiliary residual, if present, from dominating the objective

The important conceptual difference from D3IL is that this is not epsilon prediction under a reverse DDPM chain.
It is velocity prediction under an ODE-style trajectory flow.

## 4. What Changed Compared with the Original D3IL Visual Aligning Stack

This is the most important comparison for a midterm checkpoint.

### 4.1 What stayed the same

The following pieces are still conceptually the same as the original D3IL stack:
- task: aligning
- sensors: bird’s-eye and in-hand cameras
- dataset shape: image sequences plus robot state and action sequences
- simulator: `Aligning_Sim(if_vision=True)`
- visual encoder family: `MultiImageObsEncoder`

### 4.2 What changed

The following pieces changed materially:
- the training scaffolding moved into FM-PCC `Config` / `Trainer` style
- the generative core is no longer a plain D3IL DDPM path
- the checkpoint format now follows FM-PCC conventions
- the experiment is organized around `ddpm_encdec_vision` and `plan_ddpm_encdec_vision`
- the model is designed to be swappable into FMv3ODE-style flow matching
- the config and wrapper layers are now visible, testable code instead of a single monolithic policy

### 4.3 What was added

New code added by the visual checkpoint:
- [ddpm_encdec_vision/models/visual_unet.py](../../../ddpm_encdec_vision/models/visual_unet.py)
- [ddpm_encdec_vision/models/visual_gaussian_diffusion.py](../../../ddpm_encdec_vision/models/visual_gaussian_diffusion.py)
- [ddpm_encdec_vision/models/d3il_visual_bridge.py](../../../ddpm_encdec_vision/models/d3il_visual_bridge.py)
- [config/aligning-d3il-visual.py](../../../config/aligning-d3il-visual.py)
- [ddpm_encdec_vision_test/train_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/train_ddpm_encdec_vision.py)
- [ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py](../../../ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py)

That is the actual delta that converts the old D3IL visual idea into a checkpointable FM-PCC experiment.

## 5. What Changed Compared with FMv3ODE

FMv3ODE is the scaffold reference. The visual checkpoint keeps the scaffold but changes the conditioning input.

### 5.1 Same parts as FMv3ODE
- config-driven experiment definition
- dataset/config/model/diffusion/trainer split
- weighted trajectory loss structure
- ODE-style rollout concept
- checkpoint serialization through pickle files and `state_*.pt`

### 5.2 Different parts from FMv3ODE
- FMv3ODE is state-only; this checkpoint is image-conditioned
- FMv3ODE uses low-dimensional observation channels; this checkpoint uses learned visual embeddings
- FMv3ODE rollout is on abstract trajectory states; this checkpoint wires a real visual environment into the loop
- FMv3ODE planning output is based on state conditioning; this checkpoint must preserve image semantics through the encoder

### 5.3 The practical meaning

The visual checkpoint answers a specific question:

Can the FMv3ODE trajectory backbone still work when the conditioning context comes from images instead of only from state vectors?

This checkpoint is the code path for that question.

## 6. The Two Integration Surfaces in This Tree

There are two integration surfaces in the repository, and it is useful to know both.

### Surface A: Legacy bridge
The `VisualDiffusionBridge` path is the closest thing to the old D3IL visual stack.

It is useful for understanding how the vision pipeline was originally wrapped.

### Surface B: Modular FM-PCC visual path
The `VisualUNet` + `VisualGaussianDiffusion` path is the more FM-PCC-shaped version.

It is the one to read first if your goal is to understand how the current train script actually works.

If you are debugging the present checkpoint, follow Surface B first, then use Surface A as a compatibility reference.

## 7. Checkpoint and Experiment Layout

The relevant experiment names in the config are:
- `ddpm_encdec_vision`
- `plan_ddpm_encdec_vision`

The training script writes the standard FM-PCC config pickles into the seed folder, so the run is reconstructible from its checkpoint directory.

The evaluation script resolves the same experiment tree, loads the saved config objects, and runs the aligning sim from there.

That matters because the visual checkpoint is not just a script; it is a reproducible experiment layout.

## 8. How To Use This Report

If you are trying to understand the codebase from this checkpoint, use this route:

1. Open the config file and understand the experiment names.
2. Open the training script and check how the config objects are built.
3. Open `VisualUNet` and understand the visual encoder plus backbone split.
4. Open `VisualGaussianDiffusion` and check how the batch becomes a trajectory tensor.
5. Open `Aligning_Img_Dataset` and verify the 5-tuple batch structure.
6. Open `Aligning_Sim` and verify how the rollout receives actions and images.
7. Open the original D3IL agent files to compare the old DDPM visual path.

If you are trying to debug rollout behavior, start from `eval_ddpm_encdec_vision.py`, not from the simulator.

## 9. What Was Deliberately Not Done

- No verification run was executed as part of this audit write-up
- No D3IL original file was modified
- No unrelated FM-PCC task was touched
- No additional visual task was added beyond aligning

## 10. Bottom Line

This checkpoint turns visual aligning into a real FM-PCC experiment.

The original D3IL code gave the task, the images, and the working simulator.
FMv3ODE gave the trajectory-flow math and training discipline.
This checkpoint combines both:
- D3IL for the visual task semantics
- FMv3ODE for the generative/ODE scaffold
- FM-PCC for experiment management and checkpoint reproducibility

If you understand this document, you understand the core of the visual rewrite.
