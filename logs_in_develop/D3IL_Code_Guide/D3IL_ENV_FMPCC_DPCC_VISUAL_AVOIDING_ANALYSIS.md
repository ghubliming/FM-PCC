# D3IL Avoiding Environment Deep Dive

## Goal of this note

This note explains how the D3IL avoiding environment works, how FM-PCC and DPCC currently connect to it, and whether the current visual avoiding setup is likely to work as intended.

Scope:
1. Environment mechanics in D3IL
2. Dataset and observation/action contract
3. FM-PCC and DPCC integration paths
4. Justification and critique of the current visual avoiding direction

## 1) How the D3IL avoiding environment works

Primary environment implementation:
- [d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py)

Obstacle geometry:
- [d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py)

### 1.1 State, action, and sensing

The environment models obstacle avoiding in the robot Cartesian plane.

Key details:
- The core robot state returned to planners is 2D end-effector position (x, y) when vision is off.
- With vision enabled, observation is a tuple: robot position + overhead cage camera + in-hand camera.
- This behavior is toggled by if_vision in [avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py#L60) and [avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py#L121).

Action semantics used by DPCC/FM evaluation loops:
- Planner predicts delta position in 2D.
- Runtime converts to absolute desired position before stepping the env.
- This is visible in both FM and DPCC eval scripts where next desired position is computed then sent to env step.

### 1.2 Obstacles and success/failure logic

Obstacle field is fixed and layered across y-levels:
- One obstacle in level 1
- Two in level 2
- Three in level 3
- A finish line above level 3

See construction in [avoiding_objects.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py).

Termination logic:
- Failure if rod collides with any obstacle body: [avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py#L216)
- Success if current y exceeds goal line: [avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py#L232)
- Early termination on either condition: [avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py#L252)

### 1.3 Mode encoding

The env computes a 9-dimensional one-hot-ish mode encoding based on where the trajectory passes each obstacle row.
- Mode vector allocation: [avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py#L117)
- Updated in check_mode: [avoiding.py](../../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py#L185)

This mode encoding is used mainly for trajectory diversity/entropy analysis in simulation evaluation flows.

## 2) Datasets and representation contract

Dataset loaders:
- State-only: Avoiding_Dataset
- Vision-enabled: Avoiding_Img_Dataset

File:
- [d3il/environments/dataset/avoiding_dataset.py](../../../d3il/environments/dataset/avoiding_dataset.py)

### 2.1 State-only dataset

State-only sequence uses:
- observations = [x_des, y_des, x, y]
- actions = delta(x_des, y_des)

See extraction from env logs:
- des_c_pos and c_pos parsing in [avoiding_dataset.py](../../../d3il/environments/dataset/avoiding_dataset.py)

### 2.2 Vision dataset

Vision dataset extends state-only with image sequences:
- images root expected at data/images/
- overhead camera at data/images/bp-cam/
- in-hand camera at data/images/inhand-cam/

See image path loading and sequence assembly:
- [avoiding_dataset.py](../../../d3il/environments/dataset/avoiding_dataset.py#L160)
- [avoiding_dataset.py](../../../d3il/environments/dataset/avoiding_dataset.py#L207)

### 2.3 Config-level behavior

State config:
- [d3il/configs/avoiding_config.yaml](../../../d3il/configs/avoiding_config.yaml)
- Uses Avoiding_Dataset and window_size 1.

Vision config:
- [d3il/configs/avoiding_vision_config.yaml](../../../d3il/configs/avoiding_vision_config.yaml)
- Uses Avoiding_Img_Dataset, window_size 8, and if_vision true in simulation.

## 3) How FM-PCC and DPCC currently hook into D3IL

## 3.1 DPCC hook path

Main DPCC avoiding eval:
- [dpcc/scripts/eval.py](../../../../dpcc/scripts/eval.py)

What it does:
- Instantiates ObstacleAvoidanceEnv directly.
- Builds observation as concat(action[:2], obs) in state-space.
- Uses projection constraints from [dpcc/config/projection_eval.yaml](../../../../dpcc/config/projection_eval.yaml).

Dataset ingestion in DPCC:
- [dpcc/diffuser/datasets/d4rl.py](../../../../dpcc/diffuser/datasets/d4rl.py#L136)
- For avoiding, it reads only des_c_pos and c_pos from pickled logs.

Conclusion:
- DPCC is currently a state-action planner on D3IL avoiding, not a pixel planner.

## 3.2 FM-PCC visual-avoiding hook path

Current FM visual eval script:
- [FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py](../../../FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py)

Config and experiment wiring:
- [config/avoiding-d3il-visual.py](../../../config/avoiding-d3il-visual.py)
- [config/projection_eval_visual.yaml](../../../config/projection_eval_visual.yaml)

What the visual eval still does in practice:
- Creates ObstacleAvoidanceEnv without if_vision true.
- Uses state concatenation pattern obs = concat(action[:2], obs).
- Applies the same DPCC-like geometric constraints and projection logic.

Dataset ingestion in FM diffuser path:
- [diffuser/datasets/d4rl.py](../../../diffuser/datasets/d4rl.py#L136)
- Also reads des_c_pos and c_pos only.

Conclusion:
- FM visual naming/config exists, but planner input path is still largely state-based in current eval/planning loop.

## 4) Will new visual avoiding work? Justification and critique

Short answer:
- It can work as a functional avoiding controller.
- It is not yet a fully vision-grounded DPCC/FM planning stack end-to-end.

### 4.1 Why it can work

1. Same physical dynamics contract
- Both DPCC and FM rely on a stable state/action contract already validated on avoiding-d3il.

2. Same constraint projection machinery
- Halfspace/obstacle/bounds/dynamics constraints are already integrated and working in both codepaths.

3. Shared observation structure
- The state vector [x_des, y_des, x, y] and delta action model are consistent between dataset generation and eval execution loops.

### 4.2 Why it may fail to deliver true visual gains yet

1. Vision not in planner loop
- Current FM visual eval still uses state concat path and does not consume camera tensors as planner conditions.

2. Dataset alias mismatch risk
- Diffuser dataset loaders currently branch on avoiding-d3il aliases, and visual experiment ids may require alias extension to avoid runtime errors.

3. Distribution shift hidden by naming
- If training/eval still consume state-only logs, labeling as visual can overstate capability while not improving robustness to visual nuisances.

4. Constraint proxy may mask perception issues
- Strong projection can keep safety metrics high even when perception modeling is weak or absent.

## 5) Practical verdict

Verdict:
- New visual avoiding will likely run and produce valid obstacle-avoiding behavior.
- It will likely behave as state-driven DPCC/FM with visual infrastructure partially present, unless planner conditioning is explicitly switched to image-aware inputs.

Confidence: medium.
- High confidence for execution viability.
- Medium confidence for true visual generalization benefit.

## 6) What must be true to claim genuine visual avoiding

Minimum technical bar:
1. Planner policy/model receives image features (bp-cam and/or inhand-cam) in the core planning call, not only in separate imitation modules.
2. Training data pipeline for planner includes image-aligned trajectories, not only des/c_pos state logs.
3. Eval scripts set if_vision true where intended and pass vision observations through policy interfaces.
4. Ablation shows visual-conditioned policy improves under visual perturbations/occlusion/domain shift over state-only baseline.

Without these, current setup is best described as:
- visual-ready infrastructure + state-driven planning.

## 7) Cross-check list for your repo

Use this list before claiming full visual avoiding support:

- [ ] D3IL env called with if_vision true in target eval/training loop
- [ ] Policy forward pass consumes image inputs
- [ ] Diffuser/FM dataset loader for planner includes image channels or precomputed vision embeddings
- [ ] Projection metrics remain good when state shortcuts are removed
- [ ] Results report both success/collision and visual robustness stress tests

---

Author note:
This document intentionally separates runtime success from scientific claim strength. The current stack can be practically effective while still needing deeper rewiring to justify strong visual-policy claims.
