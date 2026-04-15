# 02 Implementation Plan: Gen4U2 True Visual Policy

Date: 2026-04-09
Status: Abandoned (Archive, Do Not Execute)
Notice:
1. Gen4U2 implementation is abandoned.
2. Keep this content as historical record; do not delete.
3. Active path is U5/Gen5: rewire existing visual models first, then extend to Avoiding visual if no fundamental flaw, without rebuilding wheels.

Superseding docs:
1. logs_in_develop/gen5_rewire_existing_visual_models_plan/01_gen5_reset_abandon_gen4u2.md
2. logs_in_develop/gen5_rewire_existing_visual_models_plan/02_gen5_rewire_to_existing_visual_models_plan.md

---

## Archived Gen4U2 Plan (Retained)

### 1) Planning Rule

This plan follows one hard rule:
1. do not rebuild wheels,
2. first extend existing D3IL visual logic into Avoiding planner path,
3. only add new modules if a concrete gap remains after reuse.

### 2) Correction of Last Gen4 Mistake

Explicit correction from prior Gen4 direction:
1. previous direction leaned toward creating new visual stack first,
2. that is replaced with a bridge-first approach using existing D3IL visual dataset and agent contracts,
3. visual mode must not silently run state-only when strict visual mode is requested.

Concrete implication:
1. the avoiding-d3il-visual state-only alias path is compatibility-only,
2. Gen4U2 true-visual profile must route to image-backed data and fail if image assets are missing.

### 3) Reuse Targets (What We Already Have)

#### 3.1 Existing D3IL visual data and env components

Source components to reuse:
1. d3il/environments/dataset/avoiding_dataset.py
2. d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py
3. d3il/simulation/avoiding_sim.py

#### 3.2 Existing D3IL visual agent contract

Reference pattern to reuse:
1. state = (bp_image, inhand_image, des_robot_pos)
2. preprocessing and temporal context handling from vision agents.

### 4) File-Level Implementation Map (Updated)

#### 4.1 FM dataset bridge to D3IL visual contract

Target files:
1. flow_matcher_v3_avoiding_visual/datasets/d4rl.py
2. flow_matcher_v3_avoiding_visual/datasets/sequence.py
3. flow_matcher_v3_avoiding_visual/datasets/__init__.py

Actions:
1. add a true-visual dataset mode that reuses Avoiding image folder layout and ordering logic from D3IL,
2. preserve current state-only mode for baseline compatibility,
3. return conditions with stable keys,
4. add strict visual guard for missing image assets.

#### 4.2 FM policy interface extension

Target file:
1. flow_matcher_v3_avoiding_visual/sampling/policies.py

Actions:
1. extend condition formatting to pass image keys,
2. keep state-only call path unchanged,
3. support batch repeat for both state and visual keys.

#### 4.3 FM model conditioning integration

Target files:
1. flow_matcher_v3_avoiding_visual/models/unet1d_temporal_cond.py
2. flow_matcher_v3_avoiding_visual/models/diffusion.py
3. flow_matcher_v3_avoiding_visual/models/helpers.py

Actions:
1. integrate visual conditioning using existing D3IL visual input convention,
2. start with lightweight fusion,
3. avoid large architecture rewrite.

#### 4.4 Train/eval wiring to true visual runtime

Target files:
1. FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py
2. FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py
3. config/avoiding-d3il-visual.py
4. config/projection_eval_visual.yaml

Actions:
1. add explicit mode keys,
2. true_visual eval path uses if_vision=True,
3. state_only path remains available,
4. separate outputs by mode label.

### 5) Phase Plan

1. Phase A: Reuse bridge first.
2. Phase B: Model consumes visual conditions.
3. Phase C: True visual eval rollout.
4. Phase D: Claim validation by ablations.

### 6) Mandatory Test Matrix

1. strict visual asset test,
2. compatibility test,
3. conditioning sensitivity test,
4. runtime wiring test,
5. baseline regression test.

### 7) Risk Controls

1. model ignores images,
2. image-state sequence misalignment,
3. confusion between compatibility and true-visual modes.

### 8) Definition of Done

1. image-backed dataset active in true_visual mode,
2. model forward consumes visual condition tensors,
3. eval true_visual mode uses camera observations,
4. strict mode blocks silent fallback,
5. ablations prove image dependence,
6. state-only baseline remains runnable.
