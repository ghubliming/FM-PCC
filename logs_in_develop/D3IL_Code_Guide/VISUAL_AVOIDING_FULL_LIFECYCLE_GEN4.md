# Gen4 Visual Avoiding Full Lifecycle (D3IL + FM-PCC + DPCC)

## Purpose

This document is the full lifecycle blueprint for how Gen4 visual avoiding is supposed to work in this repo, combining:
1. D3IL environment and dataset mechanics,
2. Gen4 implementation plan and execution records,
3. FM-PCC runtime path,
4. DPCC compatibility path,
5. acceptance criteria for claiming true visual capability.

Source docs merged in this lifecycle note:
- [D3IL_ENV_FMPCC_DPCC_VISUAL_AVOIDING_ANALYSIS.md](D3IL_ENV_FMPCC_DPCC_VISUAL_AVOIDING_ANALYSIS.md)
- [01_current_status_gen4_visual_camera_avoiding_d3il.md](../gen4_visual_camera_avoiding_d3il_plan/01_current_status_gen4_visual_camera_avoiding_d3il.md)
- [02_implementation_plan_gen4_visual_camera_avoiding_d3il.md](../gen4_visual_camera_avoiding_d3il_plan/02_implementation_plan_gen4_visual_camera_avoiding_d3il.md)
- [03_gen4_coding_execution_record.md](../gen4_visual_camera_avoiding_d3il_plan/03_gen4_coding_execution_record.md)
- [04_expected_results_after_gen4_visual_avoiding_upgrade_theory.md](../gen4_visual_camera_avoiding_d3il_plan/04_expected_results_after_gen4_visual_avoiding_upgrade_theory.md)
- [DPCC_VISUAL_AVOIDING_MIGRATION_GUIDE.md](../gen4_visual_camera_avoiding_d3il_plan/DPCC_VISUAL_AVOIDING_MIGRATION_GUIDE.md)

## Lifecycle Overview

The visual lifecycle has 8 stages:
1. Architecture lock and isolation strategy
2. Environment and data contract alignment (D3IL)
3. Code-path isolation (Gen4 copy-modify)
4. Config binding integrity (train/eval/load)
5. Training path (FM-PCC visual branch)
6. Evaluation and aggregation path
7. DPCC compatibility branch
8. Claim gate: does this count as true visual avoiding?

## Stage 1: Architecture Lock and Isolation Strategy

Locked strategy from Gen4 planning:
1. Keep baseline path intact.
2. Add a parallel Gen4 visual path.
3. Vendor D3IL into FM-PCC to reduce setup friction.

Isolation policy:
- Old path remains default-safe.
- New path gets unique experiment keys and prefixes.
- No destructive overwrite of baseline scripts or logs.

Relevant records:
- [01_current_status_gen4_visual_camera_avoiding_d3il.md](../gen4_visual_camera_avoiding_d3il_plan/01_current_status_gen4_visual_camera_avoiding_d3il.md)
- [02_implementation_plan_gen4_visual_camera_avoiding_d3il.md](../gen4_visual_camera_avoiding_d3il_plan/02_implementation_plan_gen4_visual_camera_avoiding_d3il.md)

## Stage 2: D3IL Environment and Data Contract

Core environment implementation:
- [d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py](../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py)
- [d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py](../../d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/objects/avoiding_objects.py)

Data contract used by both FM-PCC and DPCC diffusion planners:
- observation: [x_des, y_des, x, y]
- action: delta position in 2D

Where this is encoded:
- [d3il/environments/dataset/avoiding_dataset.py](../../d3il/environments/dataset/avoiding_dataset.py)
- [diffuser/datasets/d4rl.py](../../diffuser/datasets/d4rl.py)
- [../../../dpcc/diffuser/datasets/d4rl.py](../../../dpcc/diffuser/datasets/d4rl.py)

Visual extension in D3IL:
- `if_vision=True` returns robot position + bp_cam image + inhand_cam image.
- Vision dataset class loads aligned frame sequences from images folders.

Config toggles:
- State config: [d3il/configs/avoiding_config.yaml](../../d3il/configs/avoiding_config.yaml)
- Vision config: [d3il/configs/avoiding_vision_config.yaml](../../d3il/configs/avoiding_vision_config.yaml)

## Camera Capture and Storage Policy

Short answer:
1. Yes, camera captures can be saved.
2. They are not automatically guaranteed in every FM-PCC or DPCC eval path today.

How capture is wired in D3IL:
1. Avoiding env creates camera loggers for bp-cam in the env setup.
2. Vision mode also exposes in-hand camera frames in observation when `if_vision=True`.
3. Vision dataset loader expects persisted images under:
- `.../data/images/bp-cam/...`
- `.../data/images/inhand-cam/...`

What this means for lifecycle behavior:
1. D3IL vision data collection paths can persist camera images when run through the vision-enabled simulation/data pipeline.
2. FM-PCC visual eval script currently behaves mostly state-driven and does not, by itself, guarantee image dump artifacts unless the underlying run path includes logger-backed data recording.
3. DPCC baseline eval path is state-style and also does not, by default, produce image dataset folders.

Operational rule for this project:
1. If the goal is training/evaluating true visual policies, treat image persistence as required artifacts, not optional side effects.
2. Before visual training, verify both folders exist and are populated:
- `d3il/environments/dataset/data/avoiding/data/images/bp-cam`
- `d3il/environments/dataset/data/avoiding/data/images/inhand-cam`
3. If these are missing, regenerate data via vision-enabled D3IL pipeline before claiming visual lifecycle completeness.

Claim-safety note:
1. If camera folders are empty while running "visual" experiments, results should be labeled as state-dominant or visual-infrastructure-only.

## Stage 3: Gen4 Copy-Modify Isolation

Executed structural split (from Gen4 execution log):
1. Vendored D3IL under FM-PCC.
2. Copied FM_v3 test path into visual avoiding test path.
3. Copied flow matcher v3 into visual avoiding engine path.
4. Added visual config files with isolated experiment keys.

Key files:
- [03_gen4_coding_execution_record.md](../gen4_visual_camera_avoiding_d3il_plan/03_gen4_coding_execution_record.md)
- [config/avoiding-d3il-visual.py](../../config/avoiding-d3il-visual.py)
- [config/projection_eval_visual.yaml](../../config/projection_eval_visual.yaml)
- [FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py](../../FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py)
- [FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py](../../FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py)
- [FM_v3_avoiding_visual_test/load_results_FM_v3_avoiding_visual.py](../../FM_v3_avoiding_visual_test/load_results_FM_v3_avoiding_visual.py)

## Stage 4: Config Binding Integrity

For lifecycle correctness, these bindings must be stable and matched:
1. Python config module for visual train/eval dispatch:
- [config/avoiding-d3il-visual.py](../../config/avoiding-d3il-visual.py)

2. Projection eval yaml for visual constraints/eval seeds:
- [config/projection_eval_visual.yaml](../../config/projection_eval_visual.yaml)

3. Visual experiment IDs and plan keys:
- `avoiding-d3il-visual`
- `flow_matching_v3_avoiding_visual`
- `plan_fm_v3_avoiding_visual`

If any key drifts, training/eval/load mismatch occurs and metrics become non-comparable.

## Stage 5: Training Lifecycle (FM-PCC Visual Branch)

Training script:
- [FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py](../../FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py)

Training lifecycle steps:
1. Parse visual config and experiment key.
2. Load dataset through FM diffuser dataset path.
3. Build model and diffusion/flow components from visual config block.
4. Run seed loop with reproducibility controls.
5. Save checkpoints/loss artifacts in isolated visual prefix.

Expected first-round behavior:
1. Startup should be stable if bindings are correct.
2. Early loss can be noisier than pure state baseline.
3. No immediate large improvement is required in smoke stage.

Reference expectation:
- [04_expected_results_after_gen4_visual_avoiding_upgrade_theory.md](../gen4_visual_camera_avoiding_d3il_plan/04_expected_results_after_gen4_visual_avoiding_upgrade_theory.md)

## Stage 6: Evaluation Lifecycle

Evaluation script:
- [FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py](../../FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py)

Aggregation script:
- [FM_v3_avoiding_visual_test/load_results_FM_v3_avoiding_visual.py](../../FM_v3_avoiding_visual_test/load_results_FM_v3_avoiding_visual.py)

Eval lifecycle steps:
1. Load projection yaml and visual exps/seeds.
2. Build policy and projector variants.
3. Roll out avoiding env and collect:
- success,
- constraint satisfaction,
- violations,
- runtime.
4. Persist per-variant artifacts.
5. Aggregate across seeds/halfspace variants.

Core caution from merged analysis:
- Current FM visual eval path is still mostly state-concat in planner loop unless image-conditioned planner path is explicitly wired.

## Stage 7: DPCC Compatibility Lifecycle

DPCC baseline files:
- [../../../dpcc/scripts/train.py](../../../dpcc/scripts/train.py)
- [../../../dpcc/scripts/eval.py](../../../dpcc/scripts/eval.py)
- [../../../dpcc/scripts/load_results.py](../../../dpcc/scripts/load_results.py)

DPCC visual migration strategy:
- Add parallel visual scripts/configs, do not overwrite baseline.
- Extend dataset env alias handling for visual ID.

Migration doc:
- [DPCC_VISUAL_AVOIDING_MIGRATION_GUIDE.md](../gen4_visual_camera_avoiding_d3il_plan/DPCC_VISUAL_AVOIDING_MIGRATION_GUIDE.md)

Compatibility outcome target:
1. DPCC old avoiding remains runnable.
2. DPCC visual branch runs with same constraint/eval contract.
3. FM-PCC and DPCC remain A/B comparable under matched seeds/protocols.

## Stage 8: Claim Gate (Does Visual Really Work?)

Two levels of success must be separated.

Level A: Runtime success
- Pipeline runs train/eval/load end-to-end.
- Success and constraint metrics are reasonable.

Level B: Scientific visual success
- Planner conditioning actually consumes visual features in the decision path.
- Vision training/eval improves robustness under visual disturbances compared to state-only baseline.

Current merged assessment:
1. Runtime success likelihood: high to medium-high.
2. True visual claim confidence: medium until planner-image conditioning is fully verified.

## Full Lifecycle Failure Matrix

1. Config dispatch mismatch
- Symptom: script runs with wrong checkpoints or wrong key.
- Fix: re-check visual experiment key bindings in train/eval/load and config files.

2. Dataset alias mismatch
- Symptom: NotImplementedError in dataset loader for visual env ID.
- Fix: extend avoiding env alias branch in diffuser dataset loader.

3. Visual naming but state-only planner
- Symptom: good metrics but no visual robustness gains.
- Fix: ensure planner forward path consumes image features, not only state concat.

4. Projection masks perception deficits
- Symptom: constraints look good even when perception is weak.
- Fix: run ablations with visual perturbations and report deltas vs state baseline.

## Recommended Validation Sequence

Phase A: Smoke
1. 1 seed, 1-2 trials, visual branch only.
2. Verify no binding/runtime errors.

Phase B: Baseline parity
1. Matched seeds between old baseline and visual branch.
2. Compare success + constraints + steps + violations.

Phase C: Visual value test
1. Apply visual perturbations (lighting, blur, partial occlusion).
2. Compare state-only planner vs visual-conditioned planner behavior.

Phase D: Promotion decision
1. Promote visual path only if parity is stable and visual perturbation gains are repeatable.

## Practical Go/No-Go Rules

Go:
1. End-to-end stability with isolated visual path.
2. No regression in legacy baseline path.
3. Visual branch is at least parity, then improves under visual stress tests.

No-Go:
1. Repeated key/config mismatches.
2. Persistent regression in success or safety metrics.
3. Visual claim without evidence that planner decisions are image-conditioned.

## Final Consolidated Verdict

From combined Gen4 status/plan/record/theory + D3IL analysis:
1. The Gen4 visual avoiding infrastructure is correctly moving in the right direction.
2. The lifecycle is structurally sound due to isolation and additive changes.
3. Today, the strongest risk is not environment mechanics, but claim inflation: visual naming can outrun true vision-conditioned planning if planner input remains state-dominant.

Actionable conclusion:
- Treat current branch as visual-enabled lifecycle in progress.
- Claim full visual avoiding only after explicit planner-level image conditioning and perturbation ablations pass.
