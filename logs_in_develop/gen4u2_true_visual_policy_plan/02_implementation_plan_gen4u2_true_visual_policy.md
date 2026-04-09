# 02 Implementation Plan: Gen4U2 True Visual Policy

Date: 2026-04-09
Status: Ready for Execution After Review
Depends on: 01_current_status_gen4u2_true_visual_policy.md

---

## 1) Goal and Non-Goal

Goal:
- Convert FM v3 avoiding visual path from state-dominant behavior to true image-conditioned policy behavior, with explicit proof.

Non-goal:
- Replacing all historical state baselines.
- Breaking old train/eval scripts that rely on state-only logic.

---

## 2) Locked Design Decisions

1. Keep trajectory target unchanged for now: action + state trajectory prediction remains numeric.
2. Add visual conditioning as an additional conditioning stream, not a full decoder redesign in first pass.
3. Enable strict fail-fast mode when visual conditioning is requested but image assets are missing.
4. Keep state-only mode available behind explicit config flag.

Rationale:
- This minimizes regression risk while still forcing true visual dependence in the decision path.

---

## 3) File-Level Implementation Map

### 3.1 Data loading and batch contract

Primary files:
1. flow_matcher_v3_avoiding_visual/datasets/d4rl.py
2. flow_matcher_v3_avoiding_visual/datasets/sequence.py
3. flow_matcher_v3_avoiding_visual/datasets/__init__.py

Actions:
1. Extend avoiding visual dataset path to optionally load per-timestep images from:
   - d3il/environments/dataset/data/avoiding/data/images/bp-cam
   - d3il/environments/dataset/data/avoiding/data/images/inhand-cam
2. Introduce a visual-enabled dataset class or mode, for example:
   - SequenceDatasetVisual, or
   - SequenceDataset with use_vision_cond switch.
3. Return a batch that includes:
   - trajectories (unchanged numeric tensor),
   - conditions[0] (state condition, existing),
   - conditions['bp_imgs'] and conditions['inhand_imgs'] or single fused key conditions['vision'].
4. Add hard checks under strict mode:
   - if use_vision_cond true and expected image folders missing, raise clear error.

Acceptance:
1. Data loader prints or logs non-zero visual sample count.
2. Training fails immediately when strict visual mode is on and images are absent.

### 3.2 Visual encoder and fusion into model

Primary files:
1. flow_matcher_v3_avoiding_visual/models/unet1d_temporal_cond.py
2. flow_matcher_v3_avoiding_visual/models/diffusion.py
3. flow_matcher_v3_avoiding_visual/models/helpers.py
4. optional new file: flow_matcher_v3_avoiding_visual/models/vision_encoder.py

Actions:
1. Add a compact visual encoder (shared weights or dual-stream for bp and inhand).
2. Extract visual embedding from condition tensors each forward pass.
3. Fuse visual embedding into temporal model path using one of:
   - additive conditioning into time embedding,
   - FiLM-style scale/shift on residual blocks,
   - cross-attention block (optional in later refinement).
4. Ensure model forward actually consumes non-numeric condition keys. Current path receives cond but effectively ignores it except state overwrite in apply_conditioning.
5. Keep backward compatibility:
   - when use_vision_cond false, visual encoder path is bypassed.

Acceptance:
1. Unit-level check: changing image tensors while keeping state fixed changes model output distribution.
2. Zero-image and shuffled-image ablations produce measurable output shift.

### 3.3 Config and experiment controls

Primary file:
1. config/avoiding-d3il-visual.py

Actions:
1. Add explicit keys:
   - use_vision_cond: true/false
   - vision_strict_assets: true/false
   - vision_image_size
   - vision_backbone
   - vision_fusion
   - vision_use_bp_cam
   - vision_use_inhand_cam
2. Keep defaults safe:
   - state baseline blocks remain unchanged.
   - Gen4U2-specific blocks enable true visual mode.
3. Add metadata tag such as claim_level='true_visual_policy_candidate'.

Acceptance:
1. Config alone determines state-only vs true-visual behavior without code edits.

### 3.4 Train and eval wiring

Primary files:
1. FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py
2. FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py
3. config/projection_eval_visual.yaml

Actions:
1. Train path:
   - plumb new config flags to dataset/model constructors.
2. Eval path:
   - instantiate avoiding env with if_vision=True for true-visual runs,
   - pass image observations into policy call,
   - keep old state-only branch available by config flag.
3. Projection eval yaml:
   - add explicit run profile label for visual-conditioned runs to avoid result mixing.

Acceptance:
1. Eval in true-visual mode errors if image stream is unavailable.
2. Eval in state-only mode keeps previous behavior.

### 3.5 Policy interface update

Primary file:
1. flow_matcher_v3_avoiding_visual/sampling/policies.py

Actions:
1. Extend Policy.__call__ signature to accept optional visual tensors in conditions.
2. Normalize and batch-repeat visual tensors in _format_conditions.
3. Preserve compatibility with existing state-only condition dictionary.

Acceptance:
1. No regressions for old calls using only conditions={0: obs}.
2. Visual mode accepts and forwards camera tensors end-to-end.

---

## 4) Execution Phases

### Phase A: Data contract upgrade

Deliverables:
1. Visual-enabled dataset return structure.
2. Strict asset checks.
3. Data sanity script output.

Exit gate:
- At least one training batch contains image tensors with expected shape and non-zero variance.

### Phase B: Model conditioning integration

Deliverables:
1. Visual encoder module.
2. Fusion path in U-Net forward.
3. Config switches for enabling/disabling visual path.

Exit gate:
- Controlled perturbation test confirms output sensitivity to images.

### Phase C: Eval pipeline conversion

Deliverables:
1. True visual eval branch with if_vision=True.
2. Policy call receives state + visual conditions.
3. Result directory separation for state-only vs visual-conditioned runs.

Exit gate:
- End-to-end eval executes with live image inputs.

### Phase D: Claim validation

Deliverables:
1. Ablation report:
   - normal images,
   - zeroed images,
   - shuffled images,
   - state-only fallback.
2. Summary table for success, constraint satisfaction, and violation metrics.

Exit gate:
- Visual-conditioned run shows meaningful behavior change under image ablations.

---

## 5) Test Matrix (Must Pass)

1. Loader strictness test:
   - use_vision_cond=true + missing image folders => hard fail.
2. Loader compatibility test:
   - use_vision_cond=false => old state pipeline still trains.
3. Forward sensitivity test:
   - fixed state, varied image => different predicted action trajectory stats.
4. Eval wiring test:
   - true visual mode uses if_vision=True and passes image tensors.
5. Regression test:
   - previous state baseline metrics remain within tolerance window.

---

## 6) Risk Register and Mitigation

1. Risk: Model ignores visual features after integration.
   - Mitigation: add explicit output-sensitivity checks and fail CI on no-change.
2. Risk: Data alignment mismatch between pkl timesteps and image frame counts.
   - Mitigation: enforce deterministic frame sorting and pad/truncate policy.
3. Risk: Runtime cost increase from image encoding.
   - Mitigation: cache frame transforms, reduce image size, and benchmark per-step latency.
4. Risk: Result confusion between state and visual experiments.
   - Mitigation: isolate output prefixes and metadata tags.

---

## 7) Backward Compatibility Contract

1. Existing state-only scripts remain runnable with default false on use_vision_cond.
2. Existing checkpoints stay loadable for state-only eval.
3. New checkpoints trained with visual conditioning are stored under separate experiment prefix.

---

## 8) Definition of Done for Gen4U2 02

Gen4U2 true visual policy is considered implemented only when all conditions hold:
1. Data loader consumes camera frames during training.
2. Model forward path consumes visual embeddings.
3. Eval path uses if_vision=True and passes image inputs into policy.
4. Missing-image strict mode fails fast.
5. Ablations prove visual dependence.
6. State-only baseline remains functional.

---

## 9) Suggested Implementation Order (Low-Risk)

1. Implement data contract and strict checks first.
2. Integrate visual encoder and fusion second.
3. Wire eval true-visual branch third.
4. Run ablations and freeze claims last.

---

## 10) Review Prompt

Reviewer decision request:
1. approve this file-level plan as Gen4U2-02 baseline,
2. authorize coding execution record creation as 03,
3. authorize expected-results and risk audit update as 04 after first implementation run.
