# 01 Current Status: Gen4U2 True Visual Policy

Date: 2026-04-09
Status: Abandoned (Archive, Do Not Execute)
Notice:
1. Gen4U2 is abandoned.
2. Do not delete this file content; keep as historical record.
3. Active path is U5/Gen5: rewire existing visual models first, then extend to Avoiding visual only if no fundamental flaw, and do not rebuild wheels.

Superseding docs:
1. logs_in_develop/gen5_rewire_existing_visual_models_plan/01_gen5_reset_abandon_gen4u2.md
2. logs_in_develop/gen5_rewire_existing_visual_models_plan/02_gen5_rewire_to_existing_visual_models_plan.md

---

## Archived Gen4U2 Baseline (Retained)

### 1) Objective

Define the corrected Gen4U2 baseline.

Gen4U2 principle:
1. do not rebuild existing visual wheels,
2. extend proven D3IL visual logic into Avoiding planning path,
3. remove silent state-only fallback under visual label,
4. keep state baseline compatibility as an explicit mode.

### 2) Investigation Summary (Locked)

Based on local test and code tracing:
1. visual train/eval scripts run even when image folders are absent,
2. current FM visual path for Avoiding is still state-dominant,
3. this validates route/config only, not true image-conditioned policy.

Additional finding:
1. D3IL already has visual building blocks for Avoiding and other tasks,
2. current FM Avoiding visual path is not reusing those blocks end-to-end.

### 3) Verified Findings

#### 3.1 D3IL already contains visual logic we should reuse

Available components in this workspace:
1. visual observation path in Avoiding env via if_vision=True.
2. Avoiding image dataset support in Avoiding_Img_Dataset.
3. visual agent prediction contract using tuple input (bp_image, inhand_image, des_robot_pos).
4. image normalization and temporal context handling already implemented in D3IL vision agents.

Conclusion:
1. required visual primitives already exist and are production-shaped.

#### 3.2 Current FM Avoiding visual route bypasses those primitives

Current flow-matcher dataset route for Avoiding visual still does:
1. load pkl state only,
2. build [x_des, y_des, x, y] style observation,
3. ignore image folders during training batches.

Current eval policy call still does:
1. pass state condition only,
2. avoid camera tensor path in policy conditioning.

#### 3.3 Last Gen4 wrong direction (explicit correction)

Wrong direction from previous draft:
1. proposing new visual components first, before integrating existing D3IL visual path,
2. allowing visual alias to pass through state-only loader path as final behavior.

Correction:
1. Gen4U2 must be reuse-first,
2. visual mode must not silently degrade to state-only when strict visual is requested.

### 4) Root Cause Statement

Root cause is integration gap, not missing ecosystem pieces:
1. D3IL visual components exist,
2. FM Avoiding visual planner path is not wired to consume them,
3. therefore visual-named runs can execute without images.

### 5) Current Claim Level (Must Be Explicit)

Allowed claim now:
1. visual routing/entrypoint exists,
2. code executes in state-dominant mode.

Not allowed claim now:
1. image-conditioned planning,
2. true visual policy behavior,
3. robustness gain from camera input.

### 6) Gen4U2 Direction (Pre-02)

Gen4U2 is defined as extension, not reinvention.

Required direction:
1. reuse D3IL visual data/observation contract,
2. bridge that contract into FM planner conditioning,
3. enforce strict visual mode semantics,
4. preserve explicit state-only compatibility mode.

### 7) Constraints and Compatibility Rules

1. No wheel rebuilding before reuse path is exhausted.
2. Keep old state baselines runnable by explicit config switch.
3. Visual strict mode must fail fast when image assets are missing.
4. All visual claims must be backed by ablation evidence.

### 8) Evidence Requirements Before 02 Is Approved

02 must include explicit file-level answers for:
1. where D3IL visual dataset logic is reused,
2. how FM dataset returns camera conditions,
3. where policy/model consumes camera conditions,
4. how eval obtains and forwards camera observations,
5. how strict visual mode blocks silent state fallback,
6. which tests prove camera dependence.

### 9) Acceptance Gate for Moving to 02

Proceed to 02 only if this baseline is accepted:
1. current branch is state-dominant despite visual naming,
2. D3IL already has reusable visual components,
3. Gen4U2 must extend those components into Avoiding planner path,
4. no-rebuild-first rule is mandatory.
