# 02 Gen5 Implementation Plan: Rewire to Existing Visual Models First

Date: 2026-04-09
Status: Ready
Depends on: 01_gen5_reset_abandon_gen4u2.md

---

## 1) Objective

Primary objective:
1. validate and reuse existing D3IL vision stack,
2. then extend the same logic into Avoiding,
3. avoid new architecture unless proven necessary.

Correction objective:
1. explicitly fix wrong Gen4 Avoiding-vision method,
2. follow D3IL pattern with isolated new Gen5 folder/file paths first,
3. revert prior mixed baseline edits only where rollback matrix says required.

---

## 2) Part 1: Rollback Avoiding Only (If Needed)

This part is limited to D3IL Avoiding baseline safety.
No rewire work is performed in Part 1.
Source-of-truth baseline for rollback comparison is `/workspaces/d3il`.

### 2.0 Evidence Input From Gen4 Log

Part 1 rollback scope is derived from Gen4 execution record:
1. logs_in_develop/gen4_visual_camera_avoiding_d3il_plan/03_gen4_coding_execution_record.md
2. Use section "5) Implement additive d3il visual-avoiding path" as the authoritative list of avoiding edits.
3. Do not expand rollback scope beyond those listed avoiding touchpoints unless a new baseline break is proven.

### 2.1 Rollback Decision Matrix

Touchpoints under review:
1. d3il/environments/dataset/avoiding_dataset.py
2. d3il/simulation/avoiding_sim.py
3. d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py
4. d3il/configs/avoiding_vision_config.yaml
5. d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh

Decision per touchpoint (locked in Part 1):
1. full revert to original `/workspaces/d3il` file.

Mandatory rule:
1. all five files listed from Gen4 log 03 section 5 are reverted in Part 1,
2. no keep/partial-revert outcome is allowed in Part 1.

Exit criteria:
1. rollback matrix completed and approved,
2. baseline avoiding path validated after full-revert decisions.

### 2.2 Findings-Based Preliminary Decision

Based on current code reading and Gen5 reset policy:
1. `d3il/environments/dataset/avoiding_dataset.py`
   - full revert.
2. `d3il/simulation/avoiding_sim.py`
   - full revert.
3. `d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py`
   - full revert.
4. `d3il/configs/avoiding_vision_config.yaml`
   - full revert.
5. `d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh`
   - full revert.

### 2.3 Part 1 Acceptance Criteria

1. baseline avoiding config path runs without dependence on vision folders,
2. all five avoiding touchpoints are restored byte-equivalent to `/workspaces/d3il`,
3. rollback notes are recorded in Gen5 execution record.

### 2.4 Part 1 Concrete Rollback Procedure

Step A: Freeze scope to avoiding-only files
1. `d3il/environments/dataset/avoiding_dataset.py`
2. `d3il/simulation/avoiding_sim.py`
3. `d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py`
4. `d3il/configs/avoiding_vision_config.yaml`
5. `d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh`
6. Scope lock rule: files not listed in Gen4 log 03 section 5 remain out-of-scope for Part 1 rollback.

Step B: Diff vendored FM-PCC D3IL against baseline D3IL
1. Compare each file in `FM-PCC/d3il/...` against `/workspaces/d3il/...`.
2. Record diff summary in rollback matrix.
3. For each diff entry, map to the corresponding Gen4 log 03 bullet so decision history is auditable.

Step C: Apply rollback policy by file type
1. Revert each of the five scoped files from `/workspaces/d3il` directly.
2. Do not patch in place during Part 1.
3. Do not keep any Gen4 avoiding touchpoint during Part 1.

Step D: Safety verification after rollback decisions
1. Verify `configs/avoiding_config.yaml` still points to state dataset/sim defaults.
2. Verify each reverted file matches `/workspaces/d3il` with zero diff.
3. Verify baseline avoiding path does not require vision assets.

Step E: Decision lock
1. If baseline still breaks after full revert: treat as pre-existing baseline issue and isolate diagnosis before Part 2.
2. If baseline passes: continue to Part 2 using only isolated Gen5 vision paths.

### 2.5 Part 1 Deliverable

Create rollback appendix table with columns:
1. file,
2. diff vs `/workspaces/d3il`,
3. decision (keep/patch/revert/deprecate),
4. baseline impact,
5. verification result,
6. owner/date.

---

## 3) Part 2: Rewire to Existing Visual Models

Part 2 starts only after Part 1 acceptance is met.

### 3.1 Execution Phases

### Phase 1: Benchmark existing visual stack (no Avoiding edits yet)

Run representative visual benchmarks from existing folders:
1. aligning_vision,
2. sorting_4_vision,
3. stacking_vision.

Use benchmark scripts as-is first, then minimal fix only if required.

Outputs to collect:
1. run success/failure,
2. runtime logs,
3. checkpoint and metric artifacts,
4. quick sanity on image-conditioned path usage.

Exit criteria:
1. at least one successful end-to-end visual run,
2. no fundamental flaw.

### Phase 2: Extract reusable visual contract

From passing tasks, lock the reusable interface:
1. input contract pattern,
2. dataset contract pattern,
3. config contract pattern,
4. sim/eval contract pattern.

Target reusable baseline (expected):
1. vision tuple style `(bp_image, inhand_image, state_or_goal_state)`,
2. `if_vision: True` in simulation config,
3. image dataset class path in config.

Exit criteria:
1. one-page contract summary accepted,
2. no contradictory behavior across selected visual tasks.

### Phase 3: Rewire Avoiding to existing visual contract

Implement minimal, extension-only changes in d3il using isolated new Gen5 paths:
1. create new config file path for Gen5 avoiding vision (parallel to baseline),
2. create new simulation path for Gen5 avoiding vision (parallel to baseline),
3. create new dataset wrapper/path for Gen5 avoiding vision contract,
4. create new launcher folder for Gen5 avoiding vision benchmark,
5. ensure selected vision agent receives tuple contract directly.

Do not:
1. create a brand-new vision architecture,
2. mutate baseline avoiding files as first-choice implementation,
3. change unrelated agents/models.

Exit criteria:
1. Avoiding vision run executes end-to-end using camera observations,
2. no silent state-only fallback in declared vision mode.

### Phase 4: Link learnings back to FM planning path (next generation prep)

After Avoiding vision run is healthy:
1. map which contracts can be transferred to FM avoiding visual planner path,
2. identify exactly which FM files need rewire,
3. defer heavy FM refactor to next generation unless blocking.

Exit criteria:
1. precise handoff list for next generation,
2. evidence-backed go/no-go recommendation.

---

## 4) Concrete File Targets

### 4.1 D3IL validation and rewiring targets

1. run_vision.py
2. configs/aligning_vision_config.yaml
3. configs/sorting_4_vision_config.yaml
4. configs/stacking_vision_config.yaml
5. configs/avoiding_config.yaml
6. simulation/avoiding_sim.py
7. environments/dataset/avoiding_dataset.py
8. agents/*vision*agent.py (only chosen baseline agent)

### 4.1.1 Gen5 isolated Avoiding targets (new-folder pattern)

1. configs/avoiding_vision_gen5_config.yaml
2. simulation/avoiding_vision_gen5_sim.py
3. environments/dataset/avoiding_vision_gen5_dataset.py
4. scripts/avoiding_vision_gen5/ddpm_encdec_benchmark.sh

Notes:
1. these are preferred over direct mutation of baseline avoiding paths,
2. old Gen4 avoiding-vision paths are treated as deprecated once Gen5 isolated path passes.

### 4.2 Script launch targets

1. scripts/aligning_vision/ddpm_encdec_benchmark.sh
2. scripts/sorting_4_vision/ddpm_encdec_benchmark.sh
3. scripts/stacking_vision/ddpm_encdec_benchmark.sh

### 4.3 FM reference-only targets (do not refactor in this phase)

1. FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py
2. FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py
3. flow_matcher_v3_avoiding_visual/datasets/d4rl.py
4. config/avoiding-d3il-visual.py

### 4.4 Rollback-review targets from wrong Gen4 block

1. d3il/environments/dataset/avoiding_dataset.py
2. d3il/simulation/avoiding_sim.py
3. d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py
4. d3il/configs/avoiding_vision_config.yaml
5. d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh

---

## 5) Tests and Evidence Gates

### 5.1 Existing vision stack validation

Required:
1. benchmark run command exits successfully,
2. logs show vision config and vision agent are active,
3. no runtime path that drops image tensors unexpectedly.

### 5.2 Avoiding vision extension validation

Required:
1. Avoiding run created from vision-style config succeeds,
2. `if_vision=True` path is exercised,
3. perturbation sanity check:
   - changing image input affects action output statistics.
4. run is executed via new Gen5 isolated path, not baseline-mutated path.

### 5.3 No-fake-vision guard

Required:
1. in declared vision mode, missing image assets fail loudly,
2. fallback to state-only is allowed only in explicitly declared state mode.

### 5.4 Rollback guard

Required:
1. baseline avoiding command path remains stable after rollback decisions,
2. any non-isolated Gen4 touchpoint is either reverted or hard-guarded.

---

## 6) Risk Controls

1. Risk: existing visual scripts run but hide conditioning bug.
   - Control: add sensitivity sanity check in report.
2. Risk: Avoiding image alignment mismatch.
   - Control: deterministic file ordering and sequence length checks.
3. Risk: scope creep into FM redesign.
   - Control: FM changes limited to reference mapping in Gen5 phase.

---

## 7) Deliverables

1. Gen5 execution record (03) with benchmark commands, outcomes, and failures/fixes.
2. Gen5 expected results and risk audit (04) with go/no-go for next generation FM integration.
3. Optional: migration checklist from D3IL Avoiding-vision success to FM avoiding planner integration.
4. Rollback matrix appendix for wrong Gen4 touchpoints with keep/revert/supersede decision and reason.

---

## 8) Definition of Done

Gen5-02 is complete when:
1. existing D3IL visual stack has been validated by benchmark evidence,
2. rollback matrix is completed for wrong Gen4 avoiding-vision touchpoints,
3. Part 1 rollback/fix acceptance criteria are met,
4. Avoiding has a vision-enabled run wired via isolated Gen5 path,
5. fake-vision behavior is prevented by explicit guards,
6. baseline avoiding remains stable after rollback/guard actions,
7. next-generation FM handoff is concrete and evidence-based.
