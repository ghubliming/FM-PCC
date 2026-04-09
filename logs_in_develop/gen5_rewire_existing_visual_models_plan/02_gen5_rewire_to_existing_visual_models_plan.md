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

---

## 2) Execution Phases

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

Implement minimal, extension-only changes in d3il:
1. create Avoiding vision config aligned with existing vision config style,
2. ensure train and eval simulation use `if_vision: True`,
3. ensure dataset path uses Avoiding image dataset class,
4. ensure selected vision agent receives tuple contract directly.

Do not:
1. create a brand-new vision architecture,
2. change unrelated agents/models.

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

## 3) Concrete File Targets

### 3.1 D3IL validation and rewiring targets

1. run_vision.py
2. configs/aligning_vision_config.yaml
3. configs/sorting_4_vision_config.yaml
4. configs/stacking_vision_config.yaml
5. configs/avoiding_config.yaml
6. simulation/avoiding_sim.py
7. environments/dataset/avoiding_dataset.py
8. agents/*vision*agent.py (only chosen baseline agent)

### 3.2 Script launch targets

1. scripts/aligning_vision/ddpm_encdec_benchmark.sh
2. scripts/sorting_4_vision/ddpm_encdec_benchmark.sh
3. scripts/stacking_vision/ddpm_encdec_benchmark.sh

### 3.3 FM reference-only targets (do not refactor in this phase)

1. FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py
2. FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py
3. flow_matcher_v3_avoiding_visual/datasets/d4rl.py
4. config/avoiding-d3il-visual.py

---

## 4) Tests and Evidence Gates

### 4.1 Existing vision stack validation

Required:
1. benchmark run command exits successfully,
2. logs show vision config and vision agent are active,
3. no runtime path that drops image tensors unexpectedly.

### 4.2 Avoiding vision extension validation

Required:
1. Avoiding run created from vision-style config succeeds,
2. `if_vision=True` path is exercised,
3. perturbation sanity check:
   - changing image input affects action output statistics.

### 4.3 No-fake-vision guard

Required:
1. in declared vision mode, missing image assets fail loudly,
2. fallback to state-only is allowed only in explicitly declared state mode.

---

## 5) Risk Controls

1. Risk: existing visual scripts run but hide conditioning bug.
   - Control: add sensitivity sanity check in report.
2. Risk: Avoiding image alignment mismatch.
   - Control: deterministic file ordering and sequence length checks.
3. Risk: scope creep into FM redesign.
   - Control: FM changes limited to reference mapping in Gen5 phase.

---

## 6) Deliverables

1. Gen5 execution record (03) with benchmark commands, outcomes, and failures/fixes.
2. Gen5 expected results and risk audit (04) with go/no-go for next generation FM integration.
3. Optional: migration checklist from D3IL Avoiding-vision success to FM avoiding planner integration.

---

## 7) Definition of Done

Gen5-02 is complete when:
1. existing D3IL visual stack has been validated by benchmark evidence,
2. Avoiding has a vision-enabled run wired by extension to existing logic,
3. fake-vision behavior is prevented by explicit guards,
4. next-generation FM handoff is concrete and evidence-based.
