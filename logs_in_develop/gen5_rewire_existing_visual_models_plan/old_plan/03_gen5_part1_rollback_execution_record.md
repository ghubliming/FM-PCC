# 03 Gen5 Part 1 Rollback Execution Record

Date: 2026-04-09
Status: Completed
Scope source: logs_in_develop/gen4_visual_camera_avoiding_d3il_plan/03_gen4_coding_execution_record.md (section 5)
Baseline source of truth: /workspaces/d3il
Target repo: /workspaces/FM-PCC/d3il

---

## 1) Objective

Execute Gen5-02 Part 1 rollback for the five Avoiding touchpoints and restore FM-PCC D3IL scope to original baseline parity.

---

## 2) Scoped Touchpoints

1. d3il/environments/dataset/avoiding_dataset.py
2. d3il/simulation/avoiding_sim.py
3. d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py
4. d3il/configs/avoiding_vision_config.yaml
5. d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh

---

## 3) Commands Executed

1. Copy baseline files from /workspaces/d3il into FM-PCC for scoped paths.
2. Verify parity with diff -q for scoped paths.
3. Resolve baseline-absence mismatch for two visual-only files by deletion in FM-PCC.
4. Re-verify rollback outcomes.

---

## 4) Verification Results

### 4.1 Zero-diff against baseline

1. OK_ZERO_DIFF environments/dataset/avoiding_dataset.py
2. OK_ZERO_DIFF simulation/avoiding_sim.py
3. OK_ZERO_DIFF environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py

### 4.2 Baseline-absent files handled

Observed during verification:
1. /workspaces/d3il/configs/avoiding_vision_config.yaml does not exist.
2. /workspaces/d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh does not exist.

Action taken:
1. Removed FM-PCC/d3il/configs/avoiding_vision_config.yaml
2. Removed FM-PCC/d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh

Post-action result:
1. visual-only files are absent in both baseline and FM-PCC for scoped paths.

---

## 5) Rollback Matrix (Part 1 Final)

| file | baseline status in /workspaces/d3il | action | result |
|---|---|---|---|
| d3il/environments/dataset/avoiding_dataset.py | exists | revert by copy | zero diff |
| d3il/simulation/avoiding_sim.py | exists | revert by copy | zero diff |
| d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py | exists | revert by copy | zero diff |
| d3il/configs/avoiding_vision_config.yaml | absent | remove in FM-PCC | absent both |
| d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh | absent | remove in FM-PCC | absent both |

---

## 6) Part 1 Exit Decision

Part 1 rollback is complete.

Decision:
1. proceed to Gen5-02 Part 2 only with isolated Gen5 paths,
2. no reuse of removed visual-only baseline-touch files,
3. baseline Avoiding path remains the clean entry point.
