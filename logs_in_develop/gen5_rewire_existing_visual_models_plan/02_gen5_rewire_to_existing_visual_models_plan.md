# 02 Gen5 Implementation Plan: FMv3 Aligning Vision

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
2. follow FMv3 aligning vision pattern with isolated new folder/file paths,
3. revert prior mixed baseline edits only where rollback matrix says required.

---

## 2) Part 1: Rollback Avoiding Only (If Needed)

This part is limited to D3IL Avoiding baseline safety.
No architecture migration work is performed in Part 1.
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

Based on current code reading and reset policy:
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
3. rollback notes are recorded in execution record.

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
2. If baseline passes: continue to Part 2 using only isolated FMv3 aligning vision paths.

### 2.5 Part 1 Deliverable

Create rollback appendix table with columns:
1. file,
2. diff vs `/workspaces/d3il`,
3. decision (keep/patch/revert/deprecate),
4. baseline impact,
5. verification result,
6. owner/date.

---

## 3) Part 2: Simple Copy-Modify on FMv3 Aligning Vision (Locked)

Part 2 starts only after Part 1 acceptance is met.

This part is intentionally simple and uses the same copy-modify pattern as abandoned Gen4, while keeping old code as stable baseline.

Implementation scope note:
1. all D3IL-side integration targets the vendored folder `FM-PCC/d3il`.
2. do not edit `/workspaces/d3il`; it is baseline reference only.

### 3.0 Aligning Visual First Gate (Mandatory)

Before Entry 1 starts, run aligning visual as first validation.

Current observation state:
1. structural readiness confirmed from existing files,
2. runtime smoke result not yet recorded,
3. Part 2 remains blocked until one aligning visual run result is logged.

Required first command family:
1. `scripts/aligning_vision/ddpm_encdec_benchmark.sh` or equivalent single-run `run_vision.py` command,
2. record pass/fail and key error lines in execution record.

Verified conditioning finding:
1. D3IL visual path is `visual + state`, not visual-only.
2. Policy input contract is `(bp_image, inhand_image, des_robot_pos)`.
3. Action is predicted output target, not part of visual conditioning input tuple.

Old FM avoiding visual finding:
1. archived FM avoiding visual path conditions on state vector (`conditions={0: obs}`),
2. archived dataset route still uses pkl state stream in avoiding branch,
3. so old path behaves as state/action-dominant, not proven image-conditioned.

### 3.1 Three Entry Execution Path

Entry 1: Copy two FMv3 folders, do not edit old folders
1. copy `flow_matcher_v3_avoiding_visual` to `flow_matcher_v3_avoiding_visual_fmv3_aligning_vision`,
2. copy `FM_v3_avoiding_visual_test` to `FM_v3_avoiding_visual_fmv3_aligning_vision_test`,
3. keep original source folders untouched for rollback and A/B reference.

Entry 2: Modify only the two copied folders
1. apply FMv3 aligning vision edits only inside the two new copied folders,
2. do not modify baseline `d3il` avoiding paths,
3. do not modify old FMv3 folders directly.

Entry 2A: Mandatory integration blocks (expected broad changes)
1. Dataset block: load and align bp/inhand image sequences with state/action for avoiding.
2. Policy/model block: accept visual+state conditioning tuple end-to-end in FMv3 path.
3. Runtime block: eval/training must request vision observation path and fail loudly if image assets are missing.

Entry 3: Add two new config files
1. create `config/avoiding-d3il-fmv3-aligning-vision.py` for train/runtime,
2. create `config/projection_eval_fmv3_aligning_vision.yaml` for eval/projection,
3. keep old configs untouched.

### 3.2 Scope Locks

1. old avoiding and old FMv3 code are preserved as-is,
2. all new Part 2 edits are isolated to copied FMv3 aligning vision folders and two new config files,
3. if new path fails, disable only FMv3 aligning vision copied path and keep baseline runnable.

---

## 4) Concrete File Targets

### 4.1 Copied FMv3 Aligning Vision folders (new)

1. `flow_matcher_v3_avoiding_visual_fmv3_aligning_vision` (copied from `flow_matcher_v3_avoiding_visual`),
2. `FM_v3_avoiding_visual_fmv3_aligning_vision_test` (copied from `FM_v3_avoiding_visual_test`).

### 4.2 Two new config files (new)

1. `config/avoiding-d3il-fmv3-aligning-vision.py` (train/runtime config),
2. `config/projection_eval_fmv3_aligning_vision.yaml` (eval config).

### 4.3 Preserved old paths (read-only)

1. `flow_matcher_v3_avoiding_visual`,
2. `FM_v3_avoiding_visual_test`,
3. `config/avoiding-d3il-visual.py`,
4. baseline `d3il` avoiding files restored in Part 1.

### 4.4 Rollback-review targets from wrong Gen4 block

1. d3il/environments/dataset/avoiding_dataset.py
2. d3il/simulation/avoiding_sim.py
3. d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py
4. d3il/configs/avoiding_vision_config.yaml
5. d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh

---

## 5) Tests and Evidence Gates

### 5.1 FMv3 Aligning Vision copied path smoke validation

Required:
1. train/eval entrypoints in copied FMv3 aligning vision folders run,
2. two new config files load correctly,
3. baseline old path still runs unchanged.
4. at least one aligning visual run result is recorded before FMv3 aligning vision edits start.
5. weak-performance run is acceptable only if image perturbation changes predicted action statistics.

### 5.2 Isolation guard

Required:
1. no edits appear in preserved old FMv3 folders,
2. no new edits appear in baseline `d3il` avoiding files.

### 5.3 Rollback guard

Required:
1. disabling FMv3 aligning vision copied folders must leave baseline run path valid,
2. Part 1 rollback state remains intact.

---

## 6) Risk Controls

1. Risk: accidental edits to old baseline folders.
   - Control: edit lock to copied FMv3 aligning vision folders only.
2. Risk: config collision with old runs.
   - Control: new config filenames for FMv3 aligning vision path only.
3. Risk: FMv3 aligning vision path breaks.
   - Control: old folders remain untouched and immediately reusable.

---

## 7) Deliverables

1. execution record (03) with Part 1 rollback evidence and Part 2 copy-modify commands.
2. list of copied FMv3 aligning vision folders and modified files inside them.
3. two new config files and command examples using only FMv3 aligning vision paths.
4. quick A/B run note: old path vs FMv3 aligning vision path.

---

## 8) Definition of Done

Gen5-02 is complete when:
1. Part 1 rollback/fix acceptance criteria are met,
2. rollback matrix is completed for wrong Gen4 avoiding-vision touchpoints,
3. exactly two FMv3 folders are copied into FMv3 aligning vision folders,
4. exactly two new FMv3 aligning vision config files are added,
5. old FMv3 folders and baseline `d3il` avoiding files remain unchanged after Part 1,
6. FMv3 aligning vision copied path runs and baseline path still runs.
