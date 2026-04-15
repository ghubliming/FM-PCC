# 01 Current Status: Gen4 Visual-Camera Avoiding in d3il

Date: 2026-04-08
Status: Active Start (Re-scoped)
Scope: FM-PCC primary development, d3il as explicitly marked dependency

---

## 1) Objective

Use gen3u3-style workflow for Gen4, with one target:
1. visual camera model,
2. avoiding task,
3. minimal risk to old dpcc/FM-PCC baseline path.

---

## 2) Verified Current Code Status

### FM-PCC and dpcc relation

1. FM-PCC is derived from dpcc-style train/eval workflow patterns.
2. dpcc baseline flow relies on stable experiment keys and config dispatch (`diffusion`, `plan`).
3. Backward compatibility risk is high if old entrypoints are directly modified.

### d3il (cloned workspace)

1. Vision training entry exists (`run_vision.py`).
2. Vision agents/configs already exist for other tasks.
3. Avoiding currently has state-only task config (`configs/avoiding_config.yaml`).
4. Avoiding dataset/simulation are currently state-oriented.
5. No avoiding-vision task config exists yet.

---

## 3) Required Working Flow (Same Style as gen3u3)

Gen4 implementation will follow strict 3-step copy-modify flow, and in parallel implement d3il copy-upgrade into visual avoiding.

Method split:
1. Gen4 FM-PCC path: 3-step copy-modify for new Gen4 train/eval/load + engine folders.
2. d3il path: copy into FM-PCC, then upgrade avoiding path to visual avoiding with additive changes.

### Step 1: Copy

1. Copy FM-PCC test entry folder pattern from existing v3 path.
2. Copy FM engine folder from `flow_matcher_v3`.
3. Copy config folder/blocks into a dedicated visual-avoiding variant.
4. Copy d3il codebase into FM-PCC path as project-owned dependency baseline.

### Step 2: Rename as isolated Gen4 path

1. Create new test entry path (example naming):
	- `FM_avoiding_visual_test/`
2. Create new engine path (example naming):
	- `flow_matcher_avoiding_visual/`
3. Create visual-avoiding config namespace (do not overwrite old entries).

### Step 3: Modify only new copies

1. Apply visual avoiding changes only inside copied Gen4 paths.
2. Keep all old FM-v3/dpcc baseline paths unchanged.
3. Mark every d3il-required change explicitly in plan and execution logs.
4. In copied d3il path, implement avoiding -> visual avoiding as additive upgrade (no old-path overwrite).

---

## 4) Compatibility Decision (Point 2)

Recommendation: Yes, use another dedicated Gen4 entry.

Reason:
1. protects old dpcc/FM-PCC baselines from regression,
2. enables direct A/B comparison (old baseline vs Gen4 visual avoiding),
3. keeps rollback trivial (switch experiment entry only).

Compatibility rule:
1. old baseline config keys and scripts must remain runnable without edits,
2. Gen4 uses new experiment keys and new script entrypoints only,
3. shared utilities can be touched only when strictly necessary and backward-safe.

---

## 5) Next Step

Use this 01 as lock and execute 02 with explicit file map:
1. new copied Gen4 test folder,
2. new copied Gen4 flow-matcher folder,
3. new copied visual-avoiding config path,
4. d3il copy-upgrade list for visual avoiding,
5. explicitly marked d3il modification list.

---

## 6) Verdict on Two Gen4 Config Files

Verdict (current repository state):
1. `config/avoiding-d3il-visual.py` and `config/projection_eval_visual.yaml` are planned files and are not yet loaded by runtime scripts,
2. current mentions are documentation/planning references,
3. when Gen4 scripts are created, these two files must be bound in Gen4 folder scripts only,
4. old baseline paths stay on existing old config files.

This verdict is compatible with the selected methodology:
1. 3-step Gen4 copy-modify,
2. d3il copy-upgrade to visual avoiding,
3. old baseline path remains preserved.
