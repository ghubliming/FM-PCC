# 02 Implementation Plan: Gen4 Visual-Camera Avoiding in FM-PCC (Vendor d3il)

Date: 2026-04-08
Status: Full Detailed Plan (Review Version)
Depends on: `01_current_status_gen4_visual_camera_avoiding_d3il.md`

---

## 1) Goal

Build a Gen4 visual-avoiding path with gen3u3-style lifecycle discipline, while preserving backward compatibility with old dpcc/FM-PCC baselines.

Target outcome:
1. FM-PCC becomes self-contained (no user-side manual d3il clone required),
2. Gen4 uses strict 3-step copy-modify isolation,
3. old avoiding/state baselines remain runnable and minimally touched,
4. d3il package name remains `d3il` to avoid import-chain breakage.

---

## 2) Strategy Decision (Locked)

### 2.1 Vendor strategy

1. Copy full d3il code into FM-PCC under `FM-PCC/d3il/`.
2. Keep folder/package name `d3il` unchanged.
3. All new visual-avoiding changes are additive and guarded (do not break old paths).

### 2.2 Why this strategy

1. Removes setup friction for users.
2. Keeps import compatibility for existing scripts that already use `from d3il...`.
3. Avoids mass refactor caused by package renaming.
4. Supports A/B between old baseline and Gen4 entries.

---

## 3) Mandatory Working Flow (Same as gen3u3)

Gen4 implementation must follow exactly 3 steps:

### Step 1: Copy

1. Copy `FM_v3_test/` into a new Gen4 test folder.
2. Copy `flow_matcher_v3/` into a new Gen4 engine folder.
3. In existing `config/` folder, copy-modify into two new Gen4 config files (no folder rename).

### Step 2: Rename into isolated Gen4 paths

Proposed names (final names can be adjusted before coding, but must be unique):
1. `FM_gen4_avoiding_visual_test/`
2. `flow_matcher_gen4_avoiding_visual/`
3. experiment keys:
   - `flow_matching_gen4_avoiding_visual`
   - `plan_fm_gen4_avoiding_visual`

### Step 3: Modify only copied paths

1. Apply Gen4-specific edits only in new copied folders/keys.
2. Keep existing `FM_v3_test/`, `flow_matcher_v3/`, and old experiment keys runnable.
3. If shared utility edit is unavoidable, it must be backward-safe and explicitly logged.

---

## 4) Scope Ownership Labels (Mandatory)

Every edit in 03 execution record must be labeled:
1. `[FM-PCC]` for FM-PCC core changes.
2. `[D3IL-VENDORED REQUIRED]` for required edits inside vendored `FM-PCC/d3il/`.
3. `[SHARED-BACKSAFE]` for unavoidable shared utility edits proven backward-safe.

No unlabeled change is allowed.

---

## 5) What Will Change

### 5.1 FM-PCC Gen4 copies and entrypoints

1. New Gen4 test folder with 3 major scripts (train/eval/load_results).
2. New Gen4 flow-matcher engine folder.
3. Two new Gen4 config files in existing `config/` folder:
   - `config/avoiding-d3il-gen4-visual.py`
   - `config/projection_eval_gen4_visual.yaml`
4. Gen4 output label lines for traceability.

### 5.2 Vendored d3il additive support for visual avoiding

1. Add avoiding vision config file.
2. Add avoiding image dataset class.
3. Add avoiding simulation vision switch path.
4. Add minimal avoiding_vision script launcher.

---

## 6) What Will NOT Change

1. No rewrite of old FM-v3 folders.
2. No deletion/renaming of existing d3il package path.
3. No FM objective redesign in this phase.
4. No broad multi-task refactor across unrelated d3il tasks.
5. No forced migration of old logs/checkpoints naming.
6. No rename of `config/` folder.

---

## 7) Architecture and Compatibility Rules

### 7.1 Import and package rule

1. Keep package name `d3il` unchanged.
2. Prefer path-level isolation (which folder is on PYTHONPATH) over package renaming.
3. Gen4 scripts must resolve to vendored `FM-PCC/d3il/` by default.

### 7.2 Backward compatibility rule

1. Old baseline commands must still run without argument changes.
2. Existing old config keys remain valid.
3. Any new defaults must not alter old run behavior unless old entries explicitly opt in.

### 7.3 Data contract rule

1. Avoiding state-only dataset path remains valid.
2. New vision dataset path is additive and selected only by vision config.
3. Action dimension for avoiding remains 2 at policy output level.

---

## 8) Exact File-Level Edit List

### A) Bootstrapping vendored d3il

1. `[D3IL-VENDORED REQUIRED]` Create `FM-PCC/d3il/` by copying workspace d3il source snapshot.
2. `[D3IL-VENDORED REQUIRED]` Add a short version note file in `FM-PCC/d3il/`:
   - source commit hash,
   - copy date,
   - local modifications policy.

### B) FM-PCC Gen4 copy/rename stage

1. `[FM-PCC]` Copy `FM_v3_test/` -> `FM_gen4_avoiding_visual_test/`.
2. `[FM-PCC]` Copy `flow_matcher_v3/` -> `flow_matcher_gen4_avoiding_visual/`.
3. `[FM-PCC]` Update imports in new Gen4 test scripts to point at new engine package.
4. `[FM-PCC]` Keep old folders untouched.

### C) FM-PCC config stage

1. `[FM-PCC]` Keep folder name `config/` unchanged.
2. `[FM-PCC]` Create `config/avoiding-d3il-gen4-visual.py` by copy-modify from `config/avoiding-d3il.py`:
   - add `flow_matching_gen4_avoiding_visual`,
   - add `plan_fm_gen4_avoiding_visual`,
   - set explicit metadata tags:
     - `task=avoiding`
     - `modality=vision`
     - `engine=gen4`
     - `d3il_source=vendored`.
3. `[FM-PCC]` Create `config/projection_eval_gen4_visual.yaml` by copy-modify from `config/projection_eval.yaml`:
   - keep baseline-compatible fields,
   - add Gen4-specific defaults for visual avoiding evaluation,
   - keep key names consistent with eval/load scripts.

### C.1) Config consumer binding matrix (must stay consistent)

1. `config/avoiding-d3il-gen4-visual.py` is used by:
   - `FM_gen4_avoiding_visual_test/train_FM_gen4_avoiding_visual.py`
   - `FM_gen4_avoiding_visual_test/eval_FM_gen4_avoiding_visual.py`
   - `FM_gen4_avoiding_visual_test/load_results_FM_gen4_avoiding_visual.py`
2. `config/projection_eval_gen4_visual.yaml` is used by:
   - `FM_gen4_avoiding_visual_test/eval_FM_gen4_avoiding_visual.py`
   - `FM_gen4_avoiding_visual_test/load_results_FM_gen4_avoiding_visual.py`
3. If either filename changes, all listed consumers must be updated in the same commit.

### D) Gen4 script stage (3 major scripts)

1. `[FM-PCC]` Create/modify `FM_gen4_avoiding_visual_test/train_FM_gen4_avoiding_visual.py`:
   - set `exp = 'avoiding-d3il-gen4-visual'`,
   - parser uses module `config.avoiding-d3il-gen4-visual`,
   - parser experiment uses `flow_matching_gen4_avoiding_visual`.
2. `[FM-PCC]` Create/modify `FM_gen4_avoiding_visual_test/eval_FM_gen4_avoiding_visual.py`:
   - read `config/projection_eval_gen4_visual.yaml`,
   - set `exp = 'avoiding-d3il-gen4-visual'`,
   - parser uses `plan_fm_gen4_avoiding_visual`.
3. `[FM-PCC]` Create/modify `FM_gen4_avoiding_visual_test/load_results_FM_gen4_avoiding_visual.py`:
   - read `config/projection_eval_gen4_visual.yaml`,
   - set `exp = 'avoiding-d3il-gen4-visual'`,
   - emit explicit provenance line in outputs.

### E) Vendored d3il additive modifications

1. `[D3IL-VENDORED REQUIRED]` Create `FM-PCC/d3il/configs/avoiding_vision_config.yaml`.
2. `[D3IL-VENDORED REQUIRED]` Update `FM-PCC/d3il/environments/dataset/avoiding_dataset.py`:
   - add `Avoiding_Img_Dataset` class only,
   - keep existing `Avoiding_Dataset` unchanged.
3. `[D3IL-VENDORED REQUIRED]` Update `FM-PCC/d3il/simulation/avoiding_sim.py`:
   - add vision-mode switch (default `if_vision: False`),
   - preserve old state behavior path.
4. `[D3IL-VENDORED REQUIRED]` Add `FM-PCC/d3il/scripts/avoiding_vision/` launcher for smoke run.

### E.1) Detailed d3il upgrade plan: avoiding -> visual avoiding

This subsection is the detailed implementation blueprint for d3il only.

#### E.1.1 Config layer (`FM-PCC/d3il/configs/avoiding_vision_config.yaml`)

1. Copy baseline from `FM-PCC/d3il/configs/avoiding_config.yaml`.
2. Switch defaults to one vision-capable agent profile (start with `ddpm_encdec_vision`).
3. Keep avoiding task identity unchanged (`log_dir`, group naming, seed logic), but add visual mode fields.
4. Add both simulation blocks:
   - `train_simulation` (low rollout count for training validation)
   - `simulation` (evaluation rollouts)
5. Add/keep `if_vision: True` only in this new config file.
6. Do not modify old `avoiding_config.yaml`.

#### E.1.2 Dataset layer (`FM-PCC/d3il/environments/dataset/avoiding_dataset.py`)

1. Keep `Avoiding_Dataset` class untouched.
2. Add new class `Avoiding_Img_Dataset` with same dataset API contract used by existing image datasets:
   - returns image tensors + state + action + mask,
   - supports sequence slicing by `window_size`,
   - keeps batch compatibility with current dataloaders.
3. Input data layout should follow existing d3il vision conventions:
   - camera folder trees,
   - frame index sorting,
   - normalized image tensor conversion.
4. Keep action target definition consistent with avoiding baseline (2D action head behavior in policy output).

#### E.1.3 Simulation layer (`FM-PCC/d3il/simulation/avoiding_sim.py`)

1. Extend constructor with visual-mode flag (default false for backward compatibility).
2. Split observation building into two paths:
   - old state path (default)
   - new visual path when `if_vision=True`
3. Keep environment stepping/output metrics consistent with old avoiding path:
   - success rate,
   - entropy,
   - existing logging style.
4. Keep multiprocessing behavior unchanged unless visual pipeline requires minimal compatibility fixes.

#### E.1.4 Script layer (`FM-PCC/d3il/scripts/avoiding_vision/`)

1. Add one minimal launcher script for visual avoiding benchmark smoke run.
2. Launcher should point to `avoiding_vision_config.yaml` only.
3. Keep script scope minimal (user run only):
   - one seed,
   - low epochs,
   - ~~basic run success criterion (no crash + rollout executes).~~
   - User test/run required (environment-dependent).

#### E.1.5 Data and folder contract

1. Define expected avoiding-vision data root inside vendored d3il tree.
2. Define required subfolders for camera streams and state alignment.
3. Require frame-time alignment between image frames and action/state trajectory indices.
4. If dataset is not present, script must fail with clear missing-path message.

#### E.1.6 Backward-compatibility constraints for d3il upgrade

1. Old avoiding state config remains default and unmodified.
2. Old avoiding dataset class remains import-compatible.
3. Old simulation behavior remains identical when `if_vision=False`.
4. All visual logic is additive and activated only through new visual config/script path.

#### E.1.7 Completion criteria for d3il upgrade block

1. ~~New `avoiding_vision_config.yaml` loads successfully.~~
2. ~~`Avoiding_Img_Dataset` can iterate at least one batch.~~
3. ~~`avoiding_sim.py` runs both state mode and visual mode paths.~~
4. ~~Minimal `d3il/scripts/avoiding_vision/` launcher executes one smoke run.~~
5. ~~No regressions in old avoiding state path.~~
6. User test/run required for all runtime checks above (environment-dependent).

### F) Optional glue (only if needed)

1. `[SHARED-BACKSAFE]` Add small helper for explicit d3il root selection in Gen4 scripts.
2. `[SHARED-BACKSAFE]` Add warning if runtime resolves non-vendored d3il path during Gen4 run.

---

## 9) Compatibility Test Matrix (Must Pass)

### 9.1 Legacy baseline guard

1. ~~Run one old state-based avoiding train smoke (existing entrypoint).~~
2. ~~Run one old state-based avoiding eval smoke (existing entrypoint).~~
3. ~~Verify no import/config regression.~~
4. User test/run required (environment-dependent).

### 9.2 Gen4 smoke guard

1. ~~Gen4 train script starts with Gen4 keys.~~
2. ~~Gen4 eval script loads model and starts avoiding env path.~~
3. ~~Gen4 load_results script prints provenance tags.~~
4. User test/run required (environment-dependent).

### 9.3 d3il additive guard

1. ~~Old `avoiding_config.yaml` still works.~~
2. ~~New `avoiding_vision_config.yaml` loads without Hydra key errors.~~
3. ~~`if_vision=False` path behavior unchanged from old.~~
4. User test/run required (environment-dependent).

---

## 10) Acceptance Criteria (Go for 03 execution)

All are required:
1. Full self-contained FM-PCC repo with vendored d3il present.
2. ~~Old baseline smoke checks pass.~~
3. ~~Gen4 3-script flow (train/eval/load_results) runs end-to-end at smoke scale.~~
4. No rename/refactor needed in old script imports.
5. Execution log in 03 includes ownership labels for every change.
6. Runtime validation items are executed by user in target environment.

---

## 11) Risks and Mitigations

### Risk A: vendored d3il drift from upstream

Mitigation:
1. record copied commit hash,
2. keep local patch list in 03,
3. avoid broad edits.

### Risk B: accidental legacy break

Mitigation:
1. ~~baseline guard tests before and after d3il edits,~~
   user test/run required,
2. additive-only policy,
3. default-off vision switches.

### Risk C: import path ambiguity

Mitigation:
1. Keep Gen4 config bindings only inside Gen4 scripts,
2. keep old script paths untouched.

---

## 12) Execution Order (Strict)

1. Vendor d3il snapshot into FM-PCC.
2. ~~Run legacy baseline pre-check.~~ (User test/run required)
3. Copy and rename FM-PCC Gen4 test/engine folders.
4. Add two new config files in existing `config/` folder.
5. Add vendored d3il avoiding-vision config/dataset/sim script.
6. Wire Gen4 train/eval/load_results scripts to new config filenames.
7. ~~Run legacy baseline post-check.~~ (User test/run required)
8. ~~Run Gen4 smoke check.~~ (User test/run required)
9. Record all steps in 03.

---

## 13) Immediate Next Action

Start Section 12 step 1 first, then continue copy-modify implementation with strict ownership labels.
All runtime tests/runs are to be executed by user in the target environment.
