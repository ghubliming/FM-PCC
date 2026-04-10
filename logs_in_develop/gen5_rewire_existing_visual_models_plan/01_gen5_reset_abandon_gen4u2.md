# 01 Gen5 Reset: Abandon Gen4U2 and Reboot Strategy

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
HUGE WARNING (LOCKED):
Direct upgrade from Action model to Visual-Action model in DPCC/FM-PCC is
NOT accepted as a working path for this stage.

Reason:
1. D3IL Aligning Visual+Action evidence is tied to DDPM-ACT style pipeline.
2. DPCC current path is action-first and likely needs major rebuild for true
	image-conditioned behavior.
3. "Visual" naming without real image-action conditioning is treated as false pass.

Mandatory strategy:
1. use a real existing Visual-Action model path first,
2. inject DPCC concept into that real Visual-Action path,
3. then run Aligning test for proof.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Date: 2026-04-09
Status: Locked
Owner intent: reuse existing D3IL vision stack first, then decide Avoiding visual

---

## 1) Decision

Gen4U2 is abandoned.

Reason:
1. it drifted into partial redesign before fully exploiting existing D3IL vision pipelines,
2. this increases risk and delays proof of concept,
3. team direction is now explicit: align to existing visual models first.
4. direct Action -> Visual-Action upgrade inside DPCC/FM-PCC is now treated as high-risk rebuild path, not incremental path.

Decision lock for Gen5:
1. do not attempt direct Action -> Visual-Action conversion as first implementation step,
2. select one real Visual-Action baseline contract and keep it runnable first,
3. port DPCC concept incrementally into that baseline contract,
4. only after this, validate Aligning behavior and then consider Avoiding extension.

---

## 1.1 Explicit Admission: Wrong Gen4 Avoiding-Vision Block

The following Gen4 execution block is considered wrong in method and must be treated as non-authoritative:
1. direct additive edits into avoiding baseline files as primary strategy,
2. trying to evolve Avoiding vision by patching baseline paths first,
3. mixing baseline and vision evolution concerns in shared files.

Wrong block (admitted):
1. Added dataset class in `d3il/environments/dataset/avoiding_dataset.py` as primary path.
2. Extended `d3il/simulation/avoiding_sim.py` for vision mode as primary path.
3. Extended avoiding env vision return path in `d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py` as primary path.
4. Added `d3il/configs/avoiding_vision_config.yaml` and `d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh` on top of this mixed strategy.

Gen5 correction:
1. follow D3IL pattern using new isolated folders/files first,
2. avoid mutating baseline Avoiding files unless rollback analysis proves necessity.

---

## 1.2 Verified D3IL Findings (Rollback Decision Input)

Evidence sources used for this section:
1. Gen4 execution record: logs_in_develop/gen4_visual_camera_avoiding_d3il_plan/03_gen4_coding_execution_record.md
2. Direct file diff against baseline workspace: /workspaces/d3il vs FM-PCC/d3il for avoiding-related files.

Result summary:
1. Gen5 Part 1 decision is locked to full rollback of Gen4 avoiding touchpoints,
2. source of truth is original `/workspaces/d3il`,
3. objective is a clean baseline entry before any new Gen5 extension.

File-level findings:
1. `d3il/environments/dataset/avoiding_dataset.py`
	- Decision: full revert to original D3IL file.
2. `d3il/simulation/avoiding_sim.py`
	- Decision: full revert to original D3IL file.
3. `d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py`
	- Decision: full revert to original D3IL file.
4. `d3il/configs/avoiding_vision_config.yaml`
	- Decision: full revert to original D3IL file.
5. `d3il/scripts/avoiding_vision/ddpm_encdec_benchmark.sh`
	- Decision: full revert to original D3IL file.

Traceability to Gen4 log 03:
1. Section "5) Implement additive d3il visual-avoiding path" in Gen4 log 03 lists exactly the five avoiding touchpoints reviewed here.
2. Rollback scope is intentionally restricted to those five items to avoid accidental reversion of unrelated Gen4 FM-PCC work.

Rollback policy from findings:
1. full rollback all five Gen4 avoiding touchpoints to `/workspaces/d3il` originals,
2. no partial keep decisions in Part 1,
3. all future avoiding-vision work must restart from isolated Gen5 paths only.

---

## 1.3 Aligning Visual First: Initial Observation

Observation objective:
1. validate one existing visual pipeline first,
2. use that result as entry gate before any avoiding FMv3-align architecture work.

Static verification completed:
1. `run_vision.py` exists and is wired to visual configs,
2. `scripts/aligning_vision/ddpm_encdec_benchmark.sh` exists as executable entry,
3. `configs/aligning_vision_config.yaml` uses image dataset target and `if_vision: True` in train/eval simulation,
4. original avoiding path has no dedicated avoiding-visual config/script entry in baseline d3il.

Execution status:
1. first minimal aligning visual run was prepared,
2. runtime execution was not completed in this turn,
3. therefore current status is "entry verified by structure, runtime verdict pending".

Decision impact:
1. keep "aligning visual first" as hard first action,
2. block avoiding FMv3-align execution until one aligning visual smoke run is recorded.

---

## 1.4 Investigation: What D3IL Visual Actually Conditions On

Input-contract finding (from D3IL code):
1. D3IL vision policy is not image-only.
2. It conditions on `(bp_image, inhand_image, state)`.
3. In aligning path, state is desired robot position (`des_robot_pos`), and action is prediction target, not conditioning input.

FM old avoiding visual-path finding (from archived Gen4 paths):
1. FM visual avoiding dataset route still uses state pkl stream via `SequenceDataset` branch.
2. FM eval policy call uses `conditions={0: obs}` state vector path.
3. Therefore old FM avoiding visual path is effectively state/action-conditioned and does not prove image-conditioned planning.

Practical implication:
1. To "let visual model work" we likely need multiple coordinated changes (dataset, policy-conditioning interface, model-conditioning path, eval runtime obs path).
2. Acceptable first success criterion: even a weak model is acceptable if image tensors are truly consumed and affect outputs.
3. DPCC concept should be integrated into a real Visual-Action backbone, not used to force direct upgrade from action-only path.

---

## 2) New Gen5 Principle

Do this in order:
1. validate one real Visual-Action baseline pipeline is healthy and reproducible,
2. keep this baseline runnable as the control path,
3. integrate DPCC concept incrementally into this baseline,
4. run Aligning test as first proof,
5. then extend to Avoiding visual only if no fundamental flaw is found.

Do not do this:
1. build a new visual stack from scratch before reuse validation,
2. claim visual policy success from state-only fallback behavior.
3. attempt one-step Action -> Visual-Action conversion in DPCC/FM-PCC.

Additional lock:
1. Avoiding visual work must start from isolated Gen5 paths (new config/sim/script/data wrappers),
2. baseline avoiding paths are read-only unless explicitly approved rollback patch.

---

## 3) Existing Assets We Will Reuse

### 3.1 Vision training/eval entrypoint already available

- run_vision.py entrypoint exists and is used by benchmark scripts.

### 3.2 Vision benchmark scripts already available

- scripts/aligning_vision/*.sh
- scripts/sorting_4_vision/*.sh
- scripts/stacking_vision/*.sh

### 3.3 Vision configs already available

- configs/aligning_vision_config.yaml
- configs/sorting_4_vision_config.yaml
- configs/stacking_vision_config.yaml

### 3.4 Avoiding runtime and dataset hooks already partially available

- configs/avoiding_config.yaml (state-oriented baseline)
- simulation/avoiding_sim.py (already has if_vision pathway)
- environments/dataset/avoiding_dataset.py (includes Avoiding_Img_Dataset)

---

## 4) Why Gen4 Code Still Matters

We do not continue Gen4U2, but we reuse lessons and useful parts:
1. FM visual alias handling and path plumbing lessons,
2. existing FM avoiding visual scripts as references for eval reporting and constraints,
3. explicit warning from Gen4: visual name does not guarantee image conditioning.

Reference helpers from FM-PCC side:
1. FM_v3_avoiding_visual_test/train_FM_v3_avoiding_visual.py
2. FM_v3_avoiding_visual_test/eval_FM_v3_avoiding_visual.py
3. flow_matcher_v3_avoiding_visual/datasets/d4rl.py
4. config/avoiding-d3il-visual.py

---

## 5) Stage Gates for Gen5

### Gate A: Existing visual model health check

Pass criteria:
1. at least one existing visual benchmark runs end-to-end with expected outputs,
2. image pipeline is actually used in runtime path,
3. no immediate fundamental flaw (data/obs shape mismatch, dead conditioning, unusable latency).

### Gate B: Multi-task confidence

Pass criteria:
1. two or more existing visual tasks run reproducibly,
2. same visual contract pattern is stable across tasks.

### Gate C: Avoiding extension go/no-go

Go if:
1. Gate A and B pass,
2. no architecture blocker found for applying same visual contract to Avoiding.

No-go if:
1. fundamental flaw appears in reused stack,
2. then fix stack first before Avoiding extension.

### Gate D: Rollback and Isolation Gate

Pass criteria:
1. rollback decision recorded for each wrong Gen4 touchpoint,
2. new Gen5 Avoiding vision execution uses isolated paths first,
3. baseline avoiding path remains reproducible without Gen5 vision toggles.

---

## 6) Fundamental Flaw Definition

Any one of these counts as fundamental flaw:
1. model output is insensitive to image perturbation while marked vision-enabled,
2. runtime silently falls back to state-only in vision mode,
3. dataset-image alignment cannot be made consistent without major redesign,
4. compute cost makes rollout impractical for the target setup.

---

## 7) Immediate Next Step

Execute Gen5-02 implementation plan:
1. lock one real Visual-Action baseline contract as control,
2. define minimal DPCC concept injection points into that contract,
3. run Aligning proof on control then on injected variant,
4. benchmark existing D3IL visual models,
5. extract reusable contracts,
3. perform rollback decision matrix for wrong Gen4 avoiding-vision edits,
4. wire Avoiding visual path by isolated new-folder pattern, not reinvention.
