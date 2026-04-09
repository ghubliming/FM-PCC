# 01 Gen5 Reset: Abandon Gen4U2 and Reboot Strategy

Date: 2026-04-09
Status: Locked
Owner intent: reuse existing D3IL vision stack first, then decide Avoiding visual

---

## 1) Decision

Gen4U2 is abandoned.

Reason:
1. it drifted into partial redesign before fully exploiting existing D3IL vision pipelines,
2. this increases risk and delays proof of concept,
3. team direction is now explicit: rewire to existing visual models first.

---

## 2) New Gen5 Principle

Do this in order:
1. validate existing D3IL visual pipelines are healthy and reproducible,
2. reuse their contracts and runtime flow,
3. then extend to Avoiding visual only if no fundamental flaw is found.

Do not do this:
1. build a new visual stack from scratch before reuse validation,
2. claim visual policy success from state-only fallback behavior.

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
1. benchmark existing D3IL visual models,
2. extract reusable contracts,
3. wire Avoiding visual path by extension, not reinvention.
