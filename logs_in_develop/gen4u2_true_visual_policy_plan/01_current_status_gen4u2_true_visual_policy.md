# 01 Current Status: Gen4U2 True Visual Policy Transition

Date: 2026-04-09
Status: Locked Baseline for Review
Scope: FM-PCC Gen4U2 (true visual policy path), with DPCC compatibility awareness

---

## 1) Objective

Define the verified starting point for Gen4U2.

Gen4U2 target is no longer visual-infrastructure-only. It is:
1. image-backed data usage,
2. image-conditioned policy inference,
3. explicit separation from state-only fallback behavior,
4. preserved backward compatibility for existing state baselines.

---

## 2) Investigation Summary (Locked)

Based on local test and code tracing:
1. visual train/eval scripts run successfully even when camera image folders are missing,
2. this means current visual path is not yet truly image-conditioned,
3. current path validates config/route plumbing only.

Practical interpretation:
1. current Gen4 visual branch is operational,
2. but scientific claim level is visual-infrastructure-only,
3. true visual policy requirements remain unmet.

---

## 3) Verified Findings

### 3.1 Dataset loader behavior

In the visual flow-matcher dataset loader, avoiding visual env names are routed to the same state extraction path:
1. parse `env_state` pkl,
2. read `des_c_pos` and `c_pos`,
3. concatenate into low-dimensional state,
4. never load image files.

Result:
- no dependency on `images/bp-cam` or `images/inhand-cam` at training/eval time for current FM visual scripts.

### 3.2 Model input contract behavior

Current model path is a temporal 1D U-Net trajectory model expecting low-dimensional numeric condition vectors.

Observed contract pattern:
1. `cond_dim` aligned to state vector (4),
2. transition/action path remains numeric sequence modeling,
3. no native image encoder in the forward path.

Result:
- model cannot consume raw camera tensors in current wiring.

### 3.3 Eval rollout behavior

Current visual eval logic still uses state-concat style condition construction.

Result:
1. rollout can succeed with no images present,
2. because planner remains state-driven.

---

## 4) Root Cause Statement

Root cause is architectural/wiring mismatch, not data corruption:
1. visual environment label exists,
2. but data loader and planner path still implement state-conditioned pipeline,
3. therefore missing image assets do not break runtime.

---

## 5) Current Claim Level (Must Be Explicit)

Allowed claim right now:
1. Gen4 visual routing/config path works.
2. aliasing and script entrypoints are functional.

Not yet allowed claim:
1. true visual policy learning,
2. image-conditioned decision-making,
3. visual robustness gains attributable to camera inputs.

---

## 6) Gen4U2 Direction (Pre-02)

Gen4U2 is defined as the transition from visual-infrastructure-only to true visual policy.

Required high-level shift:
1. data path must load camera images,
2. model path must include visual encoder + fusion into policy condition path,
3. eval path must feed image-conditioned inputs end-to-end,
4. metrics must include visual stress validation.

---

## 7) Constraints and Compatibility Rules

1. Keep old state baselines runnable and unchanged by default.
2. Add true-visual path in isolated Gen4U2 keys/folders.
3. Any shared edits must be backward-safe and explicitly labeled.
4. No claim inflation: image conditioning must be verifiable in code path and ablation.

---

## 8) Evidence Requirements Before 02 Is Approved

02 implementation plan must include explicit file-level answers for:
1. which loader reads image files and batch layout,
2. where visual encoder is inserted,
3. how fusion enters the denoiser/flow module,
4. how eval passes image tensors into policy call,
5. how fallback to state-only is controlled,
6. what tests prove true visual conditioning.

---

## 9) Acceptance Gate for Moving to 02

Proceed to 02 only if this 01 is accepted as the locked baseline statement:
1. current branch is state-dominant despite visual naming,
2. Gen4U2 must implement real image-conditioned learning and inference,
3. backward compatibility must remain intact.

---

## 10) Review Prompt

Reviewer decision request:
1. confirm this diagnosis baseline,
2. confirm Gen4U2 scope boundaries,
3. then approve drafting `02_implementation_plan_gen4u2_true_visual_policy.md`.
