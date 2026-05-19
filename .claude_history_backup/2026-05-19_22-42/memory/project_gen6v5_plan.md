---
name: project-gen6v5-plan
description: Gen6V5 plan — replace VisualGaussianDiffusion (diffuser ACT-pattern) with VisualNaiveDiffusion (gc_diffusion naive pattern) inside diffuser_visual_aligning
metadata: 
  node_type: memory
  type: project
  originSessionId: c9ec8b03-7774-4a8a-a163-60517cad8227
---

Gen6V5 stays inside `diffuser_visual_aligning` suite — upgrade in place.

**What changes**: outer DDPM engine only — `VisualGaussianDiffusion` (from diffuser planner/ACT framework) → new `VisualNaiveDiffusion` (modelled on D3IL `gc_diffusion.Diffusion` naive pattern).

**What does NOT change**: VisualUNet, DPCC projector, 9D trajectory contract, dataset, eval wrapper.

**Core design of VisualNaiveDiffusion**:
- Based on `gc_diffusion.Diffusion` (d3il/agents/models/diffusion/gc_diffusion.py)
- Keeps Trainer API: `loss(trajectories, conditions)` — same as current
- Keeps VisualUNet call: `model(x_noisy, cond_dict, t)` — no change to VisualUNet
- Drops: loss_weights (action_weight per-step), returns_condition/CFG, GaussianDiffusion base class overhead
- Keeps: apply_conditioning (obs anchor snap), DPCC projector hook, action-only clamp ±5 (pitfall #4)

**Files**: New `visual_naive_diffusion.py`, update `__init__.py`, update config `diffusion` key, minor train script import change.

**Plan MD**: `logs_in_develop/Gen6_dpcc_Engine_for_visual_aligning/Gen6V5_ML_Bone_Adjust/GEN6V5_PLAN.md`
