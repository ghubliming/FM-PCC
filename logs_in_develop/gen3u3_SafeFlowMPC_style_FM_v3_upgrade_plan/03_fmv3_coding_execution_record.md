# 03 FM-v3 Coding Execution Record

Date: 2026-04-06
Status: Coding Applied
Depends on: 00_evaluation_fmv2_to_fmv3_with_safeflowmpc.md, 01_implementation_plan_fmv3_safeflowmpc_engine_in_fmpcc.md

---

## 1) Scope Rule

This document records coding execution only.
Verification phase content is intentionally out of scope for this file.

---

## 2) Executed Coding Tasks

1. Created flow_matcher_v3 by copying flow_matcher_v2.
2. Created FM_v3_test scripts by copying FM_v2_test and renaming entrypoints.
3. Added flow_matching_v3 and plan_fm_v3 blocks in config/avoiding-d3il.py.
4. Switched FM_v3_test imports from flow_matcher_v2 to flow_matcher_v3.
5. Switched FM_v3_test experiment keys:
   - train -> flow_matching_v3
   - eval/load -> plan_fm_v3
6. Updated v3 training script diffusion args to v3 names:
   - time_beta_alpha_v3
   - time_beta_beta_v3
   - flow_steps_v3
   - ode_inference_steps_v3
7. Added explicit v3 loader label output:
   - Eval ODE=<value>, FlowSteps=<value>, Beta=(alpha,beta)
8. Updated flow_matcher_v3/models/diffusion.py to SafeFlow-style continuous-time model query semantics.
9. Added flow_steps_v3 primary control and ode_inference_steps_v3 compatibility alias in v3 diffusion.
10. Updated flow_matcher_v3/models/unet1d_temporal_cond.py to ensure float time handling in forward path.

---

## 3) File-Level Change Map

### New directories and files

1. flow_matcher_v3/
2. FM_v3_test/train_FM_v3.py
3. FM_v3_test/eval_FM_v3.py
4. FM_v3_test/load_results_FM_v3.py

### Edited files

1. config/avoiding-d3il.py
2. flow_matcher_v3/models/diffusion.py
3. flow_matcher_v3/models/unet1d_temporal_cond.py
4. FM_v3_test/train_FM_v3.py
5. FM_v3_test/eval_FM_v3.py
6. FM_v3_test/load_results_FM_v3.py

---

## 4) FM-v3 Parameter Contract

Training-time parameters:
1. time_beta_alpha_v3
2. time_beta_beta_v3

Inference-time step parameters:
1. flow_steps_v3 (primary)
2. ode_inference_steps_v3 (compatibility alias)

Naming intent:
1. flow_steps_v3 is the canonical v3 rollout-step knob.
2. ode_inference_steps_v3 is maintained only for compatibility with existing calling patterns.

---

## 5) Mathematical Behavior in v3 Engine

Training objective family:
1. Remains FM velocity matching.
2. Beta-time sampling is retained using v3-named parameters.

Model-time query behavior:
1. v3 path uses continuous time values in [0, 1] directly in model query path.

Inference rollout behavior:
1. Euler flow rollout remains explicit with:
   - N = flow_steps_v3
   - dt = 1/N
   - t_i = i/N
   - x_{i+1} = x_i + v_theta(x_i, t_i) * dt

---

## 6) Non-Goals Kept Intact

1. No backbone redesign.
2. No change to FM objective class.
3. No rollout-backprop training objective.
4. No edits to existing v2 path behavior.
5. No broad refactor beyond v3 copy path.

---

## 7) Immediate Next Coding Action

Run FM-v3 train/eval/load entrypoints with the new config keys and collect generated outputs under flow_matching_v3 and plans/flow_matching_v3 directories.
