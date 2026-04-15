# 06 ODE Usage Audit Memo: Gen3v2 Scope Check

Date: 2026-04-14
Status: Completed audit
Depends on: 03_coding_execution_record_gen3v2_ode_solver_addon.md, 04_validation_record_gen3v2_ode_solver_addon.md

---

## 1) Question Answered

Does gen3v2 change only FM vector-field (VF) ODE integration behavior, and not unrelated projection/optimization ODE-like paths?

Short answer:
1. Yes for core gen3v2 model path: changes are in FM sampling ODE integrator behavior (legacy Euler vs torchdiffeq-based integration of VF).
2. Projection optimization path remains separate and is not replaced by gen3v2 ODE solver changes.

---

## 2) Core Evidence: Baseline v3 vs Gen3v2 Diffusion

Baseline FM-v3 integrator path:
1. [flow_matcher_v3/models/diffusion.py](../../../flow_matcher_v3/models/diffusion.py#L168)
2. Uses explicit Euler-style step through VF updates in sampling loop.

Gen3v2 selectable integrator path:
1. [flow_matcher_v3_ode_selectable/models/diffusion.py](../../../flow_matcher_v3_ode_selectable/models/diffusion.py#L190)
2. Adds backend switch:
   1. legacy_euler path
   2. torchdiffeq path
3. Adds method/tolerance/step-size controls for ODE integration of VF:
   1. [flow_matcher_v3_ode_selectable/models/diffusion.py](../../../flow_matcher_v3_ode_selectable/models/diffusion.py#L217)
   2. [flow_matcher_v3_ode_selectable/models/diffusion.py](../../../flow_matcher_v3_ode_selectable/models/diffusion.py#L238)
4. Fails fast if torchdiffeq backend requested but package missing:
   1. [flow_matcher_v3_ode_selectable/models/diffusion.py](../../../flow_matcher_v3_ode_selectable/models/diffusion.py#L191)

Conclusion from direct file diff:
1. The behavioral delta is ODE integration implementation inside FM sampling (VF integration path).
2. This is the intended gen3v2 scope.

---

## 3) Config Scope Check

Plan-side solver control is present in:
1. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py#L503)

Solver keys in plan entry:
1. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py#L527)
2. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py#L532)
3. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py#L533)
4. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py#L534)
5. [config/avoiding-d3il.py](../../../config/avoiding-d3il.py#L535)

Eval applies runtime override after checkpoint load:
1. [FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py](../../../FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py#L55)
2. [FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py](../../../FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py#L57)

Conclusion:
1. Solver behavior is controlled at plan/eval runtime and maps to VF ODE integrator settings.

---

## 4) Other ODE-Like Usage Found (Not Gen3v2 Core VF Integrator)

Projection/optimization modules use constrained optimization (SLSQP), not FM ODE integrator replacement:
1. [flow_matcher_v3_ode_selectable/sampling/projection.py](../../../flow_matcher_v3_ode_selectable/sampling/projection.py#L3)
2. [flow_matcher_v3_ode_selectable/sampling/projection.py](../../../flow_matcher_v3_ode_selectable/sampling/projection.py#L135)

Other project trees with torchdiffeq exist, but are separate stacks (not gen3v2 FM-v3 selectable core):
1. [d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py](../../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L8)
2. [d3il/agents/models/beso/agents/lmp_agents/lmp_modules/k_diffusion/plan_gc_sampling.py](../../../d3il/agents/models/beso/agents/lmp_agents/lmp_modules/k_diffusion/plan_gc_sampling.py#L8)

Conclusion:
1. There are other ODE/optimizer usages in repo, but they are separate systems.
2. Gen3v2 FM selectable change is isolated to its own VF integrator path.

---

## 5) Final Verdict

1. Gen3v2 core code change target is correct: VF ODE integration behavior in FM sampling.
2. No evidence that gen3v2 replaced projection optimizer logic with a different ODE solver.
3. Projection remains an independent optimization layer.
4. Current benchmark script can run VF-focused mode to stay inside this scope.

---

## 6) Recommended Scope-Safe Benchmark Mode

Use VF-only benchmark mode to evaluate only VF ODE integration behavior:

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --benchmark-mode vf_only \
  --n-trials 50 \
  --vf-batch-size 16 \
  --flow-steps 10 \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint \
  --plot
```
