# 02 Implementation Plan: Gen3v2 ODE Solver Adoption (Strict Rank)

Date: 2026-04-13
Status: Full Detailed Plan (Policy Rewrite)
Depends on: [01_current_status_gen3v2_ode_solver_addon.md](01_current_status_gen3v2_ode_solver_addon.md)

---

## 1) Goal

Adopt an ODE solver for FM-v3 evaluation by strict ranking:

1. Open-source/package solver first.
2. Paid solver options second (with educational-discount/free-license route checked).
3. Build custom solver only as final fallback.

---

## 2) Locked Execution Rank

### Rank-1 (mandatory first): package/open-source

Primary package target:

1. `torchdiffeq` backend integration.

Evidence it is already in workspace ecosystem:

1. [../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py](../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L8)
2. [../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py](../../d3il/agents/models/beso/agents/diffusion_agents/k_diffusion/gc_sampling.py#L492)
3. [../../d3il/install.sh](../../d3il/install.sh#L53)

### Rank-2 (only if rank-1 fails): paid solver route

Evaluate paid options only after rank-1 benchmark failure.

Required check before rejecting rank-2:

1. educational discount/free academic license feasibility,
2. integration effort vs measured gain,
3. runtime and stability on target tasks.

### Rank-3 (last fallback only): custom solver code

Custom solver implementation is forbidden unless rank-1 and rank-2 are both rejected with evidence.

---

## 3) Scope Boundaries

### 3.1 Injection location

Only in FM-v3 sampling path:

1. [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py)

### 3.2 Explicitly not part of this solver adoption

1. projection optimization logic,
2. safety filter optimization logic,
3. policy/environment wrapper logic.

Reason:

- optimization solvers are not ODE integrators for FM rollout.

---

## 4) Rank-1 Implementation (Package-First)

### 4.1 FM-v3 adapter design

Implement a thin backend adapter only:

1. `ode_solver_backend_v3` default `torchdiffeq`
2. `ode_solver_method_v3` (method string supported by selected package)
3. `ode_solver_rtol_v3`, `ode_solver_atol_v3`

Design rule:

- adapter/wrapper code is allowed,
- custom numerical solver algorithm implementation is not phase-1 scope.

### 4.2 File edits

1. [flow_matcher_v3/models/diffusion.py](../../flow_matcher_v3/models/diffusion.py)
    - add backend adapter function calling package solver,
    - keep existing Euler line as legacy fallback mode only.
2. [config/avoiding-d3il.py](../../config/avoiding-d3il.py)
    - add backend and tolerance keys.
3. [FM_v3_test/eval_FM_v3.py](../../FM_v3_test/eval_FM_v3.py)
    - optional pass-through CLI args for backend/method/tolerances.

### 4.3 Backward compatibility

1. Existing experiments without new keys continue current behavior.
2. New package backend is opt-in until validation is complete.

---

## 5) Rank-1 Validation Gate

Rank-1 is accepted if all pass:

1. No regression in success/constraint metrics vs baseline.
2. Runtime acceptable under target budget.
3. Numerical stability acceptable across seeds.

If fail, move to rank-2.

---

## 6) Rank-2 Evaluation Plan (Paid Options)

Only triggered by rank-1 failure.

Evaluation checklist:

1. educational/free academic licensing availability,
2. integration overhead into PyTorch FM path,
3. benchmark gain over rank-1.

Decision rule:

Use paid option only if it provides clear measurable gain that justifies cost and integration complexity.

---

## 7) Rank-3 Fallback Plan (Custom Solver)

Only allowed when both conditions are met:

1. rank-1 package route fails requirements,
2. rank-2 paid route is infeasible or underperforming.

Required approval artifact before coding:

1. written rejection summary for rank-1 and rank-2,
2. custom solver minimum scope and risk analysis.

---

## 8) Benchmark Matrix (Applies to Rank-1 and Rank-2)

Use same seeds/tasks/constraints and compare:

1. baseline Euler (current)
2. package solver configuration A
3. package solver configuration B
4. paid solver candidate (if rank-2 triggered)

Metrics:

1. success rate,
2. violation count/magnitude,
3. trajectory quality metric,
4. mean latency per step,
5. instability/failure frequency.

---

## 9) Risks and Mitigations

1. Risk: package method semantics mismatch with FM time-conditioning.
    - Mitigation: strict adapter unit checks on time domain and tensor shapes.

2. Risk: unfair comparison due to different tolerances.
    - Mitigation: report all tolerance settings and matched runtime budget.

3. Risk: hidden behavior change in legacy runs.
    - Mitigation: keep legacy default path unchanged unless explicit opt-in.

---

## 10) Acceptance Criteria

1. Plan follows strict rank order with evidence gates.
2. No immediate custom solver implementation in phase-1.
3. Package-first backend path is defined and testable.
4. Paid path has explicit trigger and evaluation criteria.
5. Custom solver remains last-resort only.
