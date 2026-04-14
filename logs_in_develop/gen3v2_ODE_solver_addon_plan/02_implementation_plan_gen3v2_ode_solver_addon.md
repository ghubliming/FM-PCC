# 02 Implementation Plan: Gen3v2 ODE Solver Adoption (Strict Rank)

Date: 2026-04-14
Status: Execution Plan (New Locked Version)
Depends on: [01_current_status_gen3v2_ode_solver_addon.md](01_current_status_gen3v2_ode_solver_addon.md)

---

## 1) Goal

Adopt ODE method selection for FM-v3 evaluation with strict ranking and strict backward compatibility:

1. Open-source/package solver first.
2. Paid solver options second (with educational-discount/free-license route checked).
3. Build custom solver only as final fallback.

---

## 1.1 Locked 3-Step Development Rule

Development must follow exactly these 3 steps:

1. Copy the full folder [FM_v3_test](../../FM_v3_test) to a new folder and work only in the copied folder.
2. Copy the full folder [flow_matcher_v3](../../flow_matcher_v3) to a new folder and work only in the copied folder.
3. Inject 2 new config parameters in [config/avoiding-d3il.py](../../config/avoiding-d3il.py).

Locked naming:

1. Use `flow_matching_v3_ode_selectable` naming style (same logic as core `flow_matching_v3`).

Locked copied-folder names:

1. `FM_v3_test` copy name: `FM_v3_ode_selectable_test`
2. `flow_matcher_v3` copy name: `flow_matcher_v3_ode_selectable`

Mandatory 2 parameters only:

1. `ode_solver_backend_v3` default `legacy_euler`
2. `ode_solver_method_v3` default `euler`

No CLI method selection in this phase.

Hard rule:

1. Do not modify original folders in this phase.
2. Modify copied folders only.

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

Only in copied FM-v3 sampling path:

1. [flow_matcher_v3_ode_selectable/models/diffusion.py](../../flow_matcher_v3_ode_selectable/models/diffusion.py)

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

1. `ode_solver_backend_v3` default `legacy_euler`
2. `ode_solver_method_v3` default `euler` (effective when backend is package mode)
3. Optional later only: `ode_solver_rtol_v3`, `ode_solver_atol_v3`, `ode_solver_step_size_v3`

Config source of truth:

1. Method selection is in [config/avoiding-d3il.py](../../config/avoiding-d3il.py) only.
2. No CLI override path.
3. Missing keys must keep original explicit Euler behavior.

Design rule:

- adapter/wrapper code is allowed,
- custom numerical solver algorithm implementation is not phase-1 scope.

### 4.1.1 Injection behavior in rollout code

Injection location:

1. [flow_matcher_v3_ode_selectable/models/diffusion.py](../../flow_matcher_v3_ode_selectable/models/diffusion.py) at the existing Euler rollout block.

Branch behavior:

1. Read `ode_solver_backend_v3` and `ode_solver_method_v3` from config-driven args.
2. If backend is `legacy_euler` (or key missing), execute the current explicit Euler line unchanged.
3. If backend is package backend (rank-1), execute package ODE stepping using selected method.
4. Keep tensor shape, dtype, and device consistent with current path.

Default guarantee:

1. Legacy Euler remains default path.
2. Alternative methods are opt-in only by config values.

### 4.1.2 Available method names for package backend

For `torchdiffeq`, method name can be one of:

1. `dopri8`
2. `dopri5`
3. `bosh3`
4. `fehlberg2`
5. `adaptive_heun`
6. `euler`
7. `midpoint`
8. `heun2`
9. `heun3`
10. `rk4`
11. `explicit_adams`
12. `implicit_adams`
13. `fixed_adams`
14. `scipy_solver`

### 4.2 File edits

1. [flow_matcher_v3_ode_selectable/models/diffusion.py](../../flow_matcher_v3_ode_selectable/models/diffusion.py)
    - add config-branch wrapper around current rollout update,
    - keep current explicit Euler update as default `legacy_euler`,
    - add package call path for opt-in backend.
2. [config/avoiding-d3il.py](../../config/avoiding-d3il.py)
    - inject 2 mandatory keys (`ode_solver_backend_v3`, `ode_solver_method_v3`) in v3 blocks,
    - create/select variant blocks with core-style naming:
      - `flow_matching_v3_ode_selectable`
      - `plan_fm_v3_ode_selectable`
3. [FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py](../../FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py)
    - parser experiment should use `flow_matching_v3_ode_selectable`.
4. [FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py](../../FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py)
    - parser experiment should use `plan_fm_v3_ode_selectable`.
5. [FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py](../../FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py)
    - parser experiment should use `plan_fm_v3_ode_selectable`.

Execution order is fixed:

1. Copy the two folders with locked names.
2. Modify copied folders only.
3. Add new config entries in [config/avoiding-d3il.py](../../config/avoiding-d3il.py).

Disallowed in this phase:

1. CLI method flags.
2. Replacing default Euler behavior.

### 4.3 Backward compatibility

1. Existing experiments without new keys continue current behavior.
2. New package backend is opt-in until validation is complete.
3. Default path remains original explicit Euler rollout.
4. Config-only switching prevents accidental runtime drift.

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
2. `torchdiffeq:dopri5`
3. `torchdiffeq:rk4`
4. `torchdiffeq:midpoint`
5. paid solver candidate (if rank-2 triggered)

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
