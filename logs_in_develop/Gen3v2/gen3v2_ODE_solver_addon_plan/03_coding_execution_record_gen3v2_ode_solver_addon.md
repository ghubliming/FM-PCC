# 03 Coding Execution Record: Gen3v2 ODE Solver Addon

Date: 2026-04-14
Status: Coding Applied (Scope-Corrected)
Depends on: 01_current_status_gen3v2_ode_solver_addon.md, 02_implementation_plan_gen3v2_ode_solver_addon.md

---

## 1) Scope Rule

This file records coding execution only.
Validation outcomes and metric interpretation are documented in 04.

---

## 2) Executed Work

1. Copied FM-v3 test folder:
   - `FM_v3_test` -> `FM_v3_ode_selectable_test`
2. Copied FM-v3 engine folder:
   - `flow_matcher_v3` -> `flow_matcher_v3_ode_selectable`
3. Updated copied test entry scripts to use selectable experiment keys.
4. Updated copied diffusion engine to support backend/method selection:
   - legacy explicit Euler path
   - package path (`torchdiffeq`) when enabled
   - runtime error if backend is `torchdiffeq` but package is not installed
   - `step_size` option is applied only for fixed-step methods
5. Added two new selectable config entries:
   - `flow_matching_v3_ode_selectable`
   - `plan_fm_v3_ode_selectable`
6. Applied strict runtime override in eval entry:
   - solver and step controls are overwritten from plan args after checkpoint load
7. Removed solver keys from selectable training config path:
   - solver selection is plan-time only

---

## 3) Files Added or Updated for Gen3v2

Copied+updated test path:
1. `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py`
2. `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`
3. `FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py`

Copied+updated engine path:
1. `flow_matcher_v3_ode_selectable/models/diffusion.py`

Config registry:
1. `config/avoiding-d3il.py`

---

## 4) Final Config Contract (Strict)

ODE solver selection keys are plan-time only and live in:

1. `plan_fm_v3_ode_selectable`

`flow_matching_v3_ode_selectable` remains for training path setup but does not carry ODE backend/method keys.

Keys used:
1. `ode_solver_backend_v3`
2. `ode_solver_method_v3`
3. `ode_solver_rtol_v3`
4. `ode_solver_atol_v3`
5. `ode_solver_step_size_v3`

Documented selectable values:
1. backend: `legacy_euler`, `torchdiffeq`
2. torchdiffeq methods: `dopri8`, `dopri5`, `bosh3`, `fehlberg2`, `adaptive_heun`, `euler`, `midpoint`, `heun2`, `heun3`, `rk4`, `explicit_adams`, `implicit_adams`, `fixed_adams`, `scipy_solver`

---

## 5) Guardrail Compliance

1. Original existing entries (`flow_matching_v3`, `plan_fm_v3`) are kept without the new ODE-selection key additions.
2. ODE option documentation and active ODE-selection keys are restricted to the new selectable plan entry (`plan_fm_v3_ode_selectable`).
3. No CLI method-selection interface was introduced.

---

## 6) Output State

Gen3v2 addon is implemented as a separate selectable path:
1. copied test entrypoint path,
2. copied diffusion engine path,
3. plan-only solver selection contract,
4. eval-time override to guarantee plan config takes effect.
