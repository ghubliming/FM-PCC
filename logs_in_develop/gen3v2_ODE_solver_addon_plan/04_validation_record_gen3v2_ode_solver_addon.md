# 04 Expected Results and Usage After Gen3v2 ODE Solver Addon

Date: 2026-04-14
Status: Pre-test expectation and run guide
Depends on: 03_coding_execution_record_gen3v2_ode_solver_addon.md

---

## 1) Baseline for Comparison

Use this baseline for first comparison:
1. `flow_matching_v3_ode_selectable` as the fixed training entry.
2. `plan_fm_v3_ode_selectable` with backend `legacy_euler` and method `euler`.

Reason:
1. This preserves old explicit-Euler behavior inside the new selectable path.
2. Then package-method changes can be measured fairly as opt-in deltas.
3. Solver backend/method is plan-time only, so retraining is not required for solver sweeps.

---

## 2) Training vs Plan Separation (Critical)

Keywords: decoupled control.

Expected:
1. Training behavior is independent from solver backend/method selection.
2. Solver comparisons are done by plan/eval runs against existing checkpoints.
3. No retraining is required when only changing ODE backend/method.

Red flags:
1. Any requirement to retrain only because `ode_solver_backend_v3` or `ode_solver_method_v3` changed.
2. Any solver setting hidden in training-only config path.

---

## 3) Eval Results We Should Expect

Keywords: selectable parity-plus, controlled differences.

Expected:
1. `legacy_euler:euler` should behave close to old FM-v3 reference behavior.
2. `torchdiffeq` methods may shift smoothness/runtime trade-off.
3. Constraint and success metrics should remain in a comparable range unless method is unsuitable.

Not expected:
1. Guaranteed improvement for every method and every seed.
2. Identical runtime between low-order and high-order methods.

Red flags:
1. Consistent success drop for all package methods.
2. Large violation increase with no compensating gain.
3. Unstable behavior only when backend changes.
4. Requested `torchdiffeq` backend runs silently as legacy backend.

---

## 4) Method-Sweep Expectation (Eval)

Recommended first sweep:
1. `legacy_euler:euler` (reference)
2. `torchdiffeq:dopri5`
3. `torchdiffeq:rk4`
4. `torchdiffeq:midpoint`

Theory expectation:
1. Quality may improve or saturate depending on task dynamics.
2. Runtime may increase for more expensive solvers.
3. Best method can be task-specific, not universal.

---

## 5) Labeling Rule for Clean Interpretation

Always log explicit labels per run:
1. `backend=<...>`
2. `method=<...>`
3. `flow_steps_v3=<...>`
4. `ode_inference_steps_v3=<...>`

And always compare with matched:
1. seed list,
2. environment and trial count,
3. projection variant set,
4. same selectable experiment entries only.

---

## 6) Decision Rule After Tests

Interpretation shortcut:
1. If `torchdiffeq` method is within noise band but more stable or better quality, keep it as candidate default for selectable path.
2. If `legacy_euler:euler` is consistently stronger for your task, keep legacy as selectable default.
3. If results are inconsistent, keep legacy baseline and expand sweep before changing default.

---

## 7) Allowed Config Surface (Strict)

Only edit this entry for solver selection:
1. `plan_fm_v3_ode_selectable`

Training entry note:
1. `flow_matching_v3_ode_selectable` does not own solver backend/method selection.

Primary keys:
1. `ode_solver_backend_v3`
2. `ode_solver_method_v3`

Optional keys:
1. `ode_solver_rtol_v3`
2. `ode_solver_atol_v3`
3. `ode_solver_step_size_v3`

Available backend options:
1. `legacy_euler`
2. `torchdiffeq`

Available method options for `torchdiffeq`:
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

---

## 8) How To Use (Run Guide)

### 8.1 Train selectable model

```bash
python FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py --seed 5
```

### 8.2 Run selectable evaluation

```bash
python FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py
```

### 8.3 Load selectable aggregated results

```bash
python FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py
```

### 8.4 Switch backend and method

Edit `config/avoiding-d3il.py` only under:
1. `plan_fm_v3_ode_selectable`

Example package run settings:

```python
'ode_solver_backend_v3': 'torchdiffeq',
'ode_solver_method_v3': 'dopri5',
'ode_solver_rtol_v3': 1e-5,
'ode_solver_atol_v3': 1e-6,
'ode_solver_step_size_v3': None,
```

Example legacy reference settings:

```python
'ode_solver_backend_v3': 'legacy_euler',
'ode_solver_method_v3': 'euler',
```

---

## 9) Practical Notes

1. If backend is set to `torchdiffeq` and package is missing, eval raises a runtime error.
2. `ode_solver_step_size_v3` is only applied for fixed-step methods; other methods ignore it with warning.
3. Eval overrides solver and step controls from plan args after loading checkpoint config.
4. Keep old existing entries unchanged.
5. Treat this addon as an isolated selectable path until validation is complete.
