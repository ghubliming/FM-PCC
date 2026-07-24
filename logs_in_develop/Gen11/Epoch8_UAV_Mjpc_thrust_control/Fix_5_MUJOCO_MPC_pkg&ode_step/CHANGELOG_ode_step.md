# ODE Step Fix — `flow_steps_v3` Missing from UAV Config

**Discovered:** 2026-06-29
**Scope:** eval-only; no retrain required

---

## Problem

`flow_steps_v3` was **never set** in `config/uav.py` across all commits since Gen11 E6 init.
`FlowMatchingODE.__init__` silently falls back to `n_timesteps` when the parameter is absent:

```python
# flow_matcher_v3_uav/models/diffusion.py:51-53
resolved_flow_steps = flow_steps_v3 if flow_steps_v3 is not None else ode_inference_steps_v3
if resolved_flow_steps is None:
    resolved_flow_steps = n_timesteps   # ← 1000 (default)
```

Result: every UAV eval ran **1000 Euler ODE steps** per FM inference call instead of ~20.
~50× slower than necessary. Results are not invalidated (more steps = more accurate ODE solve,
not wrong); only inference speed was hurt.

See also: `Fix_4/WARNING_1000_ODE_steps.md` for full impact assessment.

---

## Fix

### `config/uav.py` — plan block

```python
# added:
'flow_steps_v3': 20,
```

20 matches the `aligning-d3il-visual` production default (that config uses 16 or 20).
Without this, any eval — regardless of controller — ran 1000 steps.

### `FM_v3_uav_test/eval_fm_uav.py` — `_uav_eval_tag()`

Added `K{flow_steps_v3}` as the first token in the eval output folder name,
matching the aligning design pattern:

```
# before:
mpc4_pid_stopgo_T0.5

# after:
K20_mpc4_pid_stopgo_T0.5
```

This makes the ODE step count visible in the path and distinguishes runs at different K values.

---

## What Did NOT Change

- Model weights, training code, dataset — untouched
- `flow_steps_v3` intentionally NOT added to the train block or `args_to_watch`:
  ODE steps are inference-only; adding to train args would rename checkpoint folders
  and break loading of existing models
