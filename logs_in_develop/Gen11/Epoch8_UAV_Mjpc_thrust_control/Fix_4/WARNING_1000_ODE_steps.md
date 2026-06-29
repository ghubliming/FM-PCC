# WARNING — All UAV Evals Before U3-rev Used 1000 ODE Steps

**Discovered:** 2026-06-29 (during Fix_5/bf Fix5 run)
**Fixed in:** `config/uav.py` plan block, commit after Fix_5 bundle

---

## What Happened

`flow_steps_v3` was **never set** in `config/uav.py` across all 15 commits from Gen11 E6 init
through Fix_4b. `FlowMatchingODE.__init__` falls back to `n_timesteps` when both
`flow_steps_v3` and `ode_inference_steps_v3` are `None`:

```python
# flow_matcher_v3_uav/models/diffusion.py:51-53
resolved_flow_steps = flow_steps_v3 if flow_steps_v3 is not None else ode_inference_steps_v3
if resolved_flow_steps is None:
    resolved_flow_steps = n_timesteps   # ← falls back to 1000
```

`n_timesteps` defaults to `1000` in the constructor and was not set in the UAV config either.
Result: **every UAV eval ran 1000 Euler ODE steps per FM inference call** instead of the
intended ~20.

## Impact on Results

**Results are NOT invalidated.** 1000 steps = more accurate ODE integration, not fewer.
The trajectories are correct; only inference speed was hurt:

- ~50× slower per FM call than K=20
- Policy quality conclusions (e.g. 0% pillars success) are real — not ODE artifacts

## Fix

Added to `config/uav.py` plan block:
```python
'flow_steps_v3': 20,   # aligning-d3il-visual default; prevents fallback to n_timesteps=1000
```

Also added `K{flow_steps_v3}` as the first token in `_uav_eval_tag` in `eval_fm_uav.py`,
so future eval output folders read `K20_mpc4_pid_stopgo_T0.5` — matching the aligning pattern
and making the ODE step count visible in the path.

## Why aligning-d3il Did Not Have This Bug

`aligning-d3il-visual.py` always included `flow_steps_v3` in `args_to_watch` and explicitly
set it in every plan block (e.g. `'flow_steps_v3': 16`). The UAV config was written from
scratch and this parameter was overlooked.
