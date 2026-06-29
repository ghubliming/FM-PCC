# U3 PLAN — `pid_const_v`: Constant Speed `v_des`

**Status:** IMPLEMENTED
**Parent:** [E8 PLAN](../PLAN_MJPC_Thrust_Control.md) · [Obs/Action loop design](../DESIGN_obs_action_loop.md)

---

## Problem with Current `pid` `v_des`

```python
v_des = action / dt_fm    # dt_fm = 1/33 s, DATASET_HZ module constant
```

- **Magnitude varies** every FM step — small `action` → slow `v_des`; large `action` → fast `v_des`
- **Timing-sensitive** — assumes exactly `dt_fm` between FM calls; jitter makes it wrong
- **Inconsistent with expert collection** — expert used smooth analytic `v_des` from `traj_fn`; eval uses finite-difference approximation
- **Locked to dataset rate** — `DATASET_HZ=33` is a hardcoded constant, not config-driven

---

## New Option: `controller='pid_const_v'`

`v_des` is a **constant speed** in the direction of the FM action:

```python
norm  = ||action||
v_des = (action / norm) * v_des_magnitude    if norm > 1e-6
      = zeros(3)                              otherwise (hovering)
```

`v_des_magnitude` is a config key (default `0.4` m/s — matches expert trajectory average speed).

Properties:
- **Timing-free** — magnitude is fixed, no division by `dt_fm`
- **Directional** — always points toward the next FM waypoint
- **Tunable** — change `v_des_magnitude` in config without retraining
- **No stop-and-go** — PID sees `v_des > 0` along the flight direction → does not brake

---

## All 4 Controller Options

| `controller=` | `v_des` | Continuous? | Notes |
|---|---|---|---|
| `pid` | `action / dt_fm` | Yes | E7 default; timing-sensitive |
| `pid_stopgo` | `0` | No — strict stop-and-go | U2 |
| `pid_const_v` | `unit(action) * v_des_magnitude` | Yes | U3 — timing-free, tunable |
| `mjpc` | ignored (MJPC internal) | Yes (MPC horizon) | E8 original; cluster-only |

---

## Code Changes

### `FM_v3_uav_test/eval_fm_uav.py`

**`rollout_one` signature** — added `v_des_magnitude=0.4`

**`v_des` branch** — expanded to 3-way:
```python
if controller == 'pid_stopgo':
    v_des = np.zeros(3)
elif controller == 'pid_const_v':
    norm  = float(np.linalg.norm(action))
    v_des = (action / norm) * v_des_magnitude if norm > 1e-6 else np.zeros(3)
else:   # 'pid' default
    v_des = action / dt_fm
```

**`_run_variant`** — reads `v_des_magnitude` from config; passes to `rollout_one`.

### `config/uav.py`

Plan block: added `'v_des_magnitude': 0.4` + updated controller comment.

---

## Retrain Required?

**No.** `pid_const_v` is eval-only — only `v_des` changes. Same checkpoint as `pid` with the same `cond_mode`. Set `controller='pid_const_v'` in the **plan block only**; keep train block as `'pid'` so the checkpoint path resolves to the same trained model.

---

## Tuning `v_des_magnitude`

The expert dataset was collected at ~0.4 m/s average (from `traj_fn` speed range). Start at `0.4`. If the drone overshoots waypoints → reduce. If it's sluggish → increase. This is the only knob.
