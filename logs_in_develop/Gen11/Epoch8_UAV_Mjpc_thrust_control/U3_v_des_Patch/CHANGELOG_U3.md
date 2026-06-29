# U3 Changelog — `pid_const_v` Controller

**Branch:** `update_into_FM`
**Scope:** eval-only; no retrain required

---

## Files Changed

### `FM_v3_uav_test/eval_fm_uav.py`

**`rollout_one` signature** — added `v_des_magnitude=0.4` parameter.

**Tracker comment** — updated to document all 4 controller options.

**`v_des` branch** — expanded from 2-way (pid_stopgo / else) to 3-way:
```python
# before (U2):
v_des = np.zeros(3) if controller == 'pid_stopgo' else action / dt_fm

# after (U3):
if controller == 'pid_stopgo':
    v_des = np.zeros(3)
elif controller == 'pid_const_v':
    norm  = float(np.linalg.norm(action))
    v_des = (action / norm) * v_des_magnitude if norm > 1e-6 else np.zeros(3)
else:   # 'pid' default (and 'mjpc' — which ignores v_des)
    v_des = action / dt_fm
```

**`_run_variant`** — added:
```python
v_des_magnitude = float(config.get('v_des_magnitude', 0.4))
```
Passes `v_des_magnitude=v_des_magnitude` into `rollout_one`.

---

### `config/uav.py`

**Train block controller comment** — added `pid_const_v` entry:
```python
#   controller='pid_const_v'→ cascaded PID, v_des=unit(action)*v_des_magnitude (U3, constant speed).
```

**Plan block** — added config key:
```python
'v_des_magnitude': 0.4,   # U3 pid_const_v only: constant flight speed (m/s)
```
Also updated controller comment with all 4 options.

---

## What Did NOT Change

- FM model architecture, training code, dataset, normalizer — untouched
- `pid` and `pid_stopgo` branches — unchanged
- `_uav_exp_name`: `pid_const_v` → `_ctrlpid_const_v` suffix (non-default → new output folder, same checkpoint)

---

## Usage

```python
# config/uav.py plan block — change these two keys:
'controller':      'pid_const_v',
'v_des_magnitude': 0.4,     # tune: lower → smoother, higher → faster
```

Keep train block `controller='pid'` so the checkpoint path resolves to the existing `pid` model.

---

## 4 Controller Summary

| `controller=` | `v_des` formula | Continuous flight? |
|---|---|---|
| `pid` | `action / dt_fm` | Yes (timing-sensitive) |
| `pid_stopgo` | `zeros(3)` | No — strict brake-to-zero |
| `pid_const_v` | `unit(action) * v_des_magnitude` | Yes (timing-free) |
| `mjpc` | N/A (MJPC internal) | Yes (MPC horizon) |
