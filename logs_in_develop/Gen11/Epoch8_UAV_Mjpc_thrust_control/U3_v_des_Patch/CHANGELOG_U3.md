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

**`_run_variant`** — `v_des_magnitude` is now **auto-derived** from the dataset (U3-rev):
```python
# auto-derived: mean(|action|) × DATASET_HZ  ≡  mean(action / dt_fm)
if controller == 'pid_const_v':
    _all_acts  = dataset.actions.reshape(-1, 3)
    _act_norms = np.linalg.norm(_all_acts, axis=-1)
    _valid     = _act_norms > 1e-4          # filter zero-padding
    v_des_magnitude = float(np.mean(_act_norms[_valid])) * DATASET_HZ if _valid.any() else 0.4
else:
    v_des_magnitude = 0.0   # unused
```

Passes `v_des_magnitude=v_des_magnitude` into `rollout_one`.

**`rollout_one` signature** — default changed from `v_des_magnitude=0.4` → `v_des_magnitude=0.0`
(always set by `_run_variant`; 0.0 default makes stale callers obvious).

---

### `config/uav.py`

**Train block controller comment** — added `pid_const_v` entry:
```python
#   controller='pid_const_v'→ cascaded PID, v_des=unit(action)*v_des_magnitude (U3, constant speed).
```

**Plan block** — `v_des_magnitude` key **removed**. Speed is auto-derived from dataset; no manual knob.
Comment updated to document the derivation formula.

---

## Why Auto-Derive?

`v_des_magnitude = 0.4` was the correct value only because the dataset generator uses 0.4 m/s nominal speed.
But it obscures the coupling: any change in dataset speed → wrong hardcoded value → silently wrong eval.

The derivation `mean(|action|) × DATASET_HZ` is the algebraic mean of `action/dt_fm` — exactly what the
default `pid` controller produces. `pid_const_v` at this speed is dataset-consistent with no magic numbers.

---

## What Did NOT Change

- FM model architecture, training code, dataset, normalizer — untouched
- `pid` and `pid_stopgo` branches — unchanged

---

## Usage

```python
# config/uav.py plan block — set only:
'controller': 'pid_const_v',
# v_des_magnitude is NOT set — derived automatically from the dataset at eval time
```

Keep train block `controller='pid'` so the checkpoint path resolves to the existing model.

---

## 4 Controller Summary

| `controller=` | `v_des` formula | Continuous flight? |
|---|---|---|
| `pid` | `action / dt_fm` | Yes (timing-sensitive) |
| `pid_stopgo` | `zeros(3)` | No — strict brake-to-zero |
| `pid_const_v` | `unit(action) * v_des_magnitude` | Yes (timing-free) |
| `mjpc` | N/A (MJPC internal) | Yes (MPC horizon) |
