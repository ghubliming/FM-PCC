# U4 Test Run Guide

**As of**: fix_3 (2026-06-28)
**Config files**: `config/uav.py` (train + plan blocks), `config/uav_projection.yaml` (projection + seed/n_trials)

---

## One-time setup before any U4 test

Adding `cond_mode` to `args_to_watch` renames every checkpoint folder to include `_condp_des`.
Pre-U4 checkpoints do NOT have this suffix — the eval will fail to find them.
**Rename once, no retrain needed:**

```bash
# Example for corridor seed 6 (repeat for each scene/seed you have)
mv logs/UAV_FM/uav-corridor/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE \
   logs/UAV_FM/uav-corridor/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condp_des

# For all-scene pooled checkpoint
mv logs/UAV_FM/uav-all/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE \
   logs/UAV_FM/uav-all/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condp_des
```

---

## Phase 0 — Re-anchor (NO retrain, uses existing checkpoint)

`cond_mode = 'p_des'` (default, unchanged). Only `reanchor_alpha` in the plan block changes.
Edit **`config/uav.py`** → `plan_flow_matching_v3_uav` → `'reanchor_alpha': <value>`.

### Test 0a — Baseline (alpha = 0.0, current behavior)

```python
# config/uav.py  plan_flow_matching_v3_uav
'reanchor_alpha': 0.0,
```

```bash
python FM_v3_uav_test/eval_fm_uav.py --scene corridor
```

Console will confirm:
```
[ eval ] seed=6      (source: config/uav_projection.yaml)
[ eval ] n_trials=20 (source: config/uav_projection.yaml)
```

Results → `logs/UAV_FM/uav-corridor/plans/.../diffuser/`  (no `_reanchor` tag = alpha=0.0)

---

### Test 0b — Re-anchor alpha = 0.5 (partial grounding)

```python
# config/uav.py  plan_flow_matching_v3_uav
'reanchor_alpha': 0.5,
```

```bash
python FM_v3_uav_test/eval_fm_uav.py --scene corridor
```

Results → `logs/UAV_FM/uav-corridor/plans/.../<variant>_reanchor0.5/`

---

### Test 0c — Re-anchor alpha = 1.0 (hard reset every step)

```python
# config/uav.py  plan_flow_matching_v3_uav
'reanchor_alpha': 1.0,
```

```bash
python FM_v3_uav_test/eval_fm_uav.py --scene corridor
```

Results → `logs/UAV_FM/uav-corridor/plans/.../<variant>_reanchor1.0/`

---

**Compare 0a/0b/0c:** look at `track_err_mean`, `success_rate`, `total_over_budget` in `results.json`.
Folders are distinct (no overwrite). If 1.0 stops the OOD spiral → phase 1 is worth the retrain.

---

## Phase 1 — Plan-in-real-position (retrain required)

`cond_mode = 'real_p'` → obs changes to `[p|v]` (6D), action = `Δp`. Needs fresh model.

### Step 1 — Switch cond_mode in the TRAINING block

Edit **`config/uav.py`** → `flow_matching_v3_uav` (training block):

```python
'cond_mode': 'real_p',   # was: 'p_des'
```

Also update the plan block to match (for eval later):

```python
# plan_flow_matching_v3_uav
'cond_mode': 'real_p',
'lead_gain': 1.0,        # start with 1.0; increase if drone under-reaches goals
```

### Step 2 — Retrain

```bash
python FM_v3_uav_test/train_fm_uav.py --scene corridor --seed 6
```

Checkpoint saves to: `logs/UAV_FM/uav-corridor/flow_matching_v3_uav/H8_Dmodels.diffusion.FlowMatchingODE_condreal_p/6/`
(fully isolated from the `p_des` checkpoint — no collision)

### Step 3 — Eval (lead_gain = 1.0 first)

```python
# config/uav.py  plan_flow_matching_v3_uav
'lead_gain': 1.0,
```

```bash
python FM_v3_uav_test/eval_fm_uav.py --scene corridor
```

Results → `logs/UAV_FM/uav-corridor/plans/.../<variant>/`  (no tag = lead=1.0 default)

### Step 4 — Eval with lead_gain > 1.0 (if drone under-reaches)

```python
# config/uav.py  plan_flow_matching_v3_uav
'lead_gain': 1.5,
```

```bash
python FM_v3_uav_test/eval_fm_uav.py --scene corridor
```

Results → `logs/UAV_FM/uav-corridor/plans/.../<variant>_lead1.5/`

---

## Quick reference — what controls what

| Param | File | Block / key | Default |
|---|---|---|---|
| `seed` (checkpoint to load) | `config/uav_projection.yaml` | `seed:` | 6 |
| `n_trials` | `config/uav_projection.yaml` | `n_trials:` | 20 |
| `cond_mode` (train) | `config/uav.py` | `flow_matching_v3_uav → 'cond_mode'` | `'p_des'` |
| `cond_mode` (eval) | `config/uav.py` | `plan_flow_matching_v3_uav → 'cond_mode'` | `'p_des'` |
| `reanchor_alpha` | `config/uav.py` | `plan_flow_matching_v3_uav → 'reanchor_alpha'` | `0.0` |
| `lead_gain` | `config/uav.py` | `plan_flow_matching_v3_uav → 'lead_gain'` | `1.0` |
| `batch_size` | `config/uav.py` | `plan_flow_matching_v3_uav → 'batch_size'` | `4` |
| projection variants | `config/uav_projection.yaml` | `projection_variants:` | all 13 |

**CLI overrides** (always win over file defaults):
```bash
python FM_v3_uav_test/eval_fm_uav.py --seed 7 --n-trials 5 --scene s_curve
```

---

## Mutual exclusivity reminder

`reanchor_alpha` only applies in `cond_mode='p_des'` — ignored in `real_p`.
`lead_gain` only applies in `cond_mode='real_p'` — ignored in `p_des`.
Never set both active simultaneously.
