# U2 PLAN — 9D + PID Stop-and-Go (`controller='pid_stopgo'`)

**Status:** PLANNED
**Parent:** [E8 PLAN](../PLAN_MJPC_Thrust_Control.md) · [Option memo](MEMO_controller_options.md)
**What:** Add `controller='pid_stopgo'` as a config-selectable option — same CascadedPID, same 9D obs (`cond_mode='pos_only'`), but `v_des` forced to zero every FM step. Strict stop-and-go by construction.
**Why:** Baseline to measure the cost of dropping velocity from the FM tensor without the overhead of MJPC. If stop-and-go performance is acceptable, E7→E8 migration is unnecessary.

---

## 1 What Changes

`v_des` is the only difference from E7:

```
E7  (12D):  v_des = action / dt_fm   →  PID feedforward, continuous flight
U2  (9D):   v_des = np.zeros(3)      →  PID brakes to zero each FM step, stop-and-go
```

No new controller class. No dataset changes beyond the existing `pos_only` slice. No MJPC dependency.

---

## 2 Code Changes (2 files only)

### 2.1 `FM_v3_uav_test/eval_fm_uav.py`

**In `rollout_one` — branch `v_des` on controller:**

```python
# current (E7 + E8-MJPC):
v_des = action / dt_fm

# replace with:
if controller == 'pid_stopgo':
    v_des = np.zeros(3)          # strict stop-and-go: PID brakes to zero each step
else:
    v_des = action / dt_fm       # E7 default: velocity feedforward
```

`tracker.compute(p, q, v, om, p_des, v_des)` call is unchanged — PID receives `v_des=0`.

**In `_run_variant` — add to `eval_tag`:**

```python
# current:
eval_tag = ('_anchorP' if anchor_to_p else '') + (f'_ctrl{controller}' if controller != 'pid' else '')

# no change needed — 'pid_stopgo' != 'pid' → tag becomes '_ctrlpid_stopgo' automatically
```

**In `rollout_one` — tracker build block:**

```python
# current:
tracker = pid
if controller == 'mjpc':
    ...
    tracker = MJPCTracker(...)

# add:
tracker = pid   # pid_stopgo also uses CascadedPID — only v_des differs, handled above
if controller == 'mjpc':
    ...
    tracker = MJPCTracker(...)
# 'pid_stopgo' falls through → uses pid, v_des=0 set above
```

No new class, no new import.

### 2.2 `config/uav.py`

**In `_uav_exp_name` — no change needed.** `controller='pid_stopgo'` is non-default → suffix `_ctrlpid_stopgo` appended automatically by the existing condition:

```python
if controller != 'pid':
    name += f'_ctrl{controller}'
# 'pid_stopgo' → appends '_ctrlpid_stopgo'
```

**In both config blocks — set the new option:**

```python
'cond_mode':  'pos_only',
'controller': 'pid_stopgo',
```

**Add documentation comment:**

```python
#   controller='pid_stopgo' → CascadedPID with v_des=0 (strict stop-and-go).
#                             Tests the cost of 9D obs without MJPC overhead.
#                             See E8 U2 PLAN + MEMO_controller_options.md Option 2.
```

---

## 3 Checkpoint Paths

| Config | exp_name suffix | Train path | Eval output |
|---|---|---|---|
| E7 default | *(none)* | `…/H8_D…ODE/<seed>/` | `…/fm_only/` |
| U2 this plan | `_cmpos_only_ctrlpid_stopgo` | `…/H8_D…ODE_cmpos_only_ctrlpid_stopgo/<seed>/` | `…/fm_only_ctrlpid_stopgo/` |
| E8 MJPC | `_cmpos_only_ctrlmjpc` | `…/H8_D…ODE_cmpos_only_ctrlmjpc/<seed>/` | `…/fm_only_ctrlmjpc/` |

All three discriminated — no collision.

---

## 4 Retrain Required — But No Recollection

**Retrain: yes.** `cond_mode='pos_only'` → transition dim 9D ≠ 12D (E7). State dict shape mismatch — must train from scratch.

**Recollect: no.** The FM only sees `[p_des | p]` at each step and outputs `Δp_des`. Whether the tracker brakes to zero or flies continuously between steps is invisible to the FM — it's a runtime property set by `v_des=0`, not a training-data property. The `Δp_des` actions are pure geometry from `traj_fn`, controller-independent. The existing `data/uav_fm/v1/` slice is correct as-is.

---

## 5 SLURM

```bash
# set config/uav.py: cond_mode='pos_only', controller='pid_stopgo'

# quick test — single scene, single seed
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh pillars 6

# aggregate after multi-seed eval
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/aggregate_summaries.sh \
    "empty corridor s_curve pillars" fm_only_ctrlpid_stopgo
```

---

## 6 Files to Touch

| File | Change | Size |
|---|---|---|
| `FM_v3_uav_test/eval_fm_uav.py` | Branch `v_des` on `controller=='pid_stopgo'`; no new tracker class | ~4 lines |
| `config/uav.py` | Set `controller='pid_stopgo'` in both blocks; add comment | ~3 lines |

---

## 7 Success Criteria

- **Checkpoint loads:** 9D model trains without NaN
- **Eval runs:** no crash; `pid_stopgo` results in `fm_only_ctrlpid_stopgo/` subfolder
- **Observable stop-and-go:** per-step velocity log shows v→0 between FM steps (verify via `behavior_log`)
- **Compare vs E7:** if success rate not dramatically worse → 9D trajectory quality is sufficient; if worse → velocity conditioning in FM tensor (E7 12D) is load-bearing
