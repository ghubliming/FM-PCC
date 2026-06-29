# CHANGELOG — U2 PID Stop-and-Go (`controller='pid_stopgo'`)

**Plan:** [`PLAN_PID_Stop_Go.md`](PLAN_PID_Stop_Go.md)

---

## Files Touched (2)

### `FM_v3_uav_test/eval_fm_uav.py`

**1. Tracker build comment** (lines ~318-326):
Updated comment to document all 3 controller options (`pid`, `pid_stopgo`, `mjpc`). Added inline note that `pid_stopgo` falls through to reuse `CascadedPID` — the only difference is `v_des`, handled separately.

**2. `v_des` branch** (line ~396, the core change):
```python
# before:
v_des = action / dt_fm

# after:
v_des = np.zeros(3) if controller == 'pid_stopgo' else action / dt_fm
```
`pid_stopgo` → PID velocity error = `v_real - 0 = v_real` → actively brakes to zero each FM step.
All other controllers unchanged.

### `config/uav.py`

**Training block comment** — added `controller='pid_stopgo'` to the option list.
**Plan block comment** — noted that `pid_stopgo` uses the same checkpoint as `pos_only+pid`; `v_des=0` is an eval-only effect (no retrain needed to switch between `pid` and `pid_stopgo` when `cond_mode='pos_only'`).

---

## Key Design Points

- **No new class.** `pid_stopgo` reuses `CascadedPID` — only `v_des` differs.
- **No dataset change.** Stop-and-go is a runtime property (`v_des=0`); training data is unchanged.
- **Retrain required** only because `cond_mode='pos_only'` → 9D ≠ 12D (E7). Same requirement as E8 MJPC.
- **Checkpoint shared with `pid`+`pos_only`.** Path suffix is `_cmpos_only_ctrlpid_stopgo` — discriminated from both E7 (`H8_D…ODE`) and E8 MJPC (`…_ctrlmjpc`).
- **Eval-only switch:** to compare `pid` vs `pid_stopgo` on the same 9D checkpoint, only change `controller` in the plan block (not the train block) and re-run eval.

---

## Path Discrimination

| Config | Suffix | Eval folder |
|---|---|---|
| E7 default | *(none)* | `fm_only/` |
| U2 this | `_cmpos_only_ctrlpid_stopgo` | `fm_only_ctrlpid_stopgo/` |
| E8 MJPC | `_cmpos_only_ctrlmjpc` | `fm_only_ctrlmjpc/` |

---

## How to Run

```python
# config/uav.py — BOTH blocks
'cond_mode':  'pos_only',
'controller': 'pid_stopgo',
```

```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh pillars 6
```
