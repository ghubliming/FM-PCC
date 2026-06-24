# npz analyzer — JOB D: MPC plan-candidate analysis

**Date:** 2026-06-24
**Tool:** `npz_analysis/analyze_npz.py`
**Implements:** JOB D from `logs_in_develop/MPC_traj_saved_in_npz/PATCH_TODO_MPC_Plans_in_NPZ.md`
(+ the NEXT STEP in `CAPABILITY_GAP_plan_not_saved.md`): consume `sampled_trajectories_all`
(the candidate foresight plans), not just the executed `obs_all` path.

## Why now
The PATCH_TODO marked JOB D "N/A for UAV" because the old UAV eval wrote JSON with
`batch_size=1` and no plan fan. **That is now outdated** — Gen11 E6 U3 added an npz to
the UAV eval that *does* persist `sampled_trajectories_all` (the FM's H-step plan per
step). So the analyzer can — and now does — analyze UAV plan data too.

## What was added (follows the existing `analyze_traj` / `process_file` pattern)

### New per-trial metrics (CSV columns, `plan_*`), aggregated to `*__mean` in files_summary
- `plan_path_len`, `plan_straightness`, `plan_roughness`, `plan_max_jerk`,
  `plan_max_step` — quality of the OPEN-LOOP plans (reuses `analyze_traj` per candidate).
- **`plan_max_abs`** — largest |value| over ALL dims of the plans → **explosion detector**
  (any axis, not just xy).
- `plan_cand_spread` — candidate diversity (mean pairwise endpoint distance per snapshot;
  `NaN` when `batch=1`, e.g. UAV).
- **`plan_exec_div` / `plan_exec_div_best`** — how far the foresight plans are from what
  was actually executed (candidate-0 and best-candidate), aligning snapshot→executed step.
- `plan_n_snap`, `plan_batch` — number of plan snapshots and candidates/snapshot.

### Executed-path metric added (symmetry)
- **`traj_max_abs`** — same all-dims explosion detector on the executed `obs_all`
  (so you can compare executed-vs-plan blow-up directly).

### New flag
- `--replot-plans` — per trial, overlay the candidate plan fan (blue) on the executed
  path (black) from `sampled_trajectories_all` → one PNG per trial. Makes a plan
  explosion visible at a glance.

### Robustness
- `_plan_snapshots()` normalises both schemas to a list of `[batch, horizon, dim]`:
  avoiding (snapshots every `H/2` steps, `batch>1`) and UAV (every step, `batch=1`).
- `print_table` shows `exec_maxabs | plan_maxabs | plan_exdiv | cand_sprd | ncand`
  only when plan data is present; unchanged for plan-less npz.

## Validated on real data (Gen11 E6 pillars eval npz)
`python npz_analysis/analyze_npz.py temp/eval/fm_only/fm_only.npz --xy-cols 3 4 --replot-plans`
(UAV obs = `[p_des(0:3) | p(3:6) | v(6:9)]`, so position x,y = cols 3,4):

```
variant  n_trials  succ_rate  steps   straight  exec_maxabs  plan_maxabs  plan_exdiv  ncand
fm_only      4        0.000   431.2     0.259      248.6        3.2          1.51       1
```

**Finding the tool exposed:** `exec_maxabs ≈ 248` (executed obs explode — `p_des_z` → −262)
but `plan_maxabs = 3.2` (the FM's predicted plan stays **bounded ~3 m**). The foresight
is sane; only the *executed* path blows up → the fault is in the **action channel**
(`Δp_des_z` scaling), decoupled from the FM's observation prediction — not the FM "not
understanding" the task. `--replot-plans` writes 4 per-trial PNGs.

## Usage
```bash
# UAV (position xy in obs cols 3,4):
python npz_analysis/analyze_npz.py logs/UAV_FM/uav-<scene>/plans --xy-cols 3 4 --replot-plans
# avoiding (executed xy in cols 2,3; plans batch>1 → cand_sprd is meaningful):
python npz_analysis/analyze_npz.py <dir-with-npz> --xy-cols 2 3 --replot-plans
```
`plan_max_abs` (explosion) and `plan_cand_spread` (candidate disagreement) are the
"compare the candidates" headline; `plan_exec_div` is the plan-vs-reality gap.

## Notes / limits
- `plan_max_abs` is over all dims (axis-agnostic). The spatial metrics
  (`plan_straightness`, `plan_exec_div`) use the chosen `--xy-cols`; pick the plane you
  care about (for the UAV altitude runaway, include col 2 `p_des_z` or col 5 `p_z`).
- `plan_cand_spread` needs `batch>1`. UAV currently samples `batch=1`; to get a real
  candidate set, sample `batch>1` per step in the eval (separate change).
- Backward compatible: plan-less npz (e.g. avoiding before JOB A) just skip the `plan_*`
  columns. `py_compile` clean; existing `--replot` / `--dump-xy` unchanged.
```
