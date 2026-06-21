# CHANGELOG — MPC_NPZ_PATCH: Save Open-Loop Plans in NPZ

**Date:** 2026-06-21  
**Tag:** `MPC_NPZ_PATCH`  
**Grep to find all touched lines:** `grep -rn "MPC_NPZ_PATCH" /workspaces/FM-PCC/`  
**Reference:** [PATCH_TODO_MPC_Plans_in_NPZ.md](PATCH_TODO_MPC_Plans_in_NPZ.md)

---

## Motivation

Open-loop MPC plan snapshots (`sampled_trajectories_all`) were computed and plotted in every eval
script but **never persisted** to the `.npz` output files. This made post-hoc plan quality analysis
(plan explosion detection, fan straightness, jerk metrics) impossible from saved results alone.
Additionally, `policies.py` had a latent bug where `prev_observations` always used trajectory index
`[0]` instead of `which_trajectory`, breaking temporal consistency for `dpcc-c` selection.

---

## JOB A — `sampled_trajectories_all` added to `np.savez` (10 files)

### Priority 1 — Active gens (one-liner addition)

| File | Change |
|------|--------|
| `FM_v3_drifting_test/eval_flow_matching_v3_drifting.py` | Added `sampled_trajectories_all=np.array(..., dtype=object)` to `np.savez` |
| `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | Same |
| `FM_v3_imeanflow_test/eval_flow_matching_v3_ode_selectable.py` | Same |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Same |
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Same |

### Priority 2 — Early gens (also adds `obs_all`/`act_all` collection)

These had no `obs_all`/`act_all` at all. Added init (`obs_all=[], act_all=[]`), per-trial append,
and extended the single-line savez with all three arrays.

| File | Changes |
|------|---------|
| `FM_test/eval_FM.py` | +2 init lines, +2 append lines, updated savez (1 line) |
| `FM_v2_test/eval_FM_v2.py` | Same |
| `FM_Unet_v2_test/eval_FM_Unet_v2.py` | Same |
| `FM_hp_tune_test/eval_FM_hp_tune.py` | Same |
| `FM_v3_test/eval_FM_v3.py` | Same |

### Already fixed before this patch (not touched)

`eval_flow_matching_v3_imeanflow.py`, `eval_imf_visual_aligning.py`, `eval_fm_visual_aligning.py`,
`eval_visual_aligning_dpcc.py`, `eval_ddpm_encdec_vision.py`, `eval_fm_encdec_vision.py`

---

## JOB C — `prev_observations` reference fixed in `policies.py` (11 files)

**Bug:** `self.prev_observations = np.repeat(np.expand_dims(observations[0], ...))`  
**Fix:** `self.prev_observations = np.repeat(np.expand_dims(observations[which_trajectory], ...))`

This ensures that when `minimum_projection_cost` (`dpcc-c`) selects a non-zero trajectory, the
temporal consistency window for the **next** step references the trajectory that was actually executed,
not always candidate 0.

| File | Line |
|------|------|
| `diffuser/sampling/policies.py` | L76 |
| `flow_matcher/sampling/policies.py` | L76 |
| `flow_matcher_unet_v2/sampling/policies.py` | L76 |
| `flow_matcher_v2/sampling/policies.py` | L76 |
| `flow_matcher_v3/sampling/policies.py` | L76 |
| `flow_matcher_v3_drifting/sampling/policies.py` | L70 |
| `flow_matcher_v3_imeanflow/sampling/policies.py` | L70 |
| `flow_matcher_v3_ode_selectable/sampling/policies.py` | L70 |
| `flow_matcher_v3_uav/sampling/policies.py` | L70 (latent/harmless — batch_size=1) |
| `fm_encdec_vision/sampling/policies.py` | L76 |
| `ddpm_encdec_vision/sampling/policies.py` | L76 |

---

## Not Patched (JOB B)

**`desired_next_pos` tracking reference** (`samples.observations[0, 1, ...]` in all state-only eval
loops) was **not patched** this round. The fix requires `Policy.__call__` to return `which_trajectory`
as a third value (or attach it to `samples`), which is a breaking interface change. Deferred — only
affects tracking error metrics for `dpcc-c` variants, has no effect on the control loop.

---

## Gen11 UAV Status

`FM_v3_uav_test/eval_fm_uav.py` uses JSON output and `batch_size=1` — **all jobs exempt**. See
[PATCH_TODO_MPC_Plans_in_NPZ.md §Gen11 Epoch6 UAV](PATCH_TODO_MPC_Plans_in_NPZ.md) for details.

---

## Retrieval

```bash
grep -rn "MPC_NPZ_PATCH" /workspaces/FM-PCC/
```
