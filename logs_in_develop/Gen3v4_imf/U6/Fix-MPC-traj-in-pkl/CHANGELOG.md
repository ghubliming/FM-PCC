# Fix — save MPC open-loop plans into the Gen3v4 avoiding eval `.npz`

**Date:** 2026-06-20
**Why:** [`npz_analysis_tool/MPC_Candidate_Selection_Explained.md`](../../../npz_analysis_tool/MPC_Candidate_Selection_Explained.md)
and [`npz_analysis_tool/CAPABILITY_GAP_plan_not_saved.md`](../../../npz_analysis_tool/CAPABILITY_GAP_plan_not_saved.md)
established that the avoiding eval plots an open-loop **plan fan** (`ax[i,5]`, the blue lines — the ones
that looked exploded in the [DiT eval-explosion debug](../DEBUG_DiT_Eval_Trajectory_Explosion.md)) but never
wrote that data to the saved `.npz`. Only the (smoother) **executed** path (`obs_all`) was saved, so the
plan explosion could never be measured from saved results — only eyeballed from the PNG. This closes that
gap for **Gen3v4** (the `avoiding` task).

---

## What changed

**File:** `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`

One line added to the `np.savez` call (`~L377`):
```python
np.savez(f'{save_path}/{variant}.npz',
         n_success, n_success_and_constraints, n_steps, n_violations, total_violations,
         avg_time, collision_free_completed, args,
         obs_all=np.array(obs_all, dtype=object),
         act_all=np.array(act_all, dtype=object),
         sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object))  # ← ADDED
```

`sampled_trajectories_all` was already computed in-loop (`L249, L323-324, L338`) and used only for the
PNG — it just wasn't persisted. No change to what's computed, sampled, or plotted; this only adds it to
the saved file. Mirrors the existing pattern in `imf_visual_aligning_test/eval_imf_visual_aligning.py:2128`
(`sampled_trajectories_all=np.array(plans_all, dtype=object)`), which already saved this for the
visual-aligining task — Gen3v4/avoiding was the one task missing it.

**Verified:** `python -m py_compile FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py` → OK.
(No local torch/GPU runtime in this Docker env — actual eval run + npz inspection must happen on the
Slurm cluster.)

---

## Data shape (for downstream tooling)

`sampled_trajectories_all` is a per-trial list (`dtype=object`, ragged across trials since episode length
varies):
```
sampled_trajectories_all[trial]                     # list, len = n_snapshots_this_trial
sampled_trajectories_all[trial][snapshot]            # np.ndarray [batch_size, horizon, obs_dim]
```
- One snapshot every `save_samples_every = horizon // 2` executed steps (`L248, L323`) — the "every 4
  steps" cadence for H=8.
- `obs_dim` columns are indexed the same as `obs_all` (`obs_indices['x']`, `obs_indices['y']`, …) — these
  are **unnormalized observations**, not raw `transition_dim` (action+obs) — see `policies.py:52-53`.
- `n_snapshots_this_trial ≈ ceil(n_steps_this_trial / save_samples_every)`, varies per trial since
  episodes can terminate early (success/collision).

---

## What this unblocks (not done yet — follow-ups)

These were the two extensions flagged in `CAPABILITY_GAP_plan_not_saved.md` item 2 — only the **data**
gap is fixed here, the analyzer-side consumption is a separate increment:
- `npz_analysis/analyze_npz.py --replot-plans`: overlay the blue plan fan on the executed path, fully
  reconstructing `<variant>.png` from the `.npz` alone.
- `plan_*` quality columns (straightness / roughness / max-jerk computed **on the plans**, not the
  executed path) — this is what would finally put a number on the "plan explosion" instead of eyeballing
  PNGs, closing D6 from
  [DEBUG_DiT_Eval_Trajectory_Explosion.md](../DEBUG_DiT_Eval_Trajectory_Explosion.md).
- Obstacle geometry is still not embedded in the npz (re-read from `projection_eval.yaml` if needed) —
  unchanged, out of scope here.

## Not done
- Visual-aligining eval already saves plans (`eval_imf_visual_aligning.py:2128`) — untouched, no gap there.
- No commit/push (per policy).
- No cluster run yet to confirm the new key round-trips through a real `.npz` (only `py_compile`-level
  verification possible locally).
