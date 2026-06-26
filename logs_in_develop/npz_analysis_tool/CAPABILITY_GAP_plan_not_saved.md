# Capability gap — avoiding npz does NOT save the open-loop plans (can't fully rebuild the plot)

**Date:** 2026-06-19
**Tool:** `npz_analysis/analyze_npz.py`
**TL;DR:** the avoiding eval npz stores only the **executed** trajectory, not the per-replan **open-loop
plans** (the blue lines). So the analyzer can rebuild the executed path but **cannot fully reconstruct the
default `<variant>.png`**, and **cannot quantify the plan explosion** from the npz. Fix = inject one line
into the avoiding eval's `np.savez`.

---

## What the avoiding npz actually contains

`FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py:377` saves:
```python
np.savez(f'{save_path}/{variant}.npz',
         n_success, n_success_and_constraints, n_steps, n_violations, total_violations,
         avg_time, collision_free_completed, args,
         obs_all,   # executed observations per trial  [n_trials][steps, obs_dim]
         act_all)   # executed actions per trial
```

## What the default plot draws vs what's recoverable

| Plot panel (`<variant>.png`) | Source variable | In npz? | Rebuildable? |
|---|---|---|---|
| `ax[i,0..3]` x / y / x_des / y_des time series | `obs_all` | ✅ | ✅ |
| `ax[i,4]` **executed path** (black) + start dot | `obs_all` (cols x=2, y=3) | ✅ | ✅ (`--replot`) |
| `ax[i,5]` **open-loop plans** (blue, every H/2 steps) | `sampled_trajectories_all` | ❌ | ❌ **gone** |
| obstacle circles / halfspaces | `config/projection_eval.yaml` + exp | (config) | ✅ (re-read config) |

**The blue `ax[i,5]` plans — the ones that looked exploded — are never written to the avoiding npz**
(they live only in the saved `.png`). So:
- `--replot` / `--dump-xy` faithfully reproduce the **executed** path only.
- The **open-loop plan explosion cannot be scored from the avoiding npz** — only the (smoother) executed
  path is available. This is exactly why executed-path `traj_straightness` looked ~0.8 ("smooth") even
  when the plans were chaotic. See
  [Gen3v4_imf/U6/DEBUG_DiT_Eval_Trajectory_Explosion.md](../Gen3v4_imf/U6/DEBUG_DiT_Eval_Trajectory_Explosion.md) §2.

> Note: the **visual-aligning** schema already saves the plans
> (`eval_imf_visual_aligning.py:2109` → `sampled_trajectories_all=...`), so for that task they *are*
> recoverable — the gap is avoiding-specific.

---

## NEXT STEP — inject plan-saving so plans become replottable

**1. Save the plans in the avoiding eval (one line).** In `eval_flow_matching_v3_imeanflow.py:377`,
`sampled_trajectories_all` already exists as a local (appended at `:338`). Add it to the `np.savez`,
mirroring visual-aligning:
```python
np.savez(f'{save_path}/{variant}.npz',
         ...,                                   # existing keys
         obs_all=np.array(obs_all, dtype=object),
         act_all=np.array(act_all, dtype=object),
         sampled_trajectories_all=np.array(sampled_trajectories_all, dtype=object))  # ← ADD
```
(Each entry is a list of snapshots taken every `save_samples_every = horizon//2` steps; each snapshot is
`[batch, horizon, transition_dim]`.)

**2. Extend the analyzer** to consume `sampled_trajectories_all`:
- `--replot-plans`: overlay the open-loop plans (blue) on the executed path → fully reconstruct
  `<variant>.png`.
- `plan_*` quality columns: straightness / roughness / max-jerk on the **plans** (not just the executed
  path) — this is what finally puts a number on the explosion (D6 in the debug doc).

**3. (optional)** also stash the constraint geometry (obstacle centers/radii) into the npz so the plot is
self-contained without re-reading `projection_eval.yaml`.

**Status:** not done — flagged as the next increment. Items 1–2 are the minimum to make the plan
explosion measurable from saved results instead of eyeballing PNGs.
