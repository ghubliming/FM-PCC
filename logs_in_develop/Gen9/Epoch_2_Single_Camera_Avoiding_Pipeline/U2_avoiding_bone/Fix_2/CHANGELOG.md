# U2 Fix_2 — Col-5 MPC trajectory plots showing nothing

**Date**: 2026-06-05  
**Status**: ✅ Fixed  
**Affects**: Both FM eval and DPCC eval  
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md)

---

## Symptom

The per-trial figure (`{variant}.png`, 6 columns) has col 5 completely blank — only
environment-constraint patches (walls, obstacle circles) are drawn, with no trajectory
lines from the FM model's planned path. The user observed this in:

```
logs/avoiding-d3il-visual/plans/fm_visual_avoiding/…/results/halfspace_both-hard/diffuser.png
```

Both `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` and
`diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` have the same bug.

---

## Root cause

Both evals replaced the state-only `Policy` (which returns `samples.observations` — K
candidate FM trajectories for visualization) with a `VisualAgent.predict()` that returns
only the scalar action delta. The planned trajectory was sampled internally by
`VisualFlowMatching.forward()` and immediately discarded.

- FM eval: stored `sampled_trajectories.append(None)` — placeholder, never plotted.
- DPCC eval: had no `sampled_trajectories` / `sampled_trajectories_all` at all.

Consequence: col 5 in every per-trial plot is always empty.

---

## Fix

### 1. `VisualAgent.predict()` — expose planned c_xy trajectory (both files)

Added after the action unnormalize step:

```python
# traj shape: (1, H, 6) = [act_norm(2)|des_xy_norm(2)|c_xy_norm(2)]
obs_norm_traj = traj[0, :, 2:].detach().cpu().numpy()       # (H, 4) normalised obs
obs_raw_traj  = self.obs_normalizer.unnormalize(obs_norm_traj)  # (H, 4) raw
planned_xy    = obs_raw_traj[:, 2:4][np.newaxis, :, :]       # (1, H, 2) c_xy in metres
return action, planned_xy
```

### 2. Call sites — unpack both return values (both files)

```python
# Before:
action = agent.predict(bp_image, obs[:2].copy(), c_xy)
# After:
action, traj_plan = agent.predict(bp_image, obs[:2].copy(), c_xy)
```

### 3. Trajectory storage

**FM eval**: replaced `sampled_trajectories.append(None)` with
`sampled_trajectories.append(traj_plan)`.

**DPCC eval** (no trajectory machinery existed):
- Added `sampled_trajectories_all = []` before `for i in range(n_trials):`.
- Added `sampled_trajectories = []` inside trial loop (alongside `obs_buffer = []`).
- Added `if _ % save_samples_every == 0: sampled_trajectories.append(traj_plan)` after
  the `avg_time` accumulation.
- Added `sampled_trajectories_all.append(sampled_trajectories)` after `act_all.append()`.

### 4. Col-5 plotting (both files)

Added before the existing `plot_environment_constraints` loop:

```python
for traj_np in sampled_trajectories_all[i]:
    if traj_np is None:
        continue
    for k in range(traj_np.shape[0]):        # k=0 (1 sample per step)
        for curr_ax in [ax[i, 5], ax_all[i, variant_idx]]:
            curr_ax.plot(traj_np[k, :, 0], traj_np[k, :, 1], 'b', alpha=0.5)
            curr_ax.plot(traj_np[k, 0, 0], traj_np[k, 0, 1], 'go')
ax[i, 5].set_xlim(ax_limits[0])
ax[i, 5].set_ylim(ax_limits[1])
```

Blue lines = planned robot c_xy positions over H=8 horizon steps.  
Green dot = planned trajectory start point.  
The existing environment constraints are then drawn on top (unchanged).

---

## Files touched

| File | Change |
|---|---|
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | `VisualAgent.predict()` returns `(action, planned_xy)`; call site unpacks both; `sampled_trajectories.append(traj_plan)`; col-5 plot added |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Same `VisualAgent` change; call site; added full `sampled_trajectories_all` machinery; col-5 plot added |

---

## Limitation

Only 1 candidate trajectory is visualised per timestep (the model samples batch=1 in the
visual eval, vs K=100 in the state-only baseline). Future work could tile the input to
batch_size=K for richer visualisation, but 1 trajectory per step is sufficient for
qualitative inspection of the planned path.
