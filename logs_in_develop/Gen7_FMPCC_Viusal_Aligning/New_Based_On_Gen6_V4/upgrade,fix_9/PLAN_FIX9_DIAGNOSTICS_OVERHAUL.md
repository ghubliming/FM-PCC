# Fix 9 — Diagnostics & Output Logging Overhaul

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py` · `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`  
**Applies to**: Gen7 (FM) + Gen6V4 (DPCC) — identical changes in both eval scripts

---

## Root-Cause Analysis of Observed Issues

### I1 — "1 candidate/step" despite mpc_batch=4 in path name ✅ FIXED (session 8→9)

**Root cause**: Config used `mpc_batch_size: 4` (for path prefix), but code read
`getattr(args, 'batch_size', 4)`. Since `args.batch_size` was renamed to `mpc_batch_size`
in upgrade_8, the fallback `4` triggered — but the **original** `plan.batch_size: 1` was
still in the config, so `getattr` returned 1 (not the fallback). Path said `mpc4`, model
ran with batch=1.

**Fix applied**: code now uses `getattr(args, 'mpc_batch_size', 4)`. Config defaults:
- `plan_visual_aligning_dpcc.mpc_batch_size: 1` (DPCC single candidate — deterministic)
- `plan_fm_visual_aligning.mpc_batch_size: 4` (FM 4-candidate MPC pool)

---

### I2 — Chaotic MPC candidate visualization (CRITICAL BUG, unfixed)

**Root cause — two compounding errors**:

**Error 1 — Wrong trajectory dimensions stored.**
The model predicts a full 9D trajectory: `[act(0:3) | des_c_pos(3:6) | c_pos(6:9)]`.
`c_pos` dims (indices 6-8) are the predicted **actual robot positions** at each horizon
step — directly comparable to the DPCC/FMPCC avoiding plots, which plot
`samples.observations[:, :, obs_indices['x'/'y']]` (actual position dims) from the
sampled trajectories. Our code instead stores `traj_np[:, :, :3]` (action/delta dims,
indices 0-2) and tries to reconstruct positions via cumsum. That is unnecessary and
fragile — the predicted actual positions are already in the trajectory at dims 6-8.

**Error 2 — Normalized values stored and plotted as real-world.**
Whether using action dims or c_pos dims, `traj_np` is raw model output in
**normalized space**. The visualization applies `start + np.cumsum(normalized_deltas)`
— adding real-world meters to normalized ~[-1,1] values — producing nonsense coordinates.

**What DPCC/FMPCC reference does** (`scripts/eval.py`, lines 352-356):
```python
# avoidance task: obs_indices = {'x_des':0, 'y_des':1, 'x':2, 'y':3}
# x/y (indices 2,3) = actual robot position dims in the 4D trajectory
curr_ax.plot(
    sampled_trajectories[step][candidate, :horizon, obs_indices['x']],
    sampled_trajectories[step][candidate, :horizon, obs_indices['y']], 'b')
```
They plot the predicted actual position (indices 2,3 in 4D avoiding state) directly.
**No cumsum. No start offset. No unnormalization at plot time** — because `obs_all` is
already stored as the raw (non-normalized) env observation buffer, not the model's
normalized internal output.

**Correct fix for our code**:
1. In `get_action()`, store `c_pos` dims (indices 6-8) after **unnormalizing with
   `obs_normalizer`**, not action dims with `act_normalizer`:
```python
# After: traj_np = trajectory.detach().cpu().numpy()  # (B, H, 9)
# Store c_pos (predicted actual robot positions, dims 6:9) — analogous to DPCC obs_indices['x'/'y']
cpos_norm = traj_np[:, :, 6:9]   # (B, H, 3) normalized predicted actual positions
if self.obs_normalizer is not None:
    B_f, H_f, _ = cpos_norm.shape
    cpos_un = self.obs_normalizer.unnormalize(
        cpos_norm.reshape(-1, self.obs_normalizer.dim)
    )[:, :3].reshape(B_f, H_f, 3)   # take first 3 dims (x,y,z) of 6D obs
    self.curr_rollout_all_candidates.append(cpos_un.copy())   # (B, H, 3) real-world XYZ
else:
    self.curr_rollout_all_candidates.append(cpos_norm.copy())
```
2. In visualization, remove `start + np.cumsum(...)` entirely. Plot directly:
```python
axes[0,0].plot(cands[b, :, 0], cands[b, :, 1], ...)   # x vs y, real-world
```
3. Remove `plan_starts` from the candidate rendering entirely (no longer needed).

**Note on obs_normalizer dim**: The obs is 6D `[des_c_pos(3) | c_pos(3)]`.
`obs_normalizer.unnormalize` expects 6D input; `c_pos` is the last 3 dims (indices 3:6
of the 6D obs, or equivalently indices 6:9 of the 9D trajectory). Slice after unnorm.

**Color scheme improvement** (alongside the fix):

| Element | Current | Proposed |
|:--|:--|:--|
| Real executed path | `k-` black, lw=2 | keep, add `zorder=10` so it draws on top |
| Selected MPC candidate | `royalblue`, lw=1.5, α=0.85 | `green` (#2ca02c), lw=2.0, α=0.9 — visually distinct from real path |
| Non-selected candidates | `lightblue`, lw=0.5, α=0.35 | `gray`, lw=0.5, α=0.25 — clearly secondary |
| Goal position | (missing) | red star `r*`, markersize=14, zorder=11 (see I6) |

Rationale: black real path + green selected candidate + gray others + red star goal
reads unambiguously. Royalblue-on-lightblue is hard to distinguish when candidates overlap.

---

### I3 — Tracking error always 0.000 (misleading metric)

**Root cause**: The tracking error computes:
```python
err = |des_robot_pos_t  -  last_predicted_pos|
    = |des_robot_pos_t  -  (des_robot_pos_{t-1} + action_{t-1})|
```
Since D3IL's desired position at step t IS `desired_pos_{t-1} + action_{t-1}` (the
environment faithfully applies the commanded delta), this difference is trivially ≈ 0
by construction. The metric measures how well our open-loop mental model tracks the
commanded trajectory — which is always exact because the mental model IS the accumulator.

This is NOT a useful metric for the aligning task. It should be replaced or supplemented.

**What to show instead**:

| Metric | Description | Source |
|:--|:--|:--|
| `dist_to_target` (per-step) | Distance from robot to goal object (the real task progress) | `info['mean_distance']` per step — already recorded in `curr_rollout_dist_to_target` |
| `dist_to_target` (final) | Final mean_distance at rollout end | already in `master_rollout_history` |
| `|act|` magnitude | Raw action magnitude vs clamped — shows if max_action_delta is binding | already in `act_magnitudes` |

**Fix**: Remove or relabel the `MPC Tracking Error` subplot (row 1, col 2). Replace with
`Distance to Goal over Steps` using `dist_curve` (which is already plotted in axes[1,0]
but could be duplicated here with cleaner formatting), OR repurpose [1,2] for
`Action Magnitude over Steps`.

Also rename `max_tracking_error` key in stats files to `max_mental_model_error_m`
(or just remove it) to avoid misleading the reader.

---

### I4 — Terminal summary is legacy avoiding boilerplate

**Current output** (wrong for aligning):
```
Constraints satisfied: 1.0000          ← hardcoded
Avg number of constraint violations: 0.00 +- 0.00   ← hardcoded
Avg total violation: 0.000 +- 0.000    ← hardcoded
Tracking error: X.XXX                  ← always 0
```

**What aligning needs**:
```
--- aligning-d3il-visual [default] {variant} seed={seed} ---
Success rate:                  0.0000
Mean dist to target (final):   0.1868 m ± 0.0xxx m
Mean dist to target (avg step): x.xxx m
Avg steps (successful):        x.xx ± x.xx
Avg steps (all trials):        400.00 ± 0.00
Average inference time:        0.074 s/step
```

**Fix**: Replace the 7-line print block (lines 1110–1123 in FM eval) with a new
aligning-specific summary that reads `agent.history_rollout_mean_dist` (or equivalent)
and drops all constraint/violation/tracking-error lines.

Need to accumulate `mean_dist` per rollout in `agent.history_rollout_mean_dist`:
- Add `self.history_rollout_mean_dist = []` to `__init__`
- In `update_rollout_info()`: `self.history_rollout_mean_dist.append(float(mean_dist))`
- Print: `np.mean(agent.history_rollout_mean_dist)` ± std

---

### I5 — Duplicate stats.txt and stats.json

`diagnostics/rollout_N_stats.txt` and `realtime_diagnostics/rollout_N_stats.json`
contain **identical data** in different formats. The JSON is strictly superior:
machine-readable, load_results-compatible, no parsing needed.

**Fix**: Remove the `_save_diagnostics()` call and the `.txt` writer entirely.
Keep only the JSON in `_export_rollout_realtime()`. Consolidate the two `diagnostics`
/ `realtime_diagnostics` directory split while at it — both belong under `diagnostics/`.

**Proposed output structure** (cleaner):
```
{variant}/
  diagnostics/
    rollout_0.gif
    rollout_0_report.png        ← 9-panel realtime PNG (currently in realtime_diagnostics)
    rollout_0_stats.json        ← JSON only (no .txt duplicate)
    rollout_0_data.pkl          ← full history dict
  {variant}.npz                 ← full eval archive (keep, essential — see I7)
  {variant}.png                 ← aggregate grid PNG
  results_seed_N.pkl
```

---

### I6 — x_des/y_des and goal position on XY plot

**What x_des/y_des actually are in DPCC/FMPCC avoiding** (verified from source):

From `config/projection_eval.yaml`:
```yaml
observation_indices:
  'avoiding': {'x_des': 0, 'y_des': 1, 'x': 2, 'y': 3}
```
From `scripts/eval.py` line 256:
```python
elif 'avoiding' in exp:
    obs = np.concatenate((action[:2], obs))   # action = env.robot_state()[:2]
```
So:
- `x_des/y_des` (indices 0,1) = `env.robot_state()[:2]` = the robot's **commanded/desired
  position** prepended to the obs vector. This is the position the PD controller is
  driving toward — equivalent to our `des_c_pos` (dims 3-5 of the 9D trajectory).
- `x/y` (indices 2,3) = the D3IL environment observation = the robot's **actual position**
  at that step. Equivalent to our `c_pos` (dims 6-8 of the 9D trajectory).

The DPCC plots show 4 separate 1D time-series panels (`x_des(t)`, `y_des(t)`, `x(t)`, `y(t)`)
so you can see the commanded vs actual position track over time, plus an XY panel for
the spatial path. **This is NOT a "reference trajectory" in the sense of a pre-planned
path** — it is simply the separation of commanded vs actual position that exists naturally
in the avoiding task's observation structure.

**For our aligning task: the exact analogues already exist in the 9D state.**

| Avoiding dim | Index | Aligning analog | 9D index |
|:--|:--|:--|:--|
| `x_des` | 0 | `des_c_pos_x` | 3 |
| `y_des` | 1 | `des_c_pos_y` | 4 |
| `x` | 2 | `c_pos_x` | 6 |
| `y` | 3 | `c_pos_y` | 7 |

The 1D time-series panels `x_des(t)`, `y_des(t)`, `x(t)`, `y(t)` from DPCC are directly
reproducible for aligning using `des_c_pos` and `c_pos` from the 9D model predictions.
These are meaningful: they show whether the robot's actual position (c_pos) tracks the
commanded position (des_c_pos) and how they both evolve over the rollout.

**Fix — simple overlay on existing subplots:**

The current 9-panel layout already has `X Position over Steps` (axes[0,1]) and
`Y Position over Steps` (axes[0,2]) plotting `real_pos[:, 0/1]` = `des_c_pos` (the
commanded position). Simply overlay `c_pos` (actual robot position) on the same subplots
in a second color — no new panels, no layout changes.

Also record `robot_pos_np` (actual `c_pos`) per step alongside the existing
`des_robot_pos_np` accumulation, then store as `c_pos_history` in rollout data.

```python
# axes[0,1]: X over steps
axes[0,1].plot(real_pos[:, 0], 'k-', label='X des')
axes[0,1].plot(c_pos_history[:, 0], 'r--', label='X actual', alpha=0.7)
axes[0,1].legend(fontsize=7)
axes[0,1].set_title('X Position — des (black) vs actual (red)')

# axes[0,2]: Y over steps — same pattern
```

Goal marker on the XY plot (axes[0,0]): check what the D3IL aligning env exposes via
`info` keys. If `info.get('obj_pos')` or similar exists, record it once in
`record_step_info()` and plot a red star. If not exposed, skip — do not fabricate a proxy.

**The `x_des/y_des` of DPCC avoiding IS relevant to us** — the aligning analog
(`des_c_pos` vs `c_pos`) should be shown on the existing panels. Initial I6 analysis was
incorrect in calling these "avoiding-task artifacts."

---

### I7 — `{variant}.npz` and load_results script

**What the reference DPCC/FMPCC npz saves** (`scripts/eval.py` line 381):
```python
np.savez(f'{save_path}/{variant}.npz',
    n_success, n_success_and_constraints, n_steps,
    n_violations, total_violations,
    avg_time, collision_free_completed,
    args,
    obs_all,    # per-rollout env observation buffer (commanded+actual position, unnormalized)
    act_all)    # per-rollout action buffer
```
Note: **no** `sampled_trajectories_all` in the original. The candidate trajectories are
only used for the inline PNG and discarded. The `load_results.py` reads only the scalar
metrics and `obs_all` for trajectory overlay plots.

**What our aligning npz saves** (both eval scripts, line ~1024):
```python
np.savez(f'{save_path}/{variant}.npz',
    success_rate, entropy, mode_encoding,
    elapsed_seconds, seed,
    n_success, n_steps, avg_time,
    n_violations=np.zeros(...),          # ← FAKE: hardcoded 0, avoiding artifact
    total_violations=np.zeros(...),       # ← FAKE: hardcoded 0, avoiding artifact
    collision_free_completed=n_success,
    obs_all,    # real_robot_pos per rollout (= des_c_pos, not full 9D obs)
    act_all,    # desired_actions per rollout
    sampled_trajectories_all,            # ← EXTRA vs reference: selected trajectory only
    pos_tracking_errors,                  # ← EXTRA vs reference: always ~0, misleading
    mean_distance,                        # ← EXTRA vs reference: key aligning metric
    args)
```

**Gaps and fixes for the npz**:

| Key | Status | Action |
|:--|:--|:--|
| `n_violations`, `total_violations` | Fake zeros — avoiding artifact | Remove entirely |
| `pos_tracking_errors` | Always ~0, misleading | Remove or rename to `mental_model_error` |
| `mean_distance` | Key aligning metric ✓ | Keep — this is the primary outcome |
| `obs_all` | Only `des_c_pos` (3D) stored | Upgrade to full `[des_c_pos, c_pos]` (6D) to match DPCC's `obs_all` which includes both commanded and actual |
| `sampled_trajectories_all` | Selected trajectory only (not all candidates) | Optionally add `all_candidates_npz` from `agent.history_all_candidates` for future analysis — large but useful |
| `entropy` | Mode entropy — aligning-specific | Keep |

**`load_results_fm_visual_aligning.py` plan**:

Single script, shared between FM and DPCC variants (both save the same npz schema).
Create at `fm_visual_aligning_test/load_results_fm_visual_aligning.py`.

**Design**: mirror the structure of `FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py`
(the closest existing analog) — same seed-loop + variant-loop skeleton, swap metric
keys for aligning.

1. Reads `config/visual_aligning_eval.yaml`: `projection_variants`, `seeds`
2. Path discovery: `logs/aligning-d3il-visual/plans/{fm_visual_aligning|visual_aligning_dpcc}/[train_folder]/[plan_folder]/{seed}/results/{variant}/{variant}.npz`  
   Use the same `args.savepath` resolution pattern as the existing load_results scripts.
3. Load per-seed: `mean_distance` (primary), `n_success`, `n_steps`, `avg_time`, `entropy`
4. Aggregate across seeds and produce:
   - **Bar chart**: success rate per variant ± std across seeds
   - **Box/violin plot**: `mean_distance` per variant (main result figure)
   - **Step count**: mean ± std for successful rollouts
   - **XY trajectory overlay**: `obs_all[:, 0]` vs `obs_all[:, 1]` per variant, thin lines, one color per seed
5. Save to `[plan_folder]/plots/load_results_output/` (mirrors existing scripts' `plot_path` pattern)
6. Does **not** read `n_violations`, `n_success_and_constraints`, `total_violations`, `collision_free_completed` (avoiding artifacts)

**npz cleanup** (remove before load_results script is written, so the schema is clean):
- Remove `n_violations=np.zeros(...)` and `total_violations=np.zeros(...)` from the `np.savez` call in both eval scripts
- Remove `pos_tracking_errors` from npz (always ~0, misleading; still available in per-rollout `.pkl`)
- Keep `mean_distance` as the top-level scalar per rollout (shape: `[n_rollouts]`)

The `scripts/load_results.py` is NOT reusable — hardwired to `avoiding_halfspace_variants`
loop and reads keys absent from our npz.

---

## Change Summary Table

| ID | File | Priority | Type |
|:--|:--|:--|:--|
| I2 | both eval scripts, `get_action()` — store `c_pos` dims 6:9, unnorm via `obs_normalizer` | **P0** | Bug fix — wrong dims + normalized |
| I2 | both eval scripts, `_export_rollout_realtime()` — remove cumsum/start, plot direct XY | **P0** | Bug fix — chaotic visualization |
| I2 | both eval scripts — color scheme update | P1 | Readability |
| I3 | both eval scripts, `_export_rollout_realtime()` — replace tracking error panel | P1 | Replace misleading metric |
| I4 | both eval scripts, summary print block — aligning-specific metrics | P1 | Remove avoiding boilerplate |
| I5 | both eval scripts — remove `.txt`, consolidate `diagnostics/` dir | P2 | Remove duplicate |
| I6 | both eval scripts — overlay `des_c_pos`/`c_pos` on position panels; add goal marker | P2 | Add analogous x_des/y_des + goal |
| I7 | new file `load_results_fm_visual_aligning.py` | P3 | New analysis script |
| I7 | both eval scripts — remove fake `n_violations`/`total_violations` from npz | P2 | Clean up avoiding artifacts in npz |

---

## Additional Ideas (Pitched for Consideration)

### A1 — Candidate trajectory endpoint scatter plot
Instead of full cumsum trajectories (which are long and overlap), plot only the **endpoint**
of each candidate (position after H steps) as a scatter. Selected endpoint = colored dot,
others = gray dots. Much cleaner on a 2D XY subplot and avoids the trajectory-spaghetti
problem.

### A2 — "Candidate diversity" scalar panel
Add a panel showing `std(endpoints_xy)` per replan step as a time series. This quantifies
how much the batch of B candidates disagrees — a useful diagnostic for whether the model
is collapsing to a single mode. Expected: high diversity early, low near goal.

### A3 — Per-rollout success annotation on XY plot
Annotate the start (green circle) and end position (red X if failed, green star if
succeeded) of the real path on `axes[0,0]`. Currently the XY plot has no success/failure
readout — the reader must cross-reference the title.

### A4 — Mean-distance trajectory as a secondary metric
Plot `dist_to_target` curve on the aggregate variant PNG (currently the aggregate only
has MPC foresight). Overlapping `dist_curve` across rollouts for one variant quickly shows
whether the variant converges to the goal reliably.

### A5 — Renamed aligning-specific summary block as a function
Extract the terminal print block into `_print_aligning_summary(agent, variant, seed)`.
This makes it easy to add new metrics without hunting through a 1100-line file.

### A6 — Separate `diagnostics/` sub-dir per variant (already done structurally)
Currently all rollouts of all variants write to the same `{variant}/diagnostics/` which
is correct. Confirm `.gif`/`.png`/`.json` all land here post-I5 cleanup.

---

## What Stays Unchanged

- `.npz` save structure (kept, with targeted key removals in I7)
- `plan_start_positions` subsampling logic — **no longer needed** for candidate visualization after I2 fix (c_pos dims are already absolute positions); keep for other uses but remove from candidate rendering
- `all_candidates` / `selected_idx` accumulation in `__init__` / `reset()` / `update_rollout_info()` (Fix 8, correct — only the stored dim changes from `:3` to `6:9`)
- `trajectory_selection` logic (Fix 8, correct)
- `mpc_batch_size` rename and path structure (Fix 9 config changes, already applied)
- `.gif`/`.mp4` video recording (keep as-is)
- Per-rollout `.pkl` save in `_export_rollout_realtime()` (keep — full history for debugging)
