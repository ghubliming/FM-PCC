# U1 — real MPC candidate-fan foresight plot (replaces the Epoch-6 placeholder)

**Date:** 2026-06-25.

## What changed
`eval_artifacts.write_pcc_placeholder` (static stub SVG) is replaced by
**`write_mpc_foresight(diag_dir, idx, rollout, scene, stride=6)`** — a faithful copy
of Gen7 `fm_visual_aligning_test`'s proven `_mpc_foresight` plot.

Now that Epoch 7's `_run_variant`/`rollout_one` capture the full per-FM-step candidate
batch in `rollout['plans']` (shape `(batch, horizon, obs_dim)` per FM step — see
`Epoch7_fm_pcc_FULL_PCC_MPC/CHANGELOG.md`), there's real data to plot.

### What it draws — copied from Gen7 `fm_visual_aligning_test/_mpc_foresight`
- **Two panels**: XY top-down (left) + XYZ true 3D (right). Same as Gen7.
- **Green fan**: candidate p_des trajectories (cols 0,1,2 of each plan), every
  `stride` FM steps (default 6, matches Gen7's `mpc_foresight_stride` default).
- **Black line**: commanded p_des path (`obs_traj[:,0:3]`) — "des (commanded)".
- **Red line**: actual drone position p (`obs_traj[:,3:6]`) — "actual (p)".
  Maps to Gen7's `c_pos_hist`/`c_arr` (arm actual position).
- **Black dot**: replan anchor = actual `p` at that FM step (same as Gen7 anchor).
- **Lime ★ / Red ■**: start / end markers following actual `p` (same as Gen7).
- **Full `Line2D` legend** with all 6 elements — copied from Gen7.
- Constraint geometry blocks (bounds/halfspace/obstacles) gated on `constraint_types`
  — empty this epoch, ready for per-scene geometry without a rewrite.

**UAV vs Gen7 mapping:**
| Gen7 | UAV |
|---|---|
| `real_pos` (commanded des) | `obs_traj[:, 0:3]` (p_des) |
| `c_arr` / `c_pos_hist` (actual arm pos) | `obs_traj[:, 3:6]` (actual drone p) |
| `cands[b,:,0:3]` (3D arm candidates) | `plan[b,:,0:3]` (p_des cols of candidate) |
| anchor = `c_arr[env_step]` | anchor = `c_arr[step_i]` (spr=1, 1:1 alignment) |
| `spr = n_steps // n_replans` | `spr = 1` (FM step = replan step for UAV) |

### Bug history
**First version** (wrong): used `p` (cols 3,4,5) for BOTH candidates and executed path
→ candidates span whole arena (multi-modal FM proposals in absolute coords) while
executed `p` is a tiny static cluster → "exploded/nonsensical green lines."

**Second version** (still wrong): switched to `p_des` (cols 0,1,2) for both — single
executed p_des line only, no dual black/red overlay → dropped Gen7's key insight of
showing commanded vs actual separation.

**Final version** (this): faithful Gen7 copy — dual black (p_des) + red (p) overlay
with p_des candidate fan. Now shows: candidates anchor to where drone IS (actual p),
commanded p_des path (may explode for diffuser), actual drone path (stays sane even
when p_des explodes since PID can't track). This separation is the core diagnostic.

### Call site
`eval_fm_uav.py:_run_variant` — called while `r['plans']`/`r['obs_traj']` are still
attached (before `save_npz`/json_safe strip). No change to call order needed.

### Degenerate case
No candidate-fan data (early-exit rollout) → writes a one-line text SVG, no crash.

## Output
Same path/filename — no schema change:
```
…/plans/<variant>/diagnostics/rollout_<i>_mpc_foresight.svg
```
