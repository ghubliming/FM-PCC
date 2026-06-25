# U1 — real MPC candidate-fan foresight plot (replaces the Epoch-6 placeholder)

**Date:** 2026-06-25.

## Final design — XY top-down + XZ altitude (UAV-specific)

`eval_artifacts.write_mpc_foresight(diag_dir, idx, rollout, scene, stride=6)`

### Panel layout
```
LEFT  — XY top-down  : horizontal navigation, obstacle avoidance
RIGHT — XZ altitude  : Z profile, airborne gate, p_des-z explosion
```

Both panels use the **Gen7 dual-path convention** (copied from
`fm_visual_aligning_test/_mpc_foresight`):
- **green** = MPC candidate p_des fan (obs cols 0,1,2), every `stride` FM steps
- **black** = commanded p_des path (`obs_traj[:,0:3]`)
- **red**   = actual drone position p (`obs_traj[:,3:6]`)
- **black dot** = replan anchor at actual `p` (Gen7 convention: anchor follows where
  the drone physically IS, not where p_des commanded it to be)
- **lime ★ / red ■** = start / end (follow actual `p`)

### Why XZ not 3D
Gen7 uses XY + XYZ-3D because the arm workspace is inherently 3D and matplotlib's
projection gives useful spatial intuition.  For UAV, a static SVG 3D projection is
worse: altitude (Z) gets compressed into perspective, the candidate fan becomes a
clutter of overlapping green lines, and there's no way to rotate the view.  Replacing
3D with a dedicated **XZ altitude panel** reads the key UAV-specific stories clearly:
- Did `p_des_z` explode? (visible as red/black line diving to −228 m)
- Is the drone airborne? (orange dashed `AIRBORNE_Z` gate)
- Do dpcc candidates keep Z sane vs diffuser explosion?

### Bug history (three iterations)
1. **v1** — used `p` (cols 3,4,5) for BOTH candidates and executed path → candidates
   span whole arena, executed `p` is a tiny static cluster → "exploded green lines."
2. **v2** — "copy Gen7 exactly": XY + 3D, Gen7 dual-path convention added. Correct
   colors/anchors, but 3D panel is worse than v1 for UAV — Z buried in perspective.
3. **v3 (final)** — XY + XZ: keeps Gen7's dual-path/anchor/legend convention; replaces
   3D with the altitude panel that makes UAV's Z story readable.

## UAV vs Gen7 mapping
| Gen7 | UAV |
|---|---|
| `real_pos` (commanded des) | `obs_traj[:, 0:3]` (p_des) |
| `c_arr` / `c_pos_hist` (actual arm pos) | `obs_traj[:, 3:6]` (actual drone p) |
| `cands[b,:,0:3]` | `plan[b,:,0:3]` (p_des cols of candidate) |
| anchor = `c_arr[env_step]` | anchor = `act[step_i]` (spr=1) |
| XY + XYZ-3D | XY + XZ (altitude) |

## Call site
`eval_fm_uav.py:_run_variant` — `write_mpc_foresight(diag_dir, i, r, scene)` called
while `r['plans']`/`r['obs_traj']` are still attached (before json_safe strip).

## Output path (unchanged)
```
…/plans/<variant>/diagnostics/rollout_<i>_mpc_foresight.svg
```
