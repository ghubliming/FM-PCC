# U1 — real MPC candidate-fan foresight plot (replaces the Epoch-6 placeholder)

**Date:** 2026-06-25.

## What changed
`eval_artifacts.write_pcc_placeholder` (which only wrote a static "PLACEHOLDER —
constraint projection lands in Epoch 7" SVG) is replaced by
**`write_mpc_foresight(diag_dir, idx, rollout, scene, stride=6)`**.

Now that Epoch 7's `_run_variant`/`rollout_one` capture the full per-FM-step candidate
batch in `rollout['plans']` (shape `(batch, horizon, obs_dim)` per FM step — see
`Epoch7_fm_pcc_FULL_PCC_MPC/CHANGELOG.md`), there's real data to plot, so the stub is
no longer needed.

### What it draws
Reuses the decision-point convention from `fm_visual_aligning_test`'s
`_mpc_foresight` plot (green candidate fan + black replan-point dot, every `stride`
FM steps, overlaid on the executed path with start/end markers) — copied in spirit,
not code, since the visual-aligning version is 2-D-XY+3-D-XYZ and laced with
halfspace/obstacle constraint-drawing specific to that task. The UAV version instead
reuses **this codebase's own** top-down + altitude 2-panel convention
(`eval_artifacts.plot_overview`), since x/y/z is all there is to show — two 2-D panels
read cleaner than a 3-D one and match the existing `<variant>.png` overview style:
- top-down (x, y): every candidate in the batch, every `stride`-th FM step, in green;
  black dot at the anchor (actual position at that step); executed path in red;
  start (lime star) / end (red square) markers.
- side (x, z): same fan + executed path, plus the `AIRBORNE_Z` gate line (reused from
  `plot_overview`) so altitude collapse is visible in the foresight plot too.
- Uses the **same position columns** (`P_X,P_Y,P_Z` = actual `p`, not `p_des`) as
  `plot_overview`, so the candidate fan is directly comparable to the executed-path
  plot rather than showing the open-loop-integrated `p_des` candidates.

### Call site
`eval_fm_uav.py:_run_variant` — `artifacts.write_pcc_placeholder(diag_dir, i)` →
`artifacts.write_mpc_foresight(diag_dir, i, r, scene)`. Called before `save_npz`/
`plot_overview`, while `r['plans']`/`r['obs_traj']` (heavy keys) are still attached to
the rollout dict — no change needed to when/where it's called.

### Degenerate case
If a rollout has no candidate-fan data (e.g. `batch_size=1` or an early-exit rollout),
writes a one-line "no candidate-fan data for this rollout" SVG instead of crashing.

## Output
Same path/filename as before — no schema change for downstream tooling:
```
…/plans/<variant>/diagnostics/rollout_<i>_mpc_foresight.svg
```

## Not changed (scope)
- No constraint-geometry overlay (workspace bounds / halfspace / obstacle) — those
  stay empty placeholders this epoch per `PLAN.md`/`CHANGELOG.md`; the plot itself has
  no gating for them yet since there's nothing to draw. Adding it later is the same
  pattern as `plot_overview`'s scene-aware obstacle drawing, not a rewrite.
- `diffuser` rollouts (`batch_size` config value, default 4 for all variants including
  `diffuser` per `config/uav_eval.yaml`) still get a real multi-candidate fan plotted
  even though no projector/selection runs — useful as the "before" comparison against
  `dpcc-r/-c/-t`.
