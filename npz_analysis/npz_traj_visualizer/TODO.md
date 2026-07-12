# npz_traj_visualizer — TODO

Deferred follow-ups. Not started yet — captured so they aren't lost.

## High-value (surfaced during fix_2, 2026-07-12)

- [ ] **UAV env geometry export.** `build_scene()` only fills `background` (bounds/halfspaces/
      obstacles) for `env=='avoiding'`; UAV scenes export an empty background, so nothing draws
      ("no env showing" on s_curve/pillars). Wire in the UAV geometry — source already exists in
      `FM_v3_uav_test/eval_artifacts.py` (it draws s_curve per-segment walls clipped to their live
      x-range + a feasible-side arrow, and pillar circles for the PNG/SVG overviews). Reuse that so
      the viewer's env layer matches `constraint_overview.svg`. Needs: s_curve x-active wall sets,
      pillar centers/radii, geo/bounds. Panels are xy + xz, so walls/pillars must project into both.

- [ ] **Fan layer shouldn't be clipped when a run crashes early.** Default scale is `fit_exec`; if a
      variant crashes at the start (e.g. diffuser on s_curve: executed is a tiny blob x∈[-3.2,-2.9]
      while its plan fans span the whole arena), the fans render off-canvas and look "missing." User
      must manually switch scale to `fit_vis`/`fit_all`. Options: (a) when the MPC-fan layer is ON,
      have `fit_exec`/`fit_vis` include the visible fans in the bounds; or (b) auto-hint / auto-switch
      when executed extent ≪ fan extent; or (c) just document it. Decide which.

## From fix_2 (data consistency)

- [ ] **Cluster re-export of existing scenes** with the fixed `npz_traj_export.py` (offset detection
      baked in), to make `track_err` values fully consistent. The `resnap_plan_steps.js` stopgap only
      re-aligned `plan_steps` (x-axis); per-step analytic *values* still carry pre-fix magnitude until
      a real re-export. Fresh exports need no resnap.

- [ ] **Optional eval-side cleanup (cosmetic).** `eval_flow_matching_v3_imeanflow.py` records the
      post-step obs and never stores the reset obs → executed shifted +1 vs the plan (offset 1);
      `eval_fm_uav.py` records pre-step obs (offset 0). The exporter now auto-detects either, so this
      is optional: storing the reset obs / appending the pre-step obs would make avoiding match UAV's
      offset-0 convention.

## Carried over from fix_1 (PLAN v4 deferred list)

- [ ] symlog axis option
- [ ] edge-arrow clip markers (indicate off-canvas geometry/fans)
- [ ] INSPECT fan popup table
- [ ] view-state URL hash (shareable pan/zoom/layer state)
- [ ] per-layer opacity control
- [ ] visual-aligning env support
- [ ] multi-trial layers
