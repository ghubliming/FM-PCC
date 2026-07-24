# Fix_12 — CHANGELOG: constraint-set feasibility repair + realized-homotopy logging

**Date:** 2026-07-08. Implements the fixes recommended by
`REPORT_fix12_constraint_strictness_homotopy_rawdata.md` (same folder — read it for the
full derivations). Root problem: the planning constraint set was mathematically
near-infeasible (pillars: **both** trained channels closed; every constrained scene's
start/goal outside the workspace box), and the per-trial `homotopy` label was misleading
(it never controls the flown route).

## 1. `config/uav_projection.yaml` — geometry/inflation fixes (Q1)

- **`inflation: {r_drone: 0.36→0.31, margin_base: 0.05→0.02}`** (total margin 0.41→0.33).
  0.31 = true per-axis rotor reach at yaw=0 (`trajectories.py PILLAR_ROTOR_REACH`); 0.36
  was the worst-case diagonal, which the never-yawing drone can't present to a lateral
  surface. margin_base ≤ 0.02 is forced by the corridor expert channels themselves
  (0.45 wall − 0.31 reach − 0.12 channel = 0.02 m total slack in the data).
- **pillars: removed the synthetic y=±1.2 envelope halfspaces** (no such wall exists in
  `scene_pillars.xml`; inflated they closed the outer channels under ANY margin) and
  dropped `'halfspace'` from its `constraint_types` (family had zero entries left).
  The kept y=±1.5 workspace box bounds the field instead.
  ⚠ Output-path note: pillars' geo_tag folder becomes
  `pillars_bounds+dynamics+geo_bounds+obstacles` (new folder; old
  `...+halfspace+obstacles` results stay comparable but were produced under the broken set).
- **Workspace boxes now contain start/goal + the altitude draw:**
  pillars x ±3.0→±3.6, s_curve x ±3.5→±3.6 (paths span ±3.2), corridor x ±3.0→±3.2
  (paths span ±2.8); z ub 1.60→1.80 everywhere (altitude ~ U(0.90,1.30); the inflated
  ceiling 1.47 stays below the 1.5 m wall tops, so walls can't be hopped).
- **s_curve halfspaces moved from wall centrelines to inner faces** (∓0.35/∓1.25 instead
  of ∓0.3/∓1.3) — corridor's entry already used inner faces; this one was 5 cm off.

Resulting feasible sets (margin 0.33): pillars outer channels [1.05,1.17] (expert ±1.11
centred, was **closed**), centre |y|≤0.15 (was 0.07); s_curve band 24 cm (was 18), corner
gates ~24 cm (was ~8); corridor |y|≤0.12 (expert L/R exactly attainable, was infeasible).
All starts/goals now inside the (inflated) boxes — this also removes the guaranteed
phantom step-0 violation that forced `constraint.collision_free=False` /
`success.*_and_constraints=0` on every pillars/s_curve/corridor rollout.

## 2. `FM_v3_uav_test/eval_fm_uav.py`

- **`_realized_homotopy(scene, obs_traj)`** (new): reads the class the drone ACTUALLY flew
  from the flown path. Result dict (and therefore `results.json`, `rollout_<i>_stats.json`)
  gains **`homotopy_flown`** next to the commanded `homotopy` label (which the FM policy,
  being unconditioned, need not obey).
  - pillars: side of y=0 at each pillar column x∈{-2,0,2}, interpolated at first crossing;
    `?` if a column was never reached (per the report, the commanded label controls nothing
    physical here — same start/goal for all four classes).
  - corridor: nearest expert channel (L=-0.12/C=0/R=+0.12) to the MEDIAN flown y over the
    walled section x∈[-2,2]; `?` if the drone never entered it.
  - **Fix_12 follow-up (2026-07-09):** first cut was pillars-only and returned `null` for
    every other scene, so corridor rollouts logged a confusing `homotopy_flown: null` next
    to a meaningful `homotopy: L/C/R`. Corridor now computes it (its commanded label is a
    real start-channel bias but still not guaranteed to be the channel flown). Returns
    `None` ONLY for single-class scenes — s_curve (`['default']`) and empty (`['N/A']`) —
    where there is nothing to disambiguate and the field would be pure noise.
- **`_warn_expert_route_infeasibility(...)`** (new) + call in `eval_scene` per geo entry:
  samples each homotopy's expert route (200 pts, seeded rng) and checks it against the
  PLANNING constraint set (reuses `_exec_constraint_violations` with the planning margin
  substituted for r_drone). Prints one OK/WARNING line per homotopy before any rollout —
  the entire Fix_12 bug class would have been caught on job step 1 by this gate.
  Print-only; never blocks the run.

## 3. `FM_v3_uav_test/eval_artifacts.py`

- `write_eval_log`: per-rollout lines now show `flown=(L,R,L)` next to `homotopy=` when
  available.
- `plot_overview`: path color now keyed by `homotopy_flown` (falls back to the commanded
  label for non-pillars scenes).
- `write_mpc_foresight`: **bugfix** — the title's `success=` read `bool(rollout['success'])`,
  which after Fix_10's grouped schema is a dict → always 1. Now reads `success['strict']`
  (tolerates both schemas).

## What did NOT change

- No projector/solver logic, no npz/json schema removals (only the additive
  `homotopy_flown` field), no sbatch/CLI changes, no visual-aligning code (its chaotic
  artifact logic is explicitly deferred, per user).
- `_exec_constraint_violations` itself is untouched — after the geometry resize its
  surfaces all correspond to physical geometry (walls/pillars) or now-containing boxes,
  so the phantom-violation problem is fixed at the config level.

## Verification

- `py_compile` clean on `eval_fm_uav.py` + `eval_artifacts.py`.
- YAML: no yaml module in this container — structural greps verified (new inflation
  values present, old envelope lines gone, inner-face lines present); **full parse +
  first eval must be validated on the cluster**. The new feasibility-gate lines
  (`[ eval ] <scene> feasibility check: ...`) in the job log are the acceptance test:
  all homotopies should print OK for corridor/pillars/s_curve. Corridor L/R sit exactly
  on the inflated boundary (0 slack by construction of the expert data) — an OK there
  depends on the boundary being inclusive; if it warns with ~0.00 penetration, that is
  the known zero-slack case, not a regression.
- Feasibility numbers above are hand-derived in the report; no runtime executed here
  (**run on cluster**).

## Files touched

- `config/uav_projection.yaml` — inflation, pillars entry (envelope removed, box, ctypes),
  s_curve entry (inner faces, box), corridor entry (box).
- `FM_v3_uav_test/eval_fm_uav.py` — `_realized_homotopy` (new), `_warn_expert_route_infeasibility`
  (new), `rollout_one` result dict, `eval_scene` call site.
- `FM_v3_uav_test/eval_artifacts.py` — `write_eval_log`, `plot_overview`, `write_mpc_foresight`.
