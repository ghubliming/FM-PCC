# Epoch 9 CHANGELOG — PCC constraints brought back (real per-scene geometry)

**Date:** 2026-07-04. Implements `../Plan/PLAN_E9_PCC_constraints.md`.
Turns the E7 dynamics-only bone into a full per-scene DPCC constraint set (bounds + halfspace
+ obstacles), activation set in the yaml, `p`-only geometric binding (DPCC-faithful), and both
`bounds` sub-roles (workspace box + restored action-magnitude bound). Sampling/eval-side only —
**no retraining, no solver change**.

## What changed

### `flow_matcher_v3_uav/utils/constraints_helpers.py` — halfspace helper robustness fix
`formulate_halfspace_constraints` divided by the slope (`n = [-1, 1/m]`), crashing on a
horizontal wall (m=0, the UAV corridor walls) and undefined on a vertical wall. Split into:
- **sloped branch** (dx≠0 and dy≠0): unchanged math → **verified byte-identical** to the old
  function on all avoiding-task/arm inputs (`max|old-new| = 0.0`, enlarge ∈ {0, 0.025}).
- **horizontal branch** (dy=0): equals the m→0 limit of the sloped formula (hand-derived).
- **vertical branch** (dx=0): new; 'above' = larger-x feasible, 'below' = smaller-x feasible.
Shared with the arm evals; the equivalence check guarantees no arm regression.

### `config/uav_projection.yaml` — per-scene geometry (visual-aligning `geo_constraint_variants` pattern)
- `active_geo_variants` + `geo_constraint_variants`: **activation lives in the yaml**. Each
  entry's `name` == scene; `constraint_types` picks which families fire.
  - `empty` → `constraint_types: []` (**no constraints applied** — marked baseline / denominator).
  - `corridor` → bounds box + 2 wall halfspaces (m=0) + 4 wall-end cap balls.
  - `pillars` → bounds box + lateral envelope halfspace + 6 cylinder `sphere_outside` (r=0.12).
  - `s_curve` → bounds box + 4 per-segment walls (each with `x_active: [lo,hi]`) + 2 corner balls.
  Geometry numbers read from the scene XMLs (`scene_*.xml`).
- `inflation: {r_drone: 0.36, margin_base: 0.05}` — always-on body-clearance offset.
  **r_drone measured** from `quadrotor_modified.xml`: rotor centers (±0.14,±0.18) + rotor radius
  0.13 → √(0.14²+0.18²)+0.13 ≈ 0.36 m (conservative rotor-tip bound; reduce toward ~0.16 m body
  radius if pillar gaps prove too tight).
- `action_bounds: {lb,ub: ±0.05}` — the **restored** DPCC-avoiding action-magnitude guard on
  the action dims (0,1,2 = Δp_des), scene-independent (§2.2 of the plan). PLACEHOLDER magnitude
  — CONFIRM from the dataset's normalized action range on the cluster.
- Altitude band z∈[0.30,1.60] and arena extents are first-cut — CONFIRM flight altitude.

### `FM_v3_uav_test/eval_fm_uav.py`
- **`load_pcc_config`**: resolves the active `geo_constraint_variants[scene]` entry by scene
  name into `constraint_types` + geometry; merges `inflation`/`action_bounds`. Scene absent from
  `active_geo_variants` → dynamics-only fallback (unchanged).
- **`setup_dpcc_projector`** (E9 additions; geometric binding stays `p`-only = DPCC-faithful):
  - `bounds` builds **two** row-sets: workspace box on `p` (6,7,8), shrunk by the spatial
    margin, **and** the `action_bounds` cap on the action dims (0,1,2), **not** inflated.
  - inflation `margin = (r_drone + margin_base) + (enlarge if -tightened)` offsets every spatial
    surface (bounds shrink, halfspace shift, obstacle radius grow).
  - halfspaces accept the list form and the dict form `{line, side, x_active}`; a wall with
    `x_active` is included only if `current_x` is in its interval (`current_x=None` → all live).
  - `_normalize_halfspace(hs)` helper normalizes both formats.
- **`_exec_constraint_violations(obs_traj, geo)`** (new): exec-time violation metrics — checks
  the FLOWN path (executed `p` = obs cols 3:6) against the RAW geometry ⊕ r_drone (physical
  collision truth, not the planning margin). Returns `(collision_free, n_violations,
  total_violations)`, now wired into the rollout in place of the hardcoded `collision_free=True`.
  `empty`/dynamics-only → trivially clean.
- **s_curve per-replan switching**: `_run_variant` builds a guarded `rebuild_projector(current_x)`
  closure **only** when the scene declares `x_active` halfspaces; `rollout_one` calls it each FM
  step (`policy.projector` is read per-call, so reassignment is safe) to re-select the active
  wall set from the drone's current x. Every other scene passes `rebuild_projector=None` and
  builds once — byte-identical to before.

## Verification (Docker; no torch/MuJoCo runtime)
- `py_compile` clean on `eval_fm_uav.py` + `constraints_helpers.py`; `uav_projection.yaml` parses,
  4 geo entries resolve.
- Halfspace helper: **byte-equal** to the old formula on all 6 avoiding sloped inputs × 2 enlarge
  values (max err 0.0); horizontal/vertical degenerate cases produce correct tightened walls.
- Full projector/SLSQP path, rollouts, and per-replan rebuild are **cluster-only** (need
  torch+MuJoCo) — validate per the plan's §6 sequence.

## Not done this epoch (follow-ups)
- **Constraint drawing on the plan/overview plots** (plan §4.2 item 5) — the Gen7 geometry
  overlay (halfspace shading, obstacle discs, bounds box) is NOT yet ported to the UAV plots.
  Exec-metric numbers are recorded; the visual overlay is the remaining observability piece.
- **Cluster validation** of every scene (esp. s_curve per-replan feasibility near the crossover
  and the action-bound sizing) — untested here by construction (no runtime).
- **`action_bounds` / altitude / r_drone magnitudes** are first-cut and flagged for cluster
  confirmation from the dataset + expert trajectories.

## Files touched
- `flow_matcher_v3_uav/utils/constraints_helpers.py` — halfspace m=0/vertical fix.
- `config/uav_projection.yaml` — per-scene geo variants, inflation, action_bounds.
- `FM_v3_uav_test/eval_fm_uav.py` — per-scene resolution, both-bounds projector, inflation,
  x_active switching, exec-violation metrics.
- Working-tree only — sync to the cluster to run.
