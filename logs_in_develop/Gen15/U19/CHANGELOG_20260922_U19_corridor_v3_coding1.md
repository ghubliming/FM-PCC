# CHANGELOG — Gen15 U19 · coding 1 (2026-09-22) · `corridor_v3_tilt` (the v3) + `corridor_v3_ablation_hump`

Plan: [`PLAN_20260922_U19_corridor_v3_z_slide.md`](PLAN_20260922_U19_corridor_v3_z_slide.md) — **the plan's hump is not
the v3.** The author's v3 (this session): *"corridor v2 xy slide halfspace with strong z-axis negative angle so it
becomes xyz — the slide forces you also in the negative z direction beside the xy."* That is **`corridor_v3_tilt`**. The
plan's hump is kept as the **ablation `corridor_v3_ablation_hump`**; the pilot runs both.

Scope: everything up to submittable pilots (gates G1–G3) and the full-wave driver (G4), plus gate **G0** offline for both.
**Nothing has been executed on the cluster** — this container has no MuJoCo/torch. Run locally (python3.14 + numpy /
matplotlib / yaml): the halfspace rows, the violation scorer (extracted from the eval module), the artifact helpers, the
gate checker on synthetic data, the driver in `plan` mode, and both G0 figures. **Pure re-evaluation: no retraining**
(`scene: corridor` keeps dataset, checkpoint, routes, goals, budget; the policy never sees the constraint).

## 1 · `corridor_v3_tilt` — the v2 slide leaned over

`corridor_v2_slide` **plus one key** on the slide entry:

```yaml
- {line: [[-2.0, 0.95], [2.0, -0.05]], side: 'below', x_active: [-2.0, 2.0], z_lean: {deg: -60, z_ref: 1.11}}
```

Feasible signed distance `s(p) = s_xy(x, y) + tan(deg)·(z − z_ref)`; `deg < 0` → descending gains room (the wall leans over
the drone). At `z_ref` (the launch altitude) the cut is exactly the v2 line. Scene XML, walls, caps, **workspace box
(ceiling 1.80 — the room is below, floor 0.30 + 0.31 = 0.61, 0.50 m under the launch)**, model, checkpoint, routes, goals,
projector stack and metrics are byte-identical to v2. Virtual like the slide: scored, not a geom.

**The numbers** (`figs/make_g0_tilt_overview.py`, verified against the extracted scorer and the projector row):

| | L (y −0.12) | C (y 0) | R (y +0.12) |
| :-- | :-- | :-- | :-- |
| lateral centre limit at z_ref | 0.31·√(1+tan²60) = **0.62 m** from the line (v2: 0.31) | | |
| sideways only, at launch altitude | y → −0.69: **infeasible** (wall limit −0.64) | same | same |
| descend only, at the exit x = 2 | 0.32 m | 0.39 m | 0.45 m (< 0.50 m room) |
| min-norm correction (along the plane normal: 87 % z, 50 % x–y) | 0.24 m down + 0.14 m sideways | 0.29 + 0.17 | 0.34 + 0.20 |
| binds from x | −0.27 | −0.75 (v2: +0.53) | −1.23 |

So z is **mandatory** on every route, not optional, and the ramp is longer and gentler than v2's. **G0 PASS**.

## 2 · `corridor_v3_ablation_hump` — the plan's geometry, as an ablation

No slide; two `plane: xz` halfspaces form a roof 0 → 1.10 m at x = 0 → 0 over x ∈ [−1.5, 1.5]; ceiling raised 1.80 → 2.80.
Enforced peak 1.484 m (1.515 tightened) vs the flown band top 1.24 m; climb 0.41 m; escape slot 0.97 m. Folder suffix
`_cv3ah`; the milder rung `corridor_v3_ablation_hump_lo` (H = 0.90, `_cv3ahl`) is defined, not scheduled. **G0 PASS**.

## 3 · Code — two opt-in keys on a halfspace entry, byte-identical without them

| file · site | change |
| :-- | :-- |
| `mix_uav_test/eval_mix_uav.py` `_hs_plane` / `_hs_lean` / `_hs_normal3` (new, after `_normalize_halfspace`) | the two keys. `plane: xz` reads the line as `[[x, z], [x, z]]`; `z_lean: {deg, z_ref}` leans an x–y line about itself and returns the unit 3-D normal into the feasible side + a point on the plane. Both validate their input. |
| same · `setup_dpcc_projector`, halfspace loop | leaned entry → one row on (x, y, z): `−n3·p ≤ −(n3·P0 + margin)`, margin ⊥ to the leaned plane (the body radius). `plane: xz` → `_hs = {'x': _DIM['x'], 'y': _DIM['z']}` into the unchanged 2-D formulation. `-pdes` binding inherited through `_DIM`; HardFlow consumes the same list. **Verified**: tilt row `C[3:6] = [0.121, 0.485, 0.866]`, `d = 0.845` (unit normal, margin 0.335); hump row `0.733·x − z ≤ −1.515`; wall rows unchanged. |
| same · `_exec_constraint_violations` | leaned entry: `pen = max(0, r − n3·(p − P0))`; xz entry: z in place of y. **Verified**: level 1.11 m flights violate on all three routes from the binding x above; a descent to 0.70 m by the exit is clean; a level 0.62 m flight is clean; `corridor_v2_slide` scores exactly as before (52 steps / 9.389). |
| same · `plot_geo_constraints` | leaned entry: 3-D panel a tilted sheet, top-down the cut at z_ref with a lean note and the widened centre limit, **side panel the cut along y = 0 (raw + enforced)**; xz entry: sheet / skipped / raw + enforced roof. Legend on the side panel. |
| same · `_paint_halfspaces_into_scene` (GIF) | leaned entry: the painted wall is the z_ref cut, its centre-limit strip at r·√(1+t²); xz entry skipped. |
| `mix_uav_test/eval_artifacts.py` `_fs_hs_plane` / `_fs_hs_lean` / `_fs_hs_normal3` / `_fs_lean_cut_xz` (new) | local twins (no circular import) |
| same · `halfspace_body_boundaries(…, plane='xy')` | new keyword. `'xy'` (every existing caller): a leaned entry's z_ref cut with the widened limit. `'xz'`: roofs, **and a leaned entry's y = 0 cut** (raw + centre limit). |
| same · `step_violations` | the scorer's rule for both keys (drives the red "step in violation" dots) |
| same · `draw_projector_geometry`, `_draw_body_boundaries_xz` (new), `plot_overview`, `geometry_anchors` | the per-variant overview's **side panel now draws the plane's y = 0 cut / the roof plus the drone-centre limit**, kept in the z window — the picture to read the pilot from. Foresight SVG gets the same overlay on its real x–z axis. |
| `config/uav_projection.yaml` | `corridor_v3_tilt` (`_cv3t`), `corridor_v3_ablation_hump` (`_cv3ah`), `corridor_v3_ablation_hump_lo` (`_cv3ahl`). None in `active_geo_variants`; selected per job with `UAV_MIX_GEO_VARIANTS`. |
| `Data_Analysis/DA_UAV_v1/discovery.py` | no change needed: `geo_scene` matches the leading `corridor` token of `corridor_cv3t_…` / `corridor_cv3ah_…` |

Not touched: `Fix_16` / `action_bounds='auto'`, `SCENE_FLIGHT_ENVELOPE`, HardFlow, `constraints_helpers.py`.

## 4 · Driver — `Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3.sh` (`git add -f`; the folder is gitignored)

Copy of the U16 paper driver. `plan` (default, submits nothing) / **`smoke` = the pilots: two jobs, `corridor_v3_tilt`
and `corridor_v3_ablation_hump`** (mf, seed 6, K = 3, 3 trials L/C/R, GIF on, arms `diffuser` +
`dpcc-t-bounds_free-pdes-tightened` + `hardflow_new-t-bounds_free-pdes-tightened`; tags `u19smokecv3t`, `u19smokecv3ah`)
/ `submit` = the G4 wave for `GEO` (default tilt, tag `u19cv3t`; `GEO=corridor_v3_ablation_hump` → `u19cv3ah`; U16 layout:
mf/fm/af at K ∈ {1, 3, 5}, diffusion K = 20, 12 flights, seed 6). `PILOT_GEOS` picks the pilots. Pre-flight greps for both
keys, both helpers and the U16 fixes. `plan` ran locally for both geometries: pre-flight passes; checkpoints show MISS
here because `logs/` lives on the cluster only.

## 5 · Gates

| gate | tilt | ablation hump | how |
| :-- | :-- | :-- | :-- |
| **G0** | **PASS** | **PASS** | [`figs/`](figs/README.md) — `fig_u19_g0_tilt_overview`, `fig_u19_g0_ablation_hump_overview` |
| G1–G3 | not run | not run | `tools/check_gates_u19.py --geo tilt|hump "<geo folder glob>"` reads the pilot's `<variant>.npz` (numpy only). Per-arm table (collision-free, strict/relaxed success, S&C, violating trials, **violations per step** — the U12 artefact guard). G1 on `diffuser`. G2 per projected arm. **G3 is geometry-aware**: tilt = min executed z over x ∈ [0.5, 2.0], projected − unprojected, **< −0.15 m** (descent); hump = max z over x ∈ [−0.5, 0.5] **> +0.15 m** (climb); every flight, same trial index. Self-tested on synthetic npz. |
| G4 | not run | not run | `submit` after its pilot passes; DA with `DA_UAV_v1` as for U16 |

## 6 · To run (cluster)

```bash
bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3.sh            # plan
bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3.sh smoke      # BOTH PILOTS (≈ 2 × 20 min GPU)
python logs_in_develop/Gen15/U19/tools/check_gates_u19.py --geo tilt "logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/*/Emf_K3_*_u19smokecv3t/6/corridor_cv3t_*"
python logs_in_develop/Gen15/U19/tools/check_gates_u19.py --geo hump "logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/*/Emf_K3_*_u19smokecv3ah/6/corridor_cv3ah_*"
bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3.sh submit                                   # tilt wave
GEO=corridor_v3_ablation_hump bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3.sh submit     # hump wave
```

Also read each pilot's per-variant overview PNG (side panel) and the foresight SVGs.

## 7 · Open points, stated not hidden

- **Tilt: the strict goal.** The finish plane at x = 2.8 ignores altitude, but the strict goal (3-D, ≤ 0.30 m) may drop if the
  drone is still 0.3 m low there. The checker prints strict and relaxed side by side; report both, as Ch 5/6 do.
- **Tilt: min-norm is the projector's choice, not a law.** DPCC's per-step projection moves along the row's normal, so
  ~87 % of the correction should be in z; HardFlow's NLP may split differently. G3 reads the executed z, whichever way.
- **Hump: current-x gating.** Near the peak the rising segment is still active while the horizon already extends past it —
  conservative, as s_curve's corners.
- Every arm rebuilds the projector each step (`x_active` switching), as on the v2 slide — same cost.
- The GIF painter draws the tilt's z_ref cut only (top-down view); the overview PNG and foresight SVG show the side cut.

---
Claude (Fable 5.1, Claude Code) · 2026-09-22 · files syntax-checked (`py_compile`, `bash -n`); numeric checks local, torch-free.
