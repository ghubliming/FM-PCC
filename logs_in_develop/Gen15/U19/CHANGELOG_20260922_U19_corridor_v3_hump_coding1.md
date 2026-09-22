# CHANGELOG — Gen15 U19 · coding 1 (2026-09-22) · `corridor_v3_hump`: a constraint on altitude

Plan: [`PLAN_20260922_U19_corridor_v3_z_slide.md`](PLAN_20260922_U19_corridor_v3_z_slide.md).
Scope of this coding: everything up to a submittable pilot (gates G1–G3) and the full-wave driver (G4), plus gate **G0**
(offline). **Nothing has been executed on the cluster** — this container has no MuJoCo/torch. What *was* run locally
(python3.14 + numpy/matplotlib/yaml): the pure-numpy halfspace formulation, the violation scorer (extracted from the
eval module), the artifact helpers, the gate checker on synthetic data, the driver in `plan` mode, and the G0 figure.

## The idea, unchanged from the plan

Keep `corridor_v2_slide` (scene XML, model, checkpoint, routes, goals, projector stack `-bounds_free-pdes-tightened`,
metrics) and swap the test-time constraint: the 14° x–y slide is dropped, a **hump in the x–z plane** (roof 0 → 1.10 m at
x = 0 → 0 over x ∈ [−1.5, 1.5]) is added as two `x_active`-windowed halfspaces, virtual and scored only. The ceiling of
the workspace box is raised 1.80 → 2.80 m (no ceiling geom) so an escape slot exists.

## Code — one new key, `plane: xz`, on a halfspace entry

Entries without the key are byte-identical in behaviour (the default is `'xy'`; the list form is always `'xy'`).
Deviation from plan §2 row 1: instead of a third return value from `_normalize_halfspace` (and touching its ~6 call
sites), one helper per file reads the key — `_hs_plane(hs)` in `eval_mix_uav.py`, `_fs_hs_plane(hs)` in
`eval_artifacts.py` (which keeps its own copies to avoid the circular import). Same effect, fewer touched lines. Both
raise on a plane other than `xy` / `xz`.

| file · site | change |
| :-- | :-- |
| `mix_uav_test/eval_mix_uav.py` `_hs_plane` (new, after `_normalize_halfspace`) | the helper + the convention: `line = [[x, z], [x, z]]`, `'above'` = larger z feasible |
| same · `setup_dpcc_projector`, halfspace loop | per entry `_hs = {'x': _DIM['x'], 'y': _DIM['z'] if plane == 'xz' else _DIM['y']}`. `formulate_halfspace_constraints` reads only the two named indices, so the sloped branch, side convention and perpendicular tightening carry over; the `-pdes` binding is inherited through `_DIM`; HardFlow consumes the same list. **Verified locally**: the rising segment becomes `0.7333·x − z ≤ −1.5154` on columns (3, 5), i.e. z ≥ 1.515 m at the peak under `-tightened`, exactly plan §1's number. The wall rows are unchanged (column 4, d = 0.615). |
| same · `_exec_constraint_violations` | `q = p[2] if plane == 'xz' else p[1]` in the signed distance; same `r_drone`, same "clear when feasible ≥ r_drone". **Verified locally**: a level 1.11 m centre flight violates 36 of 200 steps, all at \|x\| < 0.49 m; a 1.60 m flight is clean; 2.60 m trips the raised ceiling; the `corridor_v2_slide` scoring is unchanged (blocked x ∈ [0.55, 1.98] vs U16's [0.53, 2.0]). |
| same · `plot_geo_constraints` | `_geo_extent` skips xz entries on the y axis; 3-D panel draws an xz entry as a sloped sheet spanning the y frame; top-down panel skips it; **side panel draws the raw roof + the enforced line (raw shifted by margin·√(1+s²)) with the feasible-side arrow and the `x_active` label**, and gets a legend. |
| same · `_paint_halfspaces_into_scene` (MuJoCo GIF) | xz entries skipped (no top-down footprint). |
| `mix_uav_test/eval_artifacts.py` `_fs_hs_plane` (new) | local twin of the helper |
| same · `geometry_anchors` extent loop | an xz entry's second coordinate goes to `zs`, not `ys` |
| same · `halfspace_body_boundaries(…, plane='xy')` | new keyword; filters by plane. Every existing caller keeps `'xy'`. |
| same · `step_violations` | same `q` substitution as the scorer (this is what marks the red "step in violation" dots) |
| same · `draw_projector_geometry` | xz entries: raw + enforced line on the **side** axis (foresight SVG has a real one), nothing on top-down; separate legend handle |
| same · `_draw_body_boundaries_xz` (new) + `plot_overview` | the per-variant overview's side panel now draws the roof (darkorange) and the drone-centre limit (crimson dashed) with the forbidden strip shaded, and keeps them in the z window — **this is the picture to read the pilot from** |
| `config/uav_projection.yaml` | `corridor_v3_hump` (suffix `_cv3h`, tag family never pooled with `_cv2s`) and the second rung `corridor_v3_hump_lo` (H = 0.90, suffix `_cv3hl`, **defined, not scheduled**). Neither in `active_geo_variants`; selected per job with `UAV_MIX_GEO_VARIANTS`. |
| `Data_Analysis/DA_UAV_v1/discovery.py` | **no change needed**: `geo_scene` matches the leading `corridor` token of `corridor_cv3h_…` (plan §2 row 7 checked, not edited) |

Not touched: `Fix_16` / `action_bounds='auto'`, `SCENE_FLIGHT_ENVELOPE` (the guard sits at 3.30 m, above the 2.80 m
ceiling), HardFlow, `constraints_helpers.py`.

## Driver — `Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3_hump.sh` (`git add -f`, the folder is gitignored)

Copy of the U16 paper driver with the geometry swapped. `plan` (default, submits nothing) / `smoke` (**= the pilot**) /
`submit` (the G4 wave, U16 layout: mf/fm/af at K ∈ {1, 3, 5}, diffusion K = 20, `diffuser` + `dpcc-{r,c,t}` + HardFlow
at K ≥ 3, 12 flights, seed 6). `GEO=corridor_v3_hump_lo` switches to the second rung (tag `u19cv3lo`). Pre-flight greps
for the U19 helpers, the `plane: xz` entry and the U16 fixes. Tag **`u19cv3`** (pilot: `u19smokecv3`).

Pilot = one job: **mf, seed 6, K = 3, 3 trials (L/C/R), GIF on, arms `diffuser` + `dpcc-t-bounds_free-pdes-tightened`
+ `hardflow_new-t-bounds_free-pdes-tightened`**. `bash … plan` ran locally: pre-flight passes for both geometries; the
checkpoint checks show MISS here because `logs/` lives on the cluster only.

## Gates

| gate | status | how |
| :-- | :-- | :-- |
| **G0** | **PASS** (offline) | [`figs/fig_u19_g0_hump_overview.png`](figs/fig_u19_g0_hump_overview.png) by `figs/make_g0_hump_overview.py`. Enforced peak 1.484 m (1.515 tightened) > band top 1.24 m; slot [1.515, 2.49] = 0.97 m ≥ 0.6 m; climb 0.41 m from launch; the limit crosses the launch altitude at \|x\| = 0.51; with the v2 ceiling the slot would be −0.03 m. Second rung: peak 1.284 / 1.315 m. |
| G1–G3 | not run | `tools/check_gates_u19.py "<geo folder glob>"` reads the pilot's `<variant>.npz` files (numpy only): per-arm table (collision-free, strict/relaxed success, S&C, violating trials, **violations per step** — the U12 artefact guard), G1 on `diffuser`, G2 per projected arm (strict, relaxed noted), G3 = apex executed z over x ∈ [−0.5, 0.5], projected − unprojected, same trial index, > 0.15 m on every flight. Self-tested on synthetic npz (level vs climbing flight). |
| G4 | not run | `bash … submit` after the pilot passes; DA with `DA_UAV_v1` as for U16 |

## To run (cluster)

```bash
bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3_hump.sh          # plan: lists jobs, checks checkpoints
bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3_hump.sh smoke    # THE PILOT (≈ 20 min GPU)
python logs_in_develop/Gen15/U19/tools/check_gates_u19.py \
  "logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/*/Emf_K3_*_u19smokecv3/6/corridor_cv3h*"
bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3_hump.sh submit   # only after G1–G3 pass
```

Also look at the per-variant overview PNG (side panel now shows the roof and the centre limit against the flown z)
and the foresight SVGs. If G2 fails while G3 passes: `GEO=corridor_v3_hump_lo bash … smoke`.

## Open points, stated not hidden

- The hump's two segments are gated by the **current** x only (the s_curve mechanism). Near x = 0 the rising line is
  still active while the horizon already extends past the peak, so the plan is asked for slightly more altitude than
  the falling line would require — conservative, not wrong, and the same as s_curve's corners.
- The strict goal (3-D, ≤ 0.30 m) may drop if the drone is still high at x = 2.8. The checker prints strict and relaxed
  side by side; the finish-plane success is unaffected by altitude.
- `x_active` switching rebuilds the projector every step for every arm, as it already does on the v2 slide (same cost).
- The GIF painter does not draw the roof (top-down view); the overview PNG and foresight SVG do.

---
Claude (Fable 5.1, Claude Code) · 2026-09-22 · files syntax-checked (`py_compile`, `bash -n`); numeric checks local, torch-free.
