# CHANGELOG — `npz_traj_visualizer`

**Date:** 2026-07-11 · **Status:** Phases 1–4 core BUILT & exporter-tested (browser QA on user side).
Plan/design: `PLAN_npz_traj_visualizer.md` (this folder). Code: `npz_analysis/npz_traj_visualizer/`.

## What it is
Rebuilds a whole eval scene from its `.npz` into an interactive HTML cockpit — every executed step-dot,
every H-step receding-horizon MPC fan, every candidate, every projection variant over the 2D env, with
DA_Code-style checkbox layers (default empty) plus built-in quantitative panels. Superset of
`compare_horizon_plans.py` (one snapshot → all snapshots, scrubbable + analyzable).

## Delivered
**`npz_traj_export.py`** (numpy+pyyaml; reuses `compare_horizon_plans` helpers):
- Reads a scene dir (flat `<v>.npz` OR nested `<v>/<v>.npz`), infers env (plans dim 4→avoiding, 6→uav),
  extracts executed path + decimated plan fans per trial, **precomputes** div_ref / violation-fraction /
  tracking-error / explosion-max / candidate-spread per step + the npz scalar-metric stat row.
- Emits `<scene>/_traj_viz/scene.json` + a **self-contained** `viewer_<scene>.html` (data inlined —
  open directly, no server). Auto `plan_every` (≤~300 fans/variant/trial). Per-file corrupt-npz
  warn+skip; graceful summary with 3 analytic spot-checks.

**`npz_traj_visualizer.html`** (vanilla JS + Canvas 2D + SVG, no CDN/frameworks, offline):
- Sidebar: trial · panel toggles · layer toggles (env bg / bone / dots / fan / mean / violation /
  horizon-end) · candidate checkboxes (0–3 + mean) · variant checklist w/ [ALL]/[NONE] · step
  scrubber + window + ▶playback · scale mode · chart toggles · PNG/SVG/CSV export. **All default OFF.**
- Canvas scene with wheel-zoom / drag-pan / dblclick-reset; linked panels (x–y and, for UAV, x–z);
  avoiding env background (halfspace triangles + obstacle circles) + red-× violation marks.
- 5 analytic charts (drift/explosion/violation/track/spread) + sortable stat table; clicking a curve
  moves the scrubber. Scale modes: fit-executed / fit-visible / fit-all / robust (1–99% clip).

## Verified (exporter, on cluster-downloaded real data)
- **avoiding** `halfspace_both-hard-imf-debug` (6 variants): env=avoiding, cols 2,3, x–y panel,
  0.13 MB; `div_ref` matches `compare_horizon_plans` exactly (dpcc-c 0.00915).
- **UAV** `s_curve...` (4 variants, nested layout): env=uav, cols 3,4,5, dual x–y/x–z panels,
  plan_every=3 (871→291 fans), 1.11 MB. **z-explosion captured** — `exec_maxabs` peaks 586.95 while
  scene panels stay on the coherent actual position (D11).
- Corrupt npz (truncated → BadZipFile) → warn+skip, good variants still export.
- Viewer JS passes `node --check`.

## Key decisions (see PLAN §13–§14)
- Code in `npz_analysis/npz_traj_visualizer/`, docs here in `logs_in_develop/` (logs = MD only).
- Selected MPC candidate is NOT stored in npz → viewer uses candidate-**mean** (labelled honestly).
- UAV scene shows actual position `p` (cols 3,4,5); `p_des_z` −587 runaway lives in the explosion curve.

## Not yet done (deferred, documented in PLAN v4)
symlog axis, edge-arrow clip markers, INSPECT fan popup table, view-state URL hash, per-layer opacity,
UAV pillar geometry, visual-aligning env, multi-trial layers.

## Run
```bash
python npz_analysis/npz_traj_visualizer/npz_traj_export.py <scene_dir> [--env uav] [--variants a,b,c]
# → open <scene_dir>/_traj_viz/viewer_<scene>.html in a browser
```
(No system Python here — test in an isolated venv `numpy pyyaml`, or run on the cluster.)
