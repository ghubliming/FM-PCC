# fix_2.2 — paste-scene.json-path workflow (verification, no code change)

**Date:** 2026-07-14
**Files touched:** none (verification only)
**Status:** confirmed working as-is

## Question

"When I run the npz-analysis export it should output under the scene folder (like the other
npz-analysis outputs), and I just want to paste the scene.json path into the viewer HTML I opened
in the browser. Are we already there?"

## Answer: yes, already there — nothing to fix.

1. **Output location is already under the scene folder.** `npz_traj_export.py` with no `--out`
   defaults to `<scene_dir>/_traj_viz/scene.json`, i.e. right beside the npz, same as the other
   per-scene analysis outputs (pngs, results.json). Confirmed present:
   ```
   temp/old_visual_avoiding/..._VTrue_mpc4_filmv2/6/results/halfspace_both-hard/_traj_viz/scene.json
   ```

2. **The viewer already loads by pasted path.** `npz_traj_visualizer.html` has a path box whose
   `buildPathCandidates()` shrinks a pasted absolute path from the front (longest match first) until
   it resolves against the HTTP server root — built exactly so a full `/workspaces/FM-PCC/...`
   filesystem path Just Works regardless of where the server was started.

## The one requirement (browser rule, not a bug)

The path box uses `fetch()`, which browsers **block over `file://`**. So paste-and-load needs a
static server:

```bash
cd /workspaces/FM-PCC && python3 -m http.server 8000
# open:  http://localhost:8000/npz_analysis/npz_traj_visualizer/npz_traj_visualizer.html
# paste: /workspaces/FM-PCC/temp/old_visual_avoiding/.../halfspace_both-hard/_traj_viz/scene.json
# -> Load
```

Alternative with **no server**: open the HTML directly and use the **file picker** — pick the same
`scene.json`. Always works over `file://`.

## Reminder of the pipeline (why you load scene.json, not npz)

`*.npz  --[python npz_traj_export.py, reads npz with numpy]-->  scene.json  --[browser]-->  viewer`.
The browser never reads npz (it can't parse the NumPy format); the export step already distilled all
17 variants' paths + plan fans + geometry into `scene.json`. See fix_2 (dropped per-scene HTML
embedding) and temp_update_1 (`--debug-visual-avoiding-temp` widens legacy width-2 avoiding plan fans).
