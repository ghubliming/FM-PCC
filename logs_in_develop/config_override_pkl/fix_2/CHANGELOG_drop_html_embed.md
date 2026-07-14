# fix_2 — npz_traj_export: drop per-scene HTML embedding (scene.json only)

**Date:** 2026-07-14
**File touched:** `npz_analysis/npz_traj_visualizer/npz_traj_export.py` (only)
**Status:** implemented, byte-compiles, run/verified locally (python3.14 + numpy 2.5.1)

## Why

The viewer is a **single reusable app**: `npz_analysis/npz_traj_visualizer/npz_traj_visualizer.html`.
It already loads a scene through its own UI — **file picker** (`<input type="file" accept=".json">`)
or **path box** (fetch by path/URL; needs an `http.server`). So stamping out a separate
`viewer_<scene>.html` (template + data inlined) per scene was redundant clutter. We now emit
**scene.json only**; you open the one app HTML and load the scene.json.

## Change

- `write_outputs(scene, out_dir, embed)` → `write_outputs(scene, out_dir)`: writes `scene.json`
  only; the template-read + `/*__SCENE_DATA__*/null` replacement + `viewer_<scene>.html` write are gone.
- Removed the now-unused `_TEMPLATE` constant and the `--no-embed` CLI flag.
- Updated module docstring, argparse description, and the run SUMMARY (now prints an `open:` hint
  pointing at `npz_traj_visualizer.html`).
- **Kept** `--debug-visual-avoiding-temp` (fix from temp_update_1) untouched.

To view any scene now:
1. open `npz_analysis/npz_traj_visualizer/npz_traj_visualizer.html` in a browser;
2. load the `scene.json` (file picker, or path box + `python3 -m http.server` as the root).

## Verify

```
/usr/local/bin/python3.14 npz_analysis/npz_traj_visualizer/npz_traj_export.py \
  "temp/old_visual_avoiding/<model>/6/results/halfspace_both-hard" \
  --env avoiding --trials all --halfspace-variant both-hard \
  --debug-visual-avoiding-temp --out logs_in_develop/config_override_pkl/fix_2
```
→ writes `fix_2/scene.json` (0.06 MB), **no HTML**. `py_compile` clean. Stray `_traj_viz/`
folders created during earlier debugging were deleted.

## ⚠️ Data note — keep temp_update_1; mpc4/6 is a fan-less outlier

`temp_update_1` (`--debug-visual-avoiding-temp`) is **NOT moot** — most old visual-avoiding scenes
DID save the MPC plan fan (`sampled_trajectories_all`, width-2 position-only), and the flag is what
lets them render. Scan of `temp/old_visual_avoiding/**/*.npz` (fans present / total npz per scene):

| scene | fans |
|---|---|
| `mpc4/6/*` (both-hard, top-left, top-right) | **0/13** — this seed saved no fans |
| `mpc4/7/both-hard` | 1/1 (partial) |
| `mpc4/7,/8,/9,/10 top-left/right` | **17/17** |
| `mpc4_filmv2/6/*` | **17/17** |

The scene exported into this `fix_2` folder happens to be `mpc4/6/halfspace_both-hard`, which is
the outlier with **no** `sampled_trajectories_all` (only `obs_all`+`act_all`) → `fans=0`, executed
paths + geometry only. That is a gap in that particular run's npz, not a viz bug and not a reason to
drop the flag. Verified working scene: `mpc4/9/halfspace_top-left-hard` → 18 fans/trial with the flag.

`fix_2/scene.json` in this folder = the both-hard export (executed paths + geometry, no fans).
