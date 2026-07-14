# temp_update_1 — `--debug-visual-avoiding-temp` for the npz trajectory viz

**Date:** 2026-07-14
**File touched:** `npz_analysis/npz_traj_visualizer/npz_traj_export.py` (only)
**Status:** implemented + run/verified locally (python3.14 + numpy 2.5.1 in-container)

## Problem

Running the npz trajectory viewer export on the **old** visual-avoiding eval outputs
(`temp/old_visual_avoiding/.../results/halfspace_*`) crashed:

```
mean_plan → snap[:, :, dims].mean(axis=0)
IndexError: index 2 is out of bounds for axis 2 with size 2
```

**Not a missing-data problem.** The old npz have everything the viewer needs:
`obs_all` = 2 trials × ~72 steps × **4** cols `[x_des, y_des, x, y]`, and
`sampled_trajectories_all` = 2 × 18 snapshots × 4 candidates × 8 horizon × **2** dims.

The crash is a **column-layout mismatch**: the old avoiding eval stored the MPC plan fan as
**position-only** (last dim = 2: x,y at cols 0,1), while the exporter assumes plans share the
obs 4-col layout and slices position at `dims = pos_cols = [2, 3]` for both obs *and* plans
(`env_col_map('avoiding')`). Width-4 obs slice fine; width-2 plan slice → out of bounds.
The newer pillars/avoiding evals store plans full-width (pos at cols 2,3), so they were unaffected.

## Change

Added an **opt-in** debug flag, off by default, that never activates in normal runs:

```
--debug-visual-avoiding-temp
```

When set, each width-2 plan snapshot is widened to a 4-col array with x,y placed at cols **[2,3]**
(cols 0,1 zero-padded), so **every existing `[2,3]`/`pos_cols` slice works unchanged** — no other
code path is modified. Implementation is one module global + one helper + three touch points, all
tagged `_DEBUG_VISUAL_AVOIDING` for easy removal:

- `_DEBUG_VISUAL_AVOIDING = False` (module global) and `_maybe_patch_avoiding_plans(snaps)` helper
  (near `_TEMPLATE`).
- `load_variant`: wrap `chp.plan_snapshots(...)` in `_maybe_patch_avoiding_plans(...)`.
- `build_scene`: set the global from `args.debug_visual_avoiding_temp` (+ one-line log).
- `main`: register the `argparse` flag.

To remove later: `grep _DEBUG_VISUAL_AVOIDING` → delete the block, flag, and wrapper call.

## Verification (run in-container)

Interpreter: `/usr/local/bin/python3.14` (numpy 2.5.1, pyyaml present). The AI container itself has
no default numpy; this standalone analysis script is not the cluster pipeline, so running it here is fine.

Scene: `temp/old_visual_avoiding/H8_K20_Meuler_T0.5_..._VTrue_mpc4/9/results/halfspace_top-left-hard`

- **Without flag:** crashes with the `IndexError` above (baseline reproduced).
- **With flag:** exports cleanly — **17 variants**, `scene.json` (0.45 MB) + `viewer_halfspace_top-left-hard.html`
  written to `.../_traj_viz/`. `off_path_frac ≈ 0.00` for nearly all fans (plan h=0 lands on the
  executed path) and `div_ref` nonzero for non-reference variants (e.g. `dpcc-c[t0]=0.283`) — i.e. the
  remap surfaces the **real** x,y, not the zero-pad.
- **Coord sanity:** exported `diffuser[t0]` plan h=0 = `[0.5249, -0.2797]` == executed path[0]
  `[0.5249, -0.2797]`; avoiding geometry loaded (1 halfspace, 1 obstacle) for `top-left-hard`.

## Usage

```bash
/usr/local/bin/python3.14 npz_analysis/npz_traj_visualizer/npz_traj_export.py \
  "temp/old_visual_avoiding/<model>/9/results/halfspace_top-left-hard" \
  --env avoiding --trials all --debug-visual-avoiding-temp
```

Output: `<scene_dir>/_traj_viz/{scene.json, viewer_<scene>.html}` (open the HTML directly; data inlined).
