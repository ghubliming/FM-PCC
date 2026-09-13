# U15 rev2 — slide halfspace + plots/GIF that show what blocks the drone

**Date:** 2026-09-13 · **Gen:** 15 · **Follows:** [`CHANGELOG_20260913_corridor_gate_n1.md`](CHANGELOG_20260913_corridor_gate_n1.md) (rev1, job 25728)
**Run with:** the same script, `Slurm_Codes/temp_bash/eval_20260913_u15_corridor_gate_n1.sh`, after deleting the remote `corridor_hggn_...` results of 25728 (same folder name).

## 1. Why rev2

After 25728, three complaints:
1. **The plots showed "0 block".** The per-variant PNG drew neither the gate nor the drone's
   width, so a gate cutting 12 cm into the body looked like it missed the route.
2. **The gate looked like a spike, not a slide.** U14's line started mid-corridor at (0.40, 0.23)
   and rose to the wall, so its free end pointed into the path.
3. **No GIF on the remote.** The GIF builder only ran locally.

## 2. Config — `config/uav_projection.yaml`

`corridor_gate_n1` gate line replaced. **U14's `corridor_gate` is unchanged**: `git diff HEAD` shows
only the `corridor_gate_n1` lines, and U14's entry still parses to `[[-2.0,-0.25],[2.0,0.55]]`, x_active [0.40, 2.0].

```yaml
# rev1 (U14 spike):  {line: [[-2.0, -0.25], [2.0, 0.55]],  side: 'below', x_active: [0.40, 2.0]}
# rev2 (slide):      {line: [[ 0.0,  0.45], [2.0, 0.212]], side: 'below', x_active: [ 0.0, 2.0]}
```

The slide leaves the top wall face flush at x = 0 (its centre limit +0.138 vs the wall's +0.140).
It descends 6.8° and releases the drone at the corridor exit (x = 2.0), where the walls end.

| scorer basis (r_drone 0.31) | value |
| :-- | :-- |
| drone-centre limit at x = 2.0 | y ≤ −0.100 |
| route L (y −0.12) | clean for all x (19.8 mm spare) |
| route C (y 0.00) | blocked x ∈ [1.16, 2.00] |
| route R (y +0.12) | blocked x ∈ [0.15, 2.00] |
| sliding needs | 5.2e-03 m/step = 24% of the FRAC = 1.0 cap |

**Why the line cannot be drawn crossing the route:** the corridor is 0.90 m and the drone 0.62 m.
Any line that visibly crosses a route leaves less than a drone underneath, so the corridor is sealed.
See `temp/geo_demo/U15_block_truth.png`. The rev2 PNG draws the drone-centre limit, which does cross the routes.

Alternatives sized: start x −1.0 / −0.5 / 0.5 / 1.0 → 4.6° / 5.5° / 8.9° / 13.0°. Steeper
is shorter, because the line can only drop 0.24 m.

## 3. Code — plotting only, no effect on any metric

| file | change |
| :-- | :-- |
| `mix_uav_test/eval_artifacts.py` | new `halfspace_body_boundaries`, `step_violations` (mirror of the scorer rule), `_draw_body_boundaries` |
| | `plot_overview(..., geo_config=None, variant_flags=None)`: top-down panel now draws the enforced geometry, the drone-centre limit with the forbidden strip shaded, each trial's body width (2 × r_drone), and violating steps in red. y is zoomed to bodies + halfspaces, legend is below the panel. Altitude panel unchanged: the geometry helper's `cz_mid` bug is kept out of it |
| | new `save_trajectory_gif(...)`: `<variant>_traj.gif`, one animated top-down per variant with all trials, drone body circles that turn red on violating steps, ≤ 110 frames, matplotlib + Pillow, no EGL |
| `mix_uav_test/eval_mix_uav.py` | `_run_variant`: passes `geo_config=config, variant_flags=variant` to `plot_overview`, then writes the GIF inside `try/except` so it can never fail the eval. `UAV_MIX_TRAJ_GIF=0` turns it off |
| `Slurm_Codes/temp_bash/eval_20260913_u15_corridor_gate_n1.sh` | header/notes only: rev2 description and pass criteria. Variables and submit line unchanged |

Unchanged: scorer, projector, HardFlow, normalizer, controller, model, every other geo entry.
`FM_v3_uav_test/eval_fm_uav.py` calls its own `plot_overview` with the old signature. The new
arguments default to `None`, and a local test of the old call signature passed.

### Why not the existing `--record gif`

It renders MuJoCo frames. The gate is a virtual constraint and not an object in the scene, so those
GIFs could never show it. It also needs EGL rendering on the GPU node.

## 4. Local verification (Docker, `python3.14`, no eval run)

Fed the rev1 25728 rollout traces into the new functions, with the rev2 slide geometry:
- `py_compile` passes for both files; `bash -n` passes for the script.
- `plot_overview` works with the new and the old signature, ~1 s.
- `save_trajectory_gif` gives 100 frames, ~15 s per variant.
- Frames inspected: slide, drone-centre limit, forbidden strip, body circles, red on violation, legend clear of the corridor.

⚠️ The legacy obstacle circles from `draw_projector_geometry` still use `r_drone + margin_base`
(0.33), while the scorer uses 0.31. This predates rev2 and is a 2 cm difference in the drawing only.

## 5. Still open from rev1 (not addressed by rev2)

The FRAC = 1.0 side effects seen in 25728 are a property of the setting, not of the geometry:
- the diffuser drifted 6–7 cm;
- `dpcc-t-bounds_free` overshot after x = 2 and missed the goal;
- `hardflow_new` flipped in both trials.

rev2 does not change them. The rerun will show whether the slide geometry makes them better or worse.
