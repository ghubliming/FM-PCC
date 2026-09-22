# U19 figures

| file | what | made by |
| :-- | :-- | :-- |
| `fig_u19_g0_tilt_overview.png` / `.svg` | gate **G0** of **`corridor_v3_tilt`** (THE v3): the v2 slide leaned −60° about z_ref = 1.11 m. Top-down: the plane's cut at the launch altitude (= the v2 line) with the drone-centre limit 0.62 m from it, and the cuts lower down (room grows with descent); the v2 vertical-wall limit for reference. Side view: the plane cut along routes L / C / R with the centre limit, floor 0.30 / 0.61 m, ceiling 1.80 m, the v2 flown band, the descend-only demand at the exit, where each route starts to bind; a table of the per-route demands (descend-only 0.32 / 0.39 / 0.45 m; sideways-only infeasible on every route at launch altitude; min-norm 0.24–0.34 m down + 0.14–0.20 m sideways). **G0 PASS**. | `make_g0_tilt_overview.py` |
| `fig_u19_g0_ablation_hump_overview.png` / `.svg` | gate **G0** of the ablation **`corridor_v3_ablation_hump`**: the x–z roof (H = 1.10 m), the drone-centre limit at +0.384 m, tightened +0.415 m, the v2 flown band, the ceiling raised to 2.80 / 2.49 m, the 0.97 m escape slot, the 0.41 m climb, the H = 0.90 second rung; top-down: walls, caps, routes, hump footprint. **G0 PASS**. | `make_g0_ablation_hump_overview.py` |

Both scripts: python3.14 + numpy/matplotlib/yaml, read `config/uav_projection.yaml`, no MuJoCo, rendered locally 2026-09-22.
Colours follow the eval's own `constraint_overview` convention (steelblue box, darkorange halfspace, crimson centre limit, tomato obstacle) so the pictures match what the cluster jobs draw.
