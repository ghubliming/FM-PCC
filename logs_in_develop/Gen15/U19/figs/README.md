# U19 figures

| file | what | made by |
| :-- | :-- | :-- |
| `fig_u19_g0_hump_overview.png` / `.svg` | gate **G0** — the `corridor_v3_hump` constraint drawn from `config/uav_projection.yaml`: side view (raw roof, drone-centre limit at +0.384 m, tightened +0.415 m, the v2 flown band 0.86–1.24 m, launch 1.11 m, ceiling 2.80 / 2.49 m, the 0.97 m escape slot, the 0.41 m climb, the H = 0.90 second rung) + top-down (walls, caps, routes, hump footprint). **G0 PASS** (enforced peak 1.484 m > band top 1.24 m; slot 0.97 m ≥ 0.6 m). | `make_g0_hump_overview.py` (python3.14 + numpy/matplotlib/yaml, no MuJoCo; rendered locally 2026-09-22) |

Colours follow the eval's own `constraint_overview` convention (steelblue box, darkorange halfspace, crimson centre limit, tomato obstacle) so the picture matches what the cluster jobs will draw.
