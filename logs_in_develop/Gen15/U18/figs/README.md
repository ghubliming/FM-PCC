# U18 figures — constraint overview of `pillars_v2` (pre-check before the pilot)

Generated locally (no MuJoCo needed) by `PYTHONPATH=. python3.14 uav_avoiding_bridge/overview_plot.py`
from `uav_avoiding_bridge/frame.py` + `config/projection_eval.yaml` — the same constants and the same geometry
selection the plant and the scorer use, so what is drawn here is what will be flown and scored.

| file | shows |
| :-- | :-- |
| `constraint_overview_s10_<geometry>.png` (×3) | left: the avoiding frame as planner / projector / scorer see it (six physical cylinders, the geometry's halfspace(s) with the forbidden side hatched, its ONE keep-out disk + tightened ring, start, finish line, drone footprint in planner units = 0.031). right: the world frame at scale 10 (pillars with the +0.31 m rotor-reach halo, the mapped keep-out and halfspaces, drone footprint at the start, arena box). |
| `constraint_overview_s10_all_geometries_world.png` | the three geometries side by side in the world frame |
| `constraint_overview_scale_ladder_world.png` | both-hard at scale 3 / 6 / 8 / 10 / 12: when the orange halos of a row touch, the drone cannot pass (red titles = negative slack) |

Reading: orange = physics (MuJoCo contact ends the episode, as the Panda rod does); blue = the planner's set
(halfspace + one disk per geometry, scored on the drone's centre point in the avoiding frame). The two are
different things by design, exactly as on the table.
