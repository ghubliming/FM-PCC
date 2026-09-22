# U18 figures — constraint overview of `pillars_v2` (pre-check before the pilot)

Generated locally (no MuJoCo needed) by `PYTHONPATH=. python3.14 uav_avoiding_bridge/overview_plot.py`
from `uav_avoiding_bridge/frame.py` + `config/projection_eval.yaml` — the same constants and the same geometry
selection the plant and the scorer use, so what is drawn here is what will be flown and scored.

| file | shows |
| :-- | :-- |
| `constraint_overview_s36_<geometry>.png` (×3) | left: the avoiding frame as planner / projector / scorer see it (six physical cylinders, the geometry's halfspace(s) with the forbidden side hatched, its ONE keep-out disk + tightened ring, start, finish line, drone footprint in planner units = 0.031). right: the world frame at scale 36 (pillars with the +0.36 m radial rotor-reach halo, the mapped keep-out and halfspaces, drone footprint at the start, arena box). |
| `constraint_overview_s36_all_geometries_world.png` | the three geometries side by side in the world frame |
| `constraint_overview_scale_ladder_world.png` | both-hard at scale 3 / 10 / 20 / 36 / 45: the title gives the slack of a path hugging an obstacle at 0.012 units (what the Panda paths do) against the 0.36 m reach; red = the drone touches. 36 is where the rod radius maps onto the reach (fix3) |

Reading: orange = physics (MuJoCo contact ends the episode, as the Panda rod does); blue = the planner's set
(halfspace + one disk per geometry, scored on the drone's centre point in the avoiding frame). The two are
different things by design, exactly as on the table.

`pilot_s10/` (one level up) keeps the scale-10 pilot: log, run JSONs and the settle-mode world PNGs that showed the contacts.
