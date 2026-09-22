# `uav_avoiding_bridge/` — Gen15 U18 · `pillars_v2`

The D3IL-avoiding planner (any of the five arms: FM, MeanFlow, CI-MeanFlow, diffusion-DPCC, HardFlow) executed by
the quadrotor in a scene that is the avoiding obstacle field scaled by `frame.SCALE` (10) and extruded into pillars.
**Nothing on the planner side changes**: models, normaliser, projectors, the three test-time geometries, the scorer
and the result layout are the avoiding ones. Only the plant (Panda + IK → quadrotor + CascadedPID) is swapped.

| file | role |
| :-- | :-- |
| `frame.py` | the similarity map avoiding ↔ world (`X = s(y_a − 0.035)`, `Y = −s(x_a − 0.5)`, `Z = 1.0`), the avoiding table constants, env knobs `FMPCC_AVOID_UAV_SCALE` / `_ALT`. Pure python. |
| `scene.py` | generates `d3il/…/quadrotor/scenes/scene_avoiding_pillars_s<scale>.xml` from `frame.py` (`python uav_avoiding_bridge/scene.py`). The committed XML is generated output. |
| `plant.py` | `UavAvoidingPlant`: the `ObstacleAvoidanceEnv` stand-in (`start/reset/robot_state/step/close`, avoiding units in and out). PID tracking for one control period per setpoint (`clock`, 5 Hz, velocity feed-forward) or until settled (`settle`). Contact / divergence end the episode as a failure. Records a per-episode sidecar. Asserts the scene matches the frame at load. |
| `scoring.py` | the avoiding violation arithmetic copied from the eval loop (halfspace / keep-out / bounds, same phases, same conventions) so Mode T scores exactly as Mode L. |
| `turbo.py` | **Mode T**: walk `logs/avoiding-d3il/plans`, replay every episode's stored setpoint sequence, rescore on the drone path, write the mirrored `<eval>_msg<tag>/…/<variant>.npz` (+ `_uav_plant.json`, `_uav_world.png`). `--dry-run` first. |
| `factory.py` | **Mode L**: `make_avoiding_env(ObstacleAvoidanceEnv)` — the one-line hook in the five eval scripts. `FMPCC_AVOIDING_PLANT=uav` selects the plant; requires `FMPCC_RUN_MSG` containing `uav`. |

## Run

```bash
# Mode T (CPU): plan, pilot, corpus  — see Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh
python uav_avoiding_bridge/turbo.py --dry-run
python uav_avoiding_bridge/turbo.py --engine flow_matching_v3_ode_selectable --eval-glob '*K20*msg20trials' \
       --seeds 6 --geos both-hard --variants diffuser dpcc-r-tightened
# Mode L (GPU): any avoiding eval sbatch, plant switched by the environment
FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=uavpv2s10 python FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py --flow-steps 20 --seed 6
```

Knobs: `FMPCC_AVOID_UAV_HZ` (5), `FMPCC_AVOID_UAV_FF` (1), `FMPCC_AVOID_UAV_GAIN` (`pid_default`),
`FMPCC_AVOID_UAV_REPLAY` (`clock`|`settle`), `FMPCC_AVOID_UAV_SCALE` (10), `FMPCC_AVOID_UAV_ALT` (1.0).

## Reading the output

Same npz keys as the source; `src_*` keep the Panda values, `avg_time` is copied (the plant is not timed),
`uav_bridge` (JSON string) records the plant settings. The sidecar carries per-episode `track_err`, the
setpoint-minus-position gap in avoiding units (`gap_a`, compare with the Panda's `panda_gap_a_p95` in the
summary — gate G3), contact steps, divergence and the world-frame path.

Plan, gates and decisions: `logs_in_develop/Gen15/U18/PLAN_20260922_U18_pillars_v2_avoiding_bridge.md`.
