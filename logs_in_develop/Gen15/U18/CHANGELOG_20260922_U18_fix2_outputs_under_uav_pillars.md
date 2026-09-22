# CHANGELOG — Gen15 U18 · fix2 (2026-09-22) · Mode T outputs live in the UAV-pillars scene folder

**Why.** `pillars_v2` is the UAV pillars scene (the author's call: "technically still uav pillars"). Coding 1 wrote the
turbo outputs beside their avoiding sources (`logs/avoiding-d3il/plans/…/<eval>-uavpv2s10turbo/`), which files them
under the D3IL task. They now go where the other UAV-pillars runs are.

**New layout (default).**
```
logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/<engine>/<train>/<eval>_msg<tag>/<seed>/results/halfspace_<geo>/
    <variant>.npz  <variant>_uav_plant.json  <variant>_uav_world.png
logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_uav_turbo_runs/turbo_<tag>_<stamp>.json
```
`<engine>/<train>/<eval>…/<seed>/results/halfspace_<geo>/` is the avoiding relative layout, kept on purpose: the
avoiding loaders and the batch reporter (`DA_Code_v3/main_da_batch.py --parent-path …/avoiding_bridge`, recursive
discovery) work unchanged when pointed at this root. The UAV reporter (`DA_UAV_v1`, `results.json` schema) does not
see these folders, which is right — they carry the avoiding npz schema.

**Changed.**
| file | change |
| :-- | :-- |
| `uav_avoiding_bridge/turbo.py` | `--out-root` (default `logs/UAV_MIX/uav-pillars/plans/avoiding_bridge`); `discover()` mirrors the source's path relative to `--root` below it; `--in-place` restores the coding-1 layout; the run JSON goes to `<out-root>/_uav_turbo_runs/`; the plan print shows sources and outputs |
| `Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh` | `OUT_ROOT` env (same default), passed as `--out-root`; header comment |
| READMEs (package + sbatch), plan §3.4 | paths updated |

**If the second pilot (after the GL fix) already wrote in place**, those folders are harmless duplicates; remove them so
nothing under the D3IL tree carries a UAV tag:
```bash
find logs/avoiding-d3il/plans -maxdepth 4 -type d -name '*uavpv2*' -exec rm -r {} +
```
(the cells are re-flown into the new root by the next run — seconds each.)
