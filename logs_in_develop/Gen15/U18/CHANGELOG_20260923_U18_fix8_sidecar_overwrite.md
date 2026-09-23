# CHANGELOG — Gen15 U18 · fix8 (2026-09-23) · live sidecars were overwritten; superseded drivers guarded

Read after the U19 chat's handover (`HANDOVER_20260923_R39_work_done_in_U19_chat.md`); nothing goes back to U19.

## Bug (mine, fix5)

The avoiding evals build a **new env, hence a new plant, per geometry × seed**. fix5's sidecar counter lived in the
closure of `make_avoiding_env`, so it restarted at 1 for every plant and every `env.close()` wrote
`uav_plant_records_<tag>_01.json` — each close overwrote the previous one. MeanFM additionally runs one python call per
seed into the same directory.

**Effect on R39 (jobs 26147–26150), had they run:** per `<engine>_K<k>` directory only the **last** geometry × seed sidecar survives
(plant stats: track_err, gap, contact steps, world path). The **npz results are unaffected** — S&C, violations, steps,
time, and the drone's executed positions (`obs_all`) are written by the eval itself. Contact vs other early ends can be
told from `obs_all` (final drone position within 0.36 m × … of a pillar surface in the world frame via `frame.py`), so
no re-run would have been needed for the table. **They were still PENDING (QOSMaxCpuPerUserLimit), so no cancel was
needed:** Slurm snapshots only the sbatch script at submit, and fix8 touches no sbatch script — `factory.py` is imported
from the repo when each job starts. A `git pull` before they start is enough; the ids 26147–26150 stay valid. The 26077 check had the pre-fix5 single-name version (last 40 episodes kept).

## Fix

`uav_avoiding_bridge/factory.py`: process-wide counter; file name
`uav_plant_records_<tag>_<process-start>_p<pid>_<nn>.json`; each file carries a `process` block (close index, pid,
argv — holds `--seed` for MeanFM —, and the relevant env: `AF_SEEDS`, `FMPCC_PROJ_CFG`, plant knobs, `SLURM_JOB_ID`).
Close order inside one process = geometry outer, seed inner. Tested locally: two plants in one process → two files.
A `git pull` on the cluster while 26147–26150 run only changes the names of later sidecars; results are untouched.

## Guarded

`live_p2_meanflow.sh`, `live_p2_dpcc.sh` (first-draft P2, superseded by `live_p23_pillars.sh`) now exit with a
SUPERSEDED message: their default tag `p23pv2live` lacks `uav` (the factory refuses it — the U19 chat's finding, correct)
and the MeanFM K1 cell would collide with R39.
