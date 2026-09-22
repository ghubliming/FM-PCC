# CHANGELOG — Gen15 U18 · coding 1 (2026-09-22) · `uav_avoiding_bridge/` + Mode T pilot driver

Plan: [`PLAN_20260922_U18_pillars_v2_avoiding_bridge.md`](PLAN_20260922_U18_pillars_v2_avoiding_bridge.md) (rev 3).
Scope of this coding: everything up to a submittable Slurm script for the **Mode T pilot (gate G1)**. Mode L (live)
is wired but has not been run. **Nothing has been executed** — this container has no MuJoCo/numpy; every file is
syntax-checked only (`py_compile`, `bash -n`). First execution = the pilot job on the cluster.

## New

| file | what |
| :-- | :-- |
| `uav_avoiding_bridge/__init__.py` | package docstring (what U18 is) |
| `uav_avoiding_bridge/frame.py` | the similarity map avoiding ↔ world and the avoiding table constants (start, finish, six cylinders, field). Pure python. Round-trip checked locally. |
| `uav_avoiding_bridge/scene.py` | MJCF generator from `frame.py`; wrote `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_avoiding_pillars_s10.xml` (committed generated output: six pillars at `T(centre)`, radius 10 × physical, 2 m tall; visual-only finish line at X = 3.15; same drone include / floor / lights as `scene_pillars.xml`) |
| `uav_avoiding_bridge/plant.py` | `UavAvoidingPlant` — the `ObstacleAvoidanceEnv` stand-in with the five-call contract (`start / reset / robot_state / step / close`), avoiding units in and out. `CascadedPID` via `generator._make_pid`; `clock` (period 1/5 Hz, feed-forward `(p_des − p_des_prev)/T`) or `settle` (until `‖p − p_des‖ < 0.05 m`, cap 2 s) replay; success = `y_a > 0.35`; contact (non-floor MuJoCo contact, `generator._is_obstacle_contact`) / arena exit / > 6 m/s end the episode as failure; `check_mode` copied verbatim; per-episode sidecar (track_err, setpoint gap in avoiding units, contacts, world path); asserts the loaded XML matches the frame. |
| `uav_avoiding_bridge/scoring.py` | the avoiding violation arithmetic copied from the eval loop for a 4-D obs / 2-D action (halfspace row, keep-out disk, bounds; same check phases and conventions, documented in the header) — no torch import needed on a CPU job. Geometry selection per name copied 1:1 (`top-left` → hs[0] + ob[3]; `top-right` → hs[1] + ob[4]; `both` → hs[2], hs[3] + ob[5]). |
| `uav_avoiding_bridge/turbo.py` | **Mode T.** Discovers `<root>/**/results/halfspace_*/<variant>.npz` (skips any eval folder containing `uav`), filters by engine / train / eval glob / seeds / geos / variants, replays each episode's stored `obs_all[:, :2]` setpoints, holds the last setpoint for `--grace-s` (2 s) if the sequence runs out, rescores on the drone path, writes `<eval>_msg<tag>/<seed>/results/halfspace_<geo>/<variant>.npz` with every source key copied and the six per-episode metrics + `obs_all` replaced (`src_*` keep the Panda values, `avg_time` copied, `uav_bridge` JSON with settings). Sidecar `<variant>_uav_plant.json`, world-frame `<variant>_uav_world.png` (physical pillars + rotor reach, mapped keep-out and halfspaces, paths coloured by outcome). Prints Panda-vs-drone per cell and a per-run JSON in `plans/_uav_turbo_runs/`. Idempotent (`--force` to redo), `--dry-run`, `--limit-episodes`, `--max-cells`. |
| `uav_avoiding_bridge/factory.py` | **Mode L** switch `make_avoiding_env(default_cls)`: `FMPCC_AVOIDING_PLANT=uav` → the plant (refuses unless `FMPCC_RUN_MSG` contains `uav`, because the live results share the Panda layout); default → `ObstacleAvoidanceEnv()` unchanged. |
| `uav_avoiding_bridge/README.md` | contract, knobs, how to run both modes |
| `uav_avoiding_bridge/overview_plot.py` | constraint overview (avoiding frame ↔ world frame, three geometries, scale ladder); no MuJoCo, rendered locally with python3.14 → [`figs/`](figs/README.md). `draw_world` is shared with turbo's per-cell PNG. |
| `Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh` + `README.md` | tracked driver (not temp_bash, so it syncs by git). CPU job, no `--gres`. `MODE=pilot` (default) = FM K20 `msg20trials`, seed 6, three geometries, `diffuser` + `dpcc-r-tightened`, 20 episodes, both replays (`clock` → tag `uavpv2s10turbo`, `settle` → `uavpv2s10turboset`). `MODE=all` = the corpus. **Dry-run unless `GO=1`.** |

## Modified (hooks only, default path byte-identical)

`env = ObstacleAvoidanceEnv()` → `env = make_avoiding_env(ObstacleAvoidanceEnv)` (+ 2 comment lines, local import) in
`FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`, `FM_v3_meanflow_test/eval_flow_matching_v3_meanflow.py`,
`FM_v3_alphaflow_test/eval_flow_matching_v3_alphaflow.py`, `scripts/eval.py`, `FM_v3_hardflow_test/eval_FM_v3_hardflow.py`.
No other line of any eval, model, projector or config file changed. `mix_uav*`, `config/uav_projection.yaml` untouched.

## Conventions reproduced (so a turbo cell is comparable with its source cell)

- checked obs = reset obs + the obs after each executed step except the last; `n_steps` = executed − 1;
  bounds overshoot for k > 0 with the previous action; violating step = halfspace-or-disk only;
  `collision_free_completed = 0` whenever the episode ends without success.
- The replay flies the stored setpoints, not the stored positions; the drone's *own* position is what gets scored.
- Grace hold: 2 s at the last setpoint if the sequence is exhausted (the Panda arrives within its step); recorded per
  episode (`grace_used`). A recorded success that the drone does not reach in time is a miss — that is the finding.

## How to submit (cluster, repo root)

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh          # dry-run: lists the 6 pilot cells
GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh     # pilot (gate G1), minutes
```
Read in the log, per cell: `panda …` vs `drone …` lines, `agree k/20`, `ended {…}`, `plant track_err / gap_a p95
(panda …) / contact eps / vmax`. Pass rule (plan §4 G1): ≥ 95 % episodes agree on (success, collision_free), no
contact on a Panda-legal episode, drone `gap_a p95` close to the Panda's. If not: `HZ=3`, `EXTRA="--gain pid_high_gain"`,
the `settle` tag, then `SCALE=12`.

## Not done / next

- Not run anywhere yet (container rule). First run = pilot; expect first-run breakage in the MuJoCo/PID glue, fix on
  the log.
- `MODE=all` after the pilot passes; the live subset L1 (`FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=uavpv2s10` on the
  FM and MeanFlow eval sbatch) after that.
- DA script `pillars_v2_grid.py`, thesis note to v3: only after G2 (plan §6). `MASTER_TEST_HISTORY.md` untouched.
