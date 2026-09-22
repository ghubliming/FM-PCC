# CHANGELOG — Gen15 U18 · fix3 (2026-09-22) · pilot 26073 read: scale 10 → 36, clock mode repaired

Pilot job 26073 (scale 10, FM K20 `msg20trials`, seed 6, 3 geometries × {`diffuser`, `dpcc-r-tightened`}, 20 episodes,
`clock` and `settle`) ran end to end; log and the settle-mode world PNGs are kept in [`pilot_s10/`](pilot_s10/).
The software worked; the numbers said two things were wrong. Both are diagnosed with the downloaded sidecars and
with **local replays** (MuJoCo now installs into the container's `python3.14`; four stored avoiding episodes from
`temp/` were flown here, which is debugging, not the pipeline).

## 1 · Contacts are geometry: the scale must map the rod onto the drone (36×, not 10×)

| evidence | value |
| :-- | :-- |
| settle mode (tracking error p95 0.05 m, speed 0.15 m/s) | **15–18 of 20 episodes end in drone–pillar contact**, in every cell |
| drone-centre distance to the pillar surface at the moment of contact (sidecar) | 0.28–0.36 m = the X2's radial rotor reach (0.36 m: rotor r 0.13 at (±0.14, ±0.18)) |
| episodes that survived: minimum clearance | 0.33–0.55 m |
| Panda's own executed paths (4 local episodes): minimum surface clearance to the row-2/3 cylinders | **0.009–0.02 avoiding units** = 0.09–0.2 m at 10× |

The demonstrations hug the obstacles at the rod's own radius (0.01). The plan's "0.05 units of centred clearance" was
wrong: the faithful similarity maps the **rod radius onto the drone's radial reach**, 0.36 / 0.01 = **36**. At 36× the
four local episodes replay contact-free in three cases; the fourth (0.009 clearance, the rod nearly touching on the
table) still touches — that is the honest transfer of a marginal path. Arena 21.6 × 25.2 m, pillars r 0.9 / 1.08 m,
keep-out 2.88 m; nothing else changes (the planner's set scales with everything).

## 2 · Clock mode: velocity feed-forward flips the drone in this PID

Trace (local, 10×, 5 Hz): the stored setpoints zig-zag (the planner's per-step actions alternate ±0.07 units in y);
the feed-forward `v_des` flipped ±0.6 m/s each period, the attitude loop saturated all four motors (u = 0 / 6.5 N
bang-bang), tilt reached 50–70°, the drone dropped and hit the 6 m/s abort. A rate-limited reference did not cure it:
**any sustained `v_des` above ≈ 0.5 m/s diverges** with `CascadedPID` (Kp [4,4,8], Kd [3,3,4]), at every scale and
rate tried. Without feed-forward the PD loop simply trails the moving reference by Kd/Kp·v ≈ 0.2–0.3 m, which at 36×
is 0.007 avoiding units — below the Panda's own setpoint gap (0.02–0.04).

| clock setting (36×) | result on the 4 local episodes |
| :-- | :-- |
| 1 Hz, ff on, v_max 0.6 / 1.0 | diverged (|v| 6 m/s) in 4 / 4 |
| 1 Hz, ff on, v_max 0.4 / 0.3 | stable but the reference lags 1–8 m behind |
| **1 Hz, ff off, v_max 1.0** | **stable; same outcomes and stop steps as settle; track_err mean 0.16–0.22 m, gap_a p95 0.007** |
| settle (v_des 0, until < 0.05 m, cap 4 s) | stable; the geometric bound |

## Changed

| file | change |
| :-- | :-- |
| `uav_avoiding_bridge/frame.py` | `SCALE` default 36 (env `FMPCC_AVOID_UAV_SCALE`); `DRONE_REACH_M = 0.36` (radial), `ROD_RADIUS_A = 0.01`; docstring numbers |
| `uav_avoiding_bridge/plant.py` | clock mode tracks a rate-limited reference (`v_max`, default 1.0 m/s) instead of jumping `p_des`; `feedforward` default **False**; `control_hz` default 1; `settle_max_s` 2 → 4 (0.5 m steps need it); sidecar gains `ref_lag_m` / `ref_lag_p95_m`; reach constant from `frame` |
| `uav_avoiding_bridge/turbo.py` | `--vmax` (1.0), `--hz` default 1, `--ff` opt-in replaces `--no-ff`; world PNG halo uses 0.36 |
| `uav_avoiding_bridge/overview_plot.py` | reach 0.36; scale ladder 3 / 10 / 20 / 36 / 45 with the hugging-path slack in the titles; header numbers |
| `Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh` | `SCALE=36`, `HZ=1`, `VMAX=1.0`; tags become `uavpv2s36turbo` / `uavpv2s36turboset` automatically |
| `d3il/…/scenes/scene_avoiding_pillars_s36.xml` | generated (s10 kept for the pilot record) |
| `logs_in_develop/Gen15/U18/figs/` | regenerated at 36× |

## Standard artifact suite per cell (same day, on request) — CPU, no GIF

Every turbo cell now writes the same kind of suite the UAV evals write, from matplotlib only:

| file | what |
| :-- | :-- |
| `<variant>.npz` | avoiding schema, rescored on the drone (`src_*` = Panda values, `avg_time` copied) |
| `<variant>.png` | the avoiding eval's per-cell figure, for the drone: rows = first 10 episodes; x / y / x_des / y_des over steps (black drone, grey dashed Panda), path against the constraints, path + the **stored** MPC foresight fans (every H/2 steps, ≤ 4 candidates) |
| `eval_<variant>.log` | Panda vs drone summary lines, plant stats, per-episode table (ended / success / S&C / cf / violations / steps / grace) |
| `<variant>_uav_world.png` | all paths in the world frame over pillars + keep-out |
| `<variant>_uav_plant.json` | sidecar: per-episode tracking, gap, contacts, world path |
| `diagnostics/<variant>/rollout_<i>_mpc_foresight.svg` | **the UAV suite's own writer** `mix_uav_test/eval_artifacts.write_mpc_foresight`, unchanged: LEFT XY top-down (green stored candidate fan, black p_des, red drone, replan anchors, start ★ / end ■), RIGHT XZ altitude; the avoiding halfspaces + keep-out disk and the six pillars are mapped into the world and passed as its `geo_config` (`artifacts.uav_geo_config`), the episode packed into its rollout schema (`artifacts.uav_rollout`). First `--foresight` episodes per cell (3; sbatch `FORESIGHT`). |

New module `uav_avoiding_bridge/artifacts.py`. Exercised locally on two stored episodes (36×, settle) — sample in
[`sample_suite_local_s36/`](sample_suite_local_s36/). The fans are the planner's decisions on the table; the replay
does not plan, so the SVG shows *what the planner foresaw* against *what the drone flew*.

## GIF recording (GPU only, on request)

Reuses the UAV-pillars machinery, not a new drawer: the overhead free camera of
`uav_expert_data_collect/generate_trajectory_gifs._render_overhead` (looking straight down at the drone; distance is a
knob, 8 m, because the 36× scene is far larger than the trained UAV scenes) and the writer
`mix_uav_test/eval_artifacts.save_rollout_gif` (imageio, palette 64, sub-rectangles). One `mujoco.Renderer` per plant,
freed in `close()`. Frames every 20 physics steps (5 sim-fps) plus one on contact, capped at 900 per episode; text
burn-in = step, z, |v|, tilt, CONTACT. Files: `<cell>/diagnostics/<variant>/rollout_<i>.gif`.

| file | change |
| :-- | :-- |
| `plant.py` | `enable_gif()`, `gif_on`, `_frame()`, `pop_frames()`, renderer freed in `close()`; capture inside `track()` |
| `turbo.py` | `--gif N` (first N episodes per cell), `--gif-fps/-res/-cam/-stride` |
| `turbo.sh` | `GIF` env: > 0 → `MUJOCO_GL=egl` + EGL device pin (as the UAV eval jobs), else `disable`; passes `--gif` |
| `turbo_gif.sh` (new) | the same job with `--gres=gpu:1` and `GIF=3` default; `exec`s `turbo.sh` |

Rendering needs a GL context, so GIF runs go through `turbo_gif.sh` (GPU) **and only when a GIF is wanted**; the
default `turbo.sh` stays CPU and writes the full suite above without GIFs. Not testable in the container (no GL); the
non-GL path (renderer unavailable → no frames, no crash) was exercised locally.

## Next

```bash
GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo_gif.sh  # pilot, 36x, + 3 GIFs per cell (GPU)
GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh      # same without GIFs (CPU)
```
Pass rule unchanged (plan §4 G1). Expect: settle ≈ Panda outcomes except the obstacle-hugging episodes; clock close
to settle with 0.2 m of lag. The s10 in-place outputs on the cluster are to be deleted (fix2 note).
