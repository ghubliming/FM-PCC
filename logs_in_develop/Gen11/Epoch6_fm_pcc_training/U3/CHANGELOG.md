# U3 CHANGELOG — legacy eval artifacts restored (npz / log / 2-D overview / GIF)

Implements U3 P1–P4 (see `PLAN.md`). Only the `fm_only` / diffuser baseline is
built; constraint/PCC fields are typed placeholders for Epoch 7.

## New file: `FM_v3_uav_test/eval_artifacts.py`
Pure-IO + matplotlib artifact writers (no torch/MuJoCo — caller supplies frames):
- `save_npz(out_dir, variant, rollouts, args)` — legacy FMv3-ODE schema:
  `n_success`, `n_steps`, `obs_all`, `act_all`, `sampled_trajectories_all`, `args`
  + **PCC placeholders** `n_success_and_constraints`, `n_violations`,
  `total_violations` (zero-filled → Epoch-7 DPCC drop-in, no schema change).
  `obs_all/act_all/sampled_trajectories_all` are `dtype=object` ragged arrays so
  existing analysis scripts read them unmodified.
- `plot_overview(...)` — **2-D overview PNG** (`<variant>.png` + `all.png`):
  top-down (x, y) path + obstacles, **plus a side (x, z) altitude panel with the
  airborne-gate line** so the U2 "never took off" failure is visible at a glance.
  Reuses the scene-aware `_draw_obstacles` / `_homotopy_color` from
  `uav_expert_data_collect/generate_overview_plots.py` (graceful plain-plot fallback
  if those deps are missing).
- `save_rollout_stats / save_rollout_gif / write_pcc_placeholder / write_eval_log`
  — per-rollout `diagnostics/rollout_<r>_stats.json`, opt-in `rollout_<r>.gif`,
  stub `rollout_<r>_mpc_foresight.svg` (Epoch-7 PCC), and `eval_<variant>.log`.

## `FM_v3_uav_test/eval_fm_uav.py`
- `--record {none,gif,all}` arg (default `none` → ~0 overhead).
- `rollout_one(...)` now buffers per-FM-step `obs_traj` (`[p_des|p|v]`), `act_traj`
  (Δp_des), and `plans` (`traj.observations`, the FM's H-step foresight — previously
  the discarded 2nd return of `policy()`). With `--record`, captures overhead frames
  via a headless `mujoco.Renderer` (free top-down camera, res 360, `frame_stride=2`).
  Heavy arrays/frames returned under `HEAVY_KEYS`, stripped from `results.json`.
- `eval_scene(...)` writes per-rollout diagnostics in-loop (GIF written then frames
  dropped → bounded memory), then `results.json` (summary, unchanged shape) + `npz`
  + `eval_<variant>.log` + 2-D overview.

## `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`
- Optional 5th arg `$5=record` (default `none`) passed through as `--record`.

## Output layout (per scene/seed/projection)
```
logs/UAV_FM/uav-<scene>/.../<seed>/eval/<projection>/
  results.json          # summary + per-rollout metrics (heavy data stripped)
  <variant>.npz         # legacy schema + zeroed PCC placeholder keys
  eval_<variant>.log
  <variant>.png  all.png # 2-D overview (top-down + altitude)
  diagnostics/
    rollout_<r>_stats.json
    rollout_<r>.gif               # only with --record gif|all
    rollout_<r>_mpc_foresight.svg # PCC placeholder (Epoch 7)
```

## Verification (Docker, numpy-level)
- `py_compile` clean on both Python files; `bash -n` clean on the eval script.
- Functional test on synthetic rollouts: npz round-trips (`obs_all[0].shape==(30,9)`,
  PCC placeholder keys present), `results.json` is JSON-serialisable (no heavy keys
  leak), and overview PNG / log / per-rollout stats / SVG / all.png all written.
- Real GIF + obstacle drawing need MuJoCo → cluster-only; here they degrade
  gracefully (plain plot, GIF skipped) — by design.

## Scope / notes
- **Diffuser baseline only.** Every constraint/PCC metric, the foresight SVG, and
  the per-variant loop are placeholders sized for Epoch-7 DPCC.
- This is artifact plumbing — it makes the U2 0%-success failure *visible* (the
  altitude panel will show the drone pinned at z≈0.08); it does **not** fix it.
- GIF is opt-in and disk-heavy (~hundreds of frames/rollout); keep `--record none`
  for routine runs.
- Working-tree only — sync to the cluster before the next test.
