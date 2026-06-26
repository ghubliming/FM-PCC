# npz_analysis tool — CHANGELOG

**Date:** 2026-06-19
**Tool:** `npz_analysis/analyze_npz.py` (+ `npz_analysis/README.md`) — under repo root `FM-PCC/`.
**Status:** working; `py_compile` clean; smoke-tested on synthetic avoiding + visual-aligning npz.

---

## What it does
Scans a path (recursively) for eval result `.npz` files and emits two CSVs + a stdout table:
- **`files_summary_<ts>.csv`** — one row per `.npz`: per-trial metric means/std, executed-trajectory
  quality aggregates, and key args.
- **`per_trial_<ts>.csv`** — one row per (file, trial): per-trial metric + per-trajectory quality.

Run: `python npz_analysis/analyze_npz.py <dir-or-file> [--out DIR] [--xy-cols 0 1] [--no-recursive]`.
Default `--out` = `<path>/_npz_analysis`.

## Why
The eval logs report **task-completion** numbers (binary success, discrete obstacle checks) that are
**blind to motion quality** — a chaotic/exploded trajectory can still score "ok"
(see `logs_in_develop/Gen3v4_imf/U6/DEBUG_DiT_Eval_Trajectory_Explosion.md` §8). This tool adds
**trajectory-quality** columns so the badness becomes a number: `traj_straightness`, `traj_roughness`
(max/median step), `traj_max_jerk`/`traj_mean_jerk` (path curvature), plus path-length/step stats —
computed on the executed path `obs_all`.

## Design notes
- **Schema-generic:** any 1-D numeric array → per-trial metric (mean/std). Handles BOTH the avoiding
  schema (`n_success`, `n_steps`, `n_violations`, `total_violations`, `collision_free_completed`, …) and
  the visual-aligning / `va_*` schema (`success_rate`, `entropy`, `mean_distance`, …) without per-schema
  code; new/renamed keys are auto-picked.
- `obs_all`/`act_all`/`sampled_trajectories_all` special-cased as trajectory payloads (not metrics).
- `args` read best-effort (argparse `Namespace` **or** `dict`, possibly a 0-d object array); failures are
  isolated to an `_args_error` column instead of crashing the run. Surfaced arg cols: objective, backbone,
  NFE (`flow_steps_v3`), `meanflow_cfg_omega`, `action_weight`, schedule, seed, loadpath, …
- Trajectory quality on the **executed** path only (the trajectory both schemas store). `--xy-cols`
  overrides which obs columns are (x, y); default `0 1`.
- Pure `numpy` + stdlib `csv` (no pandas dependency).

## Validation
- `py_compile` clean.
- Smoke test: synthetic npz in both schemas (Namespace args + dict args, smooth vs chaotic trajectories).
  `traj_straightness` / `traj_roughness` / `traj_max_jerk` correctly separated smooth from chaotic paths;
  both files summarized into the union-column CSV; per-trial rows produced.

## Update — `--replot` (regenerate the real path from the npz)
- `obs_all` is the **real executed trajectory** (the exact `(x,y)` drawn as the black line in
  `<variant>.png`), so it *is* recoverable. Added **`--replot`**: redraws the executed path per npz to a
  PNG (all trials overlaid, green start), straight from the npz — no eval rerun.
- **Avoiding columns:** the robot path is `x=col 2, y=col 3` (cols 0,1 are `x_des, y_des`, per
  `config/projection_eval.yaml`), so use `--xy-cols 2 3`. (Default stays `0 1` for generality.)
- Limitation: only the executed (black) path is in the avoiding npz; the open-loop **plans** (blue) are
  not saved there, so they can't be regenerated. Visual-aligning saves `sampled_trajectories_all`, but
  `--replot` does not draw those yet.
- matplotlib imported **lazily** (only under `--replot`), so the CSV path has no extra dependency.

## Not done / future
- Open-loop **plan** quality (visual-aligning's `sampled_trajectories_all`) is loaded but not yet scored —
  could add a `plan_*` quality block to expose the open-loop explosion directly.
- No DTW/Fréchet-vs-demo fidelity yet (D6 in the debug MD) — would need the demo set alongside.
- No commit/push (per policy).
