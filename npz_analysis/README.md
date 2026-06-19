# npz_analysis — eval `.npz` → CSV analyzer

Summarizes the eval result `.npz` files (avoiding **and** visual-aligning / `va_*` schemas) into
human-readable, DA-ready CSVs — including **trajectory-quality** numbers that the success/violation
metrics are blind to.

## Run

```bash
python npz_analysis/analyze_npz.py <path-to-dir-or-file> [--out DIR] [--xy-cols 0 1] [--no-recursive]
```

- `<path>` — a directory (scanned **recursively** for `*.npz`) or a single `.npz`. Works on paths like
  `.../results/halfspace_both-hard/` (many `diffuser.npz` / `dpcc-r.npz` / `va_diffuser.npz`).
- `--out` — output dir (default: `<path>/_npz_analysis`).
- `--xy-cols A B` — which `obs` columns are (x, y) for trajectory metrics (default `0 1`).
- `--no-recursive` — only the top dir.

## Outputs

- **`files_summary_<ts>.csv`** — one row per `.npz`: per-trial metric means (+std), executed-trajectory
  quality aggregates, and key args (objective / backbone / NFE / ω / action_weight / …).
- **`per_trial_<ts>.csv`** — one row per (file, trial): per-trial metric + per-trajectory quality.
- A compact table to stdout.

## Columns that matter

**Task metrics** (means; success-type means == rates): `n_success`, `success_rate`,
`n_success_and_constraints`, `collision_free_completed`, `n_steps`, `n_violations`, `total_violations`,
`avg_time`, `mean_distance`, `entropy` (whatever the file contains — keys are auto-detected).

**Trajectory quality** (computed on the executed closed-loop path `obs_all`):

| Column | Meaning | Smooth | Exploded/chaotic |
|---|---|---|---|
| `traj_straightness` | net displacement ÷ path length | →1 | →0 |
| `traj_roughness` | max step ÷ median step (spike index) | ~1–2 | large |
| `traj_max_jerk` / `traj_mean_jerk` | curvature (2nd difference of the path) | small | large |
| `traj_path_len`, `traj_mean_step`, `traj_max_step` | length / step stats | — | inflated |

These quantify the "smooth vs jerky/exploded" quality that binary success + discrete obstacle checks
**cannot** see. Context:
[logs_in_develop/Gen3v4_imf/U6/DEBUG_DiT_Eval_Trajectory_Explosion.md](../logs_in_develop/Gen3v4_imf/U6/DEBUG_DiT_Eval_Trajectory_Explosion.md).

## Notes
- **Schema-generic:** any 1-D numeric array is auto-treated as a per-trial metric, so renamed/new keys
  are picked up. `obs_all`/`act_all`/`sampled_trajectories_all` are special-cased as trajectories.
- Quality is on the **executed** path (`obs_all`) — the only trajectory both schemas store. (Avoiding does
  not save open-loop plans; visual-aligning saves `sampled_trajectories_all` but it is not analyzed here.)
- `args` is read best-effort (Namespace or dict); if a checkpoint's custom args class can't unpickle, the
  row still works (an `_args_error` column flags it).
