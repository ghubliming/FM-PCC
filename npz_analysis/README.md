# npz_analysis — eval `.npz` → CSV analyzer

Summarizes the eval result `.npz` files (avoiding **and** visual-aligning / `va_*` schemas) into
human-readable, DA-ready CSVs — including **trajectory-quality** numbers that the success/violation
metrics are blind to.

## Run

```bash
python npz_analysis/analyze_npz.py <path-to-dir-or-file> [--out DIR] [--xy-cols 0 1] [--replot] [--no-recursive]
```

- `<path>` — a directory (scanned **recursively** for `*.npz`) or a single `.npz`. Works on paths like
  `.../results/halfspace_both-hard/` (many `diffuser.npz` / `dpcc-r.npz` / `va_diffuser.npz`).
- `--out` — output dir (default: `<path>/_npz_analysis`).
- `--xy-cols A B` — which `obs` columns are (x, y). **For avoiding use `2 3`** (cols `0 1` are
  `x_des, y_des`; the robot path is `x=2, y=3` per `config/projection_eval.yaml`). Default `0 1`.
- `--replot` — regenerate the **executed (x,y) trajectory** as a PNG per npz (see below).
- `--no-recursive` — only the top dir.

## Can it regenerate the *real* plotted path? — yes

The npz **does** store the real trajectory: `obs_all` is the per-trial **executed path** — the exact
`(x, y)` sequence that was drawn as the **black line** in the eval's `<variant>.png`. So `--replot`
redraws that real path straight from the npz (no rerun needed):

```bash
python npz_analysis/analyze_npz.py <path> --replot --xy-cols 2 3      # avoiding: x=col2, y=col3
# → <out>/<...>__<variant>_replot.png  (all trials overlaid, green = start)
```

Caveat — **only the executed (black) path is recoverable** from the avoiding npz. The **open-loop plans**
(the blue `ax[i,5]` lines) are **not** saved there (avoiding stores only `obs_all` / `act_all`).
Visual-aligning npz *does* save `sampled_trajectories_all` (plans), but `--replot` does not draw those yet.

## Example — remote SSH, peek a results folder, output in place

You're SSH'd into the cluster (where the conda env with `numpy` lives). Point the tool at one
`results/halfspace_*` folder; it analyzes every `.npz` there (`diffuser.npz`, `dpcc-r.npz`, …) and
**writes the CSVs back into that same folder** (default `--out`), so you can `cat`/`scp` them in place.

> ⚠️ **Quote the path** — these folders contain `(…)` and `.` which the shell would otherwise mangle.

```bash
# on the remote box
cd ~/FMPCC/FM-PCC

python npz_analysis/analyze_npz.py \
  "logs/avoiding-d3il/plans/flow_matching_v3_imeanflow(1e4_beta_U4)/H8_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE_a1.5_b1.0_aw10_objmeanflow_jvp/H8_K10_Meuler_T0.5_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE/6/results/halfspace_both-hard" \
  --xy-cols 2 3 --replot       # avoiding x=2,y=3; --replot redraws the real executed paths
```

A **new `_npz_analysis/` folder is created *inside* the folder you pointed at** (the default `--out`),
and a table prints to your SSH terminal. So everything lands here:
```
logs/.../results/halfspace_both-hard/_npz_analysis/
    files_summary_20260619_150240.csv     # <ts> = real timestamp
    per_trial_20260619_150240.csv
    diffuser_replot.png                    # regenerated real path, one per .npz in the folder
    dpcc-r_replot.png
```
(The `__`-joined prefix only appears if you scan from a **parent** dir — then the png name encodes the
nested subpath, e.g. `6__results__halfspace_both-hard__diffuser_replot.png`.)
Use `--out /some/dir` to send them elsewhere instead of beside the data.

**Scan wider in one shot** — point higher up to recurse every seed × halfspace variant at once:
```bash
python npz_analysis/analyze_npz.py \
  "logs/avoiding-d3il/plans/flow_matching_v3_imeanflow(1e4_beta_U4)"        # recurses all .npz below
```

**Peek a single file:**
```bash
python npz_analysis/analyze_npz.py ".../results/halfspace_both-hard/diffuser.npz"
```

**Pull the CSV back to your laptop** (run locally; quote/escape the remote path):
```bash
scp 'user@cluster:~/FMPCC/FM-PCC/logs/.../halfspace_both-hard/_npz_analysis/files_summary_*.csv' .
```

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
