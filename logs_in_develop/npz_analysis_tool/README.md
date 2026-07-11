# npz_analysis — eval `.npz` → CSV analyzer

Summarizes the eval result `.npz` files (avoiding **and** visual-aligning / `va_*` schemas) into
human-readable, DA-ready CSVs — including **trajectory-quality** numbers that the success/violation
metrics are blind to.

## Run

```bash
python npz_analysis/analyze_npz.py <path-to-dir-or-file> [--out DIR] [--xy-cols 0 1] [--replot] [--dump-xy] [--no-recursive]
```

- `<path>` — a directory (scanned **recursively** for `*.npz`) or a single `.npz`. Works on paths like
  `.../results/halfspace_both-hard/` (many `diffuser.npz` / `dpcc-r.npz` / `va_diffuser.npz`).
- `--out` — output dir (default: `<path>/_npz_analysis`).
- `--xy-cols A B` — which `obs` columns are (x, y). **For avoiding use `2 3`** (cols `0 1` are
  `x_des, y_des`; the robot path is `x=2, y=3` per `config/projection_eval.yaml`). Default `0 1`.
- `--replot` — regenerate the **executed (x,y) trajectory** as a PNG per npz (see below).
- `--dump-xy` — write the **raw per-step (x,y) points** to `points_<ts>.csv`
  (`file, variant, trial, step, x, y`) — the actual coordinates, DA-ready.
- `--no-recursive` — only the top dir.

## Can it regenerate the *real* plotted path? — yes

The npz **does** store the real trajectory: `obs_all` is the per-trial **executed path** — the exact
`(x, y)` sequence that was drawn as the **black line** in the eval's `<variant>.png`. So `--replot`
redraws that real path straight from the npz (no rerun needed):

```bash
python npz_analysis/analyze_npz.py <path> --replot --xy-cols 2 3      # avoiding: x=col2, y=col3
# → <out>/<...>__<variant>_replot.png  (all trials overlaid, green = start)
```

## Plan-fan (MPC candidate) analysis — `sampled_trajectories_all`

When the npz stores the candidate foresight plans (visual-aligning, visual-avoiding once patched, and
Gen11 UAV), the tool also analyzes those (JOB D — see
`logs_in_develop/npz_analysis_tool/CHANGELOG_JobD_plan_candidate_analysis.md`):

- **`plan_*` CSV columns** — quality of the open-loop plans (`plan_straightness`, `plan_roughness`,
  `plan_max_jerk`, …), an all-axis **explosion detector `plan_max_abs`**, candidate diversity
  `plan_cand_spread` (`batch>1` only), and plan-vs-executed divergence `plan_exec_div` /
  `plan_exec_div_best`. `traj_max_abs` is the same explosion detector on the executed path.
- **`--replot-plans`** — per trial, overlay the candidate fan (blue) on the executed path (black).

```bash
python npz_analysis/analyze_npz.py <path> --xy-cols 2 3 --replot --replot-plans --dump-xy
```

If a `plan_*` column is missing/empty, that eval did not persist `sampled_trajectories_all` (e.g.
plain avoiding stores only `obs_all` / `act_all`); the executed-path analysis still works.

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
  --xy-cols 2 3 --replot --dump-xy    # avoiding x=2,y=3; --replot=PNG paths, --dump-xy=raw points
```

A **new `_npz_analysis/` folder is created *inside* the folder you pointed at** (the default `--out`),
and a table prints to your SSH terminal. So everything lands here:
```
logs/.../results/halfspace_both-hard/_npz_analysis/
    files_summary_20260619_150240.csv     # <ts> = real timestamp
    per_trial_20260619_150240.csv
    points_20260619_150240.csv             # raw per-step (x,y): file,variant,trial,step,x,y
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

## Reading the stdout table

The terminal table is a compact view of the most diagnostic columns. Example (UAV pillars, `--env uav`):

```
               variant  n_trials  succ_rate      steps    straight  exec_dyngap  plan_maxabs  plan_dyngap  plan_exdiv
              diffuser         5              423.000       0.882        0.056        3.201          nan       0.180
                dpcc-c         5              423.000       0.884        0.020        3.200          nan       0.046
                dpcc-r         5              423.000       0.883        0.020        3.200          nan       0.044
                dpcc-t         5              423.000       0.878        0.019        3.200          nan       0.046
              gradient         5              423.000       0.424        0.222        3.201          nan       3.558
            model_free         5              423.000       0.890        0.048        3.201          nan       0.188
       post_processing         5              423.000       0.880        0.020        3.200          nan       0.037
```

### Quick Example Interpretation (Dynamics Constraints)
If you want to know **"Did my dynamics constraints actually apply?"**, look directly at `exec_dyngap` and `plan_exdiv`:
*   **The "Constraints Applied" Winners (`dpcc-c`, `dpcc-r`, `post_processing`):** `exec_dyngap` is extremely low (`~0.020`). This proves the generated commands are physically valid and the drone can fly them. `plan_exdiv` is also low (`~0.046`), meaning the drone went exactly where the plan predicted.
*   **The "No Constraints" Baselines (`diffuser`, `model_free`):** `exec_dyngap` is higher (`0.056`). Without constraints, they generate trajectories that are physically harder for the drone to execute smoothly.
*   **The "Constraints Failed" Loser (`gradient`):** `exec_dyngap` is massive (`0.222`). The model hallucinated physically impossible commands. `plan_exdiv` is astronomically high (`3.558`), meaning the drone completely ran away and diverged from the plan.

### Column reference

| Table header | Full CSV column | What it measures |
|---|---|---|
| `variant` | `variant` | npz file stem — one row per evaluated policy variant |
| `n_trials` | `n_trials` | number of episodes in this npz |
| `succ_rate` | `success_rate__mean` | mean success rate across trials; blank = metric not stored |
| `steps` | `n_steps__mean` | mean steps per trial (lower = more direct path to goal) |
| `straight` | `traj_straightness__mean` | net displacement ÷ path length on the **executed** trajectory; 1.0 = perfectly straight, → 0.0 = wandering/exploded |
| `exec_dyngap` | `traj_dyn_gap_max` | max `\|p[t+1]−p[t]−act[t]\|` on the executed path — see below |
| `plan_maxabs` | `plan_max_abs__mean` | largest absolute coordinate ever seen in any plan waypoint (explosion detector for the open-loop fan) |
| `plan_dyngap` | `plan_dyn_gap_max` | max dynamics gap inside the plan fan; NaN when plans are obs-only (UAV, avoiding) |
| `plan_exdiv` | `plan_exec_div__mean` | mean distance between where plans predicted the agent would be vs where it actually went — the "runaway" signal |

### What to look for

**`straight` (trajectory straightness)**
- Near `1.0`: agent goes directly toward the goal.
- Near `0.0`: agent loops, wanders, or explodes. `gradient` above (0.424) is a red flag.

**`exec_dyngap` (executed dynamics gap)**
- **Avoiding env**: always ~0 by construction — the simulator applies `x_des += act` exactly. Not informative.
- **UAV env**: measures how much the drone's actual flight deviated from the commanded setpoint step `Δp_des`. DPCC variants (0.020) track tighter than unconstrained diffuser (0.056). `gradient` (0.222) shows poor tracking.

**`plan_maxabs` (plan explosion detector)**
- Should stay close to the environment's coordinate range (e.g. ~3.2 for a ±3.2 m arena).
- If it shoots to hundreds, the FM is hallucinating waypoints far outside the scene — even if the executed path looks fine (the low-level controller may have clipped it back).

**`plan_dyngap`**
- NaN for UAV and avoiding: plans store observations only, no action columns. This is expected — not a bug.
- Non-NaN only when the plan tensor explicitly stores `[act | obs]` (e.g. some visual-aligning variants).

**`plan_exdiv` (plan–execution divergence)**
- Low (~0.04): the FM's open-loop predictions closely match what the agent actually does → the model has a good internal world model.
- High (~3.5): plans diverge rapidly from reality → the controller is flying blind, correcting every step. `gradient`'s 3.558 above vs `post_processing`'s 0.037 is a stark contrast.

### What "blank" means in `succ_rate`

The metric is read from the npz by name (`success_rate`). If the eval script stores it under a different key (or doesn't store it at all), the column is blank. Check `list(np.load('variant.npz', allow_pickle=True).files)` on the cluster to see what keys are present.

## How to Interpret the Results

The tool produces a wealth of metrics beyond standard success rates. Here is a guide on how to interpret the key columns in the resulting CSVs to diagnose model behavior.

### 1. Task Success & Efficiency
**Columns:** `success_rate`, `n_violations`, `n_steps`, `avg_time`, `collision_free_completed`
These represent the headline performance of the model. 
- **High `success_rate` / Low `n_violations`**: The model achieved the objective while respecting obstacles.
- **`n_steps`**: Indicates how directly the model reached the goal. Unusually high steps paired with a high success rate might indicate overly cautious or looping behaviors.

### 2. Trajectory Quality (Smoothness vs. Chaos)
These metrics are computed on the **executed closed-loop path** (`traj_*` columns). They quantify the "smooth vs jerky/exploded" quality that binary success checks cannot see. Context: [DEBUG_DiT_Eval_Trajectory_Explosion.md](../logs_in_develop/Gen3v4_imf/U6/DEBUG_DiT_Eval_Trajectory_Explosion.md).

| Column | Meaning | Smooth Behavior | Exploded / Chaotic |
|---|---|---|---|
| `traj_straightness` | Net displacement ÷ path length | Near `1.0` | Close to `0.0` |
| `traj_roughness` | Max step ÷ median step (spike index) | ~`1.0` – `2.0` | Very large |
| `traj_max_jerk` | Curvature (2nd difference of path) | Small | Large |
| `traj_max_abs` | Largest absolute coordinate | Bound to env limits | Huge (e.g., `-227`) |

- **Why it matters:** A model might technically "succeed" but fly in erratic, jagged patterns. Low `traj_straightness` or high `traj_roughness` indicates poor trajectory conditioning or noisy diffusion outputs.

### 3. Dynamics Compliance (Physical Hallucinations)
**Columns:** `traj_dyn_gap_max`, `plan_dyn_gap_max`
These evaluate whether the model respects kinematic constraints by calculating the gap $P_{t+1} - P_t - Act_t$.
- **Near 0.0:** The model's actions perfectly explain its state transitions. It learned the underlying physics.
- **Large Gap (e.g., > 1e-3):** The model is **hallucinating physical states**. It is moving in ways that its actions do not support (e.g., relying heavily on post-hoc environmental projections rather than internalizing the dynamics). 

### 4. MPC Plan Divergence (The "Runaway" Effect)
**Columns:** `plan_exec_div`, `plan_max_abs`, `plan_cand_spread`
These evaluate the model's **open-loop foresight** (the `sampled_trajectories_all` candidate fan) against reality.
- **`plan_exec_div` (Plan-Execution Divergence):** Measures how far the planned foresight deviated from the path actually executed. High divergence means the model's predictions are rapidly breaking down across the horizon, causing the controller to constantly correct (the "runaway" effect).
- **`plan_max_abs` vs `traj_max_abs`:** If the open-loop plans explode (huge `plan_max_abs`) but the executed path does not, it means the low-level controller or environment clipping saved the run from a catastrophic model failure.
- **`plan_cand_spread`:** Measures the diversity of the candidate trajectory batch.

## Notes
- **Schema-generic:** any 1-D numeric array is auto-treated as a per-trial metric, so renamed/new keys
  are picked up. `obs_all`/`act_all`/`sampled_trajectories_all` are special-cased as trajectories.
- Both executed paths (`obs_all`) and open-loop plans (`sampled_trajectories_all`) are analyzed if present in the `.npz` file.
- `args` is read best-effort (Namespace or dict); if a checkpoint's custom args class can't unpickle, the
  row still works (an `_args_error` column flags it).

## Dynamics gap metrics — schema limitations

### `traj_dyn_gap_max` for avoiding (always ~0)
The executed-path gap checks `(x_des[t+1] - x_des[t]) - act[t]` (columns `[0,1]` of `obs_all`).
This is trivially 0 for **all** variants because the eval simulator applies `x_des += act` exactly.
It does NOT distinguish DPCC from non-DPCC models — expected behavior, not a bug.

### `plan_dyn_gap_max` for avoiding (NaN — not computable)
The avoiding plan schema (`sampled_trajectories_all`, from `eval_flow_matching_v3_imeanflow.py`)
stores `samples.observations` only — shape `(B, H, 4)` = `[x_des, y_des, x, y]`.
**Actions are not stored inside the plan tensor.** The dynamics gap formula `P[h+1]-P[h]-Act[h]`
requires explicit action columns; without them, `plan_dyn_gap_max` is reported as **NaN**.

Prior to this fix, the script used `plan_act_cols=[0,1]` thinking those were action columns,
but they are actually `x_des` (~-3.2). The formula evaluated to `(x[h+1]-x[h]) - x_des[h]`
≈ `0.02 - (-3.2) = 3.22` — a pure artifact, reported for every variant including unconstrained ones.
This **did not indicate a DPCC failure**; the dynamics constraints were working correctly.

### `plan_dyn_gap_max` for UAV (NaN — same schema limitation as avoiding)
UAV plans (`sampled_trajectories_all`, from `eval_fm_uav.py`) store `traj.observations` only —
shape `(B, H, 6)` = `[p_des(0:3) | p(3:6)]` for `cond_mode='pos_only'`.
The action (Δp_des) is available inside the FM rollout but **not saved to the plan tensor**.
`plan_dyn_gap_max` is therefore reported as **NaN** — the same situation as avoiding.

### UAV column layout (`--env uav`)
```
obs_all:                [p_des_x, p_des_y, p_des_z, p_x, p_y, p_z, v_x, v_y, v_z]  (9D)
                         cols 0,1,2             cols 3,4,5         cols 6,7,8
sampled_trajectories_all: [p_des_x, p_des_y, p_des_z, p_x, p_y, p_z]  (6D, pos_only)
```
`--env uav` uses `obs_p_cols=[3,4]` (actual drone x,y) and `plan_p_cols=[3,4]`.

### Out-of-bounds guard
If `--xy-cols` produces plan column indices beyond the plan's actual width (e.g. passing
`--xy-cols 6 7` without `--env uav` when plans have 6 columns), the script now skips those
snapshots gracefully and returns NaN rather than crashing with `IndexError`.
