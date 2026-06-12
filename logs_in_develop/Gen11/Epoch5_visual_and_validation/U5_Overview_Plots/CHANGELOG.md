# Gen11 E5 U5 — 2-D Overview Plots: Changelog

**Date:** 2026-06-12
**Branch:** update_into_FM
**Status:** COMPLETE — pending first cluster run

---

## Files changed

| File | Change |
|------|--------|
| `uav_expert_data_collect/generate_overview_plots.py` | New — standalone plot script |
| `Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh` | New — CPU-only sbatch |

---

## What this adds

A **pure matplotlib, no-MuJoCo, CPU-only** tool that generates top-down (XY-plane)
trajectory plots with obstacle geometry overlay. Complements the GIF scripts:
- GIF / physics GIF = animated 3D perspective, GPU required
- Overview plot = static 2D bird's-eye, instant, no GPU, no MuJoCo

---

## C1 — `generate_overview_plots.py`

### Two plot modes

**`summary`** — one PNG per scene, all episodes overlaid, coloured by homotopy.

```
<data_dir>/plots/<scene>_summary.png
```

- Actual trajectory `p` (from `obs[:, 3:5]`) as thin coloured lines, alpha scaled
  by density so dense datasets form a heatmap and sparse ones show individual paths.
- Homotopy colours: fixed palette (`L`=red, `C`=blue, `R`=green, pillars by label,
  stress cases by name; overflow falls through to tab10).
- Obstacle geometry: boxes as filled grey rectangles; cylinders as filled grey circles
  with a dashed red safety-margin ring at `radius + 0.31 m` (rotor reach).
- Legend: per-homotopy entry with episode count; obstacle + safety-margin entries.

**`per-episode`** — one PNG per episode, commanded vs actual.

```
<data_dir>/plots/<scene>/<homotopy_safe>/<ep_id>_plot.png
```

- `p_des` (commanded, `obs[:, 0:2]`): Blues colormap gradient darkening with time
  (earliest = light, latest = dark) — shows the planned path as a smooth arc.
- `p` (actual, `obs[:, 3:5]`): solid red — shows how well the PID tracked.
- Start marker: green circle; end marker: red star.
- Title: `ep_id`, scene, homotopy, episode duration, contact fraction, controller.
- Stress episodes: appends `STRESS=<case>  gate[C=True/False, F=True/False]`.

**`both`** — runs both modes.

### Key design choices

| Choice | Rationale |
|--------|-----------|
| `matplotlib.use('Agg')` at import | Headless — no display, no Qt/GTK install needed on Slurm |
| Actual path only in summary | Commanded path overlaid on 500 traces would be unreadable |
| Time-gradient on commanded in per-episode | Immediately shows trajectory direction without arrows |
| Safety-margin ring on cylinders | Visualises the 0.31 m rotor-reach constraint from trajectories.py |
| `plots` excluded from `discover_episodes` | Prevents re-ingesting PNG output files on second run |
| `--per-homotopy 1` = 9 plots | Matches U2/U4 selective inspection idiom |
| `--no-skip` flag | Default is skip-existing so re-runs are fast incremental updates |

### CLI

```bash
# Full scene summaries (all 4 scenes)
python uav_expert_data_collect/generate_overview_plots.py

# One scene, per-episode, 1 per homotopy
python uav_expert_data_collect/generate_overview_plots.py \
    --mode per-episode --scene pillars --per-homotopy 1

# Stress root — both modes, 1 per case
python uav_expert_data_collect/generate_overview_plots.py \
    --data-dir logs/uav_expert_data_stress --mode both --per-homotopy 1
```

### Works with stress episodes

Stress episodes carry `stress: True` and `stress_case` in the pickle.
`discover_episodes` finds them via the same `<scene>/<homotopy_safe>/<ep_id>.pkl` walk
(stress_case sits in the homotopy slot → same layout). The per-episode title shows the
stress case name and gate verdicts. No special code path needed.

---

## C2 — `generate_overview_plots.sh`

CPU-only (`MPLBACKEND=agg`, `MUJOCO_GL=disabled`, partition `gpu-1-student`, no `--gres`).
1-hour time limit (scene-summary of 1975 episodes ≈ 2 min).

| `$1` | `$2` | `$3` | `$4` | `$5` |
|------|------|------|------|------|
| mode | scene | per_homotopy | data_dir | dpi |
| `summary` | `""` = all | `""` = all | `""` = production | `150` |

---

## Quick-start commands

```bash
# Scene summaries — production data (4 PNGs, ~2 min)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh

# Selective per-episode — 1 per homotopy, all scenes (9 PNGs)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh \
    per-episode "" 1

# Stress root — both modes, 1 per case (after E4 U10 collect run)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_overview_plots.sh \
    both "" 1 logs/uav_expert_data_stress
```

Output for production: `logs/uav_expert_data/plots/`
Output for stress: `logs/uav_expert_data_stress/plots/`
