# Gen11 E5 U5 — Overview Plots: How It Works

**File:** `uav_expert_data_collect/generate_overview_plots.py`

---

## Why two lines? Commanded vs. Actual

Every episode pickle stores an `obs` array of shape `(T, 9)`:

```
obs[:, 0:3]  →  p_des   — the desired / commanded position
obs[:, 3:6]  →  p       — the actual UAV position (from MuJoCo state)
obs[:, 6:9]  →  v       — the actual velocity
```

These are **fundamentally different things**:

| | `p_des` (commanded) | `p` (actual) |
|-|---------------------|--------------|
| Source | Trajectory planner: evaluates `blended_path` or `traverse_line` at each timestep → outputs the next setpoint | MuJoCo simulation state: where the drone physically is |
| Role in training | Part of the observation the policy sees | Part of the observation the policy sees |
| Relationship | Ideal, smooth curve — the plan the PID is chasing | Lags `p_des` by PID tracking error, has small oscillations |

The gap between them is the **PID tracking error**. In a clean episode the two lines nearly overlap; in a stressed/oscillating episode they visibly diverge. Plotting both tells you at a glance how well the controller tracked the plan during collection.

There is no third "plan" line — `p_des` IS the plan, evaluated step by step.

---

## Summary mode

```
python generate_overview_plots.py --mode summary
```

Output: `<data_dir>/plots/<scene>_summary.png`

- Plots **only `p` (actual)** — one thin line per episode, coloured by homotopy.
- Overlaying `p_des` for 500 episodes would make the plot unreadable (they're nearly identical at this scale anyway).
- Alpha is computed adaptively: denser datasets → more transparent lines → the whole plot reads like a heatmap of where the UAV actually flew.
- One representative start marker per homotopy (colour-coded circle).
- Legend shows each homotopy label with episode count.

Good for: checking **dataset coverage and balance** — are all homotopy classes well-populated? Are the paths in the expected spatial region?

---

## Per-episode mode

```
python generate_overview_plots.py --mode per-episode --per-homotopy 1
```

Output: `<data_dir>/plots/<scene>/<homotopy>/<ep_id>_plot.png`

Here both lines appear because you are inspecting a single episode:

**`p_des` — dashed blue, time-gradient**
- Drawn segment-by-segment with the `Blues` colormap: early timesteps are light blue, late timesteps are dark blue.
- Shows the trajectory the planner commanded: its shape, speed profile, and whether it passed cleanly through gaps.

**`p` — solid red**
- Flat red, no time gradient.
- Shows where the drone actually was. Compare visually to `p_des` to judge tracking quality.

**Markers**
- Green circle = `p[0]` (start of actual trajectory)
- Red star = `p[-1]` (end of actual trajectory)

**Title** (two-line)
```
<ep_id>
scene=<scene>  homotopy=<h>  T=<dur>s  contact=<cf>  ctrl=<ctrl>
```
For stress episodes, a third tag is appended:
```
STRESS=wall_crossing  gate[C=True, F=False]
```

Good for: inspecting **individual episodes** — tracking error, trajectory shape, contact events, stress behaviour.

---

## Obstacle rendering

Drawn on top of trajectory lines (`zorder=3`) in both modes.

**Boxes** → filled grey rectangle using `half_extents` from the pickle.

**Cylinders** → two concentric patches:
1. Filled grey circle at actual radius.
2. Dashed orange-red ring at `radius + 0.31 m` — the **rotor-reach safety margin** from `trajectories.py`. A trajectory that crosses this ring is at risk of rotor contact even if it misses the body.

Obstacle geometry is read from `episode['obstacles']` — the same list that was embedded at collection time, so what you see exactly matches what the simulator had.

---

## Selective inspection: `--per-homotopy N`

```
python generate_overview_plots.py --mode per-episode --per-homotopy 1
```

Episodes are grouped into `(scene, homotopy)` buckets. Only the first `N` per bucket are plotted. At `N=1`:
- 4 scenes × 3 homotopies (corridor/s_curve: L/C/R) + 4 homotopies (pillars) = ~13 plots total instead of ~1975.

This is the standard selective-inspection idiom shared with `generate_gifs.sh` and `generate_physics_gifs.sh`.

---

## CLI summary

| Flag | Default | Effect |
|------|---------|--------|
| `--mode` | `summary` | `summary`, `per-episode`, or `both` |
| `--data-dir` | `logs/uav_expert_data` | Root to walk for `.pkl` files |
| `--out-dir` | `<data-dir>/plots` | Where PNGs are written |
| `--scene` | all | Restrict to one scene |
| `--per-homotopy N` | all | Keep first N per (scene, homotopy) bucket |
| `--max-episodes N` | all | Hard cap after bucketing |
| `--dpi` | 150 | PNG resolution |
| `--no-skip` | off | Default skips existing files; this flag forces regeneration |

---

## What the two lines tell you in practice

| Observation | Interpretation |
|-------------|---------------|
| Lines nearly identical | Good PID tracking; clean episode |
| `p` (red) consistently behind `p_des` (blue) | Steady-state lag — drone is slightly slower than the plan |
| `p` oscillates around `p_des` | High-frequency PID overshoot |
| Lines diverge sharply then reconverge | Contact event or near-miss; drone was knocked off trajectory |
| `p_des` is a smooth arc, `p` is jagged | Controller saturation (stress case `gain_extreme` or `extreme_speed`) |
| `p_des` teleports (discontinuous jump in blue) | Stress case `discontinuous` — commanded setpoint jumped mid-flight |
| `p_des` exits the scene boundary | Stress cases `wall_crossing`, `floor_dive`, `ceiling_climb` |
