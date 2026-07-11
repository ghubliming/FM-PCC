# Usage — `compare_horizon_plans.py` (horizontal H-step plan comparison)

**What it answers:** for the **same trial/seed**, at the **same MPC decision (snapshot)**, how does
each projection method bend the H-step foresight plan? One `.npz` per projection variant
(`diffuser, dpcc-c/r/t, gradient, model_free, post_processing, …`) in a shared results folder; this
tool slices them all at one (trial, snapshot) and overlays them.

> Sibling to `analyze_npz.py`, which is **vertical** (per-file aggregates). This one is **horizontal**
> (across files, at one point). Different question → separate script; `analyze_npz.py` is untouched.

## Run

```bash
python npz_analysis/compare_horizon_plans.py <results-dir> \
  --trial 0 --snapshot 0 --env avoiding \
  --variants diffuser,model_free,gradient,dpcc-c,dpcc-r,dpcc-t
```

Outputs go to `<results-dir>/_horizon_compare/`.

## The npz has TONS of trajectories — this tool exports a THIN SLICE

One npz alone (e.g. `diffuser.npz`) holds
`n_trials × n_snapshots × batch(candidates) × H(horizon) × dim` — e.g. `2 × 25 × 4 × 8 × 4` =
**1,600 plan waypoints**, plus the executed path. The tool does **not** dump all of it. It picks
**one trial + one snapshot** and exports just that decision point:

`rows = n_variants × batch × H` — e.g. 7 variants × 4 candidates × 8 steps = **224 rows**.

To sweep more, re-run per snapshot/trial (`--snapshot k`, `--trial i`), or use
`analyze_npz.py --dump-xy` to get the full per-step dump of everything.

## The 2 exported files (for `--trial 0 --snapshot 0`)

### `horizon_compare_t0_s0.csv` — the raw slice
One row per (variant, candidate, horizon-step). Long/tidy → pivot freely.

| column | meaning |
|---|---|
| `variant` | projection method (file it came from) |
| `trial` | trial/seed index (fixed for the whole file) |
| `snapshot` | snapshot index used (fixed for the whole file) |
| `approx_exec_step` | executed step this snapshot ≈ maps to (`snapshot × H//2`) |
| `candidate` | which of the `batch` candidates (0..3) — the fan, NOT the selected one* |
| `step` | horizon step 0..H-1 within the plan |
| `x`, `y` | plan position at that step (avoiding cols 2,3 = actual x,y) |

\* the executed/selected candidate index is **not** stored in the npz, so all candidates are exported.

### `horizon_compare_t0_s0.png` — the overlay
Each variant = one colour: candidate fan (thin) + candidate-mean (bold), ●=start ■=horizon-end.
Plus a stdout table with `path_len`, `max|c|` (explosion), plan endpoint, and **`div_ref`**
(mean per-waypoint distance of each method's mean plan to the reference `diffuser` plan).

## Key flags

| flag | default | note |
|---|---|---|
| `--trial` | 0 | trial/seed index |
| `--snapshot` | 0 | snapshot index (or target exec step if `--align step`) |
| `--align` | index | `index` = same snapshot # per method; `step` = nearest snapshot to a target exec step |
| `--env` | avoiding | column preset (avoiding→2,3; uav→3,4) |
| `--xy-cols` | by env | override plan (x,y) columns |
| `--variants` | all in dir | comma list to restrict/order (curate to ≤10 to avoid colour reuse) |
| `--reference` | diffuser | variant used as the unprojected baseline for `div_ref` |

## Read-before-you-trust caveats

- **Only snapshot 0 is state-aligned across methods.** All variants share the identical start state at
  `s=0`. For `s>0` the executed paths diverge (episodes differ in length → different snapshot counts:
  diffuser 25 vs dpcc-c 38), so the same index is a **different physical state** per method. The script
  WARNs; use `--align step` or stick to `--snapshot 0` for apples-to-apples.
- **`candidate` is the full fan, not the chosen plan** (selected index not recorded — see above).
- **Columns:** avoiding plan `dim=4 = [x_des, y_des, x, y]`; actual position = cols **2,3**.
