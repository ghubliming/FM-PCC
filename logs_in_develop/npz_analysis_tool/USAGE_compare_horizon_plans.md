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
| `--candidate` | mean | `mean` (candidate-mean + thin fan) or an int `k` (draw only candidate `k`, no fan) |
| `--draw-constraints` | off | flag plan waypoints that violate the enforced constraints (red × + `plan_viol` column). Does NOT draw the env unless `--show-env`. |
| `--show-env` | off | additionally draw the avoiding env itself (blue halfspace funnel + obstacle circle). Off by default — the env background clutters the plot |
| `--halfspace-variant` | inferred | `both-hard`\|`top-left-hard`\|`top-right-hard` (auto-read from folder name) |
| `--config` | `config/projection_eval.yaml` | source of halfspace/obstacle/ax_limits |
| `--exp` | `avoiding-d3il` | task key in the config |
| `--show-executed` | off | also draw each variant's executed path (`obs_all`) dashed, for plan-vs-reality context |
| `--full-frame` | off | lock view to the full environment (config `ax_limits`). Default auto-zooms to the plans so they stay visible; constraints are drawn either way |

## `--draw-constraints` — the projection-insight view

Flags where the plans break the enforced constraints: marks every **mean-plan waypoint that violates**
with a red ×, and adds a **`plan_viol`** column = % of ALL candidate waypoints inside a forbidden
region (halfspace triangles + obstacle circle from `projection_eval.yaml`, same variant→index
selection the eval uses). It does **not** draw the avoiding env background by default — add
**`--show-env`** if you want the blue funnel/obstacle drawn too. `plan_viol` is the column that
separates the methods:

```
variant     path_len   div_ref  plan_viol      (trial 0, snapshot 48 via --align step)
diffuser      0.712       ref      6.2%    ← raw plan pokes into forbidden zones
model_free    0.677     0.0074     3.1%    ← barely projects → still violates
gradient      0.466     0.0821     6.2%
dpcc-c        0.093     0.251      0.0%    ← hard projection: every waypoint feasible
dpcc-r        0.095     0.299      0.0%
dpcc-t        0.095     0.273      0.0%
```

At **snapshot 0** all are `0.0%` (feasible start → nothing to project → plans identical). The
violations only appear at later snapshots where the raw plan reaches the constraint.

> `plan_viol` (table) counts ALL candidates; the red × on the plot marks only the **candidate-mean**
> plan's violations (averaging hides some), so a low-but-nonzero `plan_viol` can show no ×.

**View auto-zooms to the plans by default** (the H-step plans are tiny relative to the whole scene, so
the full-env frame makes them unreadable). Constraint patches and executed paths are excluded from the
zoom bounds so they can't blow the view out. Add `--full-frame` if you want the whole environment.

## Environment safety (how it treats non-avoiding envs)

The tool is **built and verified for avoiding/D3IL**. Across other environments:

| aspect | behaviour | safe? |
|---|---|---|
| **xy columns** | `--env avoiding`→cols 2,3 (verified); `uav`→3,4 (inherited from `analyze_npz`, untested here); no visual-aligning preset. Cols exceeding the plan dim fall back **and now print a WARNING**. | pass `--xy-cols` for any non-avoiding schema |
| **`--align step`** | `save_every` (snapshot→step) is **inferred per-variant from data** (`executed_len / n_snapshots`), not hardcoded — so avoiding (every H//2) and UAV (every step) both resolve correctly. `--horizon-div` overrides. | ✅ env-agnostic |
| **`--draw-constraints`** | **avoiding-halfspace only.** On other envs the variant infer returns nothing / the config lookup `KeyError`s → it **skips constraints with a WARNING and draws none** (never draws avoiding geometry on a non-avoiding plot). | ✅ degrades safely |

**Bottom line:** safe (no crashes, no wrong-geometry drawing) everywhere; fully *correct* only for
avoiding out of the box. For UAV/visual-aligning: set `--xy-cols`, skip `--draw-constraints`, and the
plan-overlay + `div_ref` still work.

## `--candidate k` — one candidate per variant (per-candidate projection comparison)

The eval seeds RNG per trial (`torch.manual_seed(i)`), so **at snapshot 0 every variant generates the
IDENTICAL candidates and then projects them**. Verified empirically on the UAV s_curve data:
candidate `k`'s max-diff vs `diffuser`'s candidate `k` is ~0.01 for `model_free`/`dpcc-*`/`geo_free`/
`bounds_free` (= the projection nudge) — i.e. it's the **same pre-projection sample**. So
`--candidate 0` overlays "the same trajectory under each projection" → isolates the projection effect,
cleaner than the mean+fan.

**Caveats (all real):**
- **Snapshot 0 only.** After step 0 the rollouts diverge and the RNG desyncs → candidate `k` is no
  longer the same sample across variants (the tool WARNs if you pass `--candidate k` with snapshot≠0).
- **`gradient` breaks it** — its snap-0 candidates differ by ~6.4 from the others (different RNG path /
  its gradient-projection distorts even step 0). Exclude it from candidate-level comparison.
- **At snapshot 0 projection is nearly idle** (feasible start → `div_ref` ~0.01), so this shows the
  *shape* of the projection nudge, not a dramatic bend. The dramatic bends live mid-rollout — where
  candidate comparability no longer holds (the fundamental separate-rollout limitation).

```bash
# one candidate, same-input projection comparison (drop the RNG-desynced gradient)
... --snapshot 0 --candidate 0 --variants diffuser,model_free,dpcc-c,dpcc-r,dpcc-t,geo_free,bounds_free
```

## Read-before-you-trust caveats

- **Only snapshot 0 is state-aligned across methods.** All variants share the identical start state at
  `s=0`. For `s>0` the executed paths diverge (episodes differ in length → different snapshot counts:
  diffuser 25 vs dpcc-c 38), so the same index is a **different physical state** per method. The script
  WARNs; use `--align step` or stick to `--snapshot 0` for apples-to-apples.
- **`candidate` is the full fan, not the chosen plan** (selected index not recorded — see above).
- **Columns:** avoiding plan `dim=4 = [x_des, y_des, x, y]`; actual position = cols **2,3**.
