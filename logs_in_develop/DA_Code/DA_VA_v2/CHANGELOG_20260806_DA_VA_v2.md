# DA_VA_v2 — new visual-aligning DA tool (Gen7 + Gen14)

**Date:** 2026-08-06 · **Area:** `Data_Analysis/DA_VA_v2/` (new) · **Status:** written, exercised on real U7 data, **not yet run on the cluster**
**Inputs:** `logs_in_develop/Gen14/U7/SPEC_20260806_gen14_eval_outputs_for_DA_tool.md`,
`Data_Analysis/DA_Code_v3/` (template), `Data_Analysis/DA_Visual_Aligning/` (superseded)

---

## What was built

A new, parallel DA package: **`Data_Analysis/DA_VA_v2/`** (11 files, ~1700 lines).
It reads **old Gen7 visual-aligning runs and new Gen14 (Visual-Mix-ML) runs in one
pass**, carries the `DA_Code_v3` feature set, and is CSV-first — PNG generation
exists but is off unless `--plots` is given.

Plus **`Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh`** — the cluster entry point.

**Nothing existing was modified.** `DA_Code_v3/`, `DA_Visual_Aligning/`, both
`Visualizer*/index.html`, and the older sbatch scripts are untouched.

```
Data_Analysis/DA_VA_v2/
  discovery.py      candidate + result-unit discovery  (stdlib only)
  data_loader.py    one unit → per-rollout frame + scalars + quality + run config
  aggregator.py     master rollout table → units_long / agg_long / candidate stats
  reporter.py       every CSV + the TXT summary
  visualizer.py     optional PNG set (opt-in)
  config.py         names, labels, orderings
  utils.py          logging, output dirs, HTML manifest
  main_da_batch.py  batch CLI (primary)
  main_da.py        single-folder CLI
  __init__.py, README.md
```

---

## The four problems this had to solve

### 1. Layout (spec §4)

`DA_Code_v3` hard-codes `{candidate}/{seed}/results/halfspace_{hs}/{variant}.npz`.
Discovery is now **path-shape driven** and handles all four shapes in the wild:

| shape | layout | origin |
|---|---|---|
| A | `{seed}/results/{variant}.npz` | state-only flat |
| B | `{seed}/results/halfspace_{geo}/{variant}.npz` | state-only avoiding (`DA_Code_v3`) |
| C | `{seed}/results/{variant}/{variant}.npz` | Gen7 flat visual-aligning |
| D | `{seed}/results[_train_set]/{geo}/{variant}/{variant}.npz` | Gen7-with-geo and **Gen14** |

* L1 — `results_train_set/` is scanned alongside `results/` and recorded as `split=train`.
* L2 — **geometry is a first-class axis** (`geo` / `geo_base` / `tightened`), not a
  name prefix. `halfspace_top-right-hard` → `geo=top-right-hard`, keeping continuity
  with the old state-only CSVs.
* L3 — the extra variant-folder level is handled by shape detection, not a fixed join.
* L4 — the `_train_set` suffix is stripped from variant names; the split lives in its
  own column, so a train run and a test run of the same variant line up in the tables
  instead of masquerading as two variants.

Also: `*.partial.npz` crash-safety sidecars and `expert_references/` are skipped, and
a variant folder with `diagnostics/` but no npz is still loaded (from the JSONs).

A **verified consequence**: this reads the state-only avoiding trees too, so a Gen14
arm and a DPCC baseline can sit in one comparison run.

### 2. Metrics (spec §5.1–5.2)

Ingestion is **generic** — every npz key is classified by shape (1-D length-N →
per-rollout column; (N,2) → `_x`/`_y`; size-1 → run scalar), so keys future evals add
appear in the CSVs with no code change. The `DA_Code_v3` core seven:

| metric | here |
|---|---|
| `n_success`, `n_steps`, `avg_time` | verbatim |
| `collision_free_completed` | derived ← `constraint_exec_zero_violation` |
| `n_violations` | derived ← `constraint_exec_n_violated_steps` |
| `n_success_and_constraints` | derived ← `n_success × constraint_exec_zero_violation` |
| `total_violations` | **NaN on visual runs** — recommendation (a) from spec §5.2 |

Replacements for the cumulative metric the visual eval never records:
`constraint_exec_total_viol_count` (sum of the three family step-counts) and
`max_viol_depth_m` (worst depth over families). `avg_time_ms` added alongside
`avg_time` (npz stores seconds/replan).

### 3. Frozen rollouts + circuit breaker (spec §5.3)

The D1 box-obstacle guard can hold a context in place for a whole episode without
ever calling the model; those rollouts report `sat_rate=1.0`, `zero_violation=1` and
inflate every constraint aggregate. They are only visible in
`diagnostics/rollout_{r}_stats.json → context.box_obstacle_conflict`, which the loader
reads (`--no-diagnostics-scan` opts out).

Every long CSV carries a **`mask` column** — `all` and `unfrozen` — both always
written, so the toggle is a filter in the viewer rather than a re-run. Measured on the
U7 `combined_5-tightened` run: **4 of 30 rollouts per variant are frozen**, and masking
them moves `constraint_exec_sat_rate` 0.9534 → 0.9463 and `collision_free_completed`
0.733 → 0.692. The inflation the spec predicted is real and now measurable.

`projection_cb_tripped` / `projection_cb_skipped_steps` are surfaced in
`data_quality.csv` and in the summary's own section — reported, never ranked on.

The requested eval-side addition (`context_box_obstacle_conflict` as an (N,) npz array)
is **not needed** for this tool: it prefers that key when present and falls back to the
JSON scan otherwise. Cost of the scan on U7: ~1s for 115 units.

### 4. Aggregation (spec §5.4)

No `all_seeds/` directory is read or expected. One master per-rollout table, then
groupbys: per-unit (per seed), then pooled over seeds with `n_seeds` reported, so a
one-seed candidate is visibly distinct from a five-seed one. Verified on a 5-seed
state-only tree.

---

## Outputs

| file | what |
|---|---|
| `per_rollout_detail.csv` | wide, one row per rollout — everything else is a groupby of this |
| `va2_units_long.csv` | per (candidate, seed, split, geo, variant, mask, metric) |
| `va2_aggregated_long.csv` | pooled over seeds, plus `n_seeds` |
| `data_quality.csv` | frozen count, circuit-breaker trips, `npz_complete`, source |
| `run_config.csv` | engine / K / threshold / mpc size, lifted from the npz `args` |
| `candidates_multidimensional_raw.csv` / `_aggregated.csv` | `DA_Code_v3` column names |
| `va_candidates_dynamic.csv` | `DA_Visual_Aligning` long format |
| `candidates_ranking.csv`, `candidates_detailed.csv`, `candidates_per_variant.csv` | candidate-level |
| `candidates_summary.txt` | ranking + data-quality section |
| `logs/analysis.log`, `logs/loading.log`, `logs/discovery_manifest.json` | provenance |

**HTML compatibility** (both existing visualizers, verified by replaying their pandas
operations against the generated CSVs):

* `Data_Analysis/Visualizer/index.html` (DAv3) — reads the
  `candidates_multidimensional_*.csv` pair unchanged. Its schema has no geometry or
  split axis, so `halfspace_variant ← geo` (suffixed `@train_set` for train rows) and
  `constraint_type ← split`. The suffix stops a train-set run from being pooled with a
  test run of the same geometry.
* `Data_Analysis/Visualizer_Visual_Aligning/index.html` — reads
  `va_candidates_dynamic.csv` (variant carries the geo prefix) and the rollout table
  from `per_rollout_detail.csv`, which carries legacy column aliases (`success`,
  `mean_dist_m`, `steps`, `phys_err_m`, `context_xy_dist_m`, `avg_time_s`, …) for that
  purpose.

The `results_manifest.json` next to the output folder is refreshed on every run, so a
new batch shows up in both visualizers' dropdowns.

---

## Validation performed

The container has no scientific Python, so the pipeline was exercised in a **throwaway
venv in the session scratchpad** (numpy 2.5.1 / pandas 3.0.5 / matplotlib) against the
real downloaded run data in `temp/`. Nothing was installed into the container's python
or the FMPCC env.

| check | result |
|---|---|
| Gen14 U7 tree (`temp/0508`, af + mf arms, train split, 19 variants × 2 geos) | 76 units, 30 rollouts each, 0 failures |
| state-only avoiding tree in the same run (`H8_K20_T1_D...GaussianDiffusion`) | 39 units, geo `both-hard` / `top-left-hard` / `top-right-hard`, 0 failures |
| all 115 units, end to end | 3.6 s, 12 CSVs |
| 5-seed tree (`temp/Gen12/2707`) | 315 units, `n_seeds=5`, per-seed rows retained |
| **npz vs `--source json` on the same unit** | agree to float32 precision on all 9 shared metrics (max abs diff 2.9e-8) |
| frozen mask | 4/30 per tightened variant; masked aggregates differ as expected |
| `--geos` / `--variants` / `--splits` / `--candidates` / `--candidate-names` | all filter correctly |
| `--plots` | 5 PNGs render |
| `main_da.py` single-folder path | resolves and runs |
| synthetic fixtures for shapes A/C/D + `.partial.npz` + `expert_references/` | all classified correctly |
| replay of both HTML visualizers' pandas ops on the CSVs | no errors, expected shapes |

**Not yet verified:** an actual cluster run via
`Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh`, and an actual browser load of either
HTML page (only the data contract was replayed, not the rendering). → **run on cluster.**

---

## Deliberate omissions

* **Plots are not at DAv3 parity.** DAv3 ships ~15 figures incl. hierarchical/matrix
  analyses; DA_VA_v2 ships 5 (per-variant bars ×3, Pareto, data quality) and only with
  `--plots`. Per the request: the CSVs plus visual inspection in the DA_VA Viz v2 HTML
  are the intended path.
* **`total_violations` is not synthesised.** Faking a cumulative sum from per-family
  maxima would invent a number; the column stays NaN for visual runs and the two honest
  replacements are provided instead.
* **The heavy traces are never read.** `sampled_trajectories_all`, `obs_all`, `act_all`,
  `selected_idx_all`, `physical_tracking_errors` are excluded by name — `np.load` is
  lazy per key, so they are not even decompressed (~90% of file size). Candidate-fan
  analysis is out of scope for this tool.
* **`logs_in_develop/MASTER_TEST_HISTORY.md` was not touched** — offered, not added.

---

## How to run

```bash
# cluster
sbatch Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh logs/aligning-d3il-visual/plans
sbatch Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh \
    "logs/aligning-d3il-visual/plans,logs/avoiding-d3il/plans" --splits test

# discovery only — stdlib, runs in the dev container
python Data_Analysis/DA_VA_v2/discovery.py logs/aligning-d3il-visual/plans
```

Full CLI and output reference: `Data_Analysis/DA_VA_v2/README.md`.
