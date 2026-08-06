# DA_VA_v2 — Visual-Aligning Data Analysis (Gen7 + Gen14)

Replacement for `Data_Analysis/DA_Visual_Aligning/`, built on the `DA_Code_v3`
template. Reads **old Gen7 visual-aligning runs and new Gen14 (Visual-Mix-ML)
runs in the same pass**, and — because discovery is driven by path shape rather
than by a hard-coded layout — the state-only avoiding trees `DA_Code_v3` reads
as well.

Nothing in `DA_Code_v3` or `DA_Visual_Aligning` was modified; this is a parallel
tool.

**CSV-first.** The CSVs are the deliverable; PNG generation exists but is off
unless you pass `--plots`.

---

## Quick start

```bash
# from the repo root, with Data_Analysis/DA_VA_v2 on PYTHONPATH
python Data_Analysis/DA_VA_v2/main_da_batch.py \
    --parent-path logs/aligning-d3il-visual/plans \
    --output-path Data_Analysis/analysis_results/va2_$(date +%Y%m%d_%H%M%S)

# four Gen14 engine arms + the Gen7 tree in one comparison
python Data_Analysis/DA_VA_v2/main_da_batch.py \
    --parent-path logs/aligning-d3il-visual/plans,logs/visual-aligning-dpcc/plans \
    --output-path Data_Analysis/analysis_results/va2_combined

# one model folder
python Data_Analysis/DA_VA_v2/main_da.py --input-path <.../H8_K2_..._Eaf>

# what does discovery see? (stdlib only — runs anywhere)
python Data_Analysis/DA_VA_v2/discovery.py logs/aligning-d3il-visual/plans
```

On the cluster: `sbatch Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh`.

---

## What it reads

Four on-disk shapes, all auto-detected:

| shape | layout | origin |
|---|---|---|
| A | `{seed}/results/{variant}.npz` | state-only flat |
| B | `{seed}/results/halfspace_{geo}/{variant}.npz` | state-only avoiding (`DA_Code_v3`) |
| C | `{seed}/results/{variant}/{variant}.npz` | Gen7 flat visual-aligning |
| D | `{seed}/results[_train_set]/{geo}/{variant}/{variant}.npz` | Gen7-with-geo and **Gen14** |

* `results_train_set/` (from `--eval-on-train`) is picked up alongside `results/`
  and recorded as `split=train`; the `_train_set` suffix is stripped from variant
  names so a train run and a test run of the same variant line up.
* Geometry is a **first-class axis** (`geo`, `geo_base`, `tightened`), not a name
  prefix. `halfspace_top-right-hard` → `geo=top-right-hard` for continuity with
  the old state-only CSVs.
* `*.partial.npz` crash-safety sidecars and `expert_references/` are ignored.
* A variant folder that has `diagnostics/` but no npz is still loaded, from the
  per-rollout JSONs.

Source selection: `--source auto` (default) = npz when present, else JSON.
`--source json` forces the JSON path. The two agree to float32 precision on
shared fields (verified against a Gen14 U7 run).

**Two rollout-JSON schemas** are supported and auto-detected per field — the
nested Gen14/late-Gen7 one (`success.strict`, `outcome.*`, `timing.*`,
`context.*`, `constraint.exec.*`) and the flat early-Gen7 one (`success`,
`mean_distance`, `steps`, `context_info.*`, `constraint_metrics.exec_*`). A tree
mixing both loads without a flag.

---

## Metrics

Ingestion is **generic** — every key in the npz is picked up and classified by
shape, so keys future evals add appear in the CSVs without a code change.
1-D arrays of length `n_rollouts` become per-rollout columns, `(N,2)` arrays split
into `_x`/`_y`, and run-level scalars (`success_rate`, `entropy`, HardFlow's
`nlp_solves`, …) are carried through as `n=1` rows in the long CSV.

The raw traces — `sampled_trajectories_all`, `obs_all`, `act_all`,
`selected_idx_all`, `physical_tracking_errors` — are **never read**. `np.load` is
lazy per key, so they are not even decompressed (they are ~90% of file size).

### DA_Code_v3 core metrics on visual runs

| `DA_Code_v3` metric | here |
|---|---|
| `n_success`, `n_steps`, `avg_time` | verbatim from the npz |
| `collision_free_completed` | derived ← `constraint_exec_zero_violation` |
| `n_violations` | derived ← `constraint_exec_n_violated_steps` |
| `n_success_and_constraints` | derived ← `n_success × constraint_exec_zero_violation` |
| `total_violations` | **NaN on visual runs** — that eval records per-family maxima, never a cumulative sum. Column kept (state-only runs still fill it) |

Replacements for the missing cumulative metric:
`constraint_exec_total_viol_count` (bounds + halfspace + obstacle step counts) and
`max_viol_depth_m` (worst depth across the three families).
`avg_time_ms` is added next to `avg_time` (npz stores seconds per replan).

### Frozen rollouts and the projector circuit breaker

Under a tightened geometry the D1 box-obstacle guard can declare a context
unusable and hold position for the whole episode **without ever calling the
model**. Those rollouts land in the arrays with `sat_rate = 1.0` and
`zero_violation = 1` and silently inflate every constraint aggregate. They are
only identifiable from `diagnostics/rollout_{r}_stats.json →
context.box_obstacle_conflict`, which this tool reads (`--no-diagnostics-scan`
turns it off).

Every long CSV therefore carries a **`mask` column**:

* `mask=all` — every rollout
* `mask=unfrozen` — frozen rollouts removed

Both are always written, so the toggle is a filter in the viewer, not a re-run.
On the U7 `combined_5-tightened` runs this is 4 of 30 rollouts per variant and
moves `constraint_exec_sat_rate` from 0.953 to 0.946.

`projection_cb_tripped` / `projection_cb_skipped_steps` mark rollouts where the
projector stopped projecting under load — a variant with a nonzero trip count did
not run the policy it claims to. Surfaced in `data_quality.csv` and the summary,
never ranked on.

---

## Outputs

| file | what |
|---|---|
| `per_rollout_detail.csv` | **wide, one row per rollout** — the finest grain, everything else is a groupby of this |
| `va2_units_long.csv` | per (candidate, seed, split, geo, variant, mask, metric): mean/std/min/max/n |
| `va2_aggregated_long.csv` | the same pooled over seeds, plus `n_seeds` |
| `data_quality.csv` | per unit: frozen count, circuit-breaker trips, `npz_complete`, source |
| `run_config.csv` | per unit: engine / K / threshold / mpc size … lifted from the npz `args` |
| `candidates_multidimensional_raw.csv`<br>`candidates_multidimensional_aggregated.csv` | `DA_Code_v3` column names — open a VA batch in `Data_Analysis/Visualizer/index.html` unchanged |
| `va_candidates_dynamic.csv` | `DA_Visual_Aligning` long format (variant carries the geo prefix) |
| `candidates_ranking.csv`, `candidates_detailed.csv`, `candidates_per_variant.csv` | candidate-level tables |
| `candidates_summary.txt` | human-readable ranking + a data-quality section |
| `logs/analysis.log`, `logs/loading.log`, `logs/discovery_manifest.json` | what ran, what loaded, what discovery saw |

The `DA_Code_v3`-compat schema has no geometry or split axis of its own, so:
`halfspace_variant ← geo` (suffixed `@train_set` for train-split rows) and
`constraint_type ← split`. The suffix stops a train-set run from being pooled
with a test run of the same geometry in a viewer that only knows
`halfspace_variant`.

`per_rollout_detail.csv` also carries legacy aliases (`success`, `mean_dist_m`,
`steps`, `phys_err_m`, `context_xy_dist_m`, `avg_time_s`, …) so the old VA
visualizer's rollout table still renders.

### Output folder naming — do not change casually

Both visualizers build their run dropdown by regexing the `analysis_results/`
**directory listing for a leading prefix**, and they disagree about it:

| page | pattern |
|---|---|
| `Visualizer/index.html` | `href="(batch_[^/"]+)` |
| `Visualizer_Visual_Aligning/index.html` | `href="(va_batch_[^/"]+)` |

`results_manifest.json` is only a *fallback*, used when the directory-listing
fetch fails — which it does not under `python -m http.server`. So a run whose
folder name misses the prefix is **invisible in the picker even though every CSV
is present**.

One name cannot match both patterns, so each run writes `batch_va2_<timestamp>`
and symlinks `va_batch_va2_<timestamp>` beside it — same files, one copy, listed
by both pages. `--output-path` with a name that does not start with `batch_`
still works, but logs a warning and will not appear in the DAv3 picker.

The manifest is refreshed on every run regardless, for servers that do forbid
directory listings.

---

## CLI

```
--parent-path      required; comma-separated trees are merged into one run
--output-path      default writes a timestamped subfolder
--candidates       "1,3"          keep these candidate indices
--candidate-names  "fm,mf,af"     display names, in index order
--seeds            "6,7,8"        default: every seed found
--variants         "dpcc-c,..."   post-strip names
--geos             "combined_5,combined_5-tightened"
--splits           "test" | "train"
--source           auto | npz | json
--no-diagnostics-scan   skip the JSON scan (faster; loses the frozen mask)
--plots                 render the PNG set (off by default)
--verbose
```

Candidates are numbered `1..N`, sorted by path, stable across runs — the same
scheme `DA_Code_v3` settled on after the >26-candidate letter overflow.

---

## Layout

```
discovery.py     candidate + unit discovery      (stdlib only — runnable anywhere)
data_loader.py   one unit → per-rollout frame + scalars + quality + run config
aggregator.py    master rollout table → units_long / agg_long / candidate stats
reporter.py      every CSV and the TXT summary
visualizer.py    the optional PNG set
config.py        names, labels, orderings
utils.py         logging, output dirs, HTML manifest
main_da_batch.py / main_da.py   CLIs
```

Note the environment split: this container has no scientific Python, so only
`discovery.py` runs here. The full pipeline runs on the cluster (`FMPCC` env).

Changelog: `logs_in_develop/DA_Code/DA_VA_v2/`.
