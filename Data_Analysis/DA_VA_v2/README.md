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

### Legacy (bridged) trees — the `_` prefix

A pipeline that predates this API can be **bridged** into shape C instead of
being re-run. The contract is one character: a candidate under a folder whose
name starts with `_` is legacy, and `data_loader` applies that pipeline's rescues
on top of the normal load. Nothing else changes — same discovery, same tables.

| tree | root | read as |
|---|---|---|
| bridged D3IL baseline (old runs) | `logs/d3il_visual_aligning_baseline/_DA_VA_BRIDGE_d3il_baseline/` | legacy path |
| native D3IL baseline (new runs) | `logs/d3il_visual_aligning_baseline/DA_VA_d3il_baseline/` | normal Gen14 path |

```bash
# convert finished old runs (cluster; --json-only needs no numpy)
python d3il_visual_aligning_baseline_test/bridge_d3il_va_to_da_va_v2.py \
    --source-root logs/d3il_visual_aligning_baseline

# a SECOND source tree that reuses the same agent+seed needs its own root + label,
# or it overwrites the first (the bridge refuses and says so). Pin the scale when
# the run was killed before writing results_seed_*.json:
python d3il_visual_aligning_baseline_test/bridge_d3il_va_to_da_va_v2.py \
    --source-root "logs/d3il_visual_aligning_baseline(Bf_U3)" \
    --out-root    logs/d3il_visual_aligning_baseline/_DA_VA_BRIDGE_d3il_baseline_Bf_U3 \
    --label Bf_U3 --n-contexts 60 --n-trajectories 18

# then analyse any root exactly like a Gen14 tree (comma-separated = one comparison)
python Data_Analysis/DA_VA_v2/main_da_batch.py \
    --parent-path logs/d3il_visual_aligning_baseline/_DA_VA_BRIDGE_d3il_baseline
```

The D3IL visual-aligning baseline (`variant=d3il_baseline`, `geo=none`) has **no
projector**: every `constraint_*` metric, `collision_free_completed` and
`n_success_and_constraints` is NaN *by design*, never 0. `data_quality.csv` says
so per unit (`legacy`, `legacy_kind`, `has_projector`), and candidate ranking
falls back from goal+constraint to plain goal success for such a candidate — so
compare it on success / distance / time only. Its `avg_time` is **seconds per
control step** (D3IL calls the policy every step), not per replan like Gen14.

Point `--parent-path` at ONE root: both roots live under
`logs/d3il_visual_aligning_baseline/`, so passing the shared parent discovers the
bridged and native copies of the same seed as two candidates.

Regression test (stdlib, runs in this container):
`python Data_Analysis/DA_VA_v2/test_legacy_bridge_scan.py`

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
| `candidates_ranking.csv`, `candidates_detailed.csv`, `candidates_per_variant.csv` | candidate-level tables |
| `candidates_summary.txt` | human-readable ranking + a data-quality section |
| `logs/analysis.log`, `logs/loading.log`, `logs/discovery_manifest.json` | what ran, what loaded, what discovery saw |

### `LatestSnapshot` / `Latest_Snapshot` — when was this run produced?

Every eval launch drops a marker file `snapshot_<YYYYMMDD_HHMMSS>` into
`<candidate>/<seed>/config_snapshot_<config>/` (`utils/setup.py::snapshot_configs`)
and never deletes the previous ones. Discovery scans them, so the CSVs carry the
answer to "is this candidate's result from last night or from three weeks ago?"
without opening the run folder:

| where | grain |
|---|---|
| `va2_units_long.csv`, `candidates_multidimensional_raw.csv` | that **seed's** newest marker (blank if that seed has none) |
| `va2_aggregated_long.csv`, `candidates_multidimensional_aggregated.csv`, `candidates_ranking.csv` | newest marker over all the candidate's seeds |
| `candidates_detailed.csv` | plus `First_Snapshot`, `Snapshot_Count`, `Snapshot_By_Seed` (`6:2026… \| 7:2026…`) |
| `candidates_summary.txt` | human-readable, with the per-seed line when the seeds disagree |

Both HTML viewers show it as a **Last Run** column in the Path Audit Map and in
the Plot Legend, and it is repeated in the exported audit `.txt` / LaTeX. Batches
produced before this column existed simply have no column and the viewers drop
it rather than showing blanks. `Snapshot_By_Seed` is the one to read when a
candidate looks half-stale: seeds are usually launched as separate jobs, so a
single fresh seed can hide four old ones behind a recent `Latest_Snapshot`.

The `DA_Code_v3`-compat schema has no geometry or split axis of its own, so:
`halfspace_variant ← geo` (suffixed `@train_set` for train-split rows) and
`constraint_type ← split`. The suffix stops a train-set run from being pooled
with a test run of the same geometry in a viewer that only knows
`halfspace_variant`.

### Viewing it

`Data_Analysis/Visualizer_VA_v2/index.html` is the viewer for these CSVs — it
reads the four native files above directly (no compatibility shim). Serve the
repo root and open it:

```bash
python3 -m http.server 8000        # from the repo root
# → http://localhost:8000/Data_Analysis/Visualizer_VA_v2/index.html
```

#### Variant quick presets

Three checkboxes above `5. Variants` tick a whole comparison at once — **DPCC +
HF**, **DPCC + HF (tightened)**, **DPCC (tightened)** — each being `diffuser` plus
the matching `dpcc-{r,c,t}` / `hardflow_new*` arms. They are a shortcut for the
boxes below them, not a second filter. The `dpcc-c-dt*` timestep sweep and the
non-projection baselines (`gradient`, `post_processing`, `model_free`,
`geo_free`, `bounds_free`) are excluded from all three.

**On a visual-aligning batch only the first preset usually appears.** Tightening
here is a property of the **geometry**, not of the variant name, so there are no
`-tightened` variants for the other two to select and they are dropped rather
than left dead — the panel says so and points at `4. Geometry Focus`, which is
where tightening is actually chosen on this page. A batch read from a state-only
*avoiding* tree does carry `-tightened` variant names, and there all three appear.

#### Plot Legend: highlight + seed coverage

The **Plot Legend — Selected Candidates** table under the chart carries two
columns that exist only in the page, not in any CSV (inherited from the DAv3
page via `build_from_dav3.py`, so both viewers behave identically):

- **HL** — a checkbox that paints that candidate's *name* red everywhere it is
  printed: the plot's x tick label, the legend row, every row head in the Result
  Matrices **and** in the U3 run-coverage table, and the Path Audit Map. Only the
  name changes — no number, no cell background — so it can never be misread as a
  data annotation the way the `(goal, constraint)` flags can. `[CLEAR
  HIGHLIGHTS]` in the legend header resets them all, including candidates you
  have since unticked in the sidebar. The exported `.txt` / `.tex` have no
  colour, so they mark the same candidates in words.
- **Seeds** — the seeds this candidate actually has (from `va2_units_long.csv`),
  with a red **⚠ NOT FULL** naming the ones it is missing, measured against the
  batch's full seed set or against the ticked seeds in Custom Seed Compare. VA
  batches are routinely unbalanced — see the run-coverage table above the
  matrices for the rollout counts — and a bar averaged over one seed looks
  exactly like a bar averaged over five on the chart.

#### Plot: `(G, C)` failure hints

A red **`(G, C)`** stacked above a bar's value marks a bar whose run was not
clean — `(G)` = goal not always reached, `(C)` = a constraint was violated, the
same rule (and the same strict `n_success` pair) as the Result Matrices' trailing
`(goal, constraint)` flag. An unflagged number is a fully successful run; the
note under the x-axis spells the marks out and counts them.

Skipped on the four success metrics themselves — including this page's relaxed
pair (`success_relaxed`, `n_success_relaxed_and_constraints`, via `FLAG_SKIP` in
`build_from_dav3.py`) — where the flag would only restate the bar's own height.
Computed on the plot's own x-axis grouping, so it stays correct in `By
Environment` mode where the selected candidates are averaged into each bar.

#### Variant colours

Every variant gets its own colour, verified on each draw — the page used to pass
`colormap='tab10'` (ten colours), so with more than ten variants selected two
methods drew identically. Colours are keyed to the variant's position in the
batch's full variant list, so a variant keeps its colour as you tick others on
and off. The COMPARE view shares the same generator (it previously wrapped with
`i % 20`, giving the 21st series the 1st series' colour).

The old `Visualizer_Visual_Aligning/index.html` (VA v1) is superseded and is not
fed by this pipeline any more; `Visualizer/index.html` (DAv3) still opens a VA
batch through the `candidates_multidimensional_*.csv` pair.

### Output folder naming — do not change casually

The viewer builds its run dropdown by regexing the `analysis_results/`
**directory listing** for `batch_va2_*`. `results_manifest.json` is only a
*fallback*, used when the directory-listing fetch fails — which it does not under
`python -m http.server`. A run whose folder name misses the prefix is therefore
**invisible in QUICK_LIST even though every CSV is present**; open it via
CUSTOM_PATH, or name it `batch_va2_*`. The run logs a warning when it does not.

`batch_va2_*` also satisfies the DAv3 page's own `batch_*` pattern, so one folder
name serves both with no symlink or alias.

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

```
test_snapshot_scan.py        "Last Run" timestamps            (stdlib)
test_legacy_bridge_scan.py   legacy/bridged tree handling     (stdlib)
```

Note the environment split: this container has no scientific Python, so only
`discovery.py` and the two stdlib tests run here. The full pipeline runs on the
cluster (`FMPCC` env).

Changelog: `logs_in_develop/DA_Code/DA_VA_v2/`.
