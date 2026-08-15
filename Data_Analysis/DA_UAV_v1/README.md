# DA_UAV_v1 — Gen15 UAV Mix-ML Data Analysis

Analysis tool for the closed-loop UAV MuJoCo evaluation written by
`mix_uav_test/eval_mix_uav.py` (**Gen15**, `logs/UAV_MIX/`) and — same artifact
schema, different root — the **Gen11** trees under `logs/UAV_FM/` it descends
from.

Built on the `DA_VA_v2` template, which is built on `DA_Code_v3`. **Neither was
modified**; this is a third parallel tool.

**CSV-first.** The CSVs are the deliverable; PNG generation exists but is off
unless you pass `--plots`.

---

## Quick start

```bash
# from the repo root, with Data_Analysis/DA_UAV_v1 on PYTHONPATH
python Data_Analysis/DA_UAV_v1/main_da_batch.py \
    --parent-path logs/UAV_MIX \
    --output-path Data_Analysis/analysis_results/batch_uav_$(date +%Y%m%d_%H%M%S)

# the K sweep for one scene, three engines
python Data_Analysis/DA_UAV_v1/main_da_batch.py \
    --parent-path logs/UAV_MIX --scenes corridor --engines fm,mf,af

# Gen15 against the Gen11 target rows (comma-separated trees = one comparison)
python Data_Analysis/DA_UAV_v1/main_da_batch.py \
    --parent-path logs/UAV_MIX,logs/UAV_FM --scenes corridor --k 20

# one eval-tag folder
python Data_Analysis/DA_UAV_v1/main_da.py --input-path <.../Emf_K4_mpc4_pid_stopgo_T0.5>

# what does discovery see? (stdlib only — runs anywhere, including this container)
python Data_Analysis/DA_UAV_v1/discovery.py logs/UAV_MIX
```

On the cluster, **zero arguments**:

```bash
sbatch Slurm_Codes/sbatch/DA/run_da_batch_uav.sh
sbatch Slurm_Codes/sbatch/DA/run_da_batch_uav.sh "" --scenes corridor --plots
```

With no path the job scans the `AUTO_ROOTS` in that script (`logs/UAV_MIX` and
`logs/UAV_FM`), skipping the ones that are absent.

---

## Why this is a separate tool and not a flag on DA_VA_v2

| | DA_VA_v2 (visual aligning) | DA_UAV_v1 |
|---|---|---|
| path shape | `{seed}/results/{geo}/{variant}/{variant}.npz` | `{seed}/{geo_tag}/{variant}/{variant}.npz` — **no `results/` level** |
| timing | in the npz | **only** in `diagnostics/rollout_*_stats.json` |
| tightening | a geometry twin (`combined_5-tightened`) | a **variant** (`dpcc-c-tightened`) |
| rollout mask | drop D1-frozen rollouts | drop **projection-circuit-breaker** rollouts |
| extra axes | — | **scene, engine, K**, parsed out of the path |

Discovery still accepts the four aligning/avoiding shapes, so a state-only or
visual tree merged into a UAV run reads without a flag.

---

## What it reads

```
logs/UAV_MIX/uav-<scene>/plans/mix_uav_<engine>/H8_D<Class>_9D[_tokens]/E<eng>_K<k>_mpc<b>_<ctrl>_T<thr>/<seed>/<geo_tag>/<variant>/
    <variant>.npz                    per-rollout outcome arrays
    results.json                     run-level summary (timing rollup, HardFlow NFE accounting)
    eval_<variant>.log               human-readable per-rollout lines
    <variant>.png                    2-D overview
    diagnostics/rollout_<r>_stats.json   ← per-rollout TIMING lives here and nowhere else
    diagnostics/rollout_<r>_mpc_foresight.svg
    PROJECTION_CB_TRIPPED.txt        sentinel, present only when the breaker opened
```

The **candidate** is the eval-tag folder — the one holding the numeric seed
subdirs. Candidates are numbered `1..N`, sorted by path, stable across runs.

* `*.partial.npz` crash-safety sidecars, `constraint_overview.png` and
  `config_snapshot_*/` are ignored.
* A variant folder with `diagnostics/` but no npz is still loaded, from the
  per-rollout JSONs (common on `s_curve`, which brushes the 24 h SLURM limit).
* The raw traces — `obs_all`, `act_all`, `sampled_trajectories_all` — are
  **never read**. `np.load` is lazy per key, so they are not even decompressed.

### ⚠️ Timing is JSON-only

`eval_artifacts.save_npz` persists the `success` / `physical` / `constraint` /
`goal` / `projection_health` groups and stops there — **the `timing` group is
not in the npz**. `avg_time`, `fm_ms` and `proj_ms` therefore come from
`diagnostics/rollout_*_stats.json` alone.

`--no-diagnostics-scan` is accepted for a fast structural pass and then says so
loudly, but a batch run that way has **no time axis** — which is the axis the
Gen15 K sweep exists to measure. `data_quality.csv` carries `timing_missing` per
unit so a half-timed batch is visible rather than quietly NaN.

---

## The path-encoded axes

The Gen15 experiment is a K sweep across engines and scenes (PLAN §7.3), and
those axes live in folder NAMES. `discovery.parse_axes()` turns them into real
columns, which is what makes "success vs K, per engine" drawable at all:

| axis | source | example |
|---|---|---|
| `scene` | dataset folder `uav-<scene>` | `corridor` |
| `engine` | eval tag `E{engine}`, else `mix_uav_<engine>/` | `mf` |
| `K` | eval tag `K{n}` | `4` |
| `mpc_batch` | eval tag `mpc{n}` | `4` |
| `controller` | eval tag (may contain `_`) | `pid_stopgo` |
| `threshold` | eval tag `T{f}` | `0.5` |
| `backbone`, `data_proportion`, `alpha_init/end`, `train_K`, `horizon`, `diffusion_cls` | model folder | `unet`, `0.5`, … |
| `generation` | `logs/UAV_MIX` vs `logs/UAV_FM` | `Gen15` |

Candidates get a display name built from them (`corridor|mf|K4|bbunet|dp0.5`)
instead of the bare eval tag, so a K sweep is readable straight off the
candidate list. `candidate_axes.csv` prints the parse next to the raw folder
name — **read it first**; it is where a mis-parse shows up.

> 🔴 `K{n}` in the folder is truthful only because `_load_base_cfg` injects
> `flow_steps_v3` into the config from the plan block. In Gen11 that key was
> never present, so **every Gen11 folder is labelled K20 regardless of the real
> budget**. `run_config.csv` carries `path_K` next to the pickled `flow_steps`
> for exactly this check.

Filters: `--scenes`, `--engines`, `--k` are applied **before** unit enumeration,
so a rejected candidate's files are never touched.

---

## Metrics

Ingestion is generic — every key in the npz is picked up and classified by
shape, so keys a future eval adds appear in the CSVs without a code change. The
npz's Fix_10 group-prefixed names are **renamed onto** the DA_Code_v3 vocabulary
both HTML viewers speak; the raw name is kept beside it.

| npz key | canonical name |
|---|---|
| `success_strict` | `n_success` |
| `success_strict_and_constraints` | `n_success_and_constraints` |
| `success_relaxed_and_constraints` | `n_success_relaxed_and_constraints` |
| `constraint_collision_free` | `collision_free_completed` |
| `constraint_n_violations` | `n_violations` |
| `constraint_total_violations` | `total_violations` |

Derived:

| metric | from | why |
|---|---|---|
| `avg_time` | `avg_time_ms / 1000` | DA_Code_v3 reports seconds/replan and both viewers plot that name |
| `steps_to_goal` | `n_steps` on reaching episodes only | `n_steps` early-stops on goal-reach and runs the **full budget** on a miss (U_13), so averaging it over both measures misses as much as speed |
| `over_budget_frac` | `over_budget_steps / n_steps` | gate G6's "fraction of steps exceeding `1/control_hz`" |
| `nfe_effective` | HardFlow's measured `nfe_per_plan`, else `K` | `hardflow_new` evaluates the network **twice per ODE step**, so at the same K it spends 2× the generation budget of a DPCC arm (Gen15 U2). Quote this, not K, when the two are compared |

Two scene-specific readings that are not bugs:

* **`scene=empty`** has a RANDOM per-episode start→goal the state-only policy is
  never told, so goal-reaching is ill-defined there — its success is stable/safe
  flight only. Its `goal_*` columns are not a policy failure.
* **`variant=diffuser`** runs no projector by design. `proj_ms` is 0.0 and
  `has_projector=0` is recorded per unit so that is not read as a broken timer.

---

## The rollout mask

Every long CSV carries a **`mask` column**:

* `mask=all` — every rollout
* `mask=proj_valid` — rollouts whose projection circuit breaker never opened

When the sustained-slowness breaker trips (`mix_uav/sampling/projection.py`
Fix_15.2) the rollout ran, partly or wholly, on the **UNPROJECTED** trajectory.
Its constraint numbers describe a policy the variant name does not name, and
pooling it moves every constraint aggregate in whichever direction the
unprojected plan happened to fall. Both reductions are always written, so the
toggle is a filter in the viewer, not a re-run.

`projection_cb_tripped` / `projection_cb_skipped_steps` / `cb_trips` /
`backstop_hits` are surfaced in `data_quality.csv` and the summary, and never
ranked on. The eval's own `PROJECTION_CB_TRIPPED.txt` sentinel is counted at
discovery time, so the warning lands in the log before the numbers do.

---

## Outputs

| file | what |
|---|---|
| `candidate_axes.csv` | **read first** — one row per candidate: scene/engine/K/controller/…, seeds, last run, sentinel count, raw eval tag next to its parse |
| `uav_k_sweep.csv` | per (scene, engine, geo, variant, split, **K**, mask, metric) — the Gen15 table, with `n_candidates`/`candidates` flagging cells that pool more than one candidate at one K |
| `per_rollout_detail.csv` | **wide, one row per rollout** — the finest grain, everything else is a groupby of this |
| `uav_units_long.csv` | per (candidate, seed, split, geo, variant, mask, metric): mean/std/min/max/n, plus run-level scalars as `n=1` rows |
| `uav_aggregated_long.csv` | the same pooled over seeds, plus `n_seeds` |
| `data_quality.csv` | per unit: circuit-breaker counts, sentinel, `timing_missing`, `npz_complete`, source |
| `run_config.csv` | per unit: the pickled eval `args` **plus** `path_K` for the folder-vs-pickle check |
| `candidates_multidimensional_raw.csv`<br>`candidates_multidimensional_aggregated.csv` | `DA_Code_v3` column names — open a UAV batch in `Data_Analysis/Visualizer/index.html` unchanged |
| `candidates_ranking.csv` | ranked, with a `Pareto` column marking the front |
| `candidates_detailed.csv`, `candidates_per_variant.csv` | candidate-level tables |
| `candidates_summary.txt` | batch shape, ranking, **the K sweep as a text grid**, data quality, and the caveats |
| `logs/analysis.log`, `logs/loading.log`, `logs/discovery_manifest.json` | what ran, what loaded, what discovery saw |

### Pareto, and the word "best"

`candidates_ranking.csv` marks the front of (success+constraint up, ms/replan
down) **within this batch**. A candidate off the front that is cheaper OR more
accurate but not both is a **trade-off / non-dominated** result, and the summary
says so in words. Nothing here is labelled "best".

The standing comparison hierarchy applies: **MF/AF must beat naive FM**, and
there is **no diffusion-DPCC UAV checkpoint from Gen11** (PLAN §1.5) — until a
Gen15 `diffusion`-engine candidate is in the batch, the strongest available
claim is *"vs Gen11 naive FM + DPCC"*, never *"beats DPCC"*.

### `LatestSnapshot` / `Latest_Snapshot` — when was this run produced?

Every eval launch drops `snapshot_<YYYYMMDD_HHMMSS>` into
`<candidate>/<seed>/config_snapshot_<config>/` and never deletes the previous
ones (`mix_uav/utils/setup.py::snapshot_configs`, written at the eval-tag-aware
seed dir per Fix_8). Discovery scans them:

| where | grain |
|---|---|
| `uav_units_long.csv`, `candidates_multidimensional_raw.csv` | that **seed's** newest marker (blank if that seed has none) |
| `uav_aggregated_long.csv`, `candidates_multidimensional_aggregated.csv`, `candidates_ranking.csv` | newest marker over all the candidate's seeds |
| `candidates_detailed.csv`, `candidate_axes.csv` | plus `First_Snapshot`, `Snapshot_Count`, `Snapshot_By_Seed` |

`Snapshot_By_Seed` is the one to read when a candidate looks half-stale: UAV
seeds are launched as separate SLURM jobs, so a single fresh seed can hide four
old ones behind a recent `Latest_Snapshot`.

---

## Viewing it

`Data_Analysis/Visualizer_UAV_v1/index.html` reads these CSVs directly. Serve
the repo root and open it:

```bash
python3 -m http.server 8000        # from the repo root
# → http://localhost:8000/Data_Analysis/Visualizer_UAV_v1/index.html
```

It adds a **1.7 UAV Axes** panel (scene / engine / K) that filters *every* view
at once, and repoints the mask onto `projection_cb_tripped`. Everything else is
inherited from `Visualizer_VA_v2` via `build_from_va2.py` — regenerate with:

```bash
python Data_Analysis/Visualizer_UAV_v1/build_from_va2.py
python Data_Analysis/Visualizer_UAV_v1/test_page_structure.py   # stdlib
```

`Visualizer/index.html` (DAv3) also opens a UAV batch, through the
`candidates_multidimensional_*.csv` pair.

### Output folder naming — do not change casually

The viewer builds its run dropdown by regexing the `analysis_results/`
**directory listing** for `batch_uav_*`. `results_manifest.json` is only a
*fallback*, used when the directory-listing fetch fails — which it does not under
`python -m http.server`. A run whose folder name misses the prefix is therefore
**invisible in QUICK_LIST even though every CSV is present**; open it via
CUSTOM_PATH, or name it `batch_uav_*`. The run logs a warning when it does not.
`batch_uav_*` also satisfies the DAv3 page's `batch_*` pattern, so one folder
name serves both with no symlink.

---

## CLI

```
--parent-path      required; comma-separated trees are merged into one run
--output-path      default writes a timestamped subfolder
--scenes           "corridor,pillars"      path-encoded axis filter
--engines          "fm,mf,af,diffusion"    path-encoded axis filter
--k                "1,2,5,10,20"           path-encoded axis filter
--candidates       "1,3"                   keep these candidate indices
--candidate-names  "fm,mf,af"              display names, in index order
--seeds            "6,7,8"                 default: every seed found
--variants         "dpcc-c,diffuser"       projection variants
--geos             "corridor_bounds+dynamics+..."   geo_tag folders
--splits           "test" | "train"        UAV writes only `test`
--source           auto | npz | json
--no-diagnostics-scan   fast structural pass; LOSES every timing metric
--plots                 render the PNG set (off by default)
--verbose
```

---

## Layout

```
discovery.py     candidate + unit discovery + the path-axis parser  (stdlib only)
data_loader.py   one unit -> per-rollout frame + scalars + quality + run config
aggregator.py    master rollout table -> units_long / agg_long / k_sweep / candidate stats
reporter.py      every CSV and the TXT summary
visualizer.py    the optional PNG set (K sweeps, Pareto, time split, quality)
config.py        names, labels, orderings, the npz/JSON field maps
utils.py         logging, output dirs, HTML manifest
main_da_batch.py / main_da.py   CLIs
```

```
test_discovery_offline.py    path shapes + axis parsing        (stdlib)
```

Note the environment split: this container has no scientific Python, so only
`discovery.py` and `test_discovery_offline.py` run here. The full pipeline runs
on the cluster (`FMPCC` env).

Changelog: `logs_in_develop/Gen15/U4/`.
