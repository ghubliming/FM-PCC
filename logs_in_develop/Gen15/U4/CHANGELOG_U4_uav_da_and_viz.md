# CHANGELOG — Gen15 U4: the UAV data-analysis pipeline (`DA_UAV_v1`) + its viewer

**Date:** 2026-08-15 · **Type:** new analysis tool · **Status:** code complete, **NOTHING RUN ON REAL DATA**
**Files (all new):**
`Data_Analysis/DA_UAV_v1/` (9 modules + README + offline test),
`Data_Analysis/Visualizer_UAV_v1/` (`build_from_va2.py`, generated `index.html`, structure test),
`Slurm_Codes/sbatch/DA/run_da_batch_uav.sh`
**Modified:** nothing. `DA_Code_v3`, `DA_VA_v2`, `Visualizer/`, `Visualizer_VA_v2/`, `mix_uav*/`
are untouched — this is a third parallel tool in the established sibling pattern.
**Retraining:** none. Reads artifacts the Gen15 eval already writes.

Gen15 had no DA. This is it, plus the HTML explorer for it.

---

## 1. Why a third tool and not a flag on `DA_VA_v2`

`DA_VA_v2` was the template (it is the newest and the most general), but five properties of the
UAV eval are not switchable-on:

| | `DA_VA_v2` (visual aligning) | `DA_UAV_v1` |
|---|---|---|
| path shape | `{seed}/results/{geo}/{variant}/{variant}.npz` | `{seed}/{geo_tag}/{variant}/{variant}.npz` — **no `results/` level** |
| timing | in the npz | **only** in `diagnostics/rollout_*_stats.json` |
| tightening | a geometry twin (`combined_5-tightened`) | a **variant** (`dpcc-c-tightened`) |
| rollout mask | drop D1-frozen rollouts | drop **projection-circuit-breaker** rollouts |
| extra axes | — | **scene, engine, K**, parsed out of the path |

The first is structural (`eval_mix_uav.py:1438-1444` builds `seed_dir/geo_dir/out_dir` by hand,
with no results level); the second changes which files are mandatory to open; the last one is
the whole Gen15 experiment. Discovery still accepts all four aligning/avoiding shapes, so a
state-only or visual tree merged into a UAV run reads with no flag.

---

## 2. The thing this tool exists for: K, engine and scene as columns

PLAN §7.3: *"the K sweep is the experiment"* — every arm at K ∈ {1, 2, 5, 10, 20}, and the claim
under test is that `mf`/`af` hold success where `fm` collapses, with the freed wall clock showing
up as fewer circuit-breaker trips.

Those axes live in **folder names**:

```
logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/H8_DMeanFlowODE_9D_dp0.5_bbunet/Emf_K4_mpc4_pid_stopgo_T0.5/6/…
             ^^^^^^^^                ^^                              ^^^ ^^^^     ^^ ^^^^  ^^^^^^^^^^ ^^^^^
             scene                   engine                       dp  bb        K  mpc   controller  thresh
```

Discovery is the only stage that sees them, so `discovery.parse_axes()` turns them into real
columns carried on every unit and every CSV row. Without that, "success vs K per engine" is not
drawable: the K values sit in different rows with nothing tying them together.

Three consequences:

1. **`uav_k_sweep.csv`** — the only table with no `DA_VA_v2` counterpart. Grouped on
   `(scene, engine, geo, variant, split, K, mask, metric)`, i.e. K is a column, not an identity.
   Candidates are deliberately **not** a key — two candidates differing only in K are two points
   of one curve. Two differing in something else (mpc batch, controller, threshold, backbone)
   would then be silently pooled, so the table carries `n_candidates` / `candidates` and the run
   logs a warning naming every cell built from more than one.
2. **`--scenes` / `--engines` / `--k`** filter *before* unit enumeration — a rejected candidate's
   files are never opened. A 4-scene × 4-engine × 5-K tree is 80 candidates; the six rows of the
   plot being drawn is the useful working set.
3. **Candidate display names** are built from the axes (`corridor|mf|K4|bbunet|dp0.5`) instead of
   the bare eval tag, so a sweep is readable straight off the candidate checkbox list — including
   in the DAv3 page, which knows nothing about any of this.

`candidate_axes.csv` prints the parse next to the raw folder name. It is the file to read first,
and the place a mis-parse shows up.

### The Gen11 K trap, made visible

`_uav_eval_tag`'s 🔴 note: `K{n}` is truthful only because `_load_base_cfg` injects
`flow_steps_v3` from the plan block. In Gen11 that key was never present, so **every Gen11 folder
is labelled K20 regardless of the real budget.** `run_config.csv` therefore carries `path_K`
(from the folder) beside the pickled `flow_steps` (from the run) — a disagreement is the bug in
its observable form. `generation` (`Gen15` from `logs/UAV_MIX`, `Gen11` from `logs/UAV_FM`) rides
on every row so the two are never pooled by accident.

---

## 3. Timing is JSON-only — and that is the axis the generation is about

`eval_artifacts.save_npz` writes the `success` / `physical` / `constraint` / `goal` /
`projection_health` groups and **not** the `timing` group. So `avg_time`, `fm_ms` and `proj_ms`
— PLAN §7.2's per-plan wall clock and projection ms — exist **only** in
`diagnostics/rollout_<r>_stats.json`.

Consequences, all of them deliberate:

* the diagnostics scan is **mandatory in practice**, not an optional extra (`DA_VA_v2` treats it
  as a nicety that adds a frozen-rollout mask);
* `--no-diagnostics-scan` is still accepted for a fast structural pass, but the CLI, the loader
  log, `data_quality.csv` (`timing_missing`), the summary and the sbatch header all say what it
  costs;
* the sbatch script carries a ⚠️ telling the next person not to add it to save time.

A unit whose diagnostics folder is missing gets NaN timing and a `WARN` line naming it, rather
than a zero that would read as "infinitely fast".

---

## 4. The mask: the UAV hazard is the projector, not a frozen box

`DA_VA_v2`'s mask drops **D1-frozen** rollouts — the visual-aligning eval held position and never
called the model, so they report `sat_rate 1.0` and inflate every constraint aggregate.

UAV has no D1 guard. Its equivalent is the **projection circuit breaker** (`projection.py`
Fix_15.2): under sustained SLSQP slowness the eval SKIPS projection, and those steps fly the
UNPROJECTED plan. Such a rollout's constraint numbers describe a policy the variant name does not
name — and unlike a frozen rollout they can move the aggregate in *either* direction, depending
on where the unprojected plan happened to fall.

So `mask ∈ {all, proj_valid}`, keyed on `projection_cb_tripped`. Both reductions are always
written, so the toggle is a filter in the viewer rather than a re-run. `cb_trips`,
`backstop_hits`, `cb_skipped_steps` and the eval's own `PROJECTION_CB_TRIPPED.txt` sentinel are
surfaced in `data_quality.csv` and the summary, and **never ranked on**. The sentinel is counted
at discovery time, so the warning lands in the log before any number does.

---

## 5. Metric mapping

The npz uses the Fix_10 group-prefixed schema. Those keys are **renamed onto** the `DA_Code_v3`
vocabulary both HTML viewers already speak, keeping the raw name beside it — so a UAV batch opens
in `Visualizer/index.html` unchanged and `per_rollout_detail.csv` can still be read against
`eval_artifacts.save_npz` line by line.

| npz | canonical |
|---|---|
| `success_strict` | `n_success` |
| `success_strict_and_constraints` | `n_success_and_constraints` |
| `success_relaxed_and_constraints` | `n_success_relaxed_and_constraints` |
| `constraint_collision_free` | `collision_free_completed` |
| `constraint_n_violations` | `n_violations` |
| `constraint_total_violations` | `total_violations` |

Note that UAV writes the **relaxed** goal+constraint pair natively (the eval records all four
cells of the {strict, relaxed} × {with, without constraints} matrix per rollout). `DA_VA_v2`'s
viewer has to synthesise that column per rollout because `mean(a·b) ≠ mean(a)·mean(b)`; here the
synthesis is simply skipped.

Derived:

| metric | from | why |
|---|---|---|
| `avg_time` | `avg_time_ms / 1000` | `DA_Code_v3` reports s/replan and both viewers plot that name |
| `steps_to_goal` | `n_steps` on reaching episodes only | `n_steps` early-stops on goal-reach and runs the **full budget** on a miss (U_13), so averaging it over both measures misses as much as speed. This is the eval's own `steps.to_goal_mean` |
| `over_budget_frac` | `over_budget_steps / n_steps` | gate G6's "fraction of steps exceeding `1/control_hz`" |
| `nfe_effective` | HardFlow's measured `nfe_per_plan`, else `K` | see below |

### The HardFlow fairness number

U2 §"⚠️ FAIRNESS": `hardflow_new` evaluates the network **twice per ODE step** (reference step +
terminal predict), so an arm-C run at K costs 2K network evals while a DPCC arm at K costs K.
Comparing them "at the same K" compares half the generation budget on the DPCC side. The eval
records the real count in `results.json`; the loader lifts it into `nfe_effective`, which falls
back to the eval-tag K for non-HardFlow arms — one number that can honestly go on a cost axis
across every arm. The README and the summary both say to quote it rather than K.

### Two readings that are not bugs

* **`scene=empty`** has a RANDOM per-episode start→goal the state-only policy is never told
  (`generator._build_traj_and_init`), so goal-reaching is ill-defined there — success is stable /
  safe flight only. Its `goal_*` columns are not a policy failure.
* **`variant=diffuser`** runs no projector by design; `proj_ms` is 0.0 and `has_projector=0` is
  recorded per unit so that is not mistaken for a broken timer.

---

## 6. Pareto, and the word "best"

`candidates_ranking.csv` gains a `Pareto` column marking the front of (success+constraint ↑,
ms/replan ↓) **within the batch**, and the summary spells out the rule in words: a candidate off
the front that is cheaper OR more accurate but not both is a **trade-off / non-dominated**
result. Nothing is labelled "best". The Pareto plot draws the front as a dashed line and stars
its members, because a plain scatter invites reading the top-left-most point as the winner even
when nothing dominates it.

The summary's NOTES also restate the standing hierarchy: **MF/AF must beat naive FM**, and until
a Gen15 `diffusion`-engine candidate (U3) is in the batch, the strongest available claim is
*"vs Gen11 naive FM + DPCC"*, never *"beats DPCC"* (PLAN §1.5).

`run_da_batch_uav.sh` scans **both** `logs/UAV_MIX` and `logs/UAV_FM` by default, because §7.1
defines the target row as *the best Gen11 fm+DPCC row on the same scene, geo variant, projection
variant, K and seed set* — a target that is not in the same batch gets compared by eye across two
runs, which is how mismatched K and mismatched seed sets get compared.

---

## 7. Outputs

| file | what |
|---|---|
| `candidate_axes.csv` | **read first** — per candidate: scene/engine/K/controller/…, seeds, last run, sentinel count, raw eval tag beside its parse |
| `uav_k_sweep.csv` | metric vs **K** per (scene, engine, geo, variant) |
| `per_rollout_detail.csv` | wide, one row per rollout — everything else is a groupby of this |
| `uav_units_long.csv` | per (candidate, seed, split, geo, variant, mask, metric) + run-level scalars as `n=1` rows |
| `uav_aggregated_long.csv` | pooled over seeds, plus `n_seeds` |
| `data_quality.csv` | circuit breaker, sentinel, `timing_missing`, `npz_complete`, source |
| `run_config.csv` | pickled eval `args` **plus** `path_K` |
| `candidates_multidimensional_{raw,aggregated}.csv` | `DA_Code_v3` schema — opens in `Visualizer/index.html` |
| `candidates_{ranking,detailed,per_variant}.csv` | candidate-level tables |
| `candidates_summary.txt` | batch shape · candidates · ranking · **K sweep as a text grid** · data quality · caveats |
| `logs/{analysis,loading}.log`, `logs/discovery_manifest.json` | what ran, what loaded, what discovery saw |

Optional PNGs (`--plots`, off by default): three K-sweep curves (accuracy / time / cb-trips), the
Pareto scatter, per-variant bars, a **generation-vs-projection stacked time split** with the
real-time budget line — the Gen15 mechanism is that cheaper generation *releases* budget to the
projector, which is a statement about the split of the 30 ms, not its total — and the quality bar.

Output folder: `batch_uav_<ts>`, which satisfies both this viewer's `batch_uav_*` picker and the
DAv3 page's `batch_*` one with no symlink.

---

## 8. The viewer — derived, not written

`Data_Analysis/Visualizer_UAV_v1/index.html` is **generated** by `build_from_va2.py` from
`Visualizer_VA_v2/index.html`, which is itself generated from the DAv3 page. Every inherited
feature (U7 no-data messages, U8 value labels, U9 plot legend, U10 result matrices, U10.1 LaTeX,
U11 folder ZIP, U13 highlight + seed coverage, U14 `(G, C)` flags, U15 distinct variant colours,
U17 audit-map seeds, Last Run stamps, seed modes, zoom, and VA v2's per-rollout / compare /
quality views) comes for free and stays in sync by re-running the builder. Each of the 41 edits
asserts its anchor, so a moved anchor fails loudly instead of silently dropping half the page.

What this layer changes:

1. identity + the `uav_*.csv` / `batch_uav_*` wiring;
2. the mask → `proj_valid` / `projection_cb_tripped`, everywhere it is spelled (option, banner,
   `_slice`, row tint, quality columns);
3. **a new "1.7 UAV Axes" panel** — scene / engine / K checkbox groups, populated from the batch
   itself, applied inside `_slice` next to mask and split. That placement is the point: a filter
   that reached only the chart would leave the result matrices, the audit map and the
   per-rollout table describing a different subset than the plot above them. The banner prints
   the active slice, and the panel carries the *matched budget or nothing* warning;
4. UAV metrics wherever a metric list is hardcoded — the result matrices, the per-rollout default
   columns and sort options, the compare-view default axes;
5. the reference row under `N_STEPS` becomes the **step budget** (`max_episode_length`). 396
   steps means nothing alone; against a budget of 396 it means every episode ran to the wall
   without reaching the goal — the same job the INIT XY row does on the aligning page;
6. a fourth variant preset, **Constraint ablations** (`model_free` / `bounds_free` / `geo_free`
   and their combinations), which is the "which constraint family is doing the work" comparison
   the UAV yaml is set up for and the aligning page has no equivalent of. Tightening is back on
   the variant axis here, so all three inherited presets have members (on a VA batch two of them
   are dropped as empty).

One normalisation worth naming: `K` is written as an integer but read back as `float64` whenever
**any** candidate has an unparsable eval tag (its K is blank → NaN → the column is float). `"4.0"`
then never matches the `"4"` on its checkbox and the filter silently empties the page. `_norm`
coerces it to the integer spelling once, at load.

---

## 9. What was tested, and what was not

Runs **in this container** (stdlib only — no numpy/pandas here):

| test | covers |
|---|---|
| `Data_Analysis/DA_UAV_v1/test_discovery_offline.py` | 60 checks over a synthetic tree with the exact shape the eval writes: candidate = the eval-tag folder; the geo/variant split for the UAV shape **and** the four `results/`-bearing ones; scene/engine/K/mpc/controller/threshold/backbone parsing including the underscore-bearing controller token and the Gen11 no-`E`-token spelling; `.partial.npz` / `expert_references/` / `config_snapshot_*` / the geo-level `constraint_overview.png` all correctly not-a-result; a diagnostics-only variant still found; snapshot stamps and per-seed staleness; the CB sentinel; the axis filters |
| `Data_Analysis/Visualizer_UAV_v1/test_page_structure.py` | the generated page's PyScript block **compiles**; every UAV string the layer promises is present; no stale VA identifier survives that would now be wrong (a leftover `frozen` mask column reads every rollout as unmasked; a leftover `va2_*.csv` name makes the page fetch a file this pipeline never writes); the 12 inherited features are still there |

Both pass. `python -m py_compile` is clean across all 11 new Python files, and a static
import-graph check confirms every cross-module name resolves.

**NOT tested — needs the cluster:**

* the loaders, aggregator, reporter and visualizer have never touched a real npz or a real
  `rollout_*_stats.json`. Every metric mapping in §5 is read off `eval_artifacts.py` and
  `eval_mix_uav.py` by eye;
* `Visualizer_VA_v2/test_page_offline.py` (which drives the handlers against real CSVs) has no
  UAV counterpart — porting it needs a real `batch_uav_*` folder to drive it with.

**First cluster run should be:**

```bash
# 1. structural pass, no timing, fast — does discovery see what you expect?
sbatch Slurm_Codes/sbatch/DA/run_da_batch_uav.sh "" --no-diagnostics-scan
#    then read candidate_axes.csv and logs/discovery_manifest.json FIRST.

# 2. the real thing
sbatch Slurm_Codes/sbatch/DA/run_da_batch_uav.sh
```

Things to check on that first run, in order: (a) `candidate_axes.csv` — does every candidate have
a scene/engine/K, or did an eval tag fail to parse? (b) `logs/loading.log` — how many units, how
many `no_timing`? (c) `data_quality.csv` — any `cb_sentinel=1`? (d) `run_config.csv` — does
`path_K` agree with `flow_steps` on the Gen11 rows?

---

## 10. Not done / deliberately out of scope

* **`MASTER_TEST_HISTORY.md` was not touched.** It needs a Gen15 U4 row; say the word and I will
  add it.
* **No `main_da` aggregation across scenes into a single "UAV score".** Scenes have different step
  budgets, different constraint families and — for `empty` — a different definition of success.
  A mean over them would be a number with no referent.
* **No cross-generation Pareto.** Gen11 and Gen15 candidates rank in the same table (that is the
  point), but nothing auto-declares a Gen15 win: §7.1's target is *the best Gen11 row at matched
  scene / geo / variant / K / seed set*, and matching those is a judgement the tool records the
  inputs for rather than makes.
* **No port of `test_page_offline.py`.** See §9.
