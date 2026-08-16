# DA_VA_v2 Fix_1 — run invisible in the HTML picker + early-Gen7 JSON schema unread

**Date:** 2026-08-06 · **Area:** `Data_Analysis/DA_VA_v2/`, `Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh`
**Trigger:** first cluster run, `temp/0608/18_19_25_run_da_batch_va_v2_24330.log` — 198 units, 0 failures, 12 CSVs written, **but the batch did not appear in the HTML visualizer's quick list**.

Two defects. One is the reported symptom; the second was hiding in that same log
as three `RuntimeWarning` lines and is the more damaging of the two.

---

## Defect 1 — output folder name misses the prefix both visualizers filter on

**Symptom.** The run completed and wrote every CSV to
`Data_Analysis/analysis_results/va2_batch_20260806_161926/`, but the run dropdown
in the HTML page stayed empty of it.

**Cause.** Both pages build that dropdown by regexing the `analysis_results/`
*directory listing* for a leading prefix — and they disagree about which:

| page | pattern | line |
|---|---|---|
| `Visualizer/index.html` | `href="(batch_[^/"]+)/?"` | :159 |
| `Visualizer_Visual_Aligning/index.html` | `href="(va_batch_[^/"]+)/?"` | :267 |

Neither matches `va2_batch_…` — the pattern is anchored to `href="`, so a
`batch_` occurring *later* in the name does not count. `results_manifest.json`
(which the run does write, and which does contain the folder) is only a
**fallback**, reached when the directory-listing fetch fails; under
`python -m http.server` the listing succeeds, so the manifest is never consulted.
Net effect: all the data present, none of it selectable.

**Fix.** One name cannot satisfy both patterns, so:

* the run directory is now `batch_va2_<timestamp>` → listed by `Visualizer/index.html`;
* `utils.create_viewer_alias()` drops a **symlink** `va_batch_va2_<timestamp>`
  beside it → listed by `Visualizer_Visual_Aligning/index.html`. Same files, one
  copy, relative link target so it survives a repo move. Non-fatal on a
  filesystem without symlinks.
* `create_output_directory()` now builds `<prefix>_<timestamp>`, not
  `<timestamp>_<prefix>` — a timestamp-first name can never match a prefix regex.
* An `--output-path` whose basename does not start with `batch_` still runs but
  logs a warning naming the consequence.
* The reasoning is recorded at `config.py:OUTPUT_FOLDER_PREFIX` and in the sbatch
  script, so the next person to "tidy up" the folder name sees why it is shaped
  this way.

**Verified** by reproducing `http.server`'s own `list_directory()` HTML over a
real output folder and running both viewers' regexes against it:

```
DAv3 viewer  batch_*    -> ['batch_va2_20260806_161926']
VA viewer    va_batch_* -> ['va_batch_va2_20260806_161926']
```

## Defect 2 — early-Gen7 rollout JSONs were read as all-NaN

**Symptom in the log.** Three occurrences of

```
data_loader.py:231: RuntimeWarning: Mean of empty slice
  scalars = {'success_rate': float(np.nanmean(frame['n_success']))
```

for the `fm_visual_aligning` candidates. The units still counted as `loaded`.

**Cause.** The JSON reader was written against the **nested** rollout-stats
schema that Gen14 (and late Gen7) writes:

```json
{"success": {"strict": …}, "outcome": {"mean_distance": …},
 "timing": {"steps": …}, "context": {…}, "constraint": {"exec": {…}}}
```

Early Gen7 runs use a **flat** schema — the one the old `DA_Visual_Aligning`
loader was built for:

```json
{"success": true, "mean_distance": …, "steps": …,
 "context_info": {…}, "constraint_metrics": {"exec_n_violated_steps": …}}
```

Every lookup missed, so those units produced a full frame of NaN and landed in
the CSVs as silently empty rows — worse than failing, because nothing said so.
This defeated a stated requirement of the tool: *load old DA_VA Gen7 runs*.

**Fix.** Field lookup is now dual-schema: `_pick(row, (nested_path, flat_path))`
returns the first path that resolves, so a tree mixing both loads with no flag.
Covered: `success` as bool *or* `{strict, relaxed}`, `mean_distance` /
`max_physical_tracking_error` / `steps` / `avg_inference_time_per_replan` at top
level or under `outcome`/`timing`, `context` ↔ `context_info`, and
`constraint.exec.by_family.bounds.viol_count` ↔
`constraint_metrics.exec_bounds_viol_count` for all 17 constraint fields.

Three follow-ons in the same pass:

* the **npz-source diagnostics side-scan** used `context.box_obstacle_conflict`
  only — it now checks `context_info.` too, so the D1 frozen-rollout mask and the
  JSON-only final-pose columns also work on old Gen7 npz units;
* `box_obstacle_conflict` recorded as a **bare `true`** (rather than a detail
  dict) now marks the rollout frozen — an earlier draft returned `{}` there,
  which is falsy and would have read as *not frozen*;
* `_safe_nanmean()` replaces the bare `np.nanmean`, and an all-NaN frame now
  emits an explicit `WARN … unrecognised schema, metrics will be NaN` into
  `logs/loading.log` instead of a bare numpy warning on stderr.

---

## Impact on the 16:19 cluster run

* Candidates 5–9 (Gen14 af/mf, `visual_aligning_dpcc`) — npz-sourced, **unaffected**.
* Candidates 1–4 (`fm_visual_aligning`, old Gen7) — npz-sourced units are fine;
  the **3 JSON-only units** in that tree are all-NaN in that run's CSVs and are
  fixed only by re-running. Frozen-rollout masking for those four candidates was
  also inert if their JSONs use the flat schema.

**→ re-run on the cluster** (`sbatch Slurm_Codes/sbatch/DA/run_da_batch_va_v2.sh
logs/aligning-d3il-visual/plans`) to get both the correct folder name and the
Gen7 rows.

## Re-validation

| check | result |
|---|---|
| synthetic early-Gen7 flat-schema fixture (bare-bool `success`, `context_info`, `constraint_metrics`, bare-`true` conflict) | all 4 rollouts load; derived metrics correct; the conflict rollout marked `frozen=1` |
| Gen14 U7 regression (`temp/0508`, 115 units) | unchanged — 2358 rollout rows, 152 frozen, identical aggregates |
| npz vs `--source json` on the same Gen14 unit | still agree to 2.9e-8 |
| both viewers' regexes over a real `http.server` listing | each lists the run once |

Ran in the scratchpad venv (numpy 2.5.1 / pandas 3.0.5); nothing installed into
the container python or the FMPCC env. Cluster re-run still pending.
