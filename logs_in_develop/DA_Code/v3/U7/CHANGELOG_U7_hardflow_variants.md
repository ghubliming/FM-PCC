# CHANGELOG — DA_Code v3 U7: teach the DA the HardFlow (Gen12) variants

**Date:** 2026-07-26 · **Type:** fix (DA code) · **Status:** code complete, verified static
**Scope:** `config.py` + `batch_visualizer.py`. **Nothing committed.**
**Sits alongside** DA_Code/v3 fix_1..fix_6 (user asked for "U7").

> **Update (same day):** the first pass changed `config.py` only; that made hardflow load and show
> in the **single-candidate** visualizer (which is variant-generic) and the batch candidate-level /
> MAJOR-AUX plots. But the batch **robustness boxplot + constraint heatmap were hardcoded to DPCC
> variant lists**, so hardflow was invisible there. §5 adds the HardFlow subgroup to both — so
> `hardflow_new` is now a first-class variant everywhere, like `dpcc-c`.

---

## 0. The problem

After the Gen12 fix_5 path migration, the batch DA discovered the hardflow candidates
(e.g. `CAND_22 … /flow_matching_v3_hardflow/…/K10_thres0_mpc1_n2`) but **loaded 0 result
files from them / couldn't compare** them against DPCC candidates
(e.g. `CAND_78 … /flow_matching_v3_ode_selectable/…GaussianDiffusion…`).

**Root cause:** the DA's `DEFAULT_PROJECTION_VARIANTS` lists only DPCC/diffuser variants —
`hardflow_new` (and `-c/-r/-t`) were **not known variants**, so `data_loader.load_results`
never looked for `hardflow_new.npz`. The npz schema itself is fine (same metric keys as DPCC;
the loader reads scalars generically). It simply wasn't being asked to load them.

(The `MISSING [7,8,9,10]` warning is separate and expected — only seed 6 exists for the hardflow
runs; the candidate still loads its seed-6 data.)

## 1. The fix (config.py)

| change | detail |
|---|---|
| **new `HARDFLOW_VARIANTS`** | `['hardflow_new', 'hardflow_new-c', 'hardflow_new-r', 'hardflow_new-t']` |
| **`DEFAULT_PROJECTION_VARIANTS += HARDFLOW_VARIANTS`** | so the loader/discovery look for `hardflow_new*.npz` |
| **`MAJOR_VARIANTS += ['hardflow_new', 'hardflow_new-c']`** | headline arm-C lines, so `hardflow_new` sits next to `dpcc-c-tightened` in the per-variant comparison table |
| **new `HARDFLOW_METRICS`** | `nlp_solves, nlp_failures, nfe, activation_threshold, batch_size, flow_steps` — kept OUT of core `METRICS` (DPCC tables unchanged); loaded generically, available per hardflow row |
| **`METRIC_LABELS` / `METRIC_TYPES`** | labels/types for the 6 HardFlow metrics (e.g. `activation_threshold → "Activation Threshold (DPCC polarity)"`) |

**Nothing else needed:**
- `data_loader._load_result_file` already reads every scalar npz key generically (0-d arrays →
  float; strings like `variant`/`trajectory_selection` kept as-is; object arrays `obs_all`/`args`
  handled exactly as for DPCC npz). No loader change.
- `batch_aggregator` / `batch_visualizer` are driven by `MAJOR_VARIANTS` / `AUXILIARY_VARIANTS`
  and **skip any major a candidate lacks** (`isin` filter; `if not v_data.empty`), so DPCC-only
  candidates are unaffected and hardflow-only candidates don't error. `AUXILIARY_VARIANTS` is
  recomputed from the extended list, so `-r/-t` land in AUX automatically.
- `std_variants`/`tight_variants` (the DPCC group stats) are hardcoded DPCC subsets — hardflow
  cannot leak into them.

## 2. Verification (static)
- `config.py` imports; hardflow variants present in DEFAULT (4) / MAJOR (2) / AUX (2).
- core `METRICS` unchanged (still 7) — existing DPCC analysis is byte-identical.
- `config/data_loader/batch_aggregator/batch_visualizer/batch_data_loader` all `py_compile`.
- Loader key-handling reviewed against the actual hardflow npz keys (scalars, strings, object
  arrays) — no crash path.
- (numpy isn't installed in this container, so the full load was not executed here — it runs on
  the cluster / DAcodev3 env.)

## 3. How to use — hardflow@0.5 vs DPCC

Re-run the batch DA (no arg change needed; it already scans `logs/avoiding-d3il/plans`):
```
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/DA/run_da_batch_avoiding_combined.sh
```
`hardflow_new` now loads and appears as a MAJOR line. Compare:
- **arm C:** `hardflow_new` in `…/flow_matching_v3_hardflow/H8_D…aw10/K20_thres0.5_mpc1_n2/`
- **DPCC:**  `dpcc-c-tightened` in the same eval folder (both arms are written together), or the
  ode_selectable candidate `CAND_78`.

**Is it the "same NLP"?** No — at threshold 0.5 both project the *last half* (same schedule region,
DPCC polarity after Gen12 fix_6), but the NLPs are **different formulations**: DPCC = post-hoc SLSQP
projection of the sampled trajectory; hardflow = per-step prox-NLP on the *predicted terminal*. So
they are *comparable* (same success/safety/time axes, now both loaded) but **not identical solvers**.
The `activation_threshold` / `nlp_solves` metrics (now loaded) make that explicit per row.

## 5. Visualizer fix (`batch_visualizer.py`) — hardflow was hardcoded out

The batch per-variant plots each emitted exactly two subgroups, both hardcoded to DPCC:
`plot_candidate_robustness_boxplot` and `plot_candidate_constraint_heatmap` used
`['dpcc-r','dpcc-c','dpcc-t']` and the tightened trio. The config change (MAJOR_VARIANTS) does **not**
reach these literals, so hardflow never appeared in those plots — this is what "can't set
hardflow_new as a variant like dpcc-c" was hitting.

Fix: added a **third HardFlow subgroup** to each, driven by `HARDFLOW_VARIANTS`:
- `03c_candidate_robustness_hardflow.png`
- `04c_candidate_heatmap_hardflow.png`

`_generate_robustness_subgroup` / `_generate_heatmap_subgroup` already filter by
`detailed['variant'].isin(variants)` and skip empties, so DPCC-only candidates simply don't appear in
the hardflow subgroup (and vice-versa). `batch_visualizer.py` compiles.

**Where hardflow_new shows up now, as a first-class variant:**
| view | before U7 | after U7 |
|---|---|---|
| single-candidate `plot_variant_comparison` (variant bars) | not loaded | **shown** (generic; ranks all loaded variants) |
| batch candidate-level accuracy/time | not counted | **counted** (MAJOR incl. hardflow_new) |
| batch MAJOR/AUX group plots | absent | **present** |
| batch robustness boxplot / constraint heatmap | **hardcoded DPCC only** | **new 03c/04c HardFlow subgroup** |

## 6. Direct "hardflow_new vs dpcc-c" within one folder
Because the Gen12 eval writes all arms into the SAME results dir, the hardflow eval folder
(`…/K20_thres0.5_mpc1_n2/6/results/halfspace_*/`) contains **both** `dpcc-c-tightened.npz` and
`hardflow_new.npz`. Run the **single-candidate** DA on that folder to get a direct within-candidate
variant bar chart (diffuser vs dpcc-c-tightened vs hardflow_new) — the cleanest B-vs-C picture:
```
python Data_Analysis/DA_Code_v3/main_da.py \
    --root-path logs/avoiding-d3il/plans/flow_matching_v3_hardflow/H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/K20_thres0.5_mpc1_n2 \
    --seeds 6
```
(adjust flags to your main_da.py signature).

## 7. Caveats
- **Seed mismatch:** hardflow candidates have only seed 6; DPCC candidates have all 5. For a fair
  head-to-head, compare seed 6 vs seed 6 — the DA averages over whatever seeds each candidate has.
- **HARDFLOW_METRICS display:** `nlp_solves`/`activation_threshold`/… are loaded into each hardflow
  row and defined in config, but the batch *reporter's* summary table still centres on core `METRICS`.
  Surfacing them in the printed table is a small follow-up; success/safety/time (the headline) works.

---

## 8. Second visualizer bug — candidate-level plots dropped the hardflow candidate

**Symptom (user):** comparing the hardflow candidate vs the fmv3ode/DPCC candidate, only the
fmv3ode one appears; the hardflow candidate is missing from the plot entirely.

**Root cause:** the headline candidate-level plots — Pareto `00a/b`, success `01a/b`, time `02a/b` —
key on `accuracy_std_group` / `accuracy_tight_group` / `time_ms_*_group`, which the aggregator
computes **only from the DPCC variant lists** (`std_variants=['dpcc-r','dpcc-c','dpcc-t']`,
`tight_variants=[…tightened]`, `batch_aggregator.py` L102-126). A hardflow candidate has **no DPCC
variants**, so those keys are `None`, and both `_generate_pareto_subgroup` and
`_generate_bar_comparison` **skip** any candidate whose value is `None` → the hardflow candidate
vanishes. U7 §1's `MAJOR_VARIANTS` change did not reach these DPCC-only group keys.

**Fix (`batch_visualizer.py`): `plot_candidate_combined_comparison`** (wired into `plot_all`), which
includes **every** candidate using each candidate's *own* unified headline metric
(`stats['accuracy']` / `stats['time_ms']` = mean over whatever MAJOR variants it has — DPCC majors
for a DPCC candidate, `hardflow_new*` for a hardflow candidate; both are set now that U7 put hardflow
in `MAJOR_VARIANTS`). New outputs:
- `00c_candidate_pareto_combined.png` — success vs time, all candidates with both metrics
- `01c_candidate_success_combined.png` — success bar, **all** candidates
- `02c_candidate_time_combined.png` — time bar, **all** candidates

Per the user's request, **nothing is dropped**: a candidate missing the metric is drawn as a **blank
(0) bar labelled `n/a`** with its letter kept on the axis (rather than omitted). The scatter skips a
point only if *both* metrics are absent (can't place it), but the bars always keep the slot.

The DPCC-subgroup plots (`00/01/02 a/b`) are left as-is (they are DPCC-specific by design); the `c`
combined plots are the ones to read for a **HardFlow-vs-DPCC** candidate comparison.

**Verify:** `batch_visualizer.py` compiles; `plot_candidate_combined_comparison` is in `plot_all`;
uses `.get()` so a zero-file candidate can't KeyError.

---

## 9. THE ACTUAL BUG — HTML seed-verify aborted the whole plot (not the data, not the variants)

Diagnosed from a real cluster CSV (`temp/DAv3U7/batch_.../candidates_multidimensional_aggregated.csv`).
**The data was correct all along** — CAND_23 (`…/aw10/K20_thres0.5_mpc1_n2`) has all three variants
(`diffuser`, `dpcc-c-tightened`, `hardflow_new`) across all three halfspaces with correct values
(e.g. hardflow_new both-hard: `n_success_and_constraints=1.0`, `avg_time=0.497`). So U7 §1 (loading)
and the Python DA worked. The symptom "hardflow candidate won't show" was **entirely in the HTML
visualizer** (`Data_Analysis/Visualizer/index.html`), which §1/§5/§8 never touched.

**Root cause:** in **custom seed mode**, `trigger_plot` verified that *every* selected candidate has
*every* selected seed, and on any miss it did:
```python
seed-error-msg = "Error: CAND_{cand} missing seed(s) ...!"
plot-area.innerHTML = ""; plt.close('all'); return    # aborts the ENTIRE plot
```
CAND_23 is a seed-6-only run (`Missing_Seeds=[7,8,9,10]`). The moment it was included with seeds
7–10 selected, the whole plot was wiped — so the DPCC (all-seed) candidate rendered alone and adding
hardflow made everything disappear. Exactly the "leave it blank instead" ask.

**Fix (`index.html`):** replaced the hard abort with a non-fatal **warning** — plot whatever seeds
exist per candidate, blank the rest; only abort if the *entire* subset is empty:
```python
warn = [f"CAND_{c}: {sorted(missing)}" for c in checked_cands if <missing seeds>]
if warn: seed-error-msg = "Note: plotting available seeds only — missing " + "; ".join(warn)
if subset.empty: <clear + return>       # only when truly nothing
```
Now CAND_23 plots its seed-6 data alongside the multi-seed DPCC candidate, with a note that seeds
7–10 are absent. Verified the edited block compiles as Python (pyscript runs it in-browser).

**No cluster re-run needed** — the CSV is already correct. Just load the updated `index.html` and
re-SYNC. (Aggregated seed mode never had this bug; it has no per-seed verify.)

### Corrected root-cause ordering
1. §1 config — hardflow variants loadable (**needed**, real).
2. §5/§8 batch PNG plots — hardflow in robustness/heatmap/combined (**valid, but the user's pipeline
   runs `--no-plots`; the HTML is CSV-driven, so these were not the blocker**).
3. **§9 HTML seed-verify abort — the actual cause of "hardflow won't show".**

---

## 10. Follow-up — "select only the hardflow candidate → stuck on CHOOSE_EXPERIMENT_TO_START"

Second HTML regression, also introduced by U7, distinct from §9.

**Cause:** §1 added the HardFlow-only metrics (`activation_threshold`, `batch_size`, `flow_steps`,
`nlp_solves`, `nlp_failures`, `nfe`) to the CSV. In `index.html`, `populate_dynamic_filters` builds
the metric dropdown from `sorted(df_agg['metric'].unique())` and lets the browser select the **first
`<option>`**. `activation_threshold` / `batch_size` / `flow_steps` all sort **before** the core
metrics, so the default metric became a **hardflow-only** metric. Those keys exist only in
`hardflow_new*.npz` — `diffuser.npz` / `dpcc*.npz` don't have them. With the default variant still
`diffuser` (first checkbox), the `(variant=diffuser, metric=activation_threshold)` mask was empty, and
`trigger_plot`'s `if subset.empty: return` bailed **without touching `plot-area`**, so the initial
`CHOOSE_EXPERIMENT_TO_START` placeholder just stayed on screen.

**Fixes (`index.html`):**
1. **Sensible default metric** — `populate_dynamic_filters` now marks the first available of
   `["n_success_and_constraints", "n_success", "avg_time"]` as `selected` (falls back to `metrics[0]`
   only if none present). So the initial/typical selection lands on a metric every variant has.
2. **No more silent placeholder** — added `show_no_data(metric, vars, cands, env)`; both empty-subset
   branches (aggregated and custom-seed) now render a red diagnostic box naming the exact
   metric/variants/candidates/env that produced no rows, plus a tip that hardflow-only metrics only
   exist for `hardflow_new*` variants. The stuck "CHOOSE_EXPERIMENT_TO_START" can no longer occur.

Combined with §9 (seed-verify no longer aborts the whole plot), the hardflow candidate now renders on
core metrics, and any genuinely empty facet says *why* instead of showing the boot placeholder.
**Still no cluster re-run needed** — reload `index.html` and re-SYNC.
