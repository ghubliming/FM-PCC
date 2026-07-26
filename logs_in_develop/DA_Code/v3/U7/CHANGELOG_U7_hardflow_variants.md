# CHANGELOG — DA_Code v3 U7: teach the DA the HardFlow (Gen12) variants

**Date:** 2026-07-26 · **Type:** fix (DA code) · **Status:** code complete, verified static
**Scope:** `Data_Analysis/DA_Code_v3/config.py` only. **Nothing committed.**
**Sits alongside** DA_Code/v3 fix_1..fix_6 (user asked for "U7").

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

## 4. Caveats
- **Seed mismatch:** hardflow candidates have only seed 6; DPCC candidates have all 5. For a fair
  head-to-head, compare seed 6 vs seed 6 (or note the n imbalance) — the DA averages over whatever
  seeds each candidate has.
- **HARDFLOW_METRICS display:** these are loaded into each hardflow row's data dict and defined in
  config, but the batch reporter's summary table still centres on the core `METRICS`. Surfacing
  `nlp_solves`/`activation_threshold` in the printed table is a small follow-up if you want them
  in the report rather than read from the npz. The success/safety/time comparison — the headline —
  works now.
