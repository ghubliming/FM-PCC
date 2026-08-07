# DA_VA_v2 U3 — three more Result Matrices (VA distance + the relaxed success pair)

**Date:** 2026-08-07 · **Epoch:** U3
**Changed:** `Data_Analysis/Visualizer_VA_v2/{build_from_dav3.py, index.html, test_page_offline.py}`
**Untouched:** the DA pipeline (`Data_Analysis/DA_VA_v2/`), `Data_Analysis/Visualizer/index.html`,
`Data_Analysis/Visualizer_Visual_Aligning/index.html`

Works on **existing** batches — no re-run needed, `batch_va2_20260806_204620` already has
everything the new tables read.

---

## 1. What was added

"Result Matrices — Selected Candidates × All Variants" went from 4 tables to 7:

| # | metric | caption | flags |
|---|---|---|---|
| 1 | `n_success` | N_SUCCESS — strict (position AND rotation) | — |
| 2 | `n_success_and_constraints` | N_SUCCESS + CONSTRAINT | — |
| **3** | **`success_relaxed`** | **N_SUCCESS RELAXED — position only, final angle ignored** | — |
| **4** | **`n_success_relaxed_and_constraints`** | **N_SUCCESS RELAXED + CONSTRAINT** | — |
| **5** | **`mean_dist_per_rollout`** | **MIN_DIST — final box-target distance [m], lower is better** | goal / constraint |
| 6 | `n_steps` | N_STEPS | goal / constraint |
| 7 | `avg_time` | AVG_TIME | goal / constraint |

Table 1's caption now says "strict, position AND rotation" so it reads against table 3
rather than next to it.

`SUMMARY_TABLES` drives the on-page tables **and** the LaTeX export, so EXPORT_ZIP's `.tex`
gained the same three tables with no separate edit. Rows, columns, the mean ± SEM cell
format, the never-run dash and the `(goal, constraint)` fail flags are all the inherited
DAv3 machinery, untouched.

### Why these three

* `success_relaxed` is the visual-aligning eval's second success definition — position only,
  `final_xy_dist <= pos_min_dist` with the box angle ignored
  (`mix_visual_aligning_test/eval_mix_visual_aligning.py`, Json_Orgnize_C4 Finding #5). It
  was already in every CSV and plottable, but had no place in a paper table.
* `mean_dist_per_rollout` is the env's own score, `0.5 * (position error + rotation error)`
  from `d3il/.../aligning.py:check_mode()` — the quantity `pos_min_dist` / `rot_min_dist`
  are thresholds on. It is the one VA-only outcome number, and it is what separates two
  variants that both score 0.000 success.
* the relaxed **+ constraint** product had no column anywhere. See §2.

State-only avoiding trees have none of these columns, so those rows render as the honest
NULL dash — the same treatment a candidate that never ran a variant already got.

---

## 2. `n_success_relaxed_and_constraints` is derived in the page, on purpose

The pipeline only derives the STRICT product
(`data_loader._finalise_frame`: `n_success * constraint_exec_zero_violation`). The relaxed
one **cannot** be recovered from the long CSVs afterwards:

```
mean(relaxed) * mean(zero_viol)  !=  mean(relaxed * zero_viol)
```

the moment a single rollout succeeds without being violation-free. `per_rollout_detail.csv`
is the only frame that still has both flags side by side, so `derive_frames()` rebuilds the
metric there — per rollout, then reduced (mean / std / n) with the **same** grouping the
pipeline uses (`Candidate, FolderName, split, geo, variant`, plus `seed` for the per-seed
frame). Real numbers from the current batch, CAND_2 / diffuser / both-hard:

```
success_relaxed                    0.100
success_relaxed + constraint       0.033      (product of means would say 0.093)
```

Consequences that fall out of doing it this way:

* the **global mask** applies — `_slice()` drops frozen rows before the groupby, so the
  unfrozen view of this metric is real, not an approximation;
* the **split** switch applies, and with split=ALL the derived rows duplicate per split
  exactly like the native ones, so the matrix averages them identically;
* it appears in the **metric dropdown** too, so it is plottable and exportable like any
  other metric;
* per-seed mode works — the same reduction feeds `df_raw`;
* if a future DA run ever emits the column natively, the synthesis is **skipped**
  (`if DERIVED_METRIC not in df_agg['metric']`), so nothing has to be un-done later.

`per_rollout_detail.csv` carries `FolderName` but not `FullPath`, so the path column of the
derived rows is filled from `df_agg`'s candidate→path map (one path per candidate anyway).

---

## 3. Where the edits live

`Visualizer_VA_v2/index.html` is **generated**. All of the above is in
`build_from_dav3.py` — edit that, then:

```bash
python Data_Analysis/Visualizer_VA_v2/build_from_dav3.py     # 25 edits (was 20)
```

Editing `index.html` directly would be overwritten by the next DAv3 re-derive.
New build steps: `12b` (the SUMMARY_TABLES block + the three caption/comment strings that
counted "four tables"), and the `derive_frames` graft in step `10`.

---

## 4. Validation

`test_page_offline.py`, both real batches, **all checks passing** (43 on out1, up from 41):

```
out1 (3 candidates, 115 units, 5 geometries, mixed VA + state-only)   ALL CHECKS PASSED
out5 (1 candidate,   38 units, tightened twin)                        ALL CHECKS PASSED
```

New checks:

* `U3 relaxed success + constraint derived per unit` — 76 derived rows against 115
  `n_success` rows; the 39 missing are the state-only candidate, which has no
  `success_relaxed` column. On a state-only batch the expectation flips to **zero** derived
  rows (a column of zeros there would be a lie).
* `U3 derived value matches the raw rollouts` — the page's number recomputed straight from
  `per_rollout_detail.csv`, must agree to 1e-9.
* `mask applied (not double-counted)` now counts the derived rows separately, so a genuine
  mask double-count is still caught.
* `matrices mark never-run cells` only fires when this environment actually has a hole —
  it was failing on the single-candidate batch that ran every variant, which was a test
  bug, not a page bug.

Rendered output spot-checked out of the stub DOM: 7 captions in order, CAND_1 (state-only)
all dashes in tables 3–5, MIN_DIST cells carrying their `(goal)` flags.

**Still not verified: an actual browser render** — same standing caveat as U2.

---

## 5. Not done (say the word)

The DA pipeline still does not write `n_success_relaxed_and_constraints` into the CSVs.
Adding it is ~5 lines in `DA_VA_v2/data_loader._finalise_frame` plus the config lists, but
it only reaches the numbers after a cluster re-run, whereas the page-side derivation works
on the batches already on disk. If you want CSV consumers (ranking table, Colab notebooks)
to see it too, that is the follow-up.
