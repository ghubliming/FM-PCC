# DA_VA_v2 U3 — three more Result Matrices, a run tally, and a start-distance reference

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

## 2b. Run coverage — the VA batches are not balanced, and now the page says so

Candidates in a VA batch come out of **different eval jobs**: different seed counts,
different variant lists, sometimes a different geometry set entirely. Two cells in the same
column can therefore rest on very different amounts of data — and the SEM gets *narrower*
where there is more data, which reads as "steadier method" if nothing on screen contradicts
it.

A **Run coverage** table now sits at the top of the Result Matrices section, above table 1:

| Candidate | Rollouts (total) | Rollouts in <env> | Variants here | Geometries | Seeds |
|---|---|---|---|---|---|
| CAND_1 | 78 | 0 | 0 | 3 | 1 |
| CAND_2 | 1140 | 570 | 19 | 2 | 1 |
| CAND_3 | 1140 | 570 | 19 | 2 | 1 |

followed by one line of verdict — either

> **UNBALANCED** — these candidates did not run the same amount (78…1140 rollouts). Cells
> are still mean ± SEM over whatever each one actually ran; the SEM is narrower where there
> is more data, not where the method is better.

or `All selected candidates ran the same N rollouts.` Counts come from `n_success`'s `count`
(the metric every unit has) and follow the global mask and split, so switching to *unfrozen*
lowers them. A candidate with **0** rollouts in the shown environment is flagged red — that
is the case above, where CAND_1 is a state-only avoiding tree that never ran `combined_5`.

## 2c. INIT XY — where the rollouts started

`MIN_DIST = 0.33` says nothing on its own. The MIN_DIST table therefore ends with a
separated, tinted reference row:

```
INIT XY (ref)   0.4547  0.4547  0.4547  0.4547  …
```

the mean box-target distance **at rollout start** (`context_init_xy_dist`), pooled over the
selected candidates and weighted by rollout count. So a final 0.33 against a start of 0.45
is a quarter of the way closed, not a good absolute number.

Two honest caveats, both stated in the caption on the page:

* it is **position-only** — there is no composite ½pos+½rot score at t=0, while MIN_DIST is
  the composite. It is a scale reference, not a strict upper bound.
* it is **flat across columns by construction** — every variant replays the same contexts.
  That flatness is itself the check that the columns are comparable; if it ever stops being
  flat, the variants were not run on the same scenes.

The row is emitted only where the data exists: on a state-only avoiding geometry
(`both-hard` etc.) there is no `context_init_xy_dist`, and the table simply ends without it
rather than inventing a zero. It is carried into the LaTeX export too, below a `\midrule`.

---

## 3. Where the edits live

`Visualizer_VA_v2/index.html` is **generated**. All of the above is in
`build_from_dav3.py` — edit that, then:

```bash
python Data_Analysis/Visualizer_VA_v2/build_from_dav3.py     # 33 edits (was 20)
```

Editing `index.html` directly would be overwritten by the next DAv3 re-derive.
New build steps: `12b` (the SUMMARY_TABLES block + the caption/comment strings that counted
"four tables") and `12c` (the INIT XY row and the run tally, threaded through
`_summary_stats` / `_render_matrix` / `_tex_table` / `render_summary_tables` plus two CSS
rules), with the `derive_frames` graft living in step `10`.

---

## 4. Validation

`test_page_offline.py`, both real batches, **all checks passing** (47 on out1, up from 41):

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
* `U3 run-coverage table rendered` — one tallied row per selected candidate, no more, no
  fewer; `U3 coverage flags an unbalanced batch` — the UNBALANCED warning appears if and
  only if the candidates' rollout totals actually differ (out1 unbalanced, out5 balanced,
  both correct).
* `U3 INIT XY reference row on the MIN_DIST table` and `LaTeX carries the INIT XY reference
  row` — present exactly where this environment has a non-NaN `context_init_xy_dist`,
  absent on the state-only geometry. The check matches the row's CSS class, not its label:
  the label also occurs in the caption that explains the row, so a label match would pass
  with no row rendered.

Rendered output spot-checked out of the stub DOM: 7 captions in order, CAND_1 (state-only)
all dashes in tables 3–5, MIN_DIST cells carrying their `(goal)` flags, the coverage table
reporting 78 / 1140 / 1140 rollouts with the UNBALANCED verdict, and the INIT XY row flat
at 0.4547 across all 19 variants — which is exactly the shared-context property it is
supposed to demonstrate.

**Still not verified: an actual browser render** — same standing caveat as U2.

---

## 5. Not done (say the word)

The DA pipeline still does not write `n_success_relaxed_and_constraints` into the CSVs.
Adding it is ~5 lines in `DA_VA_v2/data_loader._finalise_frame` plus the config lists, but
it only reaches the numbers after a cluster re-run, whereas the page-side derivation works
on the batches already on disk. If you want CSV consumers (ranking table, Colab notebooks)
to see it too, that is the follow-up.
