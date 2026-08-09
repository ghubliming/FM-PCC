# U13 — Candidate highlight + seed coverage in the Plot Legend

**Date:** 2026-08-09
**Scope:** both HTML viewers — `Data_Analysis/Visualizer/index.html` (DAv3) and
`Data_Analysis/Visualizer_VA_v2/index.html` (regenerated from it)
**Status:** written + tested here (stdlib harness, 45/45); **the full page test
still to be run on the cluster** (needs pandas + matplotlib)
**Follows:** `../Timestamp_in_CSV_HTML/CHANGELOG_20260809_timestamp_in_csv_html.md`

---

## Why

Two separate blind spots in the same table.

**1. You cannot follow one candidate across the page.** The Result Matrices are
~18 variant columns wide and the plot puts 40 bar groups on one axis. Having
picked CAND_65 out of the Plot Legend, finding *that* row again in four
scrolling matrices — and *that* bar group on the chart — is manual, and it is
done over and over while reading a batch.

**2. A one-seed bar and a five-seed bar look identical.** The chart draws a mean;
nothing next to it says how much data is under it. The information existed only
in the `Warnings` column of the full Path Audit Map further down the page (and
only as *missing* seeds, never as present ones), so the legend right under the
plot — the table you are actually reading while looking at the bars — said
nothing about coverage at all.

## What was added

### 1. `HL` — a highlight checkbox, first column of the Plot Legend

Ticking it paints that candidate's **name** red + bold everywhere the page
prints it:

| where | what turns red |
|---|---|
| the plot | its **x tick label** (`65`), red + bold — candidate mode only |
| Plot Legend | the `ID` cell |
| **Result Matrices — Selected Candidates × All Variants** | the row head of **every** table (and, on the VA page, of the U3 run-coverage table) |
| Path Audit Map | the `ID` cell |

Nothing else changes: no number, no cell background, no row tint. That is
deliberate — the matrices already use red for the `(goal, constraint)` failure
flags, so a highlight that touched anything but the name would be readable as a
data annotation. Here it can only ever be "the row I am tracking".

`[CLEAR N HIGHLIGHT(S)]` appears in the legend header whenever anything is
highlighted. It is the only way to reach a candidate you have since **unticked**
in `6. Candidates` — that candidate has no legend row any more, so no checkbox,
but it is still red in the Path Audit Map.

The exported `.txt` and `.tex` have no colour, so they say it in words:
`- CAND_65 [HIGHLIGHTED] (Source: …)` in the audit log, and
`CAND_65  /path/…   <-- highlighted in the viewer` in the LaTeX source-path
section. The LaTeX **tables** are untouched — a screen affordance has no business
bolding a row in a paper table.

### 2. `Seeds` — coverage, with an explicit caution

New column between `ID` and `Last Run`:

```
Seeds
6, 7, 8, 9, 10
6, 7
⚠ NOT FULL — missing 8, 9, 10
```

* present seeds come from the per-seed CSV (`candidates_multidimensional_raw.csv`
  / `va2_units_long.csv`), which is the only frame with a seed axis;
* "missing" is measured against **the batch's full seed set** in Standard mode,
  and against **the seeds you ticked** in Custom Seed Compare — i.e. the same
  seeds `trigger_plot` already warns about under the sidebar, now visible in the
  table instead of only in a status line;
* with no per-seed CSV the page falls back to the discovery pipeline's own
  `Missing_Seeds` column and prints `n/a — no per-seed CSV` for the present
  list. It does **not** print `none`: that fallback knows only what is absent,
  and claiming the candidate has zero seeds would be a lie about data that is
  plainly on the plot.
* with neither source the column is omitted entirely, the same way `Last Run`
  is omitted for pre-timestamp batches.

Coverage is candidate-global (not per environment/metric), matching what
`Missing_Seeds` itself means in the Path Audit Map.

## How it is wired

The highlighted set lives in **Python** (`highlighted_cands`), not in the DOM.
All three tables are re-rendered wholesale on every redraw, so a checkbox's
`checked` attribute would not survive a single replot; the box is re-emitted
`checked` from the set instead. JS only forwards the click
(`toggle_highlight(el)` → `document.set_highlight`).

Toggling does **not** replot. `_redraw_highlight()` recolours the tick labels of
the *existing* figure (`_apply_tick_highlight`) and re-displays it, then
re-renders the three tables. No groupby on the plot path, no matplotlib redraw.

`_apply_tick_highlight` pairs tick *i* with x-category *i*, and if matplotlib
ever hands back a different number of labels it marks **nothing** rather than
paint the wrong bar group — the same defensive stance U12 takes when pairing bar
rects to `pivot_mean` cells. In environment mode there is no candidate on the
x-axis, so no tick is touched.

`highlighted_cands` is deliberately **not** cleared by `SYNC_SOURCE`: re-loading
the same batch after a re-run should keep what you were tracking. Loading a
*different* batch keeps them too, which is why `[CLEAR HIGHLIGHTS]` exists.

## Files touched

```
Data_Analysis/Visualizer/index.html                  U13 CSS, highlight state + helpers,
                                                     _seed_map/_seed_cell, legend HL + Seeds
                                                     columns, matrices + audit map use
                                                     _cand_name_html, export markers, JS wiring
Data_Analysis/Visualizer_VA_v2/build_from_dav3.py    U3 coverage table row head -> _cand_name_html
Data_Analysis/Visualizer_VA_v2/index.html            REGENERATED (33 edits, 2155 lines)
Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py   NEW — stdlib regression test
Data_Analysis/Visualizer_VA_v2/test_page_offline.py  U13 checks + a rowhead regex made
                                                     highlight-proof
Data_Analysis/DA_Code_v3/README.md                   new "Plot Legend" section
Data_Analysis/DA_VA_v2/README.md                     new "Plot Legend" subsection
```

No pipeline, CSV or config change — this release is viewer-only.

## Testing

**Ran here (container, stdlib only — pandas/matplotlib stubbed out):**

```bash
python3 Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py    # 45/45 PASS
```

Covers, on **both** pages: the name renderer (plain vs `.hl-name`, int and str
candidate ids agreeing), the tick recolouring (highlighted → red+bold, others
explicitly reset to black, environment mode marks nothing, a tick/category count
mismatch marks nothing), and the Seeds cell (complete, short-with-caution,
`n/a` fallback, unknown candidate → dash).

It also asserts the two pages have **byte-identical** implementations of the nine
highlight/seed functions and carry the same markup hooks. `Visualizer_VA_v2` is
generated from the DAv3 page, so a hand-edit to either would otherwise surface
only as the two viewers disagreeing about which candidate is red.

Both `index.html` Python blocks and both edited Python modules byte-compile clean.

**Still to run on the cluster:**

```bash
python Data_Analysis/Visualizer_VA_v2/test_page_offline.py <batch_va2_dir>
```

Its new `[U13 …]` sections check the seed map against `df_raw` itself, that the
`⚠ NOT FULL` caution appears **exactly** when a candidate is short (never as
decoration), that one HL checkbox marks every matrix row head and leaves the
other candidates plain, that the checkbox comes back checked after the
re-render, that the x tick really is `HL_COLOR` + bold, and that
`clear_highlights` wipes all of it.

Then, by eye in a browser: tick `HL` on a candidate, confirm the bar group's x
label and every matrix row head go red together, and that `EXPORT ZIP`'s `.txt`
carries `[HIGHLIGHTED]`.

## Known limitations

* Highlight is a **viewing** state — it is not in any CSV and does not survive a
  page reload.
* Environment mode (`2. Analysis Mode → By Environment`) highlights the tables
  but not the plot: the x-axis is the environment there, candidates are averaged
  into each bar group, so there is no candidate label to paint.
* The `Seeds` column reports whether the *candidate* has a seed anywhere in the
  batch, not whether it has it for the currently plotted metric/environment. A
  bar can still be thinner than the column suggests; the matrices' per-cell
  `n=` hover and (on the VA page) the run-coverage table are the finer answer.
