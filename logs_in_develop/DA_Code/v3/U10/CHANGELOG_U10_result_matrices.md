# DA_Code v3 — U10: empty default selection + paper-style Result Matrices + LaTeX export

**Scope:** `Data_Analysis/Visualizer/index.html` only. No Python DA change, no cluster re-run.
**Depends on:** U9 (plot legend), U8 (bar value labels), U7 (hardflow variants + render fixes).
Toolbar tag bumped `SCIENTIFIC_SUITE_v3.7` → `v3.11`.

> **U10.1 revision** (same session, folded in below): the matrices are restricted to the
> **checked candidates** instead of the whole batch, and the download button exports the
> tables as LaTeX. Points marked *(U10.1)* changed relative to the first U10 pass.

## 1. Nothing is pre-checked any more

`populate_dynamic_filters()` used to auto-check the **first** variant and the **first**
candidate (`" checked" if i == 0`) so an initial lightweight plot would draw. Both are now
rendered unchecked, for both lists.

Consequence: on SYNC the plot area is empty. `trigger_plot` no longer silently `return`s in
that state — new **`show_empty_selection()`** paints an explicit panel ("select ≥1 variant
and ≥1 candidate", `[ALL]` hint, note that nothing is pre-checked by design), idles the
scorecard, and hides the U9 plot-legend box. `current_fig` is set to `None` so
`DOWNLOAD_PNG` stays a no-op instead of exporting a stale figure.

Seed checkboxes are **unchanged** (still all checked) — they are a sub-filter, not a
selection.

## 2. New section: Result Matrices (selected candidates × all variants)

New block `#summary-section` / `#summary-container`, placed exactly between the U9
**Plot Legend** box and the **Path Audit Map**.

**Selection rules** *(U10.1)* — asymmetric on purpose:

- **Rows = the checked candidates only** (`6. Candidates`, numeric-aware sort). The first
  U10 pass used the whole batch; with 70+ candidates that is not a paper table.
- **Columns = every variant**, always. The variant checkboxes are ignored — a paper table
  wants all methods side by side, and this is also what keeps ticking through variants for
  the plot cheap (no matrix rebuild).
- A checked candidate with no run in the selected environment is kept as an **all-`—` row**
  rather than silently dropped: it was explicitly asked for, so its absence is the result.
- No candidates checked → the section shows "No candidates selected — tick some in the
  sidebar" instead of tables.

Four tables:

| # | Metric | Failure markers |
|---|--------|-----------------|
| 1 | `n_success` — goal reached | no |
| 2 | `n_success_and_constraints` — goal reached **and** constraints satisfied | no |
| 3 | `n_steps` — episode length | **yes** |
| 4 | `avg_time` — computation time per planning step [s] | **yes** |

### Seed handling (mirrors the plot's own branch)

- **Standard seed mode** → aggregated CSV: `mean`, `std`, `count` are already across seeds.
- **Custom Seed Compare** (only when `candidates_multidimensional_raw.csv` loaded) → raw CSV
  grouped over the checked seeds; zero seeds checked shows a prompt instead of empty tables.
- A cell built from **one** seed prints the plain number; from **>1** seed it prints
  `mean ± SEM` with `SEM = std / √n`. Each cell carries a `title="n=… seed(s)"` tooltip.

### NULL handling

A candidate/variant combination that was never run renders as a grey `—` (`--` in LaTeX).
Columns come from the env-filtered frame; rows come from the checkbox selection, so a
candidate with no data in the current environment yields a full `—` row (e.g. `CAND_1` has
no `both-hard` rows at all in `batch_avoiding_combined_20260726_164945`).

### Failure markers (tables 3 & 4)

A cell "wins" only when `n_success == 1` **and** `n_success_and_constraints == n_success`.
Otherwise the number is suffixed with a red `(…)` listing the reason(s):

- `goal` — `n_success < 1` (goal not always reached)
- `constraint` — `n_success_and_constraints < n_success` (a constraint was violated)
- `(goal, constraint)` — both.

Tolerance `1e-9`. If `n_success` is missing but `n_success_and_constraints` exists, the
latter is compared against `1.0`. Rationale: a fast `avg_time` / short `n_steps` is
meaningless if the run didn't actually succeed, so the timing tables carry the verdict of
the success table inline.

Example (`both-hard`, CAND_20): `diffuser` → `n_success 1.000`, `n_success+constraint 0.500`
→ its `n_steps 91.0 (constraint)` and `avg_time 0.0498 (constraint)`.

### Implementation

New module-level `SUMMARY_TABLES` spec + helpers next to `render_path_map`:
`_seed_context()`, `_summary_stats(env, seed_mode, seeds)` (returns a
`{(metric, cand, variant): {mean,std,count}}` dict via `stats.to_dict('index')`, plus the
available candidate/variant lists), `_cell_stats()`, `_fmt_cell()`, `_fail_flags()`,
`_render_matrix()`, and the entry point `render_summary_tables(force=False)`.

*(U10.1)* **`_summary_context()`** is the single source of truth shared by the on-page
matrices **and** the LaTeX export — it reads env + seed mode + checked candidates off the
DOM and returns either a ctx dict (`env, seed_mode, seeds, rec, cands, variants`) or a
human-readable error string. The `.tex` therefore cannot disagree with what is on screen.

`render_summary_tables()` is called from `trigger_plot` **before** the empty-selection
early-return: with candidates but no variants ticked the plot cannot draw, yet the matrices
are perfectly well defined.

**Caching:** key is `(env, seed_mode, seeds, checked_candidates, data_version)`. The variant
checkboxes are deliberately *not* in the key — they fire `trigger_plot` on every click but
never change the tables. A module-level `summary_cache_key` skips the rebuild unless that key
changes. New global `data_version`, incremented in `load_data`, invalidates on SYNC.

## 3. LaTeX export *(U10.1)*

Sidebar button `DOWNLOAD_PNG` → **`EXPORT PNG + LATEX`** (`#download-btn`, same
`py-click="download_plot"`). One click now writes up to three files:

| File | When |
|------|------|
| `{base}.png` | only if a figure is currently drawn |
| `{base}_tables.tex` | whenever `_summary_context()` succeeds (candidates ticked) |
| `{base}.txt` | always — the existing audit log, now also carrying `BATCH:` |

`download_plot` no longer bails out on `current_fig is None`: the tables depend on the
candidate selection alone, so exporting LaTeX with zero variants ticked is legitimate. The
three near-identical blob/anchor blocks collapsed into one `_download_bytes(payload,
filename, mime)` helper. The toolbar status reports what was written, e.g.
`EXPORTED: PNG + TEX + TXT`.

### The .tex file

`build_latex(ctx, batch_label, stamp)` emits a **standalone, compilable** document:
`\documentclass[10pt]{article}` + `geometry` (landscape, 1.5 cm margins) + `booktabs` +
`graphicx`, a header block (batch / environment / seeds / timestamp), the marker legend,
the four `table` floats, and a `Candidate source paths` audit section.

- `_tex_table()` — `\begin{table}[htbp]` · `\caption` · `\label{tab:n-success}` (underscores
  → hyphens, they are not label-safe) · `\resizebox{\textwidth}{!}{...}` around a
  `booktabs` `tabular{l r…r}`. `\resizebox` is required: 18 variant columns never fit
  `\textwidth`, and without it LaTeX silently overflows the margin.
- `_tex_cell()` — `$0.500$` or `$67.7 \pm 1.5$`, `--` for NULL, failure markers as
  `\,{\tiny (goal, constraint)}`.
- `_tex_escape()` — `\ & % $ # _ { } ~ ^`. Applied to captions, variant names and candidate
  labels. Source paths instead go inside a `verbatim` block, which needs no escaping at all.
- Captions in `SUMMARY_TABLES` were changed to plain ASCII with `" -- "`, so one string feeds
  both outputs: HTML renders it as `&mdash;`, LaTeX as an en dash.
- **Everything emitted is ASCII** — the file carries no `inputenc`, and a stray em dash even
  inside a `%` comment trips older TeX installs.

### CSS

New `.paper-tbl` / `.paper-tbl-wrap` classes. These deliberately override the global
`table { table-layout: fixed; width: 100% }` — with ~18 variant columns, fixed layout wraps
every `mean ± sem` cell into an unreadable stack. Sticky header row + sticky `Candidate`
column inside a scrollable `overflow:auto; max-height:560px` wrapper, `white-space: nowrap`,
row hover highlight, `.nullcell` (grey `—`), `.sem` (grey `± …`), `.flag` (red `(…)`).

### Other

`df_agg` / `df_raw` are now initialised to `None` at module level (previously only created
inside `load_data`, so any pre-SYNC call to `trigger_plot` would have raised `NameError`
instead of returning cleanly).

## Verify

Static checks done here (no browser, no cluster):

- The `<script type="py">` block compiles **and executes** — the whole module was `exec`'d
  in CPython with `pandas` / `matplotlib` / `js` / `pyscript` / `pyodide` stubbed out.
- HTML tags balance.
- `build_latex()` and `_render_matrix()` were then **run for real** against
  `analysis_results/batch_avoiding_combined_20260726_164945` (`top-right-hard`, candidates
  2 / 6 / 20 plus a deliberately absent 999): 4 `table`/`tabular` environments open and
  close, brace balance 0, every data row carries exactly 18 `&`, the absent candidate is an
  all-`--` row, 79 cells flagged, output 100 % ASCII.
- Flag derivation cross-checked against a pure-`csv` mirror of the aggregation, e.g.
  `both-hard` CAND_20 / `diffuser`: `n_success 1.000`, `+constraint 0.500` →
  `n_steps 91.0 (constraint)`.

In the browser: serve `Data_Analysis/` and open `Visualizer/index.html`, then

1. SYNC a batch → plot area shows the "select at least one variant and one candidate" panel,
   sidebar lists all unchecked, and the Result Matrices section says "No candidates selected".
2. Tick a few **candidates only** → the four matrices appear (all variants as columns) while
   the plot still asks for a variant.
3. Tick variants → the plot draws; the matrices do **not** change or rebuild.
4. Switch **4. Environment Focus** → matrices rebuild for the new env.
5. Switch **2.5 Seed Mode** to *Custom* → cells flip between plain numbers and `mean ± SEM`;
   unticking all seeds shows the prompt.
6. Cross-check a marked cell: its `(goal|constraint)` suffix must match tables 1 and 2 for
   the same candidate/variant.
7. **EXPORT PNG + LATEX** → `.png` (if a plot is drawn) + `_tables.tex` + `.txt` land in
   Downloads; `pdflatex {base}_tables.tex` should build in one pass.
