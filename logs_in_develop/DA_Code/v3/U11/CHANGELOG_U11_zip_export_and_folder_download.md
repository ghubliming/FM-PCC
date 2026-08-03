# DA_Code v3 — U11: single-ZIP export + per-row "download this source folder" buttons

**Scope:** `Data_Analysis/Visualizer/index.html` only. No Python DA change, no cluster re-run,
no new dependency (Pyodide ships `zipfile`/`urllib.parse` in its stdlib).
**Depends on:** U10 (Result Matrices + LaTeX export), U9 (plot legend), U7/U8.
Toolbar tag bumped `SCIENTIFIC_SUITE_v3.11` → `v3.12`.

Two requests, both about getting data *out* of the visualizer:

1. The export button fired **three separate downloads**; it should hand over **one .zip**.
2. The **Source Path** columns should offer a download of the run folder itself — the browser
   equivalent of right-click → Download on that folder in VS Code.

---

## 1. `EXPORT PNG + LATEX` → `EXPORT ZIP (PNG + LATEX + LOG)`

`download_plot()` previously called `_download_bytes()` three times, i.e. three synthetic
`<a download>` clicks in a row (PNG, `.tex`, `.txt`). Browsers throttle or prompt on
multi-file auto-downloads ("this site wants to download multiple files"), and the three
pieces only make sense together anyway.

Now each artifact is appended to a `bundle = [(name_in_zip, bytes)]` list, and a single
`zipfile.ZipFile(..., ZIP_DEFLATED)` written to an in-memory `BytesIO` is downloaded as
`plot_<metric>_<env>_<Nv>_<Nc>_<stamp>.zip`, containing:

```
plot_..._<stamp>.png          # only when a figure is actually on screen
plot_..._<stamp>_tables.tex   # only when _summary_context() succeeds
plot_..._<stamp>.txt          # audit log, always
```

Unchanged: **what** goes into each file, the U10.1 behaviour that PNG and TEX are each
optional (export still works with no plot drawn, or with no candidates ticked), and the
`_summary_context()` single-source-of-truth guarantee that the `.tex` matches the screen.
The status line reports `EXPORTED ZIP: PNG + TEX + TXT` listing only what was actually
included.

## 2. New: `⇓ ZIP` button per row in **both** path tables

Added to the **Source Path** cell of the *Plot Legend — Selected Candidates* table
(`render_selection_map`) and the **Source Path (Audit)** cell of the *Path Audit Map*
(`render_path_map`). Both call the same code path.

### How this can work at all

The CSV's `Full_Path` is an **absolute path on the machine that ran the analysis**, e.g.
`/data/home/llim/FMPCC/FM-PCC/logs/avoiding-d3il-visual/plans/.../<run>`. The visualizer is
documented (`Data_Analysis/README.md`) to be served with `python3 -m http.server` **from the
repo root**, so the very same server that hands out this page also hands out
`<repo-root>/logs/...`. The button therefore:

1. **Resolves** the absolute path back to a root-relative one — `_rel_path_candidates()`
   produces ordered guesses (everything after the last `FM-PCC/` segment → from the first
   `logs/` segment → the whole path with the leading `/` dropped), and each is probed as
   `../../<guess>/` (the page sits at `<root>/Data_Analysis/Visualizer/`). First one that
   returns a real directory listing wins.
2. **Crawls** it recursively (`_crawl_dir`) by parsing `http.server`'s `<a href="…">` listing;
   entries ending in `/` recurse, the rest are collected as files.
3. **Zips** client-side, fetching each file as an `ArrayBuffer` → `bytes`, writing it under
   `<folder-name>/…` so unzipping never litters the cwd.
4. **Downloads** as `CAND_<id>__<folder>.zip` (falls back to `<folder>.zip` with no label).

Nothing is added server-side and no path is trusted blindly — a guess that doesn't resolve to
a directory listing is simply skipped.

### Guards

| Guard | Value | Why |
|---|---|---|
| `MAX_CRAWL_DEPTH` | 8 | runs nest ~2-3 levels; 8 is far past that |
| `MAX_CRAWL_FILES` | 4000 | a mis-resolved path could otherwise crawl the whole repo |
| `CONFIRM_FILE_COUNT` | 150 | above this, `window.confirm()` first — `.npz` rollout dumps get big, and the archive is built in browser memory |
| `_folder_dl_busy` | flag | one folder at a time; all `⇓ ZIP` buttons are `disabled` while a download runs |

Directory detection requires **both** `Content-Type: text/html` **and** the literal
`Directory listing` from `http.server`'s `<title>`, so a stray `index.html` (or an SPA
fallback) can't be mistaken for a directory full of files.

### Failure modes are explicit, not silent

- Path not reachable over HTTP → status `FOLDER_NOT_SERVED` + an alert stating the actual
  requirement (serve from the repo root **of the machine holding the runs**, directory
  listing enabled). This is the expected outcome when browsing a cluster-generated CSV from a
  laptop that never downloaded those log folders — there is no way around it, the bytes
  aren't there.
- Resolved but empty → `FOLDER_EMPTY`. Individual files that 404 mid-crawl are skipped and
  logged to the console rather than aborting the archive.
- During the zip the status line shows live progress: `ZIPPING 37/412 — 18.4 MB`.

### Wiring detail (why not inline `onclick`)

The path travels in `data-path` / `data-cand` attributes (HTML-escaped by `_esc_attr`) and is
picked up by a **delegated** `click` listener on `document` matching `.dl-folder`. Two
reasons: both tables are re-rendered wholesale on every redraw (a delegated listener survives
that), and real run folders contain characters that break naive inline-attribute quoting —
e.g. `fm_visual_avoiding(unfull)`. The JS handler calls `document.download_folder`, a
`create_proxy`'d **sync** Python wrapper that `asyncio.ensure_future`s the coroutine, so JS
never has to handle (or swallow) a promise rejection.

## Files touched

- `Data_Analysis/Visualizer/index.html`
  - `<style>`: `.dl-folder` button styling (+ `:hover`, `:disabled`).
  - imports: `+ re, zipfile, asyncio`, `+ urllib.parse.quote/unquote`
    (`import asyncio` moved to the top; it was previously a local import at the bottom).
  - new: `_esc_attr`, `_folder_btn`, `_rel_path_candidates`, `_enc`, `_fetch_listing`,
    `_listing_entries`, `_crawl_dir`, `_download_folder`, `download_folder`
    (+ `MAX_CRAWL_DEPTH` / `MAX_CRAWL_FILES` / `CONFIRM_FILE_COUNT` / `_folder_dl_busy`).
  - modified: `download_plot` (bundle → one zip), `render_selection_map` /
    `render_path_map` (button in the path cell), the summary note text and the button label
    (`EXPORT ZIP (PNG + LATEX + LOG)`), `document.download_folder` proxy registration,
    delegated JS click handler, toolbar version tag.

## Verification

Syntax of the embedded `<script type="py">` block checked with `ast.parse` locally.
**Browser-side behaviour is untested here** (this container has no browser and no Pyodide) —
needs a manual pass when serving a batch: `python3 -m http.server 8000` from the repo root,
open `/Data_Analysis/Visualizer/index.html`, confirm (a) EXPORT produces one `.zip` with 1–3
members, (b) `⇓ ZIP` on a candidate whose `logs/` folder exists locally produces the folder
archive, (c) a candidate whose folder is absent shows `FOLDER_NOT_SERVED` rather than hanging.
