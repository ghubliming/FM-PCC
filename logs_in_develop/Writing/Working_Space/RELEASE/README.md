# RELEASE — the submission-clean thesis, assembled from the live v2 / v3 / v4 drafts

**Created:** 2026-09-24 · **Revised:** 2026-09-25 (per-build folders, front-matter decisions, title-page fit) ·
**Tool:** [`tools/make_release.py`](tools/make_release.py) · **Every build:** [`CHANGELOG.md`](CHANGELOG.md) · **Outputs:** `output/`

A release is the whole thesis as one LaTeX project on the TUM template, with **no comments, no flags,
no drafting notes** — ready to compile and hand in. It is built mechanically from the owners' live
files; nothing in it is edited by hand, and nothing is taken from a draft's `bundle/` (those may be
stale).

| chapter | taken from | owner |
| :-- | :-- | :-- |
| preamble, front matter, **abstract**, Ch 1–4, acronym list, `bibliography.bib` | `v2/thesis_v2.tex`, `v2/bibliography.bib` (content-based split, the markers of `v3/tools/split_v2.py`) | v2 |
| Ch 5–6 | `v3/chapters/05_setup.tex`, `06_results.tex` (+ `parts/00_preamble_v3.tex`, `bibliography_v3.bib`) | v3 |
| Ch 7–8, appendix A–C | `v4/chapters/07_conclusion.tex`, `08_discussion.tex`, `09_appendix.tex` and its nested `app_*` inputs (+ `parts/00_preamble_v4.tex`, `bibliography_v4.bib`) | v4 |
| template files | `../../Template_DONT_CHANGE/` — `settings.tex`, `pages/`, `logos/`, `Makefile`, `.latexmkrc`, `main.xmpdata` (read-only source; copies only) | template |

The abstract is the author's and is copied as it stands. **Content is never changed by a release**: a
content problem goes to the owning draft through `../cross_draft/`, and the release is rebuilt.

## Quick build

```bash
cd logs_in_develop/Writing/Working_Space/RELEASE
python3 tools/make_release.py --tag GOLDEN_TEMPLATE      # a full build: output/<build>/ with latex/, the zip, the notes
python3 tools/make_release.py                            # the same without a tag
python3 tools/make_release.py --note "..."               # a line recorded in the notes and the changelog (repeatable)
python3 tools/make_release.py --acknowledgments          # keep the (empty) Acknowledgments page
python3 tools/make_release.py --no-cover                 # drop the cover page; the title page comes first
python3 tools/make_release.py --appendix-short           # hide the long-data tables of the appendix
python3 tools/make_release.py --standalone               # fallback: v2's standalone preamble, no TUM template
python3 tools/make_release.py --dry-run                  # assemble and check in memory, write nothing
python3 tools/make_release.py --list                     # the builds in output/
python3.14 tools/make_release.py --attach-pdf ~/main.pdf --release output/<build>   # record the real page count
python3 tools/make_release.py --rezip output/<build>     # rebuild a build's zip
```

Python 3 only, no third-party package (`--attach-pdf` counts pages with `pypdf`, present under
`python3.14`). No TeX toolchain exists in this container: **a release is never compiled here.**
Compile it on Overleaf (upload the zip; pdfLaTeX + Biber, main document `main.tex`) or locally
(`make pdf`, or `pdflatex main; biber main; pdflatex main; pdflatex main`).

## What a build produces — every build is kept, never overwritten

```
output/<stamp>_thesis_release_<v2>_<v3>_<v4>[_<TAG>][_standalone][_appendixshort][_PAGEWARN]/
  <same name>.zip          the LaTeX project (latex/ at the zip root) — upload this to Overleaf
  latex/                   the project itself, browsable:
    main.tex               preamble + front matter + \input order + acronym list + lists + bibliography
    settings.tex, main.xmpdata, pages/{cover,title,disclaimer,abstract}.tex (+ acknowledgments.tex when used), logos/
    chapters/01..09 (+ app_long/, app_ntrial20_feasible.tex)
    bibliography.bib       ONE file: v2 + v3 + v4 entries, dev fields (file=) and comments removed
    figures/               exactly one file per referenced figure (.pdf preferred, else .png)
    Makefile, .latexmkrc
  RELEASE_NOTES_<stamp>.md the record of the build: page estimate, HOLES, bugs, cleaning report,
                           sources with SHA-256, structure / list-of-figures / list-of-tables preview
  WARNING_PAGE_LIMIT.md    only when the estimate (or an attached PDF) is outside 60–200 pages
  <build>.pdf              a compiled PDF attached later with --attach-pdf
```

The folder name carries the date-time and the three draft revisions (highest `## vN.M` heading of
each draft's `CHANGELOG.md`). `_PAGEWARN` is appended when the page estimate falls outside the
60–200-page limit; the compiled count, recorded with `--attach-pdf`, adds a warning file instead.

## Front matter — the decisions (2026-09-25, checked against the template's own compiled PDF)

The template's `build/main.pdf` has: p1 cover (title + author, logos), p2 title page (German title,
author / examiner / supervisor / submission date), p3 the declaration *"I confirm that this … thesis is
my own work and I have documented all sources and material used"* at the **bottom** of the page with
the place, date and name line, p4 a blank Acknowledgments page, p5 the abstract, then the contents.

- **Cover and title page — both kept** (template default; the cover is the outer sheet of the bound
  copy, the title page the inner one). `--no-cover` drops the cover.
- **Declaration — kept, always.** It is TUM's mandatory statement of authorship; it is signed in the
  printed copy. It prints from `pages/disclaimer.tex` at the bottom of its page.
- **Acknowledgments — dropped unless there is text.** The page is optional. Put the text into
  `RELEASE/front/acknowledgments.tex` (plain LaTeX paragraphs) and it is included automatically;
  `--acknowledgments` includes the empty page.
- **Title page fit.** With the real titles (two `\huge` lines English, three German) the template's
  title page overflows and the faculty logo lands alone on page ii (seen in the first compiled
  release). The release copy sets the German title in `\LARGE` and reduces the vertical gaps
  (20/15/15/10 mm → 10/8/8/6 mm); fonts and order of everything else are the template's.

## What "clean" means here

- Every `%` comment is removed. A bare line-ending `%` (no text) is TeX syntax and stays; a comment
  glued to code becomes a bare `%`, so no space token is introduced.
- Drafting macros: `\srcnote`, `\dataref`, `\hole`, `\longdata`, `\flawed`, `\outdated`, `\todofigure`
  are removed; `\guard` and `\provisional` are unwrapped to their text (it is thesis prose);
  `deadblock` / `pillarsflawed` blocks are unwrapped. Their definitions and `\newif` switches leave
  the preamble. The presentation macros (`\mainconfig`, `\selectedmark`, `\baselinemark`) stay.
- Build switches are resolved statically: `\ifstandalone` → the template branch (`--standalone`: the
  other one), `\ifappendixfull` → tables printed (`--appendix-short`: hidden).
- The v2 placeholders for Ch 5–8 and the appendix stubs are not used; the live text is v3's / v4's.
- One preamble reorder for the template build: `nag` before `\documentclass` (as the template's
  `main.tex`), `amsmath/amssymb/amsthm` before `settings.tex` (whose `pdfx` loads `hyperref`), so that
  equation anchors are patched. `\getDoctype` is shortened to `Master's Thesis` because the template's
  cover and title pages append `in \getDegree` themselves.
- Every `\hole`, `\longdata`, `\provisional`, TODO metadata value, empty template page and raster-only
  figure is listed under **HOLES** in the notes; every failed mechanical check under **Bugs**.

## Page count

The notes carry a model estimate (370 words per page, figures by width, tables by rows; calibrated on
the first compiled release: 185 pages on 2026-09-25 at v2.28 / v3.99 / v4.1a, including the two pages
that are gone since). Treat it as ±15 %; only an attached PDF gives the real count. The author's limit
is 60–200 pages; the institute's orientation for a master's thesis is 60–80.

## Rules

- **Never edit a file under `output/`.** Fix the owning draft (or this tool) and rebuild.
- **Never delete a build** unless the author says so (the two builds of 2026-09-24/25 were deleted on
  the author's word before the front-matter fixes; their changelog rows say so).
- **Never edit `../../Template_DONT_CHANGE/`.** The tool only reads it.
- A compiled PDF is attached with `--attach-pdf`; it lands in the build folder next to the zip and the
  notes (never inside the zip), and its page count is checked against the 60–200 limit.
