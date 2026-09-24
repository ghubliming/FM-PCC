# RELEASE init — the release tool and the first (golden) release · 2026-09-24

**Task (author):** a RELEASE folder under `Working_Space`; aggregate the v2 / v3 / v4 outputs into a
clean, submission-ready LaTeX zip (no comments, no flags, no dev notes; each chapter from its owner's
own files, not from a bundle; abstract untouched; aggregated bibliography; TOC / abbreviations / lists
intact); scripts preserved for the next quick build; page limit 60–200 warned; every build recorded
with date-time and the three draft versions; a notes MD per build listing the holes and any bug noticed.
First release tagged as the golden template.

## Created

| file | what |
| :-- | :-- |
| `RELEASE/tools/make_release.py` | the build tool (Python 3, stdlib only). Splits `v2/thesis_v2.tex` content-based (preamble · front matter with the abstract · Ch 1–4 by chapter label · acronym back matter; v2's Ch 5–8 placeholders and appendix stubs dropped); takes `v3/chapters/05, 06` + `parts/00_preamble_v3.tex`; takes `v4/chapters/07, 08, 09` + nested `app_long/`, `app_ntrial20_feasible.tex` + `parts/00_preamble_v4.tex`. Cleans every file: comments stripped (escape-aware; bare `%` kept only when glued to code), drafting definitions and `\newif` switches dropped from the preambles, `\ifstandalone` / `\ifappendixfull` resolved statically (nested-aware), `\srcnote \dataref \hole \longdata \flawed \outdated \todofigure` removed with balanced-brace parsing, `\guard \provisional` unwrapped, `deadblock`/`pillarsflawed` unwrapped, lines that held only a removed macro deleted (no paragraph break introduced). Assembles `main.tex` on the TUM template (settings/pages/logos/Makefile/.latexmkrc/xmpdata copied from `Template_DONT_CHANGE`, comments stripped; `pages/abstract.tex` generated from v2's abstract), one `bibliography.bib` (v2+v3+v4 entries, `file=` fields and comments removed, duplicate keys refused), `figures/` with one file per figure (.pdf preferred, else .png; searched in v4, v3, v2 and the DA store). Acronym list = v2's, plus any used-but-missing declaration found in v4/v3 (none needed). Checks: input tree, labels/refs, citations, acronyms, environments, braces, figures, leftover marks/switches/comments, dev residue in prose. Page estimate (430 words/page, figures by width, tables by rows, chapter starts, front/back matter); `_PAGEWARN` suffix + `WARNING_PAGE_LIMIT.md` outside 60–200. Writes `RELEASE_NOTES_<stamp>.md` (estimate table, HOLES, bugs, residue, cleaning report, sources with SHA-256, structure / LoF / LoT preview, build instructions), zips the LaTeX project only, prepends a row to `RELEASE/CHANGELOG.md`. Options: `--tag`, `--appendix-short`, `--standalone` (v2's standalone preamble, no template — fallback), `--no-zip`, `--outdir`, `--dry-run`, `--no-log`, `--list`, `--rezip FOLDER`, `--attach-pdf PDF [--release FOLDER]` (copies a compiled PDF in, counts pages with pypdf under python3.14, records the verdict). |
| `RELEASE/README.md` | purpose, source map, quick build, output layout, what "clean" means, rules |
| `RELEASE/CHANGELOG.md` | created by the tool; one entry per build (newest first) |
| `RELEASE/output/20260924_223438_thesis_release_v2.28_v3.99_v4.1a_GOLDEN_TEMPLATE/` + `.zip` (7.8 MB) | **the golden release**: `main.tex` + 19 text files, 48 binary files (41 figures, 5 logos, Makefile, .latexmkrc); notes `RELEASE_NOTES_20260924_223438.md` |
| `.claude/memory/thesis-release-builds.md` (+ index line in `MEMORY.md`) | the author's release rules for later sessions |

Nothing under `v2/`, `v3/`, `v4/` or `Template_DONT_CHANGE/` was changed.

## Decisions made in the tool (not content)

- **Template build is the default** (`\ifstandalone` → else branch): that is what "ready to submit"
  means and what v2's preamble was designed for. Preamble reorder for it: `nag` before
  `\documentclass` (as the template's `main.tex`); `amsmath,amssymb,amsthm` before `settings.tex`
  (whose `pdfx` loads `hyperref`) so equation anchors are patched. Per-draft `\addbibresource` lines
  removed — `settings.tex` registers the single `bibliography.bib`.
- **`\getDoctype` shortened to `Master's Thesis`** in the template build: the template's cover, title
  and disclaimer pages append `in \getDegree` themselves, so v2's value would print "Master's Thesis
  in Informatics in Informatics". Recorded as a finding for v2. Untouched in `--standalone`.
- **Acknowledgments page kept but empty** (the template's TODO only) — a HOLE, the author's text.
  `pages/software_used.tex` (the template's AI-tools declaration example) not included — author's call.
- **Figures ship as PNG**: no PDF of any figure exists anywhere (drafts or DA store); SVG cannot be
  read by pdflatex. Once `DA_in_Paper/plotting/svg/svg2pdf.sh` has run on a machine with Inkscape, the
  tool picks the PDFs up automatically.
- The zip holds the LaTeX project only; notes, warnings and attached PDFs stay in the folder.

## Result of the golden build (from its notes)

- Built on **v2.28 · v3.99 · v4.1a**, 13 source files (SHA-256 prefixes in the notes).
- **Estimate ~159 pages (136–191): within the 60–200 limit, above the 60–80 orientation** of the
  institute. Not compiled — no TeX toolchain here; the template build has never been compiled by
  anyone in this project (only v3/v4 standalone bundles were, on Overleaf), so `--standalone` is the
  fallback if the first template compile fails.
- **10 HOLES:** author / supervisor / advisor / submission date are `TODO:` values printed on the
  cover, title and disclaimer pages; the German title is marked "TODO: confirm"; one `\provisional`
  (Ch 6, alignment distances); the web-link `\hole` and the `\longdata` banner of Appendix B.4;
  the empty Acknowledgments page; 41/41 figures raster-only.
- **1 finding:** the `\getDoctype` duplication (fixed in the release, v2 to change).
- **9 residue lines** for the owners: repo file names in `\texttt` inside the prose of Ch 6 (lines
  181, 450, 1025) and Appendix B (127, 131) — `DA_in_Paper/plotting/sources.py`, `figures/MANIFEST.md`,
  `data_status/PENDING_…md`, `DA_20260923_R2fix_R37_aligning.md`, `eval_mix_visual_aligning.py`; and
  the four TODO metadata values.
- All mechanical checks pass: 276 labels, every reference resolves, 53 bibliography keys all cited,
  11 acronyms all declared, environments and braces balanced, every figure file present, no
  drafting macro, switch or comment text left in any released `.tex`.

## Verification

- `--dry-run`, template and `--standalone` builds run into the scratchpad (`--no-log`) and inspected
  before the golden build; a grep over the golden folder finds no `%` comment, no bare-`%` line, no
  drafting macro, no `\if…` switch; `unzip -t` clean; zip = 68 files at the root (Overleaf-ready).
- Tool fixes during testing: a loop variable clobbered the release name; the template's guarded
  `logos/…-\fg.pdf` includes were flagged as missing figures; empty comment lines left bare `%`
  lines; the outline walked nested appendix inputs after their file (twenty-episode section shown
  under C) and charged a file's words to its last chapter; `\appendix{}` glued to Ch 8's segment.

*Written 2026-09-24 by Claude (Fable 5.1) for the author. Not compiled — no TeX toolchain in the
container. Content untouched.*
