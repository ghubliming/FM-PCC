# `bundle/` — one flat `.tex`, built from the split draft

v3 is split by chapter so that v2 and v3 can be worked on in parallel. That is right for editing and
wrong for uploading. **This folder holds the flattened build**, and a tool that produces it
mechanically, so the flat file is never edited by hand and cannot drift from the split one.

```bash
python3 bundle/make_bundle.py                 # NEW sections only -- v2 chapters collapsed
python3 bundle/make_bundle.py --full          # the complete document
python3 bundle/make_bundle.py --clean-notes   # hide [src: ...] notes in the review copy
python3 bundle/make_bundle.py --svg-package   # same, but render the SVG figures directly
python3 bundle/make_bundle.py --verify        # prove the newest bundle matches the tree
python3 bundle/make_bundle.py --list
python3 bundle/make_bundle.py --prune 5       # keep only the 5 newest
```

## Default: new sections only

v3 inherits Chapters 1–4 from v2 and normally never edits them, so by default the bundle **does not
re-typeset them**. Each unchanged v2 chapter is collapsed to its numbered `\chapter`/`\section`
headings plus a grey **"v2 SECTION — not built in this bundle"** box naming the file, its length and
the v2 revision it came from. Result: roughly half the length (2 347 against 4 108 lines today), with
the v3 text on the page.

**Nothing about numbering changes.** The kept headings carry their labels, so chapter and section
numbers are identical to the full build — checked: the 58 numbered headings appear in the same order
in both — and every `\autoref` from a v3 chapter into a v2 one still resolves. The tool reports it if
a reference ever lands inside collapsed text and would print `??`.

Two guards keep "the v2 chapters are untouched" a checked fact rather than an assumption:

| guard | why |
| :-- | :-- |
| only `chapters/` files that `tools/sync_v2.py` marks `inherit` or `merge` are candidates; `parts/` (preamble, front matter, back matter) is always inlined | the document does not compile without them |
| a candidate collapses **only if byte-identical** to `inherited/v2_base/`; otherwise it is inlined in full and the build prints `inlined IN FULL because v3 has edited them` | `02_background` and `04_method` are *allowed* to be edited by v3. Collapsing an edited chapter would hide exactly the edit under review |

**Use `--full` for the complete thesis** — the Overleaf upload of record, anything sent to a
supervisor, and before submission. `--full` and `--svg-package` combine.

Each run writes a timestamped pair into `output/`, named `thesis_v3_<stamp>_new` or `_full`, and appends a row to [`BUNDLE_LOG.md`](BUNDLE_LOG.md):

```
output/thesis_v3_<YYYYMMDD_HHMMSS>_new.tex        new sections only (default)
output/thesis_v3_<YYYYMMDD_HHMMSS>_new_clean.tex  new sections, source notes hidden
output/thesis_v3_<YYYYMMDD_HHMMSS>_full.tex       the complete document (--full)
output/thesis_v3_<...>.zip                        that .tex + both .bib files + figures/
```

> **Git status**: Generated outputs in `output/` (flat `.tex`, `.zip`, staging assets) are excluded by `.gitignore` so build outputs do not clutter the repository. The bundling script (`make_bundle.py`), documentation, and logs are tracked in git.

## For Overleaf

**Upload the `.zip`** (New Project → Upload Project). It already contains everything the build needs:
the flat `.tex`, `bibliography.bib`, `bibliography_v3.bib` and `figures/`. Overleaf picks the file
carrying `\documentclass` as the main document. Set the compiler to **pdfLaTeX** and the bibliography
tool to **Biber** — the draft uses `biblatex` with `backend=biber`, inherited verbatim from the TUM
template's `settings.tex`.

The normal bundle is annotated and shows `[src: ...]` notes. `--clean-notes` creates a second bundle
with the supported `\submissiontrue` switch already active. In Overleaf, the equivalent manual switch
is to place `\submissiontrue` after the preamble inputs and before `\begin{document}`. This hides only
`\srcnote`; unresolved `\hole` text remains visible and warns so missing content cannot be concealed.

### Figures — two modes, and which to use

The figures are generated as SVG in [`Data_Analysis/DA_in_Paper/`](../../../../../Data_Analysis/DA_in_Paper/README.md) and copied into `../figures/` by its `plotting/export_to_draft.py` and `\includegraphics` cannot
read SVG. No SVG converter exists in the container the thesis is written in, so the tool cannot just
convert them for you. Instead it rewrites every `\includegraphics` call to `\fmpccgraphic` and
injects that macro, which resolves in order: `.pdf` → `.png` → *(mode-dependent)* → placeholder box.

| you want | run | what happens |
| :-- | :-- | :-- |
| **it must compile anywhere** *(default)* | `make_bundle.py` | Each figure becomes a framed box naming the file it could not find. The document always builds. |
| **real figures on Overleaf** | `make_bundle.py --svg-package` | Adds `\usepackage{svg}`; Overleaf runs Inkscape at build time and renders the SVGs. |
| **real figures anywhere** | `DA_in_Paper/plotting/svg/svg2pdf.sh`, `export_to_draft.py v3`, then `make_bundle.py` | PDFs are produced once in the store, exported into `figures/`, and picked up automatically. Best option if you have `rsvg-convert`, `inkscape` or `cairosvg` on any machine. |

`--svg-package` is **opt-in, not the default**, because it is a property of the *build host* and not
of the document: a TeX installation without Inkscape fails outright on it, and the default has to
build everywhere. In every mode a real `.pdf` or `.png` wins if one exists, so converting the figures
never requires touching the source.

## What the tool actually does

1. **Inlines every resolvable `\input`**, recursively, wrapping each in `BEGIN`/`END` banners that
   name the source file and its SHA-256 prefix.

   An `\input` whose target **does not exist** is left verbatim, and that is correct rather than a
   fallback: the inherited preamble and front matter carry `\input{settings}` and
   `\input{pages/cover}` inside the `\ifstandalone … \else` branch, for the day the draft is merged
   into the TUM template. `\standalonetrue` is set, so LaTeX never reads them.

2. **Verifies before writing, and refuses to write on failure** — no resolvable `\input` left, no
   content lost, braces balanced, environments balanced, exactly one `\documentclass`,
   `\begin{document}` and `\end{document}`.

3. **`--verify` is the check that matters.** It extracts every inlined source back out of a bundle
   and diffs it against the tree, undoing the one transformation the tool applies. A collapsed
   chapter has no text to diff, so for those it checks that the source's SHA-256 still matches the
   one recorded at build time. That proves the
   flattening is byte-faithful rather than merely well-formed. Run on an *older* bundle it answers a
   different and equally useful question — *was this built from what is on disk now?* — and a
   mismatch there is information, not a bug.

## Rules

- **Never edit a file in this folder.** It is build output. An edit here is lost on the next run and,
  worse, silently diverges from the draft everyone else is reading. Edit `../chapters/` or
  `../parts/`, then rebuild.
- **Nothing reads a bundle back.** No tool in this workspace consumes one; they exist to be handed to
  Overleaf, a supervisor, or a reviewer.
- **The header of every bundle lists its sources with SHA-256 prefixes**, so any bundle can be traced
  to exactly the files it was built from, even months later.
- Bundles accumulate on purpose — the timestamps are the point. `--prune N` when the folder gets
  noisy.

## Known state at the time of writing

`\hole`, `\provisional`, `\guard` and `\dataref` are drafting macros and render in colour: red for
what is not yet written, orange for a claim below the strength the thesis wants, violet for a caveat
that must travel with a claim. They are defined in `../parts/00_preamble_v3.tex` and must all be gone
before submission; `../tools/check.py` counts them on every run.

**This has never been compiled here** — the container has no TeX toolchain. The tool checks structure,
not typesetting, and says so on every run. The first real build is yours.
