# `bundle/` — one flat `.tex`, built from the split draft

v3 is split by chapter so that v2 and v3 can be worked on in parallel. That is right for editing and
wrong for uploading. **This folder holds the flattened build**, and a tool that produces it
mechanically, so the flat file is never edited by hand and cannot drift from the split one.

```bash
python3 bundle/make_bundle.py                 # flatten, verify, zip
python3 bundle/make_bundle.py --svg-package   # same, but render the SVG figures directly
python3 bundle/make_bundle.py --verify        # prove the newest bundle matches the tree
python3 bundle/make_bundle.py --list
python3 bundle/make_bundle.py --prune 5       # keep only the 5 newest
```

Each run writes a timestamped pair and appends a row to [`BUNDLE_LOG.md`](BUNDLE_LOG.md):

```
thesis_v3_<YYYYMMDD_HHMMSS>.tex     one self-contained file, ~3 700 lines
thesis_v3_<YYYYMMDD_HHMMSS>.zip     that .tex + both .bib files + figures/
```

## For Overleaf

**Upload the `.zip`** (New Project → Upload Project). It already contains everything the build needs:
the flat `.tex`, `bibliography.bib`, `bibliography_v3.bib` and `figures/`. Overleaf picks the file
carrying `\documentclass` as the main document. Set the compiler to **pdfLaTeX** and the bibliography
tool to **Biber** — the draft uses `biblatex` with `backend=biber`, inherited verbatim from the TUM
template's `settings.tex`.

### Figures — two modes, and which to use

The figures are generated as SVG ([`../plots/`](../plots/README.md)) and `\includegraphics` cannot
read SVG. No SVG converter exists in the container the thesis is written in, so the tool cannot just
convert them for you. Instead it rewrites every `\includegraphics` call to `\fmpccgraphic` and
injects that macro, which resolves in order: `.pdf` → `.png` → *(mode-dependent)* → placeholder box.

| you want | run | what happens |
| :-- | :-- | :-- |
| **it must compile anywhere** *(default)* | `make_bundle.py` | Each figure becomes a framed box naming the file it could not find. The document always builds. |
| **real figures on Overleaf** | `make_bundle.py --svg-package` | Adds `\usepackage{svg}`; Overleaf runs Inkscape at build time and renders the SVGs. |
| **real figures anywhere** | `../tools/svg2pdf.sh` then `make_bundle.py` | PDFs are produced once, land in `figures/`, and are picked up automatically. Best option if you have `rsvg-convert`, `inkscape` or `cairosvg` on any machine. |

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
   and diffs it against the tree, undoing the one transformation the tool applies. That proves the
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
