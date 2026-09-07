# Bone — thesis structural skeleton

**Status:** initial phase. Structure only, no prose.

| File | What it is |
| :-- | :-- |
| `thesis_bone_broad.tex` | **Coarse view.** Title + ToC + chapters and top-level sections only. No subsections, no budgets, no specs. Use this to agree the overall shape. |
| `thesis_bone.tex` | **Detailed view.** Same chapters, plus subsections, per-chapter page budgets, and "what this chapter must deliver" comments. Use once the coarse shape is settled. |

Both share the same metadata block, the same fixed title, and the same
standalone/template switch, so they stay in sync by construction — if a chapter
or top-level section changes, change it in both.

## How to use it

1. **Start broad.** Settle the chapter list and top-level sections in
   `thesis_bone_broad.tex` first. Only then look at `thesis_bone.tex`, where
   every chapter block carries a `BUDGET ~N pages` line and a short spec of
   what it must deliver — there the comments, not the headings, are the
   contract.
2. **Validate the architecture with the advisor before writing prose.** The
   thesis guide's core warning is that text is expensive to refactor — a wrong
   outline is the costliest mistake at this stage.
3. When the outline is approved, copy `Template_DONT_CHANGE/` to a new folder
   under `Working_Space/` (e.g. `Working_Space/thesis/`), move the metadata
   macros and the chapter skeleton from this file into that copy's `main.tex`,
   split the chapters into `chapters/NN_*.tex`, and flip `\standalonefalse`.

## Build

No LaTeX toolchain in this dev container. Compile in Overleaf, TUM ShareLaTeX,
or on a machine with TeX Live:

```bash
pdflatex thesis_bone_broad.tex && pdflatex thesis_bone_broad.tex   # twice, for the ToC
```

Standalone mode uses a minimal preamble (no `pdfx`/PDF-A, no biblatex, no
external `pages/` files), so it compiles inside this folder with nothing else
present. The real thesis build uses the template's `settings.tex`.

## Open decisions blocking further structure work

See `../../Auxiliary/NOTES_open_questions.md`.
