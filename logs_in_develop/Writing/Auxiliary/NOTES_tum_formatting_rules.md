# TUM I6 formatting rules — the operative subset

Condensed from `../Writing_Hints/tum_i6_thesis_submission_reference.md`. These
are the rules that change how text gets written, so they matter *now*, not at
submission time. Verify against the current official pages before handing in.

## Hard rules

- **Template:** use the TUM I6 / TUM-Dev LaTeX template. Already vendored at
  `../Template_DONT_CHANGE/`.
- **Citation style:** `alpha` with biblatex/biber (or `alphadin` with BibTeX).
  The template's `settings.tex` already sets `style=alphabetic`, backend biber
  — that is the correct setting, leave it.
- **Direct quotations require a page number** in the citation.
- **Figures:** vector graphics (TikZ/PGF, Inkscape → PDF). Text in figures
  should come from LaTeX so typography matches the body.
- **Every figure must be referenced in the text.** An unreferenced figure is,
  by the institute's own phrasing, unnecessary.
- **Figure citations** carry the source *and* the page number, in the caption.
- **Length:** no formal limit. Master's theses are typically **60–80 pages**
  including front matter and bibliography. Above ~100 pages the advisor will
  not read all of it. The `thesis_bone.tex` budgets sum to ~72 body pages.

## Math notation (single letters for variables, always)

| Category | Style | Example |
| :-- | :-- | :-- |
| Scalar | lowercase italic | `a` |
| Vector | lowercase **bold** italic | `\boldsymbol{v}` → macro `\vect{v}` |
| Matrix | uppercase **bold** italic | `\boldsymbol{M}` → macro `\matr{M}` |
| Function name | upright text | `\textrm{sin}(x)` |
| Unit | upright, half-space after number | `42\,\textrm{Hz}` |
| Angle | Greek | `\alpha` |
| Variable index | plain subscript | `\vect{x}_i` |
| Static index | upright subscript | `\vect{x}_\textrm{min}` → `\sidx{min}` |
| Transpose | upright superscript T | `\vect{x}^\textrm{T}` → `\trans` |
| Identity matrix | upright bold | `\textbf{Id}_n` → `\Id{n}` |
| Unit vector | upright bold | `\textbf{e}_x` → `\unitvec{x}` |
| Absolute value | vertical bars | `\mid a \mid` |
| Modulo | upright | `\textrm{mod}` |
| Cross / dot product | `\times` / `\cdot` | |
| Set | uppercase italic, braces | `A = \{1,2,3\}` |
| Sequence | calligraphic, angle brackets | `\mathcal{A} = \langle 1,2,3\rangle` |
| Set difference / union | `A \backslash \{x\}` / `A \cup \{x\}` | |

`thesis_bone.tex` defines `\vect`, `\matr`, `\trans`, `\sidx`, `\Id`,
`\unitvec` for exactly this. Use them; never hand-roll `\mathbf` in a chapter.

⚠️ **Notation collision to resolve early.** This field's papers overload `x`,
`t`, and `u`: flow-matching time `t`, control horizon time `k`, the velocity
field `u`/`v`, and the control input `u`. Fix a single global convention in
`\ref{sec:method:formal}` before writing any other method text, and record it
in `NOTES_notation_decisions.md` once decided.

## Deliverables at submission

- Official copy per programme rules **plus one extra copy for the advisor**.
- Advisor's copy: storage medium on the last inner page with thesis PDF **and
  LaTeX sources, the code, and the evaluation datasets** — enough to reproduce
  every reported result. This is what Appendix `Reproducibility` is for; keep
  it current instead of assembling it at the end.
- No non-disclosure clause (*Sperrvermerk*) is permitted for I6 theses.

## Talks

Two English talks: an initial topic presentation (5 min + 5 min Q&A, within the
first ~2 months) and the final defence (**20 min + 5 min Q&A** for a Master's
thesis, compulsory and graded). ~1 slide per minute. Both at the monthly I6
Defense Day; the advisor adds you to the speakers list.
