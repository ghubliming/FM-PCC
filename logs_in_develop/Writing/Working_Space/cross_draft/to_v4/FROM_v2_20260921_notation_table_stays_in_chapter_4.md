# FROM v2 → v4 · 2026-09-21 · v2.25 · the notation table stays in Chapter 4, and why

**Author question (2026-09-21):** *"I don't know if such a giant table should be put in the main
text — Table 4.1, Notation used throughout. If the rules do not say and consensus puts it here, keep
it; if the appendix is better, move it and notify v4."*

**Answer: the rules do say, and they say Chapter 4. Not moved.** Recorded here so the appendix owner
does not reopen it.

## The two things that decided it

1. **`Auxiliary/NOTES_tum_formatting_rules.md`**, under *Math notation*, carries a standing
   instruction about exactly this table:

   > ⚠️ **Notation collision to resolve early.** This field's papers overload `x`, `t`, and `u` …
   > Fix a single global convention in `\ref{sec:method:formal}` before writing any other method
   > text.

   The method formalisation section is §4.1, and `tab:notation` sits in §4.1.3 with the two
   paragraphs that state the decisions it encodes (the plan is $\vect{x}$ and not $\tau$;
   $\ftime$ runs data-at-one; the three decorated collisions).

2. **The TUM template has no slot for a symbol list.** `Template_DONT_CHANGE/main.tex` prints an
   `acronym` environment, `\listoffigures` and `\listoftables`, and nothing else in the front
   matter. Moving the table to an appendix would put the thesis's own notation convention *behind*
   the results that use it, and there is no front-matter alternative that the template supports.

## What the table actually is

One `table` float, 25 rows in two blocks (the plan and the sampler, then the low-level control block
introduced by `\autoref{sec:method:deployment}`), inside `tabularx` at `\linewidth`. It is within the
text width by the width heuristic. It is read once, as the reader enters the mathematics, and
referred back to by `\autoref{tab:notation}` thereafter.

## If v4 wants to revisit it

The only change worth considering is a **duplicate** — the table in §4.1.3 as now, plus a short
symbol list in the reproducibility appendix for a reader who arrives at Chapter 6 cold. That would be
v4's call and would need a template addition. Do not simply move it: §4.1.3's prose reads into the
table and would have to be rewritten with it.

Related and already on this table: `FROM_v2_20260918_acronym_list_placement.md` asks the neighbouring
question about the **acronym** expansions. Both are front-matter/appendix placement decisions and can
be settled together.

**Written by:** Claude Opus 5 (Claude Code) · 2026-09-21 · FM-PCC dev container. **Not compiled.**
