# HANDOVER → v2 — two naming changes Chapter 4 has to make

**From v3, 2026-09-15.** v3 owns Chapters 5–6, 8 and the appendix; Chapter 4 is v2's. Both items below
are inside Chapter 4, so v3 has **not** touched them. Until v2 acts, Chapter 4 and Chapter 6 disagree in
wording — that is the cost of not cross-editing, and it is deliberate.

## 1. "genuine steps" → "guiding steps"

The author ruled *genuine* out as jargon on 2026-09-15. It is a value word: it says a step is real
without saying what it does. The quantity counts active steps that are **not** the final one — the only
steps where a later step still exists for the network to react to, so the only steps where projection
steers the sample being generated instead of correcting a finished one. That is what the name should say.

| where | now (Ch. 4) | should become |
| :-- | :-- | :-- |
| `04_method.tex:1001` | `\subsection{Genuine Steps}` | `\subsection{Steps at Which Projection Guides Sampling}` |
| `04_method.tex:1012` | `the \emph{genuine step count}` | `the \emph{guiding step count}` |
| `04_method.tex:1035` | `at the first genuine step` | `at the first guiding step` |
| `eq:method:degen:ngen` and its uses | `n\sidx{gen}` | `n\sidx{guide}` |

`04_method.tex:404` ("a genuine hyperparameter") is ordinary English about something else — leave it.

**Already done in v3** (Chapter 6 §6.1.5 heading, its equation and prose, Chapter 8 RQ3, the appendix
name table, which keeps the artefact token `n_genuine`). The label `sec:method:degenerate` is
**unchanged**, so the `\autoref` from §6.1.5 into Chapter 4 keeps working either way.

## 2. "Three regimes follow" — *regime* is on the banned list

`04_method.tex`, just below the boundary derivation, introduces the cases as *regimes* and the table
labels them *degenerate / thin / admissible*. Both are the tier-and-stage vocabulary the author removed
from Chapters 5–6 on 2026-09-14. Name the three cases by the count itself — $n\sidx{guide} = 0$, $= 1$,
$\ge 2$ — and say what each supports, without a tier word.

Recorded in
`Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md` §3 (two rows, both marked as v2's).
