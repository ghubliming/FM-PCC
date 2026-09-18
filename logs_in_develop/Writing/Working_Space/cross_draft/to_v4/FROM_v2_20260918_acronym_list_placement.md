# FROM v2 → v4 · 2026-09-18 · v2.24 · where the abbreviation expansions belong

**Author question (2026-09-18), raised against `tab:embodiments`:** the table used short forms — `IK`,
`MJPC`, `sim substeps`, `env. step` — several of them at their first appearance in the running text.
v2.24 fixed the table itself: the short forms are now `\ac{IK}` and `\ac{MJPC}` (so the acronym
package expands them at first use), the informal ones are spelled out, and the caption defines every
remaining symbol in place, with a forward pointer to `sec:setup:metrics`.

**What is left for v4.** The author also asked whether the abbreviation expansions should live in an
appendix list rather than only in the front-matter acronym list. That is a whole-thesis decision —
the acronym list is shared by all drafts and the appendix is not v2's — so it is recorded here rather
than acted on. v2 currently declares the acronyms in `\begin{acronym}` in the preamble and prints the
list in the front matter; nothing in Ch 1–4 depends on where it is printed.

No action needed in v2 either way; if v4 moves the list, v2 needs no edit.
