# REMINDER for v4 — build a glossary appendix (do last)

**Created:** 2026-09-14, by the v2 chat, at the author's request. Notes only — v4 has not started
(`../DRAFT_OWNERSHIP.md`). Do this after v2 and v3 are declared finished.

## Why this exists

v2.13 removed *"Scope and Terminology"* (then §1.5) from the Introduction. v2.14 restructured Chapters 2–4 — the table below is updated. A terminology list in the
Introduction is not where theses put it: each term is defined **at first use in the body**, and a
**collected glossary** goes to the back matter. The body definitions are done in v2; the collected
list is appendix material, which is v4's.

## What to build

An appendix *"Glossary and Notation"* (or extend the template's *Abbreviations* page), listing each
term with a one-line definition and a pointer to where the body defines it:

| term | defined in the body at (v2.14 — labels are stable, numbers may move) |
|---|---|
| **NFE / step budget `K`** — network evaluations per plan | §1.1 `sec:intro:motivation` (first use); §4.3.2 `sec:method:fm` (formal) |
| **number of candidate plans `B`** | §4.5.5 `sec:method:selection` |
| **iterate projection / endpoint projection** | §2.5 `sec:bg:mpc` (concept); §4.5 `sec:method:projection` (math) |
| **unguided** (no projection) | §2.5 `sec:bg:mpc`; §4.5 `sec:method:projection` |
| **genuine steps `n_gen`**, degenerate / thin / admissible | §4.5.4 `sec:method:degenerate` |
| **tightened constraints / tightened geometry** | §4.5.2 `sec:method:dpcc`, *Constraint tightening* |
| **activation threshold `η`** | §4.5.1 `sec:method:proj`, *The activation schedule* |
| every symbol of Table 4.1 (notation) | §4.1.3 `sec:method:formalisation` |

Names must follow `../../Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md`.

## Check before closing

- [ ] Every glossary entry points to a real body definition (labels still resolve after v3/v4 edits).
- [ ] Every `\ac{}` acronym used in the body appears in the abbreviation list.
- [ ] No dev token (arm, fan, S&C, flagship, …) slipped into the glossary.
