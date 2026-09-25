# O014 — B WRITING (length): the relaxed B-ii / B-iii rows applied in v5 (v5.5)

**Opened:** 2026-09-25 21:05 · **Kind:** advance (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.4** (the thesis) · last release: `20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "Go. and after finish mark what is been done and which is not, in a new column, keep the current relaxed comments. Update the v5 with
> changelog then release mark as B writing." (author, 2026-09-25) — the relaxed B-ii / B-iii verdicts of job O013.

## Scope check (before touching anything)

- [x] **Advance (kind `advance` / `absorb`):** 47 edits in `v5/chapters/{01,03,04,05,06,07,09}` → `## v5.5` "B WRITING (length)"; no claim, number, figure content or drafting macro changed (two orphaned `\srcnote`s went with the claims they sourced).
- [ ] **Advance (kind `advance` / `absorb`):** the edit is made in `../../v5/` — the Orchestra owns it (no ownership check, no cross note); one `## v5.N` entry in `v5/CHANGELOG.md` + `v5/changelogs/`, `tools/check.py`, `tools/make_release_v5.py --dry-run`; INBOX rows resolved there are marked `🔀 v5.N`. The items below are for the legacy flow.
- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [ ] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [ ] No README / CROSS_STATE / SYNC_STATE / MANIFEST of a draft edited; no `RELEASE/output/`, no `Template_DONT_CHANGE/`.
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| v5 | `chapters/01_introduction.tex` (6), `03_related_work.tex` (3), `04_method.tex` (1), `05_setup.tex` (15), `06_results.tex` (17), `07_conclusion.tex` (2), `09_appendix.tex` (4 sections added) | the relaxed rows B-ii-1, 2, 3, 4, 5, 7 (in part), 9, 10 (in part), 11; B-iii-3 — before → after per edit in the record | `v5/CHANGELOG.md` **v5.5**, `v5/changelogs/v5.5_20260925_review_B_writing_length.md` |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` | a *done (v5.5)* column on the B-ii and B-iii tables, the relaxed comments kept; B-iii-9 the honest size; header **Applied (B-ii / iii)**; signature | the MD itself |
| v2 / v3 / v4 | **nothing** | untouched | — |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7764 lines, 277 labels, 53/53 citations, 41 figures, all pass · `python3 tools/make_release_v5.py --dry-run` → 19 + 48 files, 9 holes, 1 finding, 4 residue hits · `absorb.py status` not re-run (no legacy file changed)
- One slip caught by `check.py` and fixed before recording: a guessed label (`sec:res:uav:pillars:pro`) in a new appendix lead sentence → `sec:res:uav:pillars:projected`.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | — |

## Release

the build is job O015 (the author: "then release mark as B writing").

## Closed

2026-09-25 21:05 · versions after: v2.28 · v3.100b · v4.2 · v5.5 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
