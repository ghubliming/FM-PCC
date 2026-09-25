# O016 — Answer MD: conservative DO / NOT verdicts on every C row (content), with the limit of each DO

**Opened:** 2026-09-25 21:10 · **Kind:** check (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.5** (the thesis) · last release: `20260925_210607_thesis_release_ORCH_v5.5_B_WRITING`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "Now A/B should be checked, for the PartC, which fixes should I done and which not? using conservative method, add a new column." (author, 2026-09-25)

## Scope check (before touching anything)

- [x] Kind `check`: only the answer MD edited; no thesis file touched.
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
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` §9 C | a *conservative verdict* column on C-i and C-ii (24 rows): ✅ DO 14 (every 🟢; Ci-1 / Ci-3 corrected against code; Ci-4 / Ci-5 where the thesis contradicts its own table or sourcing rule), 🟡 DO in part 7 (the named minimum only: Ci-7 hedge, Cii-1 clauses not the abstract, Cii-3 captions not x/9, Cii-5 geometry not the 4.8 cm sentence, Cii-9 the deletion not the sweep, Cii-12 one phrase, Cii-16 Ch 2 not the abstract), ⛔ NOT 3 (Cii-11, 14, 17); the legend under the C heading; "How this proceeds"; header **Revised (5)**; signature | the MD itself |
| thesis | **nothing** | untouched | — |

## Checks

- No thesis file touched. Tables: no column mismatch.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | the author says go for v5.6 |

## Release

none.

## Closed

2026-09-25 21:10 · versions after: v2.28 · v3.100b · v4.2 · v5.5 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
