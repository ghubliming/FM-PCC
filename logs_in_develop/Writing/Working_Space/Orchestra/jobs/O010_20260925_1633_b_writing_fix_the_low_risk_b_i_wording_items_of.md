# O010 — B WRITING FIX: the low-risk B-i wording items of the review answer applied in v5 (v5.4)

**Opened:** 2026-09-25 16:33 · **Kind:** advance (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.3** (the thesis) · last release: `20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "start on B, are the lowrisk B-i mutual exlusive vs B-ii? if so me may no risk fix B-i first. but first answer me. And in B udpate, we build a
> new changlog and mark as B writing fix." → "Go. do B-i." (author, 2026-09-25)

## Scope check (before touching anything)

- [x] **Advance (kind `advance` / `absorb`):** 27 wording edits in `v5/chapters/{01,02,04,06,07,08}` → `## v5.4` "B WRITING FIX"; no claim, number, table cell, figure or drafting macro changed.
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
| v5 | `chapters/01_introduction.tex` (2), `02_background.tex` (1), `04_method.tex` (7), `06_results.tex` (15), `07_conclusion.tex` (1), `08_discussion.tex` (1) | the B-i rows 1, 2, 3, 5, 6, 7, 8, 9 (Ch 6), 10, 11, 14 — before → after per edit in the record | `v5/CHANGELOG.md` **v5.4**, `v5/changelogs/v5.4_20260925_review_B_writing_fix.md` |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` | B-i status column (11 rows ✅ v5.4; held: B-i-4 Ci-7, B-i-9 abstract, B-i-12, B-i-13 decision, B-i-15 / 16 DA, B-i-17 decision); B-i-13 and B-i-17 re-tagged 🟠 decision with the conflict named; header **Applied (B)**; signature | the MD itself |
| v2 / v3 / v4 | **nothing** | untouched | — |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7 835 lines, 275 labels, 53/53 citations, 41 figures, all pass · `python3 tools/make_release_v5.py --dry-run` → 19 + 48 files, 9 holes, 1 finding, 4 residue hits (as v5.3) · `absorb.py status` not re-run (no legacy file changed)
- Before editing, two rows were checked against the author's own rules and held: B-i-13 (v3.83 rates-± instruction vs the Ch 5 counts rule), B-i-17 (v3.49: protocol in captions). Every applied edit was re-read in place.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | — |

## Release

none — not asked for this revision.

## Closed

2026-09-25 16:36 · versions after: v2.28 · v3.100b · v4.2 · v5.4 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
