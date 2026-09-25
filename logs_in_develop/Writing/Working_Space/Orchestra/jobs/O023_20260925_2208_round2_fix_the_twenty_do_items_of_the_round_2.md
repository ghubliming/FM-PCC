# O023 — ROUND2 FIX: the twenty DO items of the round-2 answer applied in v5 (v5.8)

**Opened:** 2026-09-25 22:08 · **Kind:** advance (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.7** (the thesis) · last release: `20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "ship. and v5 changelog and realse" (author, 2026-09-25) — the round-2 answer's list R2-A

## Scope check (before touching anything)

- [x] **Advance (kind `advance` / `absorb`):** 28 edit operations in `v5/chapters/{05,06,07,08}` → `## v5.8` "ROUND2 FIX"; no number of record, table cell or figure changed; six regressions of v5.5–v5.7 undone.
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
| v5 | `chapters/05_setup.tex` (6), `06_results.tex` (14), `07_conclusion.tex` (6), `08_discussion.tex` (2) | the twenty R2-A items | `v5/CHANGELOG.md` **v5.8**, `v5/changelogs/v5.8_20260925_round2_fix.md` |
| RELEASE feedback | `…_v5.7_DATA_FIX/feedback/CLAUDE_ANSWER_to_ROUND2_…md` | header **Applied** / **Released** lines | the MD itself |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7802 lines, 277 labels, 53/53 citations, 41 figures, all pass · `python3 tools/make_release_v5.py --dry-run` → 19 + 48 files, 9 holes, 1 finding, 4 residue hits
- Two anchor slips in the edit script (a comment line inside the guiding-step equation; a self-including span) were caught by the script's own assertions before any partial write and fixed; the passages re-read after.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | — |

## Release

the build is the next job.

## Closed

2026-09-25 22:08 · versions after: v2.28 · v3.100b · v4.2 · v5.8 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
