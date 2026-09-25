# O011 — Temporary review release from v5.4: a marker build to read the B-i writing fixes in the PDF

**Opened:** 2026-09-25 20:10 · **Kind:** release (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.4** (the thesis) · last release: `20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "can you release it (as a temp marker, I will use forreview now for the B-i changes in PDF)" (author, 2026-09-25)

## Scope check (before touching anything)

- [x] Kind `release`: `cd v5 && python3 tools/make_release_v5.py --job O011 --tag TEMP_REVIEW_Bi --note …`; no draft file touched.
- [ ] **Advance (kind `advance` / `absorb`):** the edit is made in `../../v5/` — the Orchestra owns it (no ownership check, no cross note); one `## v5.N` entry in `v5/CHANGELOG.md` + `v5/changelogs/`, `tools/check.py`, `tools/make_release_v5.py --dry-run`; INBOX rows resolved there are marked `🔀 v5.N`. The items below are for the legacy flow.
- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [ ] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [x] Under `RELEASE/output/` only the new build folder was written; no existing build touched (builds are never deleted — the marker stays beside the others).
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| RELEASE | `output/20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi/` — `latex/` (main.tex + 18 text files, 48 binary files), the Overleaf zip (7 841 KB), `RELEASE_NOTES_20260925_201059.md` | built from v5 **v5.4** (B WRITING FIX); tag `TEMP_REVIEW_Bi`; two notes say it is a temporary review marker for the B-i edits, not a milestone | `RELEASE/CHANGELOG.md` entry `20260925_201059` |
| v5 | `CHANGELOG.md` (v5.4 entry), `changelogs/v5.4_…md` | the build named in the revision it was built from | — |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` | header: **Released (B-i review)** line | — |
| v2 / v3 / v4 | **nothing** | untouched | — |

## Checks

- Build: 19 text + 48 binary files; page estimate ~180 (153–215); 9 holes, 1 finding (`\getDoctype`), 4 residue hits — as the v5.3 build.
- `diff -rq` against `…_ORCH_v5.3_BUGFIX_A/latex`: 6 file(s) differ — `chapters/01_introduction.tex`, `chapters/02_background.tex`, `chapters/04_method.tex`, `chapters/06_results.tex`, `chapters/07_conclusion.tex`, `chapters/08_discussion.tex`; no file only on one side. Exactly the six chapter files v5.4 edited; no figure, no other file.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | — |

## Release

`RELEASE/output/20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi/` — a temporary review marker for the B-i wording edits (the author's words above); to compile on Overleaf and read against `…_ORCH_v5.3_BUGFIX_A`.

## Closed

2026-09-25 20:11 · versions after: v2.28 · v3.100b · v4.2 · v5.4 · notes left: — · release: `20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi` · signed: Orchestra (Claude Fable 5.1, Claude Code)
