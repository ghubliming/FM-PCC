# O024 — Release from v5.8 marked ROUND2 FIX

**Opened:** 2026-09-25 22:08 · **Kind:** release (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.8** (the thesis) · last release: `20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "ship. and v5 changelog and realse" (author, 2026-09-25)

## Scope check (before touching anything)

- [x] Kind `release`: `cd v5 && python3 tools/make_release_v5.py --job O024 --tag ROUND2_FIX --note …`; no draft file touched; no existing build touched.
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
| RELEASE | `output/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX/` — `latex/`, the Overleaf zip, the notes | built from v5 **v5.8**; tag `ROUND2_FIX`; page estimate ~179 (152-215) | `RELEASE/CHANGELOG.md` entry `20260925_220823` |
| v5 | `CHANGELOG.md` (v5.8 entry), `changelogs/v5.8_…md` | the build named | — |

## Checks

- Build: 19 + 48 files; page estimate ~179 (152-215); 9 holes, 1 finding, 4 residue hits.
- `diff -rq` against the v5.7 build: 4 file(s) differ — `chapters/05_setup.tex`, `chapters/06_results.tex`, `chapters/07_conclusion.tex`, `chapters/08_discussion.tex`; no file only on one side.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | — |

## Release

`RELEASE/output/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX/` — the ROUND2 FIX build; to compile on Overleaf and read against `…_ORCH_v5.7_DATA_FIX`.

## Closed

2026-09-25 22:08 · versions after: v2.28 · v3.100b · v4.2 · v5.8 · notes left: — · release: `20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX` · signed: Orchestra (Claude Fable 5.1, Claude Code)
