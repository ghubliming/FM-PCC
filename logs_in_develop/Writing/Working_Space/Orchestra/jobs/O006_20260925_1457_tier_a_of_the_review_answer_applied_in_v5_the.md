# O006 — Tier A of the review answer applied in v5: the format and consistency bugs A1-A20 (v5.2)

**Opened:** 2026-09-25 14:57 · **Kind:** advance (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.1** (the thesis) · last release: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "Init a Changlog and update in v5 for the tier A. and no need to realse. after fix also mark in the Answer md saying been fixed."
> (author, 2026-09-25) — tier A = group A of the review answer (job O005's re-cut of the O002 answer): the pure writing / format bugs and
> the nine prose-vs-table contradictions, A1–A20.

## Scope check (before touching anything)

- [x] **Advance (kind `advance` / `absorb`):** the edit is made in `../../v5/` — the Orchestra owns it (no ownership check, no cross note); one `## v5.N` entry in `v5/CHANGELOG.md` + `v5/changelogs/`, `tools/check.py`, `tools/make_release_v5.py --dry-run`; INBOX rows resolved there are marked `🔀 v5.N`. The items below are for the legacy flow.
- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [x] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [x] No README / CROSS_STATE / SYNC_STATE / MANIFEST of a draft edited; no `Template_DONT_CHANGE/`; under `RELEASE/output/` only the answer MD in `…_GOLDEN_TEMPLATE/feedback/` (the author asked for the fixed marks there); the build's `latex/` untouched.
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| where | file(s) · lines (v5.2) | change | record |
| :-- | :-- | :-- | :-- |
| v5 | `chapters/04_method.tex:684–687, 876` | A1: the FM-loss display had two `\label`s (amsmath drops the first → three `??`); one label, the one `\eqref{eq:method:engine:fm}` re-pointed to `eq:bg:fm:loss` | `v5/CHANGELOG.md` **v5.2**, `v5/changelogs/v5.2_20260925_review_tier_a_format_bugs.md` |
| v5 | `chapters/05_setup.tex:1112–1115, 514, 705` | A12: "Every model … the same checkpoint at every budget" → "Every flow-based model …; the diffusion baseline's budget is fixed with its noise schedule at training, so each of its budgets is a separately trained checkpoint"; A5: Figures 5.8 and 5.11 capped at `height=0.70\textheight,keepaspectratio` | same |
| v5 | `chapters/06_results.tex` — 15 edits: A13 (l. 66–69), A14 (96), A15 (148), A16 (161–162), A20 (351–352), A7 (Fig 6.2 caption 485–488), A8 (Table 6.6 caption), A17 (908), A10 (911), A2 (guard at 960–963 + comment), A18 (966–969), A19 (1267–1270 and 1323), A3 (Table 6.12 caption 1749–1752), A4 (Fig 6.7 `height=0.70\textheight` cap) | the nine contradictions set to the tables' values; the two captions corrected; the identifier removed; the three floats capped; the caption cut | same |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` | a *status* column on both A tables (17 rows `✅ v5.2`, A6 / A9 `⏳ DA`, A11 `⏳ your metadata`, A4 split cap ✅ / relayout ⏳); an "Applied in v5.2" note under the A heading; a status line under §0's P1 list; the header's **Applied** line; the signature | the MD itself |
| v2 / v3 / v4 | **nothing** | untouched (v5 now differs from its inherited base in these lines; `absorb.py` merges any later legacy change three-way) | — |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7 833 lines, 275 labels (276 − the dropped duplicate), 53 of 53 citations, 41 figures, all pass; drafting macros unchanged · `python3 tools/make_release_v5.py --dry-run` → 19 + 48 files, 9 holes, 1 finding (`\getDoctype`), 4 residue hits, as v5.1 · `absorb.py status` not re-run (no legacy file changed; the base is untouched)
- Every edit re-verified against the v5 table / definition it cites before it was made (record in the v5.2 MD); the float fixes sized from the reviewer's page images and the PNG aspect ratios — confirmed only by a compile.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | an Advance edit needs no cross note (the thesis is v5); the DA items (A6, A9, Fig 6.7 relayout) wait for the author's go — not yet a DA note |

## Release

none — the author: "no need to release".

## Closed

2026-09-25 15:03 · versions after: v2.28 · v3.100b · v4.2 · v5.2 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
