# O012 — Answer MD: conservative DO / NOT verdicts on every B-ii and B-iii row (bounded edits named, rebuilds declined)

**Opened:** 2026-09-25 20:40 · **Kind:** check (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.4** (the thesis) · last release: `20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "The B-ii / iii can you update your answer md again? we take the conservative approch, try to minize the risk of the change, we audit/build the most of
> the tex. but I wont say it is good! but at least we viewed once, now soem of the critic in B-ii/iii indeed make sense, but I dont want make very
> big/catasphoic change mark in the table we better do what and what NOT." (author, 2026-09-25)

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
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` §9 B-ii / B-iii | a *conservative verdict* column on both tables: ⛔ NOT (no rebuild) on 8 B-ii and 7 B-iii rows; ✅ DO bounded on B-ii-4 (one cross-reference sentence at §5.2's head), B-ii-7 (cut the repeated numeric tour inside §6.4: `06:2074–2075` and two of four "one episode in thirty"), B-iii-3 (one clause at `07:13`); ✅ later on B-ii-13 (vector export at the submission build), B-ii-11 (only the small DA tweaks of B-i-15 / 16), B-iii-8 (a guideline); an explanatory note under each heading; "How this proceeds", header **Revised (3)**, signature | the MD itself |
| thesis | **nothing** | untouched | — |

## Checks

- No thesis file touched. The bounded edits were sized on the released v5.4 text: 59.2 / 70.1 / 18.1 / 553.4 restated at `06:1969–1970` and `06:2074–2075`; "one episode in thirty" at `06:1969, 1970, 2028, 2038`; "dominat…" first in Ch 7 at `07:13`, defined at `06:351`; §5.2 opens at `05:79`. Tables: no column mismatch.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | the author reads the verdicts and says go for v5.5 |

## Release

none.

## Closed

2026-09-25 20:40 · versions after: v2.28 · v3.100b · v4.2 · v5.4 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
