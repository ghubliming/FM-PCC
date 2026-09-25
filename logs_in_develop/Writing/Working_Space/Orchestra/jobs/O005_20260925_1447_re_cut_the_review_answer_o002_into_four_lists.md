# O005 — Re-cut the review answer (O002) into four lists for the author: format bugs, tone, content, runs

**Opened:** 2026-09-25 14:47 · **Kind:** check (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.1** (the thesis) · last release: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "end of advanced Orchestra init job. it is good, we will work on this later. Now back to the audit md. now edit that md. Job: 1. There is
> TOO MUCH info the process in the Answer to Audit. at end modify or add new section for: List the Pure Writing/Format issue first, ie pure
> bugs. Then the Tone/Writing Style/Redundant Text; Then list the Content Factual Issue/ suggested Changes. Then the suggested more
> runs/datas. 2. the Current version is been highly curated, I need to be caution on the factual changes." (author, 2026-09-25)

## Scope check (before touching anything)

- [x] Not an Advance job: kind `check` — only the answer MD in the release's `feedback/` folder is edited (the author asked for exactly that); no `.tex`, no `.bib`, no figure, nothing in v5, v2, v3, v4 or `latex/`.
- [ ] **Advance (kind `advance` / `absorb`):** the edit is made in `../../v5/` — the Orchestra owns it (no ownership check, no cross note); one `## v5.N` entry in `v5/CHANGELOG.md` + `v5/changelogs/`, `tools/check.py`, `tools/make_release_v5.py --dry-run`; INBOX rows resolved there are marked `🔀 v5.N`. The items below are for the legacy flow.
- [x] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [x] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [x] No README / CROSS_STATE / SYNC_STATE / MANIFEST of a draft edited; no `Template_DONT_CHANGE/`; under `RELEASE/output/` only the answer MD in `…_GOLDEN_TEMPLATE/feedback/` (the author's explicit instruction; the build's `latex/`, notes and zip untouched).
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_to_THESIS_FIRST_READING_REVIEW_20260925_1310_v2.28_v3.100b_v4.2.md` | **§9 replaced**: the per-owner change list (v2 / v3 / v4 / DA / RELEASE tables + author's decisions) → "The list for the author — four groups": **A** pure writing / format bugs (A1–A11 + the nine prose-vs-table contradictions A12–A20, each with the sentence, the table's value and the fix; known-intentional items; two tooling items) · **B** tone / style / redundant text (B1–B9, author's call) · **C** content: C-i wrong as printed (Ci-1…7, smallest fix, "touches" cell), C-ii claim wider than evidence / spec implicit (Cii-1…17, ordered, every one the author's), C-iii declined · **D** runs and data: D1–D6 from stored data (with "changes printed numbers?"), D7–D11 cluster runs; "How this proceeds" (A in one v5 pass → v5.2; B/C ticked items only; D as DA note / run list). **Header**: Status line rewritten for the v5 flow + a Revised line. **§4 W6 corrected**: `07:21, 63` already carry the exact form ("reduces the median final distance by 84 % of the mean initial distance"); the shorthand "closes N % of the distance" is Ch 6's (`06:636, 705, 710–711, 1100–1101, 1972–1973`); the first version had named `07:63` wrongly. Signature line extended. §0–§8 otherwise unchanged. | the MD itself (header "Revised" line) |
| thesis | **nothing** — v5, v2, v3, v4, DA, `latex/` untouched | — | — |

## Checks

- No thesis file touched, so no draft check was due. The re-cut was re-verified against the release copy before writing: every quoted sentence of A12–A20 re-read at its line (`05:906–907`; `06:11–12, 39, 84–85, 98–99, 255–257, 729, 777, 1027`), the 0 % definition (`05:630–635`), the inclusive gate (`05:1064–1068`), the context-7 disclosure (`06:1020–1026`), the sampler disclosure (`06:437–440`), the front-matter holes (`main.tex:12–17`), the 84 % forms (`grep` over Ch 6–8: Ch 7 exact, Ch 6 shorthand).
- The new §9's tables: 67 rows, no column-count mismatch (checked by script). File 396 lines.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | the author ticks items in §9; only then does anything move (A → v5 in one pass; B / C ticked items; D → a DA note / run list) |

## Release

none.

## Closed

2026-09-25 14:48 · versions after: v2.28 · v3.100b · v4.2 · v5.1 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
