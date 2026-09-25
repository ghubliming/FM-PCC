# O020 — DATA FIX: D4 of the review answer — the endpoint non-convergence rate stated in §6.1.2 from the DA record (v5.7)

**Opened:** 2026-09-25 21:25 · **Kind:** advance (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.6** (the thesis) · last release: `20260925_212026_thesis_release_ORCH_v5.6_FACTUAL_CHANGE`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "ship the D with v5 changelog and build release as Data fix, also mark the ANSWER md." (author, 2026-09-25) — D4 under the O019 verdicts

## Scope check (before touching anything)

- [x] **Advance (kind `advance` / `absorb`):** one sentence + a `\dataref` in `v5/chapters/06_results.tex` §6.1.2 → `## v5.7` "DATA FIX"; the number is of record (DA_20260924_R16_R36_must_need.md §5.3); nothing run.
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
| v5 | `chapters/06_results.tex` §6.1.2 | D4: the endpoint solves' non-convergence rate, 0.1–1.1 % over the six endpoint cells of Table 6.3, with a data note of the per-cell counts | `v5/CHANGELOG.md` **v5.7**, `v5/changelogs/v5.7_20260925_review_D_data_fix.md` |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` | a *done (v5.7)?* column on both D tables; header **Applied (D)** / **Released (DATA FIX)**; signature | the MD itself |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7798 lines, 277 labels, 53/53 citations, 41 figures, all pass · `python3 tools/make_release_v5.py --dry-run` → 19 + 48 files, 9 holes, 1 finding, 4 residue hits
- The number re-read in the DA record (§5.3 table: six cells, 0.09–1.08 %).
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | — |

## Release

the build is the next job (the author: "build release as Data fix").

## Closed

2026-09-25 21:25 · versions after: v2.28 · v3.100b · v4.2 · v5.7 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
