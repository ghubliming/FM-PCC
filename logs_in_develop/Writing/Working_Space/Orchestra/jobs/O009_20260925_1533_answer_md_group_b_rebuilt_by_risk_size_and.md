# O009 — Answer MD: group B rebuilt by risk, size and target (minor / giant / no target); group C tagged dangerous / careful / minor

**Opened:** 2026-09-25 15:33 · **Kind:** check (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.3** (the thesis) · last release: `20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "Update the Answer md; 1. which Writing in Category B is low risk and minor changes, which is Giant changes and maybe related to massive
> rebuild? and which is even not a claim(has no specific target, if has); Rebuild the answer md B section. 2. and in the C section which is
> dangerous and which is just minor changes? also classify them." (author, 2026-09-25)

## Scope check (before touching anything)

- [x] Kind `check`: only the answer MD in the release's `feedback/` folder edited; no thesis file, no figure, no v5 change.
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
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` §9 | **B rebuilt** in three tables: B-i low-risk minor (17 rows, each with file:line / page, the edit, size S/M, risk 🟢/🟠), B-ii giant (13 rows: rebuilds of §1.4 + Ch 7, Ch 2–3, Ch 4 displays, §4.6 / §5.2, Tables 6.6–6.8, 6.11, §6.4, the W3 sweep with hit counts, captions / headers pass, table restructures, figure redesigns, ❌ Discussion-before-Conclusion, vector export), B-iii no target (9 rows: what the reviewer said vs what the text shows, with word counts — "proves" 0 hits, "dominat…" 31 / 3 / 2, model names 44 / 25 / 5 in Ch 6). **C tagged**: a risk column on C-i and C-ii (🔴 6 · 🟠 9 · 🟢 9) with a one-line reason each; the legend under the C heading; "How this proceeds" rewritten (🟢 in one pass → v5.4, 🟠 one by one, 🔴 per decision); header **Revised (2)** line; signature | the MD itself |
| thesis | **nothing** | untouched | — |

## Checks

- No thesis file touched; no draft check due. Word counts taken over the released text `…_ORCH_v5.3_BUGFIX_A/latex` (grep); every quoted W1 / W2 phrase located (`01:5`, `06:2057, 2061, 79, 83, 95`, `07:31`, `08:3`; the reviewer's second W1 quote is not verbatim). Tables of §9: no column-count mismatch (script).
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | the author ticks |

## Release

none.

## Closed

2026-09-25 15:36 · versions after: v2.28 · v3.100b · v4.2 · v5.3 · notes left: — · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
