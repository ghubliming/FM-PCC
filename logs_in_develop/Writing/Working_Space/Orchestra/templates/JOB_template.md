# {{JOB}} — {{TITLE}}

**Opened:** {{OPENED}} · **Kind:** {{KIND}} (edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **{{V2}}** · v3 **{{V3}}** · v4 **{{V4}}** · last release: `{{RELEASE_LAST}}`
**Sync chain at open:** {{SYNC}} · **INBOX open rows:** {{INBOX_OPEN}}

## Asked

> <the author's words, verbatim or close; the prompt is NOT thesis text — see Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md>

## Scope check (before touching anything)

- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [ ] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [ ] No README / CROSS_STATE / SYNC_STATE / MANIFEST of a draft edited; no `RELEASE/output/`, no `Template_DONT_CHANGE/`.
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| draft | file(s) · lines / labels | change | entry in its CHANGELOG |
| :-- | :-- | :-- | :-- |
| | | | |

## Checks

- v2: `cd RELEASE && python3 tools/make_release.py --dry-run` (the only mechanical check that reads v2's live file) → <result>
- v3: `cd v3 && python3 tools/check.py` → <result> · `python3 bundle/make_bundle.py` → <stamp> · `--verify` → <result>
- v4: `cd v4 && python3 tools/check.py` → <result> · `python3 bundle/make_bundle.py` → <stamp> · `--verify` → <result>
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| | | | |

## Release

none — or `RELEASE/output/<build>/` built with `--tag ORCH_{{JOB}} --note "Orchestra {{JOB}}: <why>"` (only on the author's word)

## Closed

{{CLOSED}}
