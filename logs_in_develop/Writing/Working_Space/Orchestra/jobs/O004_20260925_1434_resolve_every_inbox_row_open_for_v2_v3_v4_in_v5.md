# O004 — Resolve every INBOX row open for v2 / v3 / v4 in v5 (v5.1): absorbed by construction, answered, or not applicable

**Opened:** 2026-09-25 (see the CHANGELOG row) · **Kind:** advance · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.0** (the thesis) · last release: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
**INBOX open rows at open:** → v2: 2, → v3: 4, → v4: 4 (three of them the O003 FYIs), → v5: 0

## Asked

> "5. First job is to resolve any the INBOX for v2,v3,4. to v5.1" (author, 2026-09-25 — point 5 of the Advance Orchestra instruction; points 1–4 are job O003)

## Scope check (before touching anything)

- [x] **Advance (kind `advance`):** the work is done in v5 — the Orchestra owns it; no cross note; one `## v5.1` entry in `v5/CHANGELOG.md` + `v5/changelogs/v5.1_20260925_inbox_rows_resolved.md`; the INBOX rows resolved are marked `🔀 v5.1`.
- [x] No legacy draft file touched (the notes asked the *legacy* owners to sync or to edit their tools; in v5 that is by construction or not applicable).
- [x] No thesis text needed a change: every substantive item is either already in v5 (v2.28) or already stated in the text (Ch 5 / Ch 6 passages named in the record).
- [x] No figure, no `RELEASE/output/`, no template touched.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| cross_draft | `INBOX.md` — the seven rows open on 2026-09-25 (→ v2: O001 · → v3: O001, v4.2, v2.28 · → v4: O001, v3.100b, v2.28) | status `⏳` → `🔀 v5.1 (<resolution>)`; the three O003 FYI rows left `⏳` for the owner chats | `v5/changelogs/v5.1_…` (the table with evidence per row) |
| v5 | `CHANGELOG.md` (**v5.1**), `changelogs/v5.1_20260925_inbox_rows_resolved.md` | the revision record; **no `.tex` / `.bib` changed** | — |
| v2 / v3 / v4 | **nothing** | untouched | — |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7 824 lines, 276 labels, 53 of 53 citations, 41 figures, all pass · `python3 tools/make_release_v5.py --dry-run` → 19 + 48 files, 9 holes, 1 finding (`\getDoctype`), 4 residue hits · `python3 tools/absorb.py status` → nothing to absorb. All unchanged from v5.0 (no text edited).
- INBOX: 7 rows `🔀 v5.1`, 3 rows `⏳` (the O003 FYIs), `## → v5` empty.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | the seven rows carry their resolution in the status cell; the O003 FYI notes (job O003) already tell the owners that their rows are resolved in v5 | nothing |

## Release

none (the author has not asked for one; `tools/make_release_v5.py --dry-run` is clean and ready).

## Closed

2026-09-25 14:34 · versions after: v2.28 · v3.100b · v4.2 · v5.1 · notes left: none (resolutions in the INBOX status cells; the O003 FYIs cover the owners) · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
