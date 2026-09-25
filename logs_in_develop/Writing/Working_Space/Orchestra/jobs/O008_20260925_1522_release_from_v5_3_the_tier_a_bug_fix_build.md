# O008 — Release from v5.3: the tier-A bug-fix build (first-reading audit, answer group A; v5.2 text fixes + v5.3 figure fixes)

**Opened:** 2026-09-25 15:22 · **Kind:** release (advance | absorb | edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · **v5 v5.3** (the thesis) · last release: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 1, → v4: 1, → v5: 0

## Asked

> "release it. (mark from v5.2/bug fix A)" (author, 2026-09-25) — after "end of A tier?": tier A of the first-reading audit is done but for A11 (the author's metadata).

## Scope check (before touching anything)

- [x] Kind `release`: `cd v5 && python3 tools/make_release_v5.py --job O008 --tag BUGFIX_A --note …` on the author's word; no draft file touched.
- [ ] **Advance (kind `advance` / `absorb`):** the edit is made in `../../v5/` — the Orchestra owns it (no ownership check, no cross note); one `## v5.N` entry in `v5/CHANGELOG.md` + `v5/changelogs/`, `tools/check.py`, `tools/make_release_v5.py --dry-run`; INBOX rows resolved there are marked `🔀 v5.N`. The items below are for the legacy flow.
- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [ ] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [x] Under `RELEASE/output/` only the new build folder was written (the tool); no existing build touched; `Template_DONT_CHANGE/` read only (the tool copies it).
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| RELEASE | `output/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A/` — `latex/` (main.tex + 18 text files, 48 binary files), the Overleaf zip (7 841 KB), `RELEASE_NOTES_20260925_152307.md` | built from v5 **v5.3** (v2.28 · v3.100b · v4.2 carried); tag `BUGFIX_A`; two notes name the source audit and the pure-bug-fix kind | `RELEASE/CHANGELOG.md` entry `20260925_152307` (marked Orchestra / v5) |
| v5 | `CHANGELOG.md` (v5.3 entry), `changelogs/v5.3_…md` | the build named in the revision it was built from | — |
| RELEASE feedback | `…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md` | header: **Released** line naming this build | — |
| v2 / v3 / v4 | **nothing** | untouched | — |

## Checks

- Build: 19 text + 48 binary files; page estimate ~179 (153–215), inside the 60–200 limit; 9 holes (the front-matter metadata among them), 1 finding (the standing `\getDoctype` patch), 4 residue hits — the golden build's counts.
- `diff -rq` against the golden release's `latex/`: 6 file(s) differ — `chapters/04_method.tex`, `chapters/05_setup.tex`, `chapters/06_results.tex`, `figures/fig_aligning_projected_tradeoff.png`, `figures/fig_expert_aligning.png`, `figures/fig_expert_uav.png`; no file only on one side. Every differing file is one that v5.2 / v5.3 changed; nothing else moved.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| — | none | none | — |

## Release

`RELEASE/output/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A/` — the tier-A bug-fix build (the author: "release it. (mark from v5.2/bug fix A)"); to be compiled on Overleaf; check pp. 61, 67, 122, 131 (the float fixes) and pp. 29–31 (the three former `??`) first.

## Closed

2026-09-25 15:24 · versions after: v2.28 · v3.100b · v4.2 · v5.3 · notes left: — · release: `20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A` · signed: Orchestra (Claude Fable 5.1, Claude Code)
