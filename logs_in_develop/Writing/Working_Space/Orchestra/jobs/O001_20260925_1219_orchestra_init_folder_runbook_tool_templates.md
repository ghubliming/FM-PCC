# O001 — Orchestra init: folder, runbook, tool, templates, first snapshot

**Opened:** 2026-09-25 12:19 · **Kind:** init (edit | todo | release | sync | check) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · last release: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 0, → v3: 2, → v4: 2

## Asked

> "Build a Orchestra folder to init a working flow in logs_in_develop/Writing. You are the Orchestra for manage minor changes
> (or cross linked changes btw different chapters) in v2/v3/v4/Release, you work direct inside their own folders […]. Main Job is
> update v2/v3/v4 and update their Changelog (and bundle building for v3/4) on your tiny jobs. Problem is they are normally
> maintained by own agent, you interfere will cause confusion, so after changes, in their cross note inbox leave a msg for
> v2/3/4 if changed something. (the big changes I will direct into the own agent chat to solving). Then If need Release […],
> run a release with changelog mark that is run be Orchestra. Only release when I ask you to. Need to have a changelog to
> maintain each Orchestra Job with date-time, v2/3/4 current version. And maybe the case that there is no direct update of
> tex/thesis just distr. heavy jobs to each v2/3/4 as sep TODO list. This is also Orchestra Job. No concrete job to do, init
> and prepare ready for jobs." (author, 2026-09-25)

## Scope check (before touching anything)

- [x] Init job: no draft file touched; the checklist below is the standing one for edit jobs.
- [ ] **Minor or cross-linked** → the Orchestra does it here. **Big** (a section rewritten, a new result, a restructuring) → not here: becomes a TODO distribution (`new-todo`) for the owner chat.
- [ ] Every file to touch is in the owner's **owns** column of `../../DRAFT_OWNERSHIP.md` (v2: `thesis_v2.tex`, `bibliography.bib` · v3: `chapters/05, 06`, `parts/00_preamble_v3.tex`, `bibliography_v3.bib` · v4: `chapters/07, 08, 09`, `app_long/`, `parts/00_preamble_v4.tex`, `bibliography_v4.bib`).
- [ ] No inherited copy touched (v3's Ch 1–4 / v4's Ch 1–6, `inherited/`, inherited parts and `.bib`) — a change there is made upstream and synced.
- [ ] No figure drawn or edited (figures come only from `Data_Analysis/DA_in_Paper/` via `export_to_draft.py`).
- [ ] No README / CROSS_STATE / SYNC_STATE / MANIFEST of a draft edited; no `RELEASE/output/`, no `Template_DONT_CHANGE/`.
- [ ] A cross-linked change (a label, a term, a number restated elsewhere) — every other place listed below, each in its owner's file or as a note.

## Done

| draft | file(s) · lines / labels | change | entry in its CHANGELOG |
| :-- | :-- | :-- | :-- |
| — (Orchestra) | `Orchestra/README.md`, `CHANGELOG.md`, `STATE.md`, `tools/orchestra.py`, `templates/{JOB,CROSSNOTE,TODO}_template.md`, `jobs/`, `todo/` | the folder, the runbook (rules 1–10, the procedure, the conventions), the job changelog (date-time + v2·v3·v4 after each job), the tool (`status`, `new-job`, `bump`, `note`, `new-todo`, `close-job`), the three templates | this row (O001) |
| — (shared) | `cross_draft/README.md` (Rules) | one line: the Orchestra as a fourth note source, how its rows are closed | — |
| v2 / v3 / v4 | **nothing** | the drafts' files, changelogs, READMEs and state files are untouched (verified with `git status`) | — |
| — (memory) | `.claude/memory/thesis-orchestra-role.md` + index line | the role for later Orchestra sessions | — |

**Surveyed before building (read-only):** `DRAFT_OWNERSHIP.md`, `cross_draft/README.md` + `INBOX.md` (rows: → v2 31, → v3 28, → v4 43; open 0 / 2 / 2), the three drafts' README + CHANGELOG heads and `changelogs/`, `v3/tools/sync_v2.py` (regex reads **no** letter suffix → the v2 numbering rule), `v4/tools/sync_v3.py` and `RELEASE/tools/make_release.py` (both read letters), the bundle tools and logs, `RELEASE/README.md` + `CHANGELOG.md` + both changelog MDs, the sync states, the memories on ownership / release / style.

## Checks

- `RELEASE --dry-run` (2026-09-25 12:14): would write 19 text + 48 binary files; one standing BUG for v2 (`\getDoctype` carries `in \getDegree`; the tool patches the release copy). No Orchestra change affects it.
- v3 `tools/check.py`: 11 files, 6777 lines, 246 labels, 53/53 citations, 37 figures, all pass (13 references into v4's Ch 7–9 resolve in the release). v4 `tools/check.py`: 17 files, 7788 lines, 276 labels, 53/53, 41 figures, all pass. Bundles not rebuilt (nothing changed): newest v3 `thesis_v3_20260925_111503/111504`, v4 `thesis_v4_20260925_110246`.
- `tools/orchestra.py`: `status` run; `bump` dry-run gives v2.28 → v2.29, v3.100b → v3.100c, v4.2 → v4.2a (nothing written to the drafts); `note` ×3 written; `new-job` / `close-job` exercised by this job; `new-todo`, `note --from-todo`, `bump` (real insertion) and `close-job` (row shows the bumped v3.100c) exercised on a scratch copy of the workspace, nothing written to the real drafts.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| v2 | `cross_draft/to_v2/FROM_Orchestra_20260925_O001_orchestra_introduced.md` | `⏳ · Orchestra O001 · 2026-09-25` (FYI) | nothing now: the protocol (tagged changelog entry — v2: next number; note per job; TODO lists; ORCH_O### releases); mark ✅ when read |
| v3 | `cross_draft/to_v3/FROM_Orchestra_20260925_O001_orchestra_introduced.md` | same (FYI) | same (v3: next letter) |
| v4 | `cross_draft/to_v4/FROM_Orchestra_20260925_O001_orchestra_introduced.md` | same (FYI) | same (v4: next letter) |

## Release

none (the author did not ask; last build stays `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`).

## Closed

2026-09-25 12:23 · versions after: v2.28 · v3.100b · v4.2 · notes left: v2, v3, v4 (FYI) · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
