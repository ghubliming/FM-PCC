# O003 — Advance Orchestra: v5 aggregate initialised from v2.28 / v3.100b / v4.2 (thesis in one folder, release from v5)

**Opened:** 2026-09-25 13:35 · **Kind:** init (the Advance flow; later jobs in v5 are `advance` / `absorb`) · **Asked by:** the author
**Versions at open:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** · v5 — (did not exist) · last release: `20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE`
**Sync chain at open:** v3 carries v2.27 (v2 moved); v4 carries v3.100 / v2.27 (v3 moved) · **INBOX open rows:** → v2: 1, → v3: 3, → v4: 3

## Asked

> "DONT care the Audit for now. 1. since the major parts of paper is been set, we will use less (but kept it) the Distr. Job and
> v2/3/4 sep. working flow. 2. Init a Aggregate folder inside Orchestra Folder and *a working flow* (as *Advance Orchestra*,
> distinguish our current legacy orchestra workflow. **IF put into another location is better allow to do it! even like create a
> NEW folder near the current Orchestra! ie not inside**), copy the current v2/3/4 into your own materials/chapters folder. record
> the init date-time and the v2/3/4 version, and if pending inbox infos for v2/3/4 version (ie. if unfinished jobs). 3. Then we
> will use here Orchestra as the major workspace, from the thesis→release. no more v234, all together (ie the Advance Orchestra
> now IS a v2+3+4 together, but RELEASE kept), then the build is the RELEASED folder (if build from us in the Release folder need
> mark is from the Orchestra, so may be also need special script to direct build from the Orchestra Aggregate folder not the
> current RELEASE script from v234), RELEASE version use majorly datetime to distinguish. not vXX. (and dont delete old built
> latex) 4. Maintain Changlog for advanced Orchestra each update. from v5.0 (init with v2,3,4 version) to v5.1…etc. (v5 record
> in the Writing folder notes near to v2,3,4) 5. First job is to resolve any the INBOX for v2,v3,4. to v5.1" (author, 2026-09-25;
> point 5 is job O004)

## Scope check (before touching anything)

- [x] **Kind init:** a new draft folder and its workflow; no legacy draft file edited (v2 / v3 / v4 are read and copied only).
- [x] **Location:** `Working_Space/v5/`, a sibling of v2 / v3 / v4 / Orchestra — the author allowed a folder near the Orchestra;
      the reasons (the v5.N scheme, the changelog next to the drafts', the tools' revision reading) are in `v5/changelogs/v5.0_…`.
- [x] No figure drawn or edited (copies only, from the drafts' exported sets); `RELEASE/output/` untouched (the equivalence test
      build went to the session scratchpad with `--no-log` and was deleted); `Template_DONT_CHANGE/` read only.
- [x] Shared files edited, each saying so: `DRAFT_OWNERSHIP.md` (the author re-divided the work), `cross_draft/README.md`,
      `cross_draft/INBOX.md` (legend, `## → v5`), `RELEASE/README.md`; the Orchestra's own runbook, changelog, tool and template.
- [x] No README / state file of a legacy draft edited.

## Done

| where | file(s) | change | record |
| :-- | :-- | :-- | :-- |
| **v5** (new) | `thesis_v5.tex`, `parts/` (5), `chapters/` (11 incl. `app_long/`), the three `.bib`, `figures/` (78), `inherited/INIT_STATE.json` + `MANIFEST.md` + `materials/`, `tools/check.py`, `tools/make_release_v5.py`, `tools/absorb.py`, `README.md`, `CHANGELOG.md`, `changelogs/v5.0_20260925_init_from_v2.28_v3.100b_v4.2.md` | the aggregate of v2.28 (split at its markers) · v3.100b · v4.2; the init record with date-time, revisions, SHA-256 and the seven open INBOX rows; the build script (imports the legacy release tool; `…_thesis_release_ORCH_v5.N`); the absorb tool for later legacy changes; the runbook | `v5/CHANGELOG.md` **v5.0** |
| Orchestra | `README.md` (the Advance section, the session line, rule 7, the map), `CHANGELOG.md` (column `v2 · v3 · v4 · v5`, kinds), `tools/orchestra.py` (v5 in status / versions, `bump v5` = next number, `--link`, kinds `advance` / `absorb`, `## → v5` rows), `templates/JOB_template.md` | the legacy flow kept, marked as such; the Advance flow added | this file |
| Working_Space | `DRAFT_OWNERSHIP.md` (v5 section + revised line), `cross_draft/README.md` (`to_v5/` rule), `cross_draft/INBOX.md` (🔀 status, `## → v5` section, folder `to_v5/`), `RELEASE/README.md` (builds from v5) | the map and the rules say where the thesis lives now | — |
| v2 / v3 / v4 | **nothing** | untouched (read and copied) | — |

## Checks

- v5: `python3 tools/check.py` → 17 files, 7 824 lines, 276 labels, 53 of 53 citations, 41 figures, **all pass** ·
  `python3 tools/make_release_v5.py --dry-run` → 19 text + 48 binary files, 9 holes, 1 finding (the standing `\getDoctype` patch), 4 residue hits — the golden release's counts ·
  **equivalence:** a test build (scratchpad, `--no-log`, tag EQTEST, stamp 20260925_142637, deleted afterwards) gave a `latex/` tree **byte-identical** to the golden release's (67 files, `diff -rq` clean; estimate ~179 pages) ·
  `python3 tools/absorb.py status` → nothing to absorb, every legacy source at its init revision.
- Orchestra: `python3 tools/orchestra.py status` → v5 **v5.0** shown as the thesis; `## → v5` parsed (0 rows); the three FYI rows inserted.
- Incident, recorded: the first patch of `tools/orchestra.py` truncated the file (a wrong `newline` argument made `open()` fail after
  truncating); restored with `git checkout` from HEAD 8db0cdc2 (identical to the session's version) and re-patched; syntax and `status` verified.
- **Not compiled** (no TeX toolchain here). Nothing committed.

## Messages left (one per draft changed, or per draft that gets a TODO)

| to | note file | INBOX row | what it asks |
| :-- | :-- | :-- | :-- |
| v2 | `to_v2/FROM_Orchestra_20260925_O003_advance_orchestra_v5.md` | ⏳ Orchestra O003 · 2026-09-25 (FYI) | nothing now; nothing of v2 changed; how a later v2 job reaches v5 (a `## → v5` row, absorbed by the Orchestra) |
| v3 | `to_v3/FROM_Orchestra_20260925_O003_advance_orchestra_v5.md` | ⏳ Orchestra O003 · 2026-09-25 (FYI) | the same for v3; v2.28 need not be merged for a release any more |
| v4 | `to_v4/FROM_Orchestra_20260925_O003_advance_orchestra_v5.md` | ⏳ Orchestra O003 · 2026-09-25 (FYI) | the same for v4; v3.100b need not be merged for a release any more |

## Release

none — the equivalence test build was written to the session scratchpad with `--no-log` and deleted; `RELEASE/output/` and `RELEASE/CHANGELOG.md` untouched.

## Closed

2026-09-25 14:32 · versions after: v2.28 · v3.100b · v4.2 · v5.0 · notes left: v2, v3, v4 (FYI: the Advance flow; nothing asked) · release: — · signed: Orchestra (Claude Fable 5.1, Claude Code)
