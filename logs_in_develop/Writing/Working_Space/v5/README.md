# v5 — the thesis in one folder (the aggregate · Advance Orchestra)

**Created:** 2026-09-25 13:43 (v5.0, Orchestra job O003) · **Authority:** the author · **Master:** [`thesis_v5.tex`](thesis_v5.tex) ·
**Revisions:** [`CHANGELOG.md`](CHANGELOG.md) (`## v5.N`, one per working pass) + [`changelogs/`](changelogs/) (one detailed, signed MD per revision) ·
**Init record:** [`inherited/INIT_STATE.json`](inherited/INIT_STATE.json), [`inherited/MANIFEST.md`](inherited/MANIFEST.md) ·
**Build:** [`tools/make_release_v5.py`](tools/make_release_v5.py) → `../RELEASE/output/` · **Jobs:** [`../Orchestra/CHANGELOG.md`](../Orchestra/CHANGELOG.md)

> **Start of every session (the Orchestra chat):** `python3 ../Orchestra/tools/orchestra.py status` · `python3 tools/absorb.py status` ·
> read `../cross_draft/INBOX.md`, section `## → v5`. Then `python3 tools/check.py` before and after every pass.

---

## What v5 is (author, 2026-09-25)

> "since the major parts of paper is been set, we will use less (but kept it) the Distr. Job and v2/3/4 sep. working flow. […]
> the Advance Orchestra now IS a v2+3+4 together, but RELEASE kept, then the build is the RELEASED folder […] RELEASE version
> use majorly datetime to distinguish. not vXX. (and dont delete old built latex) […] Maintain Changelog for advanced Orchestra
> each update. from v5.0 (init with v2,3,4 version) to v5.1…etc."

v2 wrote Chapters 1–4 (with the abstract, the preamble, the acronyms and `bibliography.bib`), v3 wrote Chapters 5–6, v4
wrote Chapters 7–8 and the appendix, each in its own chat with a one-way sync between them and a release tool that read
the three live drafts. **v5 is those three put together, once, and edited as one thesis from then on** by the Orchestra
chat. It was initialised from the drafts' live files at these revisions — the same files, byte for byte, that the golden
release of 2026-09-25 (`20260925_115125_…_GOLDEN_TEMPLATE`) was built from:

| in v5 | from | revision | how |
| :-- | :-- | :-- | :-- |
| `parts/00_preamble.tex` · `parts/01_frontmatter.tex` (the abstract) · `chapters/01–04` · `parts/99_backmatter.tex` (the acronym list) · `bibliography.bib` | v2 `thesis_v2.tex`, `bibliography.bib` | **v2.28** (2026-09-24) | the monolith split at its content markers (`v3/tools/split_v2.py`); v2's Ch 5–9 placeholders are not used |
| `chapters/05_setup.tex` · `06_results.tex` · `parts/00_preamble_v3.tex` · `bibliography_v3.bib` | v3 | **v3.100b** (2026-09-25) | copied |
| `chapters/07_conclusion.tex` · `08_discussion.tex` · `09_appendix.tex` · `app_long/uav_corridor_rules.tex` · `app_ntrial20_feasible.tex` · `parts/00_preamble_v4.tex` · `bibliography_v4.bib` | v4 | **v4.2** (2026-09-25) | copied |
| `figures/` | v4 (+ v3, v2 where v4 had no copy) | the exports of 2026-09-25 | copies; the store is `Data_Analysis/DA_in_Paper/figures` |

File by file with SHA-256 prefixes: [`inherited/MANIFEST.md`](inherited/MANIFEST.md). Every file is v5's own now: **there is
no inherited copy, no sync tool and no owner chat to notify.** The three legacy drafts are kept ("used less"): what the author
has changed there is absorbed with [`tools/absorb.py`](tools/absorb.py) (below).

## The rules (binding on the Orchestra chat when it works in v5)

1. **v5 is the thesis.** Edits are made here, directly, in the chapter and part files. No cross note, no ownership check —
   but **every working pass is one revision**: a `## v5.N` entry in `CHANGELOG.md` (newest first; the next number, never a
   letter) and a detailed `changelogs/v5.N_<YYYYMMDD>_<slug>.md`, signed (who, model, date, *not compiled*). The entry names
   the files and lines changed, the author's words that asked for it, the checks run, the INBOX rows resolved and the
   release built, if any. v5.0 is the init; v5.1 the first pass.
2. **The drafts' own rules travel with their chapters.** Ch 1–4: every equation names its source (a paper, or a repo file with
   lines) — "from memory" is never a source (`../v2/README.md`, *The sourcing rule*). Ch 5–6: every number names its evidence
   of record — the batch and the DA that published it, or a file with lines (`../v3/CHANGELOG.md` rules). Ch 7–8: no number of
   their own; every one restates Chapter 6 and names its table or section in a `\dataref` (`../v4/CHANGELOG.md` rules).
   No `p`-values or test names anywhere (author). Figures are never drawn or edited here.
3. **The style rules of the author hold in every sentence:** the request is never restated, justified or claimed as done in
   the thesis ([`../../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md`](../../Writing_Hints/HINT_20260920_prompt_is_not_thesis_text.md));
   the naming table [`../../Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md`](../../Auxiliary/Naming/TRANSLATION_20260914_dev_jargon_to_scientific.md);
   the storyline of every conclusion [`../GUIDE_20260924_results_storyline_author.md`](../GUIDE_20260924_results_storyline_author.md).
4. **Checks, every pass, never a compile:** `python3 tools/check.py` (labels, duplicates, citations, environments, braces,
   figures, nested inputs, the appendix switch, drafting-macro counts) and `python3 tools/make_release_v5.py --dry-run` (the
   whole thesis assembled and cleaned in memory: holes, bugs, residue). There is no TeX toolchain in this container;
   "checked" never means "compiled".
5. **The release is the build, only on the author's word:** `python3 tools/make_release_v5.py --job O### [--tag …] --note "…"`
   writes `../RELEASE/output/<YYYYMMDD_HHMMSS>_thesis_release_ORCH_v5.N[_TAG]/` (`latex/`, the Overleaf zip, the notes),
   marked as the Orchestra's build from v5, and prepends an entry to `../RELEASE/CHANGELOG.md`. **The date-time is the identity
   of a build**; the v5 revision is information. **No build is ever edited or deleted** — a content problem is fixed here, the
   revision recorded, the release rebuilt. Read `../RELEASE/README.md` before the first build of a session (front-matter
   decisions, the page limit 60–200 and the 60–80 orientation, `--attach-pdf` for the compiled count).
6. **The legacy drafts are sources, not targets.** Nothing under `../v2`, `../v3`, `../v4` is edited from v5. When the author
   takes a big job to an owner chat, that chat announces the change under `## → v5` in `../cross_draft/INBOX.md`; the
   Orchestra runs `tools/absorb.py status` → `diff` → `merge` (a three-way merge against `inherited/materials/`, the owner's
   file as v5 last absorbed it), checks, records a new v5.N and marks the row `🔀 v5.N`. A legacy Orchestra job *inside* a
   draft (kinds `edit` / `todo` / `sync`) follows `../Orchestra/README.md` as before.
7. **Figures** come only from `Data_Analysis/DA_in_Paper/figures` (`plotting/make_figs.py`, then a copy into `figures/`;
   `export_to_draft.py` knows the drafts by name — for v5 copy the changed files by hand or pass the folder path, and say so
   in the revision). `figures/EXPORTED.md` is v4's list of 2026-09-25, kept as the record of the init.
8. **Every pass is an Orchestra job:** `python3 ../Orchestra/tools/orchestra.py new-job "<title>" --kind advance` (or
   `absorb`) before, `close-job` after, with the revisions after the job (`v2 · v3 · v4 · v5`); `status --write` refreshes
   `../Orchestra/STATE.md`. The report to the author names: v5.N before → after, files and lines, checks, INBOX rows resolved,
   the build (if any), and what was left out and why.

## The procedure

```bash
cd logs_in_develop/Writing/Working_Space/v5
python3 ../Orchestra/tools/orchestra.py status                       # 1. where are we (v2 · v3 · v4 · v5, INBOX, builds)
python3 tools/absorb.py status                                       # 2. has a legacy draft moved? (merge before editing the same file)
python3 ../Orchestra/tools/orchestra.py new-job "<title>" --kind advance   # 3. the job O###
#                                                                    # 4. the edit: chapters/, parts/, *.bib
python3 tools/check.py                                               # 5. mechanical checks
python3 tools/make_release_v5.py --dry-run                           # 6. the whole thesis assembled in memory: holes, bugs, residue
python3 ../Orchestra/tools/orchestra.py bump v5 --job O### --title "<what>" --body-file <body.md> \
        --link changelogs/v5.N_<YYYYMMDD>_<slug>.md                  # 7. the ## v5.N entry (next number), then write that MD
python3 tools/make_release_v5.py --job O### --note "Orchestra O###: <why>"   # 8. only when the author asks for a release
python3 ../Orchestra/tools/orchestra.py close-job O### --summary "<one line>" [--release <build folder>]   # 9.
python3 ../Orchestra/tools/orchestra.py status --write               # 10. STATE.md
```

Body files for `bump` are written to the session scratchpad; the entry lives in `CHANGELOG.md`.

## Layout

```
thesis_v5.tex          master: preamble parts, \input order — structure only, no prose
parts/
  00_preamble.tex      v2's preamble: TUM metadata, packages, notation macros, the drafting macros (\hole, \srcnote, \ifsubmission)
  00_preamble_v3.tex   v3's additions (\dataref, \guard, \provisional, \flawed, \outdated, deadblock, presentation macros, placeins)
  00_preamble_v4.tex   v4's additions (the appendix long-data switch \ifappendixfull, \longdata, the bib registration)
  01_frontmatter.tex   title page (standalone) / template pages (release), the ABSTRACT, contents, \mainmatter
  99_backmatter.tex    the acronym list, lists of figures / tables, \printbibliography
chapters/01..09        the thesis; 09_appendix inputs app_long/uav_corridor_rules.tex and app_ntrial20_feasible.tex
bibliography.bib / bibliography_v3.bib / bibliography_v4.bib   three files, one bibliography (the build aggregates them)
figures/               copies of every figure the thesis uses (+ EXPORTED.md, v4's list at init)
inherited/
  INIT_STATE.json      the init: date-time, the v2 / v3 / v4 revisions, every file's source and SHA-256, the open INBOX rows
  MANIFEST.md          the same, readable
  ABSORB_STATE.json    (after the first absorb) which legacy revisions v5 carries now
  materials/           the owner files as v5 last absorbed them — the merge base of tools/absorb.py (v2_split/ = the split monolith)
tools/
  check.py             mechanical checks (v4's, re-pointed) — NOT a compiler
  make_release_v5.py   the build: RELEASE/output/<stamp>_thesis_release_ORCH_v5.N/ (imports ../RELEASE/tools/make_release.py)
  absorb.py            status / diff / merge / versions — a legacy change into v5
changelogs/            one signed MD per revision (v5.0_…, v5.1_…)
CHANGELOG.md           ## v5.N entries, newest first
```

## The build

```bash
python3 tools/make_release_v5.py --dry-run                                   # assemble + check in memory, write nothing
python3 tools/make_release_v5.py --job O004 --note "..."                     # output/<stamp>_thesis_release_ORCH_v5.N/{latex/, .zip, RELEASE_NOTES_<stamp>.md}
python3 tools/make_release_v5.py --tag GOLDEN                                # a tag after the revision
python3 tools/make_release_v5.py --appendix-short | --standalone | --no-cover | --acknowledgments | --no-zip | --no-log
python3 tools/make_release_v5.py --list                                      # every build in RELEASE/output/, legacy and v5
python3.14 tools/make_release_v5.py --attach-pdf ~/main.pdf --release ../RELEASE/output/<build>   # the compiled page count
```

The tool imports `../RELEASE/tools/make_release.py` and reuses its cleaner (comments and drafting macros removed, `\guard` /
`\provisional` unwrapped, switches resolved), its template assembly (`Template_DONT_CHANGE`, read-only), its checks, its page
model, its notes and its zip; only the sources and the name / marking differ. Compile on Overleaf (upload the zip; pdfLaTeX +
Biber, `main.tex`) or with `make pdf`; never here.

## State at init (v5.0, 2026-09-25 13:43)

- `tools/check.py`: 17 files, 7 824 lines, 276 labels, 53 of 53 citations, 41 figures — all mechanical checks pass. Drafting
  marks carried over: 1 `\hole` (the web link of the long-data section), 1 `\provisional`, 19 `\guard`, 78 `\srcnote`,
  60 `\dataref`, 2 `\flawed`, 1 `\longdata` — all to be gone before submission.
- `tools/make_release_v5.py --dry-run`: 19 text + 48 binary files, 9 holes, 1 finding (the standing `\getDoctype` patch:
  `Master's Thesis in \getDegree` → the template's pages append the degree themselves), 4 residue hits — the same as the
  golden release; a v5.0 test build reproduces the golden release's `latex/` byte for byte (`changelogs/v5.0_…`).
- `tools/absorb.py status`: nothing to absorb — every legacy source is at its init revision.
- **Carried over, open:** the seven INBOX rows open on 2026-09-25 (→ v2 one, → v3 three, → v4 three; resolved in v5.1);
  the author's decisions in `../v4/notes/OPEN_20260924_v4_open_items.md` (the web link of the long-data section; hiding its
  tables; the glossary; the raster figures); the parked answer to the first-reading review
  (`../RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/CLAUDE_ANSWER_…md`, job O002 — "don't care the audit for
  now", author); the standing `\getDoctype` finding (a one-word change in `parts/00_preamble.tex` when the author wants it).
