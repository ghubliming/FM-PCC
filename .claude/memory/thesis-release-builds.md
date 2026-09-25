---
name: thesis-release-builds
description: "Thesis RELEASE builds — since 2026-09-25 built FROM v5 with Working_Space/v5/tools/make_release_v5.py (imports RELEASE/tools/make_release.py; folder output/<stamp>_thesis_release_ORCH_v5.N, date-time = identity, marked Orchestra); the legacy make_release.py (live v2/v3/v4 files) stays; submission-clean (every comment/flag stripped, one aggregated bib, TUM template); one folder per build (latex/ + zip + notes), every build kept, never deleted or edited; front-matter decisions of 2026-09-25; page limit 60–200"
metadata:
  type: project
---

Since 2026-09-24 the thesis is released with `logs_in_develop/Writing/Working_Space/RELEASE/tools/make_release.py`
(`python3 tools/make_release.py --tag ...`). Author's rules for a release:

- **Each chapter comes from its owner's live file** — v2's `thesis_v2.tex` (split content-based: abstract,
  Ch 1–4, acronyms, bib), v3's `chapters/05, 06`, v4's `chapters/07, 08, 09` + nested `app_*` — **never
  from a draft's `bundle/`** (may be stale) and never from the inherited copies (v3/v4 lag v2 by a revision).
- **The .tex must be submission-clean**: no `%` comments, no `\hole`/`\srcnote`/`\dataref`/`\guard` marks,
  no `\if...` build switches, no dev notes. Content is not the release's job — a content issue goes to the
  owning draft via `cross_draft/`, then rebuild.
- **The abstract is the author's**: copied as is, never edited.
- **Output layout (author, 2026-09-25):** `output/<stamp>_thesis_release_<v2>_<v3>_<v4>[_TAG]/` holds
  `latex/` (the project), `<build>.zip`, `RELEASE_NOTES_<stamp>.md`, later the attached PDF. **Every build is
  kept**; delete only on the author's word (the first two builds were deleted 2026-09-25 as "not golden yet").
- **Front matter (author, 2026-09-25, checked against the template's own PDF):** cover + title page both
  kept (`--no-cover` exists); the authorship declaration (`pages/disclaimer.tex`, bottom of its page) is
  mandatory; Acknowledgments dropped unless `RELEASE/front/acknowledgments.tex` has text; the title page is
  shrunk in the release copy (German title `\LARGE`, gaps 10/8/8/6 mm) because the real titles overflowed
  and left the faculty logo alone on page ii.
- **Page limit 60–200** (institute orientation 60–80): the tool estimates pages (calibrated: the first compile
  was **185 pages** on 2026-09-25) and appends `_PAGEWARN` when outside; a compiled PDF is recorded with
  `--attach-pdf`. Page count is close to the limit — say so when reporting.
- **Every build**: a row in `RELEASE/CHANGELOG.md` (date-time + v2/v3/v4 revisions) and the notes MD listing
  HOLES (critical lacking things) and bugs noticed.

- **Since 2026-09-25 the build comes from v5** (the aggregate; [[thesis-orchestra-role]]): `cd Working_Space/v5 && python3
  tools/make_release_v5.py --dry-run`, then `--job O### [--tag …] --note "…"` on the author's word. Same cleaner/checks/notes (the
  legacy tool is imported); name `<stamp>_thesis_release_ORCH_v5.N` — the DATE-TIME is the identity ("not vXX"); a marked entry
  in RELEASE/CHANGELOG.md with the v2/v3/v4 revisions v5 was initialised from. Verified: a v5.0 build reproduced the golden
  release's latex/ byte for byte. Content problems are fixed in v5 (a new v5.N), then rebuild.

**Why:** the author reviews and submits from the release, so it must be reproducible from the owners' files
in one command and must not silently carry drafting residue; the harness runs `rm` in the session cwd, so
deletions under `output/` must use absolute paths.

**How to apply:** for any "release / zip / clean build of the thesis" request, run the tool (read
`RELEASE/README.md` first) instead of hand-assembling; extend the tool when a new drafting macro or
switch appears in a draft. Related: [[thesis-draft-ownership]], [[master-thesis-writing-tum]],
[[no-unrequested-code-edits]].
