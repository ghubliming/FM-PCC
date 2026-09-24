---
name: thesis-release-builds
description: Thesis RELEASE builds (since 2026-09-24) — Working_Space/RELEASE/tools/make_release.py assembles the submission-clean thesis from the LIVE v2/v3/v4 files (never bundles), strips every comment/flag, one aggregated bib, TUM template; outputs are never edited; first build tagged GOLDEN_TEMPLATE
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
- **Page limit 60–200** (institute orientation 60–80): the tool estimates pages (no TeX here) and appends
  `_PAGEWARN` to the folder name when outside; a compiled PDF is recorded with `--attach-pdf`.
- **Every build**: a row in `RELEASE/CHANGELOG.md` (date-time + v2/v3/v4 revisions) and a
  `RELEASE_NOTES_<stamp>.md` in the output folder listing HOLES (critical lacking things) and bugs noticed.
- **Outputs are never edited**; the first release (2026-09-24) is tagged `GOLDEN_TEMPLATE` as the reference.

**Why:** the author reviews and submits from the release, so it must be reproducible from the owners' files
in one command and must not silently carry drafting residue.

**How to apply:** for any "release / zip / clean build of the thesis" request, run the tool (read
`RELEASE/README.md` first) instead of hand-assembling; extend the tool when a new drafting macro or
switch appears in a draft. Related: [[thesis-draft-ownership]], [[master-thesis-writing-tum]],
[[no-unrequested-code-edits]].
