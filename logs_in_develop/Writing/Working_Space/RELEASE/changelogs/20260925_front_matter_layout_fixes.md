# RELEASE — per-build folders, front-matter decisions, title-page fit, recalibrated estimate · 2026-09-25

**Task (author):** delete the old outputs (not golden yet) and rebuild after fixes; from now on keep every
build and put zip and folder together in a subfolder; recheck the template and the writing hints for the
four confusions (blank Acknowledgments; where the "I confirm that this master's thesis is my own work…"
statement is and whether it is required; why two front pages; the page ii that holds only a logo); the
drafts may have moved since the last build.

## Facts established first

- The template's own compiled PDF (`Template_DONT_CHANGE/build/main.pdf`, 11 pages): p1 cover (title,
  author, logos) · p2 title page (German title, author / examiner / supervisor / submission date) · p3
  the declaration *"I confirm that this … thesis is my own work and I have documented all sources and
  material used. Munich, Submission date — Author"* at the **bottom** of the page · p4 a blank
  Acknowledgments page · p5 the abstract · p6 contents. So the declaration is in the template's build
  (bottom of page 3, easy to miss), and the blank Acknowledgments is the template's own.
- The author's Overleaf compile of the first release (`output/Flow_Matching_Predictive_Control_with_
  Constraints - 2026-09-25T105619.791.pdf`, read with pypdf): **185 pages**; p1 cover, p2 title page,
  **p3 = folio "ii" with no text** (the faculty logo alone — the title page overflowed: the real English
  title takes two `\huge` lines and the German title three, against one each in the template), p4 the
  declaration, p5 blank Acknowledgments, p6 abstract, p7–9 contents; Abbreviations p176, List of
  Figures p177, List of Tables p178–179, Bibliography p180–185.
- `Writing_Hints/tum_i6_thesis_submission_reference.md` says nothing about cover/title/acknowledgments;
  its only page guidance is the 60–80-page orientation and the warning above 100 pages.
- Drafts moved: v3 v3.99 → **v3.100b** (Ch 5/6 audit applied; v3's Ch 7–9 copies archived), v4 v4.1a →
  **v4.2** (audit applied on v3.100; the Extended Results heading now carries the author's long-data
  flag as plain text). v2 unchanged at v2.28. No new drafting macro, switch or environment appeared.
- The v4 session had already run the tool once (`20260925_110504_…_v4.2`, committed in 64505797).

## Changed — `tools/make_release.py`

- **Output layout:** `output/<build>/` now holds `latex/` (the project), `<build>.zip` (zip root =
  `latex/`), `RELEASE_NOTES_<stamp>.md`, `WARNING_PAGE_LIMIT.md` and attached PDFs. `--rezip`,
  `--attach-pdf`, `--list` follow the layout. Nothing is ever deleted by the tool (it refuses to overwrite).
- **Acknowledgments dropped by default** (optional page, no text): the `\input{pages/acknowledgments}`
  line and the page file are left out. Text in `RELEASE/front/acknowledgments.tex` (new folder) includes
  the page automatically with that text; `--acknowledgments` includes the empty page.
- **Cover page kept** (template default; the outer sheet of the bound copy); `--no-cover` drops it.
- **Declaration kept, always** (TUM's mandatory authorship statement; `pages/disclaimer.tex`).
- **Title page fit** (`fit_title_page`): in the release copy of `pages/title.tex` the German title is set
  in `\LARGE` and the vertical gaps go 20/15/15/10 mm → 10/8/8/6 mm; every other font and the order stay
  the template's. Estimated height 227 mm → ≈183 mm against ≈197 mm of type area. Each replacement is
  counted; a template change that breaks a pattern is reported in the notes instead of silently skipped.
- **Page model recalibrated** on the 185-page compile: 370 words per page (was 430), 4.6 lines per
  bibliography entry (was 3.3), front pages counted from the pages actually included. New estimate
  ~179 (152–215) for a build expected at ≈183.
- `--note TEXT` (repeatable): a line recorded in the notes and in the changelog row.

## Changed — files and outputs

- `README.md` rewritten for the layout, the flags and the front-matter decisions; `front/README.md` added.
- `output/20260924_223438_…_GOLDEN_TEMPLATE` and `output/20260925_110504_…_v4.2` (folders + zips) deleted
  on the author's word; their `CHANGELOG.md` rows are annotated **DELETED 2026-09-25** (files remain in git
  history, commits 65bca136 / 64505797). The author's compiled PDF in `output/` is untouched.
- **New golden build:** `output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/`
  (`latex/` 19 text + 48 binary files, zip 7.8 MB, notes). 9 holes (the five TODO metadata values incl.
  the unconfirmed German title, one `\provisional` in Ch 6, the web-link `\hole` and the `\longdata`
  banner of Appendix B.4, 41/41 raster figures), 1 finding (v2's `\getDoctype`, fixed in the release),
  4 residue lines (the TODO metadata). All mechanical checks pass; no comment, mark or switch left.
- `.claude/memory/thesis-release-builds.md` updated (layout, front-matter decisions, the 185-page anchor).

## Verification

- Scratch builds (`--no-log`, template and the new options) inspected before the golden build:
  front matter sequence, `pages/title.tex` after the fit, zip root, notes, `--list`.
- The deletion was first attempted with relative names while the harness kept the session cwd — nothing
  was removed (the names did not exist there); redone with literal absolute paths and checked before/after.
- Not compiled here (no TeX toolchain). The changes to the title page are geometric estimates; the next
  Overleaf compile confirms them (expected: no logo-only page, ≈183 pages).

*Written 2026-09-25 by Claude (Fable 5.1) for the author. Content untouched; nothing committed.*
