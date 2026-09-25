# CHANGELOG -- `Working_Space/RELEASE`

One entry per build of `tools/make_release.py`, newest first: when, which v2 / v3 / v4 revisions, what was produced, the page estimate, how many holes and findings. The full record of a build is the `RELEASE_NOTES_<stamp>.md` inside its output folder.

---

## 20260925_220823 -- 20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX

- **Built by the Orchestra from v5 v5.8** (v5.8 — 2026-09-25 · ROUND2 FIX · source: the second reading of v5.7 (ROUND2_FEEDBACK…), answer list R2-A · twenty factual and consistency corrections in Ch 5–8, six of them undoing regressions of v5.5–v5.7; the controller reference, the guard, the Steps definitions, the dominance definition; no number of record changed (Orchestra O023)) · job O024 · `v5/tools/make_release_v5.py`
- **v5 was initialised from:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** (initialised 2026-09-25 13:43:19, job O003); what changed in v5 since is in `v5/CHANGELOG.md`
- **Mode:** TUM template; tag ROUND2_FIX
- **Output:** `output/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX/latex/` (main.tex + 18 text files, 48 binary files), `output/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX.zip` (7842 KB)
- **Estimate:** ~179 pages (152-215), inside the 60-200 limit, but the uncertainty band touches it; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 9 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 277
- **Notes:** `output/20260925_220823_thesis_release_ORCH_v5.8_ROUND2_FIX/RELEASE_NOTES_20260925_220823.md`
- **Note:** ROUND2 FIX build — v5.8: the twenty DO items of the answer to the second reading of v5.7 (feedback/ROUND2_FEEDBACK_TO_RESPONSE_AND_V5.7_20260925.md → CLAUDE_ANSWER_to_ROUND2_…md, list R2-A): the quadrotor controller reference corrected (stop-and-go tracker, zero velocity feed-forward), the guard kept visible, the Steps population and dominance defined, four summaries narrowed to their tables, the unsupported projected-loop share deleted; six regressions of v5.5–v5.7 undone; no number of record, table cell or figure changed. Not in it: the abstract, the corridor wording, the optional cuts, the 10/10 cells (the author's word). The author: 'ship. and v5 changelog and release'.
- **Note:** Record: v5/changelogs/v5.8_20260925_round2_fix.md. First look when compiled: §5.5.3 (the controller sentence), §5.4.1 and Table 6.11 (Steps), §6.1.3 (the dominance definition), §8.1 and §8.3.

## 20260925_212528 -- 20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX

- **Built by the Orchestra from v5 v5.7** (v5.7 — 2026-09-25 · DATA FIX · source: the first-reading audit (M9 / §8-3), answer group D under the conservative verdicts — D4 only: the endpoint solves' non-convergence rate (0.1–1.1 %, a number of record in the DA) stated in one sentence of §6.1.2; D7–D11 rejected by the author, D1 / 2 / 3 / 5 not made (Orchestra O020)) · job O021 · `v5/tools/make_release_v5.py`
- **v5 was initialised from:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** (initialised 2026-09-25 13:43:19, job O003); what changed in v5 since is in `v5/CHANGELOG.md`
- **Mode:** TUM template; tag DATA_FIX
- **Output:** `output/20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX/latex/` (main.tex + 18 text files, 48 binary files), `output/20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX/20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX.zip` (7841 KB)
- **Estimate:** ~179 pages (152-215), inside the 60-200 limit, but the uncertainty band touches it; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 9 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 277
- **Notes:** `output/20260925_212528_thesis_release_ORCH_v5.7_DATA_FIX/RELEASE_NOTES_20260925_212528.md`
- **Note:** DATA FIX build — v5.7: D4 of the review answer, one sentence in §6.1.2 stating the endpoint solves' non-convergence rate (0.1–1.1 % over the six endpoint cells of Table 6.3) from the DA record; nothing run, no table cell or figure changed. D7–D11 rejected by the author (no cluster run for this review); D1 / 2 / 3 / 5 not made. On top of v5.6 (FACTUAL CHANGE), v5.5 (B WRITING), v5.4 (B-i) and the tier-A fixes. The author: 'build release as Data fix'.
- **Note:** Source: RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md via feedback/CLAUDE_ANSWER_…md §9 D (the done column); record: v5/changelogs/v5.7_20260925_review_D_data_fix.md.

## 20260925_212026 -- 20260925_212026_thesis_release_ORCH_v5.6_FACTUAL_CHANGE

- **Built by the Orchestra from v5 v5.6** (v5.6 — 2026-09-25 · FACTUAL CHANGE · source: the first-reading audit, answer group C under the conservative verdicts (14 DO + 7 minima) · two self-contradictions scoped or removed, three technical statements corrected against the code, the CI-MeanFM mechanism marked as an untested account, scope and specification qualifiers added; no number of record changed (Orchestra O017)) · job O018 · `v5/tools/make_release_v5.py`
- **v5 was initialised from:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** (initialised 2026-09-25 13:43:19, job O003); what changed in v5 since is in `v5/CHANGELOG.md`
- **Mode:** TUM template; tag FACTUAL_CHANGE
- **Output:** `output/20260925_212026_thesis_release_ORCH_v5.6_FACTUAL_CHANGE/latex/` (main.tex + 18 text files, 48 binary files), `output/20260925_212026_thesis_release_ORCH_v5.6_FACTUAL_CHANGE/20260925_212026_thesis_release_ORCH_v5.6_FACTUAL_CHANGE.zip` (7841 KB)
- **Estimate:** ~179 pages (152-214), inside the 60-200 limit, but the uncertainty band touches it; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 9 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 277
- **Notes:** `output/20260925_212026_thesis_release_ORCH_v5.6_FACTUAL_CHANGE/RELEASE_NOTES_20260925_212026.md`
- **Note:** FACTUAL CHANGE build — v5.6: the conservative C pass of the review answer (14 DO rows + 7 minima): two self-contradictions scoped or removed, three technical statements corrected against the code, the CI-MeanFM mechanism marked as an untested account, scope and specification qualifiers added; no number of record, table cell or figure changed. On top of v5.5 (B WRITING), v5.4 (B-i) and the tier-A fixes. The author: 'release, mark as Factual Change'.
- **Note:** Source: RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md via feedback/CLAUDE_ANSWER_…md §9 C (the done column); every edit: v5/changelogs/v5.6_20260925_review_C_factual_change.md. First look when compiled: §4.3.4 (the residual sentence), §4.6 (the keep-out row), §6.2.1 and §6.2.3 (the CI-MeanFM paragraphs), Fig. A.1's caption.

## 20260925_210607 -- 20260925_210607_thesis_release_ORCH_v5.5_B_WRITING

- **Built by the Orchestra from v5 v5.5** (v5.5 — 2026-09-25 · B WRITING (length) · source: the first-reading audit, answer groups B-ii / B-iii relaxed for length · 47 edits: duplicates cut in Ch 1, 3, 4, 5, 6, 7, three tables merged, two tables and three figures to the appendix; no claim or number changed (Orchestra O014)) · job O015 · `v5/tools/make_release_v5.py`
- **v5 was initialised from:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** (initialised 2026-09-25 13:43:19, job O003); what changed in v5 since is in `v5/CHANGELOG.md`
- **Mode:** TUM template; tag B_WRITING
- **Output:** `output/20260925_210607_thesis_release_ORCH_v5.5_B_WRITING/latex/` (main.tex + 18 text files, 48 binary files), `output/20260925_210607_thesis_release_ORCH_v5.5_B_WRITING/20260925_210607_thesis_release_ORCH_v5.5_B_WRITING.zip` (7840 KB)
- **Estimate:** ~177 pages (151-213), inside the 60-200 limit, but the uncertainty band touches it; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 9 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 277
- **Notes:** `output/20260925_210607_thesis_release_ORCH_v5.5_B_WRITING/RELEASE_NOTES_20260925_210607.md`
- **Note:** B WRITING build — v5.5, the length pass: the relaxed B-ii / B-iii rows of the review answer (contributions without result narratives, Related-Work duplicates, §5.2 duplicates, §6.4 compressed, Tables 6.14–6.16 merged, Tables 6.6 / 6.7 and Figures 6.1 (full grid) / 6.5 / 6.7 to the appendix) on top of v5.4 (B-i) and the tier-A bug fixes; no claim or number changed. The author: 'then release mark as B writing'.
- **Note:** Source: RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md via feedback/CLAUDE_ANSWER_…md §9 B-ii / B-iii (the done column); every edit: v5/changelogs/v5.5_20260925_review_B_writing_length.md. First look when compiled: the appendix's new sections B.1, B.4, B.5, B.8 and the merged Table in §6.3.3.

## 20260925_201059 -- 20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi

- **Built by the Orchestra from v5 v5.4** (v5.4 — 2026-09-25 · B WRITING FIX · source: the first-reading audit (answer group B-i, the low-risk wording items) · 27 wording edits in Ch 1, 2, 4, 6, 7, 8; no claim, no number, no table cell changed (Orchestra O010)) · job O011 · `v5/tools/make_release_v5.py`
- **v5 was initialised from:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** (initialised 2026-09-25 13:43:19, job O003); what changed in v5 since is in `v5/CHANGELOG.md`
- **Mode:** TUM template; tag TEMP_REVIEW_Bi
- **Output:** `output/20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi/latex/` (main.tex + 18 text files, 48 binary files), `output/20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi/20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi.zip` (7841 KB)
- **Estimate:** ~180 pages (153-215), inside the 60-200 limit, but the uncertainty band touches it; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 9 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 275
- **Notes:** `output/20260925_201059_thesis_release_ORCH_v5.4_TEMP_REVIEW_Bi/RELEASE_NOTES_20260925_201059.md`
- **Note:** TEMPORARY REVIEW MARKER — built from v5.4 (B WRITING FIX) so the author can read the 27 B-i wording edits in the PDF; the author: 'release it (as a temp marker, I will use for review now for the B-i changes in PDF)'. Not a milestone build: it carries the tier-A bug fixes (v5.2 + v5.3) plus the B-i wording; no claim, number or figure changed since …_ORCH_v5.3_BUGFIX_A.
- **Note:** Source of the edits: RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md via feedback/CLAUDE_ANSWER_…md §9 B-i; the list with before → after is v5/changelogs/v5.4_20260925_review_B_writing_fix.md.

## 20260925_152307 -- 20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A

- **Built by the Orchestra from v5 v5.3** (v5.3 — 2026-09-25 · **PURE BUG FIX** · source: the same first-reading audit (answer group A, items A6 and A9) · Figures 5.10 / 5.11 (clipped text) and 6.4 (palette) rebuilt in the DA store and re-exported; no text change (Orchestra O007)) · job O008 · `v5/tools/make_release_v5.py`
- **v5 was initialised from:** v2 **v2.28** · v3 **v3.100b** · v4 **v4.2** (initialised 2026-09-25 13:43:19, job O003); what changed in v5 since is in `v5/CHANGELOG.md`
- **Mode:** TUM template; tag BUGFIX_A
- **Output:** `output/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A/latex/` (main.tex + 18 text files, 48 binary files), `output/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A.zip` (7841 KB)
- **Estimate:** ~179 pages (153-215), inside the 60-200 limit, but the uncertainty band touches it; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 9 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 275
- **Notes:** `output/20260925_152307_thesis_release_ORCH_v5.3_BUGFIX_A/RELEASE_NOTES_20260925_152307.md`
- **Note:** PURE BUG FIX build — the first-reading audit's tier A applied: v5.2 (twenty text fixes in Ch 4–6: the lost FM-loss label, the printed audit identifier, four floats over the page, two captions, one retired wording, nine prose-vs-table contradictions) + v5.3 (Figures 5.10, 5.11, 6.4 fixed at the DA source). No content change against the golden release 20260925_115125; the float fixes are sized, not compiled — check pp. 61, 67, 122, 131 first.
- **Note:** Source: RELEASE/output/20260925_115125_…_GOLDEN_TEMPLATE/feedback/THESIS_FIRST_READING_REVIEW.md, answered in feedback/CLAUDE_ANSWER_…md §9 group A (status column = what this build carries).

## 20260925_115125 -- 20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE

- **Built on:** v2 **v2.28** (v2.28 — 2026-09-24 · the seven open cross notes applied (v3.73–v3.96, v4.0, v4.1)) · v3 **v3.100b** (v3.100b — 2026-09-25 · v3's copies of Ch 7–9 and the appendix archived (author: "also archive the v3 chapters for 07/08/09/appendix")) · v4 **v4.2** (v4.2 — 2026-09-25 · The ChatGPT audit of v4.1a applied (§15) on v3.100; author's long-data heading flag and planning-horizon item; figures re-exported; bundles and release rebuilt)
- **Mode:** TUM template; tag GOLDEN_TEMPLATE
- **Output:** `output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/latex/` (main.tex + 18 text files, 48 binary files), `output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE.zip` (7828 KB)
- **Estimate:** ~179 pages (152-215), inside the 60-200 limit, but the uncertainty band touches it; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 9 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 276
- **Notes:** `output/20260925_115125_thesis_release_v2.28_v3.100b_v4.2_GOLDEN_TEMPLATE/RELEASE_NOTES_20260925_115125.md`
- **Note:** Previous build compiled on Overleaf (2026-09-25 10:56): 185 pages incl. the blank Acknowledgments page and the overflow page ii, both gone in this build -> expect about 183 pages; inside 60-200 with little headroom, far above the 60-80 orientation.

## 20260925_110504 -- 20260925_110504_thesis_release_v2.28_v3.100_v4.2

- **DELETED 2026-09-25** (author: not the golden standard yet; rebuilt after the front-matter fixes). The files remain in git history (commits 65bca136 / 64505797). The compiled PDF of the first build (`Flow_Matching_Predictive_Control_with_Constraints - 2026-09-25T105619.791.pdf`, 185 pages) stays in `output/`.

- **Built on:** v2 **v2.28** (v2.28 — 2026-09-24 · the seven open cross notes applied (v3.73–v3.96, v4.0, v4.1)) · v3 **v3.100** (v3.100 — 2026-09-25 · The audit of v3.98 applied in Ch 5/6; the alignment halfspace added to the demonstration check; four figures rebuilt; `check.py` follows nested inputs) · v4 **v4.2** (v4.2 — 2026-09-25 · The ChatGPT audit of v4.1a applied (§15) on v3.100; author's long-data heading flag and planning-horizon item; figures re-exported; bundles and release rebuilt)
- **Mode:** TUM template
- **Output:** `output/20260925_110504_thesis_release_v2.28_v3.100_v4.2/` (main.tex + 19 text files, 48 binary files), `output/20260925_110504_thesis_release_v2.28_v3.100_v4.2.zip` (7829 KB)
- **Estimate:** ~163 pages (138-195), within the 60-200 limit; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 10 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 276
- **Notes:** `output/20260925_110504_thesis_release_v2.28_v3.100_v4.2/RELEASE_NOTES_20260925_110504.md`

## 20260924_223438 -- 20260924_223438_thesis_release_v2.28_v3.99_v4.1a_GOLDEN_TEMPLATE

- **DELETED 2026-09-25** (author: not the golden standard yet; rebuilt after the front-matter fixes). The files remain in git history (commits 65bca136 / 64505797). The compiled PDF of the first build (`Flow_Matching_Predictive_Control_with_Constraints - 2026-09-25T105619.791.pdf`, 185 pages) stays in `output/`.

- **Built on:** v2 **v2.28** (v2.28 — 2026-09-24 · the seven open cross notes applied (v3.73–v3.96, v4.0, v4.1)) · v3 **v3.99** (v3.99 — 2026-09-24 · The v4.0 and v4.1 cross notes applied (Ch 5/6 appendix references); bundle) · v4 **v4.1a** (v4.1a — 2026-09-24 · Synced to v3.99 (carrying v2.27); alias labels removed; bundle rebuilt)
- **Mode:** TUM template; tag GOLDEN_TEMPLATE
- **Output:** `output/20260924_223438_thesis_release_v2.28_v3.99_v4.1a_GOLDEN_TEMPLATE/` (main.tex + 19 text files, 48 binary files), `output/20260924_223438_thesis_release_v2.28_v3.99_v4.1a_GOLDEN_TEMPLATE.zip` (7826 KB)
- **Estimate:** ~159 pages (136-191), within the 60-200 limit; above the 60-80 guideline. NOT compiled (no TeX toolchain here).
- **Holes recorded:** 10 · **bugs/findings:** 1 · figures 41 (41 raster) · bibliography 53 entries, 53 cited · labels 276
- **Notes:** `output/20260924_223438_thesis_release_v2.28_v3.99_v4.1a_GOLDEN_TEMPLATE/RELEASE_NOTES_20260924_223438.md`

