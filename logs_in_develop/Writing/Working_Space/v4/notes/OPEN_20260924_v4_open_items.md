# OPEN — what v4.0 leaves for the author (2026-09-24)

Holes are `\hole{…}` in the text (red in the annotated bundle, nothing in the clean one); decisions are
taken in the draft and can be reversed.

## Holes in the text (1)

| where | what |
| :-- | :-- |
| `chapters/09_appendix.tex`, Appendix B lead paragraph | **the web link** to the full record of the long-data section (UAV-corridor under every selection rule; URL). Until it exists its tables are printed (`\appendixfulltrue`). |

~~The simulator name in Future Work~~ — resolved at v4.1: NVIDIA Isaac Sim (author).

**Structure since v4.1 (author):** Ch 7 = Conclusion (summary, RQ answers); Ch 8 = Discussion
(limitations, towards deployment, future work). The v4.0 Interpretation, Negative Results and Threats
sections are archived in `withheld/20260924_v4.1_archive/`; the author called them duplicates of Ch 6
and "suicide". Do not bring them back.

## Switches and decisions taken (reversible)

1. **Long-data tables printed** (`\appendixfulltrue` in `parts/00_preamble_v4.tex`). To hide them:
   flip to `\appendixfullfalse` (or build with `--appendix-short`); the section keeps its reading and
   its alias labels. Since v4.1 only UAV-corridor under every selection rule is a long-data section
   (the author kept five sections of Appendix B and archived the other four); its `\longdata` banner
   stays until submission.
1b. **Appendix A is titled *Supplementary Material*** (the author rejected "Derivations"); it holds the
   sampling laws and, since v4.1, the quadrotor dimensions figure. Another title is one word to change.
1c. ~~Alias labels~~ — removed at v4.1a after v3.99 re-pointed every Ch 5/6 reference.
1d. **v2.28 is not in v4 yet**: it reaches v4 after v3's `sync_v2.py merge`; then `sync_v3.py merge` and a
   rebuild. Until then Ch 1's outline sentence (v2.27) still reads the old chapter order.
2. **The twenty-episode dead block is dropped** (v3.67 dead; `app:avoiding-twenty` replaces it, v3.92).
   Text in `withheld/20260924_v4.0_archive/`; alias label kept. If it comes back, re-insert the archive
   and delete the alias `\label{app:avoiding-twenty-episode}` after the ntrial20 input.
3. **Derivations kept** (v3.58 keep instruction supersedes `notes.txt`).
4. **Reproducibility = facts only**; the removed prose and the two tables are in `notes/MOVED_…tex`.
5. **Ch 8 §8.2 *Towards Deployment*** (`sec:disc:practice`) carries the author's "real-world practice"
   and, since v4.1, the selection-rule paragraph from `notes.txt` (one candidate → identical rules;
   four → projection ×3.2–4.9; temporal consistency for the average-velocity models).
6. **References from Ch 7–8 into Ch 1–4 are section-level** (a Ch 4 table or figure collapses in the
   working bundle; sections survive as headings).

## What the other drafts owe (INBOX rows written)

- **v3:** re-point the three `\autoref{app:avoiding-twenty-episode}` (05_setup.tex ×2, 06_results.tex
  ×1) to `app:avoiding-twenty` and rewrite 05_setup.tex:917–922 (the baseline's twenty-episode run is no
  longer "kept for the record" in the appendix); re-point or drop the 05_setup.tex:1091–1092 sentence
  "how its properties bear on reading them is given in `app:repro`" (the reading is now Ch 7
  `sec:disc:threats`).
- **v2:** the abbreviation findings (`notes/AUDIT_…`): MJPC vs "MuJoCo MPC"; PD undeclared. Since v4.1
  also Ch 1's outline sentence (Ch 7 concludes, Ch 8 discusses; future work is in the Discussion).

## Later, before submission

8. **Glossary reminder** (`REMINDER_20260914_glossary_appendix.md`): v4 recommends **no separate glossary
   appendix** — every term is defined at first use, Table 4.1 holds the notation (v2.25 decision), the
   acronym list prints in the front matter, and the two meanings of $K$ are in Table 4.1 and Ch 6 §6.1.1.
   The author decides; if wanted, it is a short `\addchap` after the acronyms.
9. **`bibliography_v4.bib`** is empty; uncomment its `\addbibresource` in `00_preamble_v4.tex` with the
   first entry (an empty resource is not given to biber).
10. **TUM-template merge:** `00_preamble_v4.tex` (the switch and `\longdata`) folds into `settings.tex`
    with v3's and v2's additions; the `\longdata` banners and both holes must be gone.
11. **`fig_hardflow_endpoint_generation`** is v2's vendored figure and is not in the DA_in_Paper store
    (as in v3); `figures/` carries v3's copy.
12. **Page budget.** Ch 7 ≈ 6 pages, Ch 8 ≈ 4 at 11 pt (estimate, not compiled); the thesis as a whole is
    well above the 60–80-page guide because of Ch 5–6 — an author's call, not v4's.
