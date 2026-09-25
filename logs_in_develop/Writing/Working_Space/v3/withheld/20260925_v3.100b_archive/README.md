# withheld/20260925_v3.100b_archive

v3's frozen copies of the files after Chapter 6, archived at **v3.100b** (author, 2026-09-25: "also archive the
v3 chapters for 07/08/09/appendix"), moved here unchanged from `chapters/`:

| file | state |
| :-- | :-- |
| `chapters/07_discussion.tex` | v3's Ch 7 as handed to v4 at v3.98 (v4 now has `07_conclusion.tex`) |
| `chapters/08_conclusion.tex` | v3's Ch 8 as handed over (v4 now has `08_discussion.tex`) |
| `chapters/09_appendix.tex` | v3's appendix at v3.98, incl. the dead twenty-episode block v4 dropped |
| `chapters/app_ntrial20_feasible.tex` | the twenty-episode section the appendix input; byte-identical to v4's |

Ch 7–9 are v4's since the v3.98 handover. `thesis_v3.tex` no longer inputs them; `tools/check.py` checks Ch 5/6
references into them against v4's live labels; the bundle ends after Chapter 6. Not input by any master.
