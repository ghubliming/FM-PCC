# From v3 to v4 — Ch 7, Ch 8 and the appendix at **v3.98**, copied 2026-09-24 20:19

**Author, 2026-09-24:** "Handle the Chap 7, 8 and Appendix from current v3, give to v4 folder in a new folder, mark the
v3 version (i.e. copy them to a v4 folder with current v3 versions and date-time, not delete ours, the v4 will start to
build)."

- **This folder is a copy.** v3 keeps its own files, unchanged, in `v3/chapters/`.
- **Nothing else under `v4/` was touched.** `notes.txt` and `REMINDER_20260914_glossary_appendix.md` are as before.
- **An exception to `DRAFT_OWNERSHIP.md`** ("v3 never edits anything under v4/"), made at the author's request. It
  applies only to the creation of this folder.

## Files (byte-identical to `v3/chapters/` at v3.98)

| file | lines | sha256 (prefix) | state |
| :-- | --: | :-- | :-- |
| `chapters/07_discussion.tex` | 21 | `062a1847b716` | **Headings and labels only**: Interpretation, Negative and Inconclusive Results, Threats to Validity, Limitations, plus a `\hole{Written in v4.}`. Ch 4 references these labels. v3's earlier discussion prose is in `cross_draft/to_v4/FROM_v3_20260914_discussion_prose.tex`. |
| `chapters/08_conclusion.tex` | 81 | `f15ff4e05b69` | **v3's concise draft of 20-09**: Summary, Answers to the Research Questions, Future Work, with three `\hole`s ("Refine in v4", "Written in v4"). **Stale against Ch 6 as of v3.98.** Rewrite it from the notes below, not from this text. |
| `chapters/09_appendix.tex` | 1076 | `1e11470b4722` | A Derivations; B Extended Results; C Reproducibility (two `\hole`s: compute total, job ids in `tab:corpora`). |
| `chapters/app_ntrial20_feasible.tex` | 149 | `ddde881204c5` | D3IL-avoiding at twenty episodes per seed (`app:avoiding-twenty`). `09_appendix.tex` inputs it as `\input{chapters/app_ntrial20_feasible}`, just before `\chapter{Reproducibility}`. Curated by another agent; used as is. |

**Appendix B, in order:**
- D3IL-avoiding at every evaluated budget.
- D3IL-avoiding executed paths.
- D3IL-aligning executed paths.
- UAV-corridor, the flights along the corridor.
- The retired twenty-episode block (`deadblock`, `app:avoiding-twenty-episode`).
- UAV-corridor under every selection rule.
- UAV-s-curve under every selection rule.
- UAV-s-curve at the budgets on disk (`\outdated`).
- The ntrial20 section.

## What these files need in order to build

- **Macros**, from `v3/parts/00_preamble_v3.tex`: `\dataref`, `\guard`, `\provisional`, `\outdated`, `\flawed`,
  `deadblock`, `\mainconfig`, `\selectedmark`, `\baselinemark`, `\todofigure`, the column type `L`.
- **From v2's preamble** (`v3/parts/00_preamble.tex`): `\hole`, `\srcnote`, `\ifsubmission`.
- **Labels in Ch 1–6.** The appendix and Ch 8 `\autoref` labels there (`tab:avoiding-dpcc-protocol`,
  `sec:res:ablations`, `sec:setup:metrics:spread`, `tab:uav-scurve`, …). They resolve in a build that also holds v3's
  Ch 5–6 (or v4's merge of them).
- **Figures** in the appendix: `fig_avoiding_paths`, `fig_aligning_paths`, `fig_uav_corridor_side`,
  `fig_platform_x2_dimensions`. Figures are never copied by hand: once v4 has its master file, run
  `python3 Data_Analysis/DA_in_Paper/plotting/export_to_draft.py v4`, which reads the `\includegraphics` of the draft.
- **Bundle tool:** `v3/bundle/make_bundle.py` (two variants since v3.94) is written for `v3/`. A v4 bundle needs its own
  copy, with `V3`/`MASTER` pointed at `v4/`.

## Read before writing Ch 7 and Ch 8

1. `Working_Space/GUIDE_20260924_results_storyline_author.md`: the author's storyline per environment, **the thesis in
   three steps** (v3.97) and **the general conclusion** (v3.96).
2. `Working_Space/INSIGHTS_20260924_results_per_environment.md`: the insights of each Ch 6 section, with sources.
3. `cross_draft/to_v4/FROM_v3_20260924_v3.96_general_conclusion_handover.md`, including its v3.97 addendum: the state of
   Ch 5–6, and three corrections to earlier notes:
   - UAV-corridor RQ2 is a trade-off;
   - D3IL-aligning is 0/10 in position, not "box moved";
   - UAV-s-curve: the controller matters, and ours is the better one.
4. `cross_draft/INBOX.md`, section → v4. Newer rows supersede older ones.

## Open decisions (the author's), carried over

- **Drop the retired twenty-episode block** (`app:avoiding-twenty-episode`) now that `app:avoiding-twenty` replaces it.
  Then re-point its three references: `05_setup.tex` twice, `06_results.tex` once. Their lines are in the header of
  `app_ntrial20_feasible.tex`.
- **Keep or drop** the `\outdated` "UAV-s-curve at the budgets on disk".
- **The two Reproducibility `\hole`s:** bookkeeping, not runs.
- **`DRAFT_OWNERSHIP.md`** still lists the appendix as v3's "for now". Whether it now moves to v4 is the author's call.

Copied by Claude (Opus 5.5, Claude Code, v3) · 2026-09-24 20:19.
