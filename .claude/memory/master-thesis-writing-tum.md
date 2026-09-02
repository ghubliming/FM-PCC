---
name: master-thesis-writing-tum
description: Master's thesis writing task (TUM, School of CIT, Chair I6) — where the template, hints, reference papers and working space live, and the rules for each
metadata:
  type: project
---

The user is writing a **Master's thesis at TUM (School of Computation, Information
and Technology, Chair I6)** on this repo's FM-PCC work. Started 2026-09-02; as of
that date it is in the **initial/structuring phase** (skeleton only, no prose).

**Fixed title (user-set, verbatim, no hyphens, no subtitle):**
**Flow Matching Predictive Control with Constraints**
Never paraphrase or "improve" it. The user knows it under-covers the visual,
UAV and non-FM-engine (MeanFlow / α-Flow / diffusion) parts of the work and
chose it anyway; that gap is handled in the text by `sec:intro:scope`, not by
renaming. German rendering for the registration form is still unconfirmed.

**The work is based on DPCC** (Diffusion Predictive Control with Constraints,
L4DC 2025, arXiv 2412.09342; upstream at `/workspaces/aux_repo/dpcc`), which
itself builds on Diffuser's temporal U-Net and the D3IL environments. For the
thesis this matters because DPCC is *both* the ancestor of the codebase and the
reference baseline — the bone keeps those roles in separate chapters
(`sec:method:dpcc` = inherited-vs-own delineation; `sec:setup:baselines` =
comparison). Details in `Writing/Auxiliary/NOTES_dpcc_lineage.md`.

Locations:

- **Template — read-only:** `logs_in_develop/Writing/Template_DONT_CHANGE/`
  (TUM-Dev LaTeX thesis template). **Never edit or build in place.** Copy into
  `Working_Space/` for a real build.
- **All writing work:** `logs_in_develop/Writing/Working_Space/` — this is where
  user-requested writing deliverables go, one subfolder each.
  `Working_Space/Bone/thesis_bone.tex` is the structural skeleton (title, ToC,
  chapter tree with page budgets); it compiles standalone via a
  `\standalonetrue` switch and drops into a template copy when flipped false.
- **Notes/decisions:** `logs_in_develop/Writing/Auxiliary/` — layout, TUM I6
  formatting + math-notation rules, paper→chapter map, open questions.
- **Guidelines:** `logs_in_develop/Writing/Writing_Hints/` — condensed TUM I6
  submission rules + a general thesis guide.
- **Reference papers:** `/workspaces/aux_repo/PAPERS/` (outside the repo):
  `in Proposual/` (DPCC, Diffuser, Flow Matching, D4RL), `Recommand_Paper/`
  (HardFlow — incl. **full LaTeX source + `HF/reference.bib`**, the best
  bibliography bootstrap — SafeFlowMatcher, constrained sampling),
  `auxiliary_papers/` (`DGM/`, `D3IL_relevant/`, `ODE/`, `VLA/`, `Drone/`, …).
  Sibling `.xopp` files are the user's own Xournal++ annotations — read them.

Key format constraints: biblatex `alpha`/biber citation style; vector figures
only, every figure referenced in text; TUM I6 single-letter math notation
(macros `\vect`/`\matr`/`\trans`/`\sidx` are defined in the bone); target
60–80 pages; defence is 20 min + 5 min Q&A in English.

**Why:** the thesis is a separate deliverable from the code, with its own rules
(untouchable template, institute formatting) that are easy to violate by
accident.

**How to apply:** put thesis work in `Working_Space/`, notes in `Auxiliary/`,
never touch `Template_DONT_CHANGE/`. Check
`Auxiliary/NOTES_open_questions.md` before structural work — scope (which of
Gen0–Gen16 is in the thesis) and the RQ set are still open. Claims follow the
existing repo conventions: [[architecture-matched-beat-is-the-strong-claim]],
[[benchmark-hierarchy-who-beats-whom]], [[pareto-definition-of-good]],
[[da-target-is-best-baseline-variant]], [[hardflow-low-K-degeneracy]],
[[uav-budget-ms-not-a-goal]]. Repo navigation: [[fmpcc-dev-logs-navigation]].
