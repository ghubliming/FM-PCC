# Writing workspace — layout and rules

Everything thesis-writing related lives under `logs_in_develop/Writing/`.

| Path | Role | Rule |
| :-- | :-- | :-- |
| `Template_DONT_CHANGE/` | Official TUM-Dev LaTeX thesis template (TUM I6 recommended). | **Read-only. Never edit, never build in place.** Copy it into `Working_Space/` when a real build is needed. |
| `Working_Space/` | All actual writing work. One subfolder per deliverable. | This is where user-requested writing work goes. |
| `Working_Space/Bone/` | Structural skeleton: title, ToC, chapter/section tree. | Current phase. |
| `Writing_Hints/` | Condensed TUM I6 submission guidelines + a general thesis-writing guide. | Reference; treat the linked official pages as authoritative. |
| `Auxiliary/` | Notes, decisions, checklists, paper maps — this folder. | Markdown only, no LaTeX. |

## Reference papers

`/workspaces/aux_repo/PAPERS/` (outside this repo, not version-controlled here):

| Subfolder | Contents |
| :-- | :-- |
| `in Proposual/` | The core proposal set: DPCC, Diffuser (Janner), Flow Matching (Lipman), FM guide, D4RL. |
| `Recommand_Paper/` | HardFlow (PDF **and full LaTeX source + `reference.bib`** in `HF/`), SafeFlowMatcher, physics-constrained FM sampling, HF-related optimal-control papers. |
| `auxiliary_papers/` | Everything else, grouped: `DGM/` (MeanFlow, Improved MeanFlow, AlphaFlow, Drifting, …), `D3IL_relevant/` (D3IL, X-IL, ACT, real-time action chunking), `ODE/` (solvers), `VLA/` (DiT, FiLM, π0), `Drone/` (UAV-Flow, CGD, PID), `Mujuco/`, `GraphModel/`. |

`.xopp` files next to some PDFs are Xournal++ annotation layers — the user's own
margin notes. Worth reading before summarising a paper; they mark what mattered.

**`Recommand_Paper/HF/reference.bib` is the single most useful bootstrap for the
bibliography** — it is a real, curated `.bib` from a paper in exactly this area.
Harvest from it rather than hand-typing entries.

## Cross-references into the repo

- `logs_in_develop/MASTER_TEST_HISTORY.md` — generation ↔ folder ↔ status index.
  Source of truth for *what was actually built and tested*. Do not edit it while
  writing; it is the engineering log, not a thesis artefact.
- `Data_Analysis/` — the plotting/aggregation pipeline that produces the numbers
  and figures the Results chapter will cite.
- `Slurm_Codes/logs/important_runs/important_runs.md` — which cluster runs the
  reported numbers actually came from. Needed for the reproducibility appendix.
