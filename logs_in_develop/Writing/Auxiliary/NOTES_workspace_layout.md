# Writing workspace — layout and rules

Everything thesis-writing related lives under `logs_in_develop/Writing/`.

| Path | Role | Rule |
| :-- | :-- | :-- |
| `Template_DONT_CHANGE/` | Official TUM-Dev LaTeX thesis template (TUM I6 recommended). | **Read-only. Never edit, never build in place.** Copy it into `Working_Space/` when a real build is needed. |
| `Working_Space/` | All actual writing work. One subfolder per deliverable. | This is where user-requested writing work goes. |
| `Working_Space/Bone/` | Structural skeleton: title, ToC, chapter/section tree. | Current phase. |
| `Working_Space/v1/`, `v2/` | The earlier drafts (`thesis_v*.tex`), each with its own README/CHANGELOG. | `v2` is the live **mathematics** draft and remains upstream for Chapters 1–4. |
| `Working_Space/v3/` | 🟢 **The live draft.** v2 split by chapter, plus the experiments: Chapters 5–8 are written. Master file `thesis_v3.tex`. | **v2 and v3 run in parallel.** v3 inherits Chapters 1–4 from v2 and owns 5–8; `v3/tools/sync_v2.py status` says whether v2 has moved, and `merge` three-way-merges it down. The dependency is one-way — a fix for both drafts is made in **v2** and synced down. See `v3/inherited/MANIFEST.md`. |
| `Working_Space/v3/plots/`, `v3/figures/` | The figure pipeline and its output. | Figures are **generated from the batch CSVs**, not drawn. New data lands ⇒ edit `v3/plots/sources.py` (the only file holding a path) and re-run `python3 plots/make_figs.py`. |
| `Working_Space/data_status/`, `fallback_target/` | Entry-readiness and fallback-claim analyses feeding the TARGET ladder. | Analysis, not thesis prose. |
| `Working_Space/future_work/` | Outlook material for `sec:conc:future`: perception→constraint front-end, zero-shot constraint re-tasking / sim2real. | 🟡 **Ideas, not results.** Deliberately outside `thesis_v2.tex`; see its README before citing anything from it. |
| `Writing_Hints/` | Condensed TUM I6 submission guidelines + a general thesis-writing guide. | Reference; treat the linked official pages as authoritative. |
| `Auxiliary/` | Notes, decisions, checklists, paper maps — this folder. | Markdown only, no LaTeX. |
| `Auxiliary/Naming/` | 🟢 The canonical **code │ paper │ thesis** name table. | Read before writing any table header or method sentence. |

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
- `logs_in_develop/Rebuild_repo/` — 🔴 **read-only insight, never a source.** A concept sketch for a
  unified rebuild: unbuilt, unstable, self-labelled *"CONCEPT / IDEAS — not a finalized plan"*.
  **Do not cite it in the thesis and do not point a chapter at it.** Its one contribution to the
  writing is a naming audit, which has been re-checked against the code and copied into
  `NOTES_naming_and_rebuild.md`. Use that file; re-read the original only when it moves.
