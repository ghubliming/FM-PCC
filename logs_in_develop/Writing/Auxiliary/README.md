# Auxiliary — writing notes

Markdown side-notes for the thesis. Not thesis content; things to keep straight
while writing.

| Note | What it holds |
| :-- | :-- |
| `NOTES_workspace_layout.md` | Where everything lives: template (read-only), working space, hints, reference papers in `aux_repo/PAPERS`. |
| `NOTES_tum_formatting_rules.md` | The operative subset of the TUM I6 rules — citation style, figures, math notation table, length, submission deliverables, talk formats. |
| `NOTES_dpcc_lineage.md` | The work is based on DPCC. What that means for the writing: the Diffuser→D3IL→DPCC→FM-PCC chain, DPCC's dual role as ancestor *and* baseline, and where to delineate inherited vs. own work. |
| `NOTES_paper_map.md` | Which reference paper belongs in which chapter/section; bibliography bootstrap. |
| `NOTES_open_questions.md` | Decisions still open, split into blocking vs. shaping. Start here each session. |
| **`NOTES_naming_and_rebuild.md`** | 🔴 **What the thesis calls things, and why the code's names are not it.** The `film_mode='v1'`-is-not-FiLM trap and the full artefact→thesis translation table. Read before writing any table header or any `sec:method:backbone` sentence. Also states why `logs_in_develop/Rebuild_repo/` is **never** cited. |
| **`Methodology_Sources/`** | 📁 **Source maps for the auxiliary apparatus the thesis owes** (TARGET §5) — UAV model & scenes, the PID/MJPC control stack, expert-data generation, the visual-aligning env, rendering & state injection, and constraint geometry incl. honest geometry. Each file says what is documented, where, and **what is missing**. |

Add `NOTES_notation_decisions.md` once the symbol convention is fixed (the working convention is
already fixed in `Working_Space/v2` Table 4.1 — write the note from it).

> **A name is a claim.** If a config flag, a folder tag or a class name asserts something the code
> does not do, the thesis must not repeat it. `NOTES_naming_and_rebuild.md` is the standing rule and
> carries the worked example.

> **Writing a `sec:method:*` or `sec:setup:*` section?** Start in `Methodology_Sources/` — those
> files exist so the method chapter does not have to be reconstructed from ~1000 dev-log MDs.
