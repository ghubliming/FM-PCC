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
| **`Naming/`** | 📁 🟢 **THE naming folder — start at [`Naming/NAMING_20260910_master_table.md`](Naming/NAMING_20260910_master_table.md).** Three columns: *code token* │ *paper name* │ *what it really is → thesis name*, for every engine, arm, rule, backbone, geometry knob and embodiment, plus the five standing rules. **Canonical — where it and any note below disagree, it wins.** |
| `NOTES_naming_and_rebuild.md` | Rationale: the code-flag traps (`film_mode='v1'` is not FiLM), and why `logs_in_develop/Rebuild_repo/` is **never** cited. Superseded as a *table* by `Naming/`. |
| `NOTES_method_naming.md` | Rationale: the argument behind the method names — options weighed, and why *endpoint projection* beat the alternatives. Superseded as a *table* by `Naming/`. |
| **`NOTES_method_naming.md`** | 🏷️ **The thesis-facing names of the *methods*** (the companion covers code-flag traps). Records the author's 2026-09-10 position — mechanism over brand, claim only what is ours — the code-verified account of what arm C actually computes, and the recommendation **iterate projection / endpoint projection** in place of "HF-SLSQP". Decision still open. |
| **`Methodology_Sources/`** | 📁 **Source maps for the auxiliary apparatus the thesis owes** (TARGET §5) — UAV model & scenes, the PID/MJPC control stack, expert-data generation, the visual-aligning env, rendering & state injection, and constraint geometry incl. honest geometry. Each file says what is documented, where, and **what is missing**. |

Add `NOTES_notation_decisions.md` once the symbol convention is fixed (the working convention is
already fixed in `Working_Space/v2` Table 4.1 — write the note from it).

> **A name is a claim.** If a config flag, a folder tag or a class name asserts something the code
> does not do, the thesis must not repeat it. `NOTES_naming_and_rebuild.md` is the standing rule and
> carries the worked example; `NOTES_method_naming.md` applies the same test to the *methods* — and
> adds the second half of it: **credit only what is theirs, claim only what is ours.**

> **Writing a `sec:method:*` or `sec:setup:*` section?** Start in `Methodology_Sources/` — those
> files exist so the method chapter does not have to be reconstructed from ~1000 dev-log MDs.
