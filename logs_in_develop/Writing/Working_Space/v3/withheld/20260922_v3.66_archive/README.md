# Archive of v3.66 (2026-09-22) — what was taken out of the built chapters, verbatim

Author's instruction (v3.66 items 1 and 5): *remove* the endpoint-vs-per-step subsection of D3IL-avoiding and
*blank* the UAV-corridor data of Chapters 5/6, but keep everything for reference. Nothing here is built.

| file | what | why it left the chapter |
| :-- | :-- | :-- |
| `sec_6_1_2_6_endpoint_perstep_avoiding_v3.65.tex` | §6.1.2.6 as of v3.65: prose, Fig 6.5 `fig:projector-cost`, Table 6.3 `tab:hf-ladder` | rewritten at v3.66 around the single-seed, four-candidate K3/K10 comparison (`tab:avoiding-projectors`); the single-candidate ladder and the cost figure are no longer in the text |
| `uav_corridor_ch5_ch6_v3.65.tex` | every UAV-corridor passage of Ch 5 and Ch 6 (corridor_v2_slide corpus) | the corridor is being re-evaluated on a new grid (corridor v3, `data_status/PENDING_20260922_corridor_v3_run.md`); the chapters keep the structure with the data blanked |
| `figures/fig_avoiding_projector_cost.{svg,png}` | Fig 6.5 | see above |
| `figures/fig_uav_corridor_{tradeoff,paths,altitude}.{svg,png}` | Figs 6.9–6.11 | see above; the builders in `DA_in_Paper/plotting/builders/` still produce them from the v2 corpus |

The numbers in these files remain valid for the corpus they cite; they are not wrong, they are superseded by the
author's decision on scope. §6.4 (🔒 locked) still quotes corridor numbers from this corpus.
