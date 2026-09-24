# Archive — v4.1 (2026-09-24): what the restructuring of the endings dropped and moved

Author, 2026-09-24: the Conclusion becomes Chapter 7 with the summary and the research-question answers
only; the Discussion becomes Chapter 8 with Limitations, Towards Deployment and Future Work; the
Interpretation, Negative and Inconclusive Results and Threats to Validity sections of the v4.0
Discussion are dropped as duplicates of Chapter 6 and "not a must".

| file | what it is |
| :-- | :-- |
| `07_discussion_v4.0.tex` | the complete v4.0 Discussion (411 lines). **Dropped:** §7.1 Interpretation (six subsections), §7.2 Negative and Inconclusive Results, §7.3 Threats to Validity. **Kept:** §7.4 Limitations, now `chapters/08_discussion.tex` §8.1 (one scope sentence on the seeds added). The selection-rule paragraph of §7.1.4 survives, compressed, as one paragraph of §8.2 Towards Deployment. |
| `08_conclusion_v4.0.tex` | the complete v4.0 Conclusion (168 lines). §8.1–8.2 are now `chapters/07_conclusion.tex`, unchanged; §8.3 Towards Deployment and §8.4 Future Work are `chapters/08_discussion.tex` §8.2–8.3 (labels `sec:disc:practice`, `sec:disc:future`; the simulator hole resolved to NVIDIA Isaac Sim). |

Nothing here is built. Labels `sec:disc:interpretation`, `sec:disc:negative`, `sec:disc:threats`
existed only in the dropped text; no file of Chapters 1–6 references them (checked).

## Second round (author's point 11): Appendix B trimmed to five sections

| file | what it is |
| :-- | :-- |
| `09_appendix_v4.1a_before_B_trim.tex` | the appendix as it stood after the first v4.1 round (A Derivations, B nine sections, C Reproducibility with the X2 figure). |
| `appendix_B_dropped_sections.tex` | the four dropped sections verbatim: D3IL-avoiding at every evaluated budget, endpoint projection with one candidate plan, UAV-s-curve under every selection rule, UAV-s-curve at the budgets on disk (readings, `\longdata` banners, `\ifappendixfull` blocks). |
| `app_long/` | the data records those sections input: `avoiding_budgets.tex`, `avoiding_endpoint_ladder.tex`, `uav_scurve_budgets.tex` (extracted byte-for-byte from the v3.98 handover at v4.0). |

Kept in the thesis: the executed paths of D3IL-avoiding and D3IL-aligning, the flights of UAV-corridor
along the corridor, UAV-corridor under every selection rule (with its long-data marker), D3IL-avoiding at
twenty episodes per seed. The X2 dimensions figure moved from Reproducibility to Appendix A.

