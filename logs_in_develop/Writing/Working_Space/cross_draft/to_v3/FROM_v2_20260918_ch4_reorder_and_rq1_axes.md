# FROM v2 → v3 · 2026-09-18 · v2.24 · Chapter 4 reordered, RQ1 reworded, capacity column in Table 3.1

**Chapter 4 section order changed** (author instruction, 2026-09-18). The chapter now runs
overview → dynamics including control → generative model → constraint projection → environments:

| was | now |
| :-- | :-- |
| 4.3 Generative Models | **4.3 Plant Dynamics and Control** (former 4.4 + 4.6, merged) |
| 4.4 Plant Dynamics | 4.4 Generative Models |
| 4.5 Constraint Projection | 4.5 Constraint Projection |
| 4.6 Low-Level Control | 4.6 Environments |
| 4.7 Environments | — |

`\section{Low-Level Control}` no longer exists as a section: its six subsections sit under
`\section{Plant Dynamics and Control}`, whose header carries **both** labels
`\label{sec:method:dynamics}\label{sec:method:deployment}`. Every reference in v3 to
`sec:method:deployment` therefore still resolves, but it now points at the merged section rather than
a section of its own. **No label was added or removed in v2.24.** The next `sync_v2.py` must take the
new order wholesale rather than patching hunks.

**RQ1 reworded** (`sec:intro:rqs`). It no longer says "improve the quality--cost frontier"; it asks
whether flow matching *reaches the task success and the constraint satisfaction of the diffusion model
it replaces at fewer solver steps and less planning time per control step*, and whether the number of
steps at which it does so orders the three objectives. Ch 1 of v3 and the RQ answers in Ch 8 should
use the same axes.

**`tab:related-loops` (Table 3.1)** now has the column *Backbone and capacity*. None of Diffuser,
Diffusion Policy, DPCC, HardFlow or SafeFlowMPC states a parameter count (checked in HardFlow's
LaTeX source, DPCC's hyperparameter table and SafeFlowMPC), so those rows read "size not reported";
the row for this thesis states **$4.0$\,M parameters for every generative model**, matching
`v3/chapters/05_setup.tex:532`.

**Other v2.24 edits that the sync will carry:** the §4.3.1 bullet "The model is deliberately crude,
and that is the point." is rewritten as fact ("The model carries no knowledge of the robot.");
`rem:eulermodel` loses "the weakest defensible model" and the `\srcnote` sentence "the distinction
matters and the thesis should not overstate it"; `tab:embodiments` expands its short forms and its
caption points forward to `sec:setup:metrics`; the velocity-setpoint policy of record (brake-to-rest)
is stated in `sec:method:geometric`, answering v3.32.
