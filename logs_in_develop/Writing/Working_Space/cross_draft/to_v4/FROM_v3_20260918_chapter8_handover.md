# TO v4 — Chapter 8 is yours in full; v3 no longer builds it

**2026-09-18 · from v3.36 (author instruction).** `\input{chapters/08_conclusion}` is commented out in
`thesis_v3.tex` and `inherited/MANIFEST.md` marks the file **own, not built**. The file is untouched
otherwise and stays in `v3/chapters/` as the version you inherit. `tools/check.py` passes without it,
so nothing in Chapters 5–7 or the appendix references a Chapter 8 label.

## What is in the file you are picking up

Drafted through v3.35. The three RQ answers match v2's current three-RQ structure (the budget RQ was
folded into RQ1 at v2.22), and **RQ1's answer was rewritten at v3.35** to the axes v2.24 gave it: the
level of the diffusion model reached at fewer solver steps and less planning time, and whether the step
count at which that happens orders the three objectives. If v2 moves RQ1 again, that paragraph is the
one to re-check.

## One result changed after that text was written

v3.36 rebuilt the UAV-pillars section (§6.3.1.2) from the batch CSV. The conclusion's sentence about
UAV-pillars is now narrower than the old "mean over eleven configurations" reading, and Chapter 8
should follow Chapter 6 rather than the earlier phrasing:

* on the **seven per-step configurations** the two leading models share at $\nfe=5$, analytic
  average-velocity matching and instantaneous-velocity matching are **level** (mean $0.586$ each; each
  ahead in three of the seven cells, one level);
* instantaneous-velocity matching's advantage is **entirely in the endpoint block** ($0.950$ against
  $0.700$ over four configurations), and that block acts on a plan that was **already collision-free on
  all ten flights before any projection ran**;
* at $\nfe=2$ the order reverses — analytic $0.429$, instantaneous $0.343$,
  consistency-interpolated $0.300$.

So UAV-pillars supports "endpoint projection preserves an already feasible plan where per-step
projection damages it", not "instantaneous-velocity matching is the better constrained planner".
