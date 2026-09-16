# FROM v3 → v4 · 2026-09-16 · findings that belong in the Discussion (Ch 7)

Each is established in v3 Ch 5–6; v4 decides what to argue. Evidence in parentheses.

1. **Downstream metrics hide plan quality.** Projection and tracking absorb large errors; DPCC's own Table 2
   (dynamics model wrong by 4× still S&C 0.77) quantifies the margin. Smoothness metric still owed.
   (§6.1.3, `NOTES_plan_quality_and_smoothness.md`)
2. **Consistency training is never ahead of MeanFlow**: level where the task saturates (avoiding, corridor,
   alignment K2), behind where it separates (alignment K20, pillars — behind flow matching there). Mechanism:
   bootstrapped target inherits network error, does not shrink with K. (§6.2.3, §6.3.3)
3. **Endpoint projection needs guiding steps** — absent at the budget avoiding runs at; faster where it runs,
   less collision-free on the corridor, physically safer on pillars. (§6.1.4, §6.4.2)
4. **The tracking controller can be the bottleneck** (s-curve: same plan, MPC 3/3 vs brake-to-rest 0/10,
   inversion at altitude). A limitation of the whole stack, not the planner. (§6.3.4)
5. **Threats to validity:** single seed outside obstacle avoidance; alignment on *training* contexts;
   non-uniform training settings (`tab:train`); no tightened diffusion run on alignment; D3IL's own image
   policy reproduced in our pipeline barely moves the box (0.44 m from 0.45 m, 44 % untouched) — author
   has not decided whether it is reported.
6. **Flow matching ≈ diffusion on alignment**, while MeanFlow separates — the gain is the objective, not
   the ODE family alone. (§6.2.1)
