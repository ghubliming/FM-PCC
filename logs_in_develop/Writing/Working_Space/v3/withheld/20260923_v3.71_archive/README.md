# withheld / 20260923_v3.71_archive — the s-curve as it stood at v3.70 (rollback copy)

Archived by the author's instruction on 2026-09-23 ("archive current s_curve sections in case we need rollback")
before the v3.71 rebuild of §6.3.3–6.3.4.

- `scurve_sections_v3.70.tex` — §6.3.3 *UAV-s-curve* (pilot table `tab:uav-scurve`, four cells) and §6.3.4 *Caveat: The
  Tracking Controller* (`tab:uav-controller` MeanFM K10, 10 vs 3 flights, plain-boundary per-step rows; Fig 6.9;
  `tab:uav-controller-cost`) verbatim, lines 1628–1812 of `chapters/06_results.tex` at v3.70.
- `uav_intro_velocity_setpoint_v3.70.tex` — the §6.3 intro paragraph "*Brake-to-rest* is the third of three
  velocity-setpoint policies…" and its srcnote (deleted at v3.71: the setpoint policies are v2 §method:geometric).
- `ch5_protocol_uav_v3.70.tex` — §5.6.3.3 quadrotor protocol paragraph before the s-curve sentence was rewritten.
- `fig_expert_uav_v3.70.svg` — Fig 5.11 with the UAV-pillars panel (dropped at v3.71).
- `fig_uav_scurve_paths_v3.70.svg` — Fig 6.9 with the panel titles "Sampling-based MPC" / "Proportional stop-go".

To roll back: paste the .tex back over the v3.71 sections (labels are unchanged: `sec:res:uav:scurve`,
`sec:res:uav:controller`, `tab:uav-scurve`, `tab:uav-controller`, `tab:uav-controller-cost`, `fig:uav-scurve-paths`;
v3.71 adds `sec:res:uav:scurve:models`, `sec:res:uav:scurve:projection`, `tab:uav-scurve-projection`), restore the
two SVGs in the figure store (`figures/env/`, `figures/da/`) and re-export.
