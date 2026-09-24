# TO v3 — UAV-s-curve's finish line is now the end of the walls in the code (x = 3.0 m); no number changes

**2026-09-24 · from the DA side (R44 run chat), at the author's request.** Nothing in `v3/` was touched.
- **Code:** `mix_uav_test/eval_mix_uav.py`, `SCENE_CLEAR_LINE_X = {'corridor': 2.0, 's_curve': 3.0}`, the corridor's
  rule applied to the s-curve.
- **Changelog:** `logs_in_develop/Gen15/U19/CHANGELOG_20260924_U19_scurve_finish_line_coding3.md`.
- **Re-score:** all 160 R44 s-curve flights re-scored from their flown paths: **0 change**. Tables 6.14, 6.15 and 6.16
  stand as printed (`DA_in_Paper/analysis/scurve_r44_raw.py` §6b).

## What Chapter 5 can now say

1. **`fig:env-uav` caption.** "the green line at the far end is the finish line, at the end of the walls on UAV-corridor
   and UAV-s-curve (UAV-s-curve is scored $0.2$\,m further on, at its goal point)". The parenthetical can go. Both scenes
   are now scored at the end of their walls: corridor x = 2.0 m, s-curve x = 3.0 m.
2. **`sec:setup:metrics:uav`, the success item.** "On UAV-corridor the finish line is the end of the corridor, $x=2.0$\,m,
   …" can name both. Suggested: "On UAV-corridor and UAV-s-curve the finish line is the end of the walls, $x=2.0$\,m and
   $x=3.0$\,m, where the walls and every test-time constraint end; a flight is scored the moment it passes it and flies
   on as before."
   - The corridor's specific reason (the episode limit catching slowed projected flights) stays corridor-only.
   - For the s-curve, the moved line changes no count: its successful flights end at the goal sphere, about 0.1 m before
     the wall end, and its failed flights that pass x = 3.0 are unsafe.
3. **`\srcnote` of the corridor rule** (`SCENE_CLEAR_LINE_X = {'corridor': 2.0}`): now `{'corridor': 2.0, 's_curve': 3.0}`,
   with the changelog above.

**Dataref for the s-curve tables:** the R44 flights were flown under the goal-point rule. The re-score at x = 3.0 gives
identical counts, so one clause, "identical under the end-of-wall line (0 of 160 flights change)", covers it.
