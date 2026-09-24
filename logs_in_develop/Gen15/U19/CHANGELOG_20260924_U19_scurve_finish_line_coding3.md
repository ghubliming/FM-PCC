# CHANGELOG 2026-09-24 · Gen15 U19 coding 3 — the s-curve's finish line moves to the end of the walls (x = 3.0 m)

**Why.** Author, 24-09: "set the s_curve goal line in the repo code like the thesis Ch 5 claims, at the end of the wall,
like the corridor". Chapter 5 draws the finish line of UAV-s-curve at the end of its walls (`fig:env-uav`, v3). The eval
scored it at the route's goal point: the goal plane at x = 3.2, or the 0.30 m goal sphere. This applies the corridor's rule
(coding 2, `CHANGELOG_20260924_U19_corridor_finish_line_coding2.md`) to the s-curve.

## Change — `mix_uav_test/eval_mix_uav.py` only (11 lines)

- **`SCENE_CLEAR_LINE_X = {'corridor': 2.0, 's_curve': 3.0}`.** Where 3.0 comes from:
  - both second-segment walls end at x = 3.0 (`scene_s_curve.xml` `seg2_wall_neg` / `seg2_wall_pos`: pos x 1.75,
    half-size 1.25);
  - every s-curve geometry ends there too (`s_curve`, `s_curve_hg`: halfspace `x_active` [0.5, 3.0]);
  - the route's goal point, (3.2, 0.8), lies 0.2 m further on (`trajectories.s_curve_scene_path`);
  - x ≥ 3.0 is reached only at the route's end: the first segment ends at x = −0.5.
- **Mechanics:** exactly coding 2's latch.
  - `crossed_line` also latches when the vehicle's x reaches the line, tested after physics.
  - The goal-plane / goal-sphere rule stays beside it as `goal.crossed_goal_line` and `success.relaxed_goal_line(_and_constraints)`.
  - `goal.clear_line_x` now records 3.0 on s-curve rollouts.
- **Comments:** the explanatory block above the constant, the module docstring (one line), and the three inline comments
  that named only the corridor (lookup, latch, `results.json` key).

## Unchanged

- **The flights.** No new early stop: a flight still ends at the 0.30 m goal sphere or at the 871-step budget. The control,
  recorded paths, violation counts, strict `success` and timing are all untouched. The eval stays deterministic per
  configuration: re-flying an s-curve cell gives the same flights, and only `success.relaxed*` can differ.
- **Not touched:** `FM_v3_uav_test/eval_fm_uav.py` (the Gen11 sibling), the sbatch scripts (no new flag), `DA_UAV_v1`
  (it reads `success.relaxed*`, which now carries the new rule).

## ⚠️ Depends on coding 2, which is staged but not committed

The latch this constant feeds, `clear_line_x` / `crossed_goal_line`, is coding 2's. On 24-09 it is **staged, not
committed**: HEAD `5092e056` has no `SCENE_CLEAR_LINE_X`. This edit sits on top of it in the working tree, so commit both
together. The R44 s-curve runs (git `0df6df71` / `5092e056`) therefore ran without either rule: their `results.json` has
no `goal.clear_line_x` key.

## Effect on the existing s-curve results: none

All 160 R44 flights were re-scored from their flown paths: crossed = the stored latch or flown x ≥ 3.0, success =
crossed and safe (`Data_Analysis/DA_in_Paper/analysis/scurve_r44_raw.py` §6b; the 16 cells of Tables 6.14–6.16).
- **0 of 160 flights change.** Every success count of Tables 6.14, 6.15 and 6.16 is the same under the new line.
- One flight comes within 3 cm without crossing (B1 per-step random, flight 7, max x 2.9932). It inverted, so it is
  unsafe under any line.
- Successful s-curve flights end at the goal sphere (about x 2.9–2.95), and the failed ones that pass x = 3.0 are unsafe.
  The moved line therefore changes no count here. It exists so that the code says what the thesis says, and so that a
  slow flight that clears the walls but runs out of budget before the goal point scores like the corridor's.

## Checks

- `python3.14 -m py_compile mix_uav_test/eval_mix_uav.py` passes.
- Latch replay (the expression of `eval_mix_uav.py:1926–1929` with the constant read from the edited source):
  - s-curve stopping at x 2.999: not crossed;
  - passing x 3.001 off the goal sphere: crossed under the new rule only;
  - reaching the goal sphere at x 2.93: crossed under both;
  - corridor x 2.0: crossed (unchanged);
  - pillars (no line): equal to the old latch.
- Line endings are unchanged (LF in HEAD, index and working tree); the Git CRLF warning is the container's
  `core.autocrlf=true`.
- **Run on cluster (optional):** any s-curve cell. `results.json` must show `goal.clear_line_x = 3.0` and the same flights
  as its R44 twin.

For v3: the Ch 5 caption's "(UAV-s-curve is scored 0.2 m further on, at its goal point)" can become the wall-end rule,
with no number changing. Note: `cross_draft/to_v3/FROM_DA_20260924_scurve_finish_line_wall_end.md`.
