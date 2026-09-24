# CHANGELOG 2026-09-24 · Gen15 U19 coding 2 — the corridor's finish line moves to the end of the corridor (x = 2.0 m)

**Why.** R45a (author, 24-09) moved the corridor's success line to the end of the corridor, where the walls and every
corridor constraint end. The goal plane at x ≈ 2.8 lies 0.8 m past the last constraint, in free space. On corridor v3
the projected diffusion baseline ran out of its 396 steps between the two lines while still moving forward. The author
asked for the rule in the eval itself, so that every later corridor run scores it natively. Data side:
`Data_Analysis/DA_in_Paper/analysis/DA_20260924_corridor_v3_R45a_clear_line.md`.

## Change — `mix_uav_test/eval_mix_uav.py` only

- **The constant.** `SCENE_CLEAR_LINE_X = {'corridor': 2.0}` sits beside `GOAL_RADIUS`. 2.0 is where both wall boxes end
  (`scene_corridor.xml` and `scene_corridor_v2.xml`: pos x 0, half-size 2.0). In `config/uav_projection.yaml` every
  corridor geometry (corridor, `_hg`, `_ball*`, `_gate*`, `v2_slide`, `v3_tilt`, `v3_ablation_hump*`) has its halfspaces
  at x_active ≤ 2.0 and its wall-end caps at x = 2.0.
- **The latch** (`rollout_one`, physics loop):
  - `crossed_goal_line` is the old latch: the goal plane or the 0.30 m sphere.
  - `crossed_line = crossed_line or crossed_goal_line or x ≥ clear_line_x`.
  - The x test runs after physics, like the goal-plane latch, which matches the reading the R45a DA uses.
  - For any scene without a clear line (s-curve, pillars, empty), `crossed_line == crossed_goal_line`, which is the old
    behaviour exactly.
- **Success.**
  - `success_relaxed = crossed_line and safe`, so corridor success now uses x ≥ 2.0.
  - The original rule is kept beside it as `success_relaxed_goal_line` and `…_goal_line_and_constraints`.
  - `success ⇒ success_relaxed` still holds.
- **`results.json`, per rollout**, new keys:
  - `goal.crossed_goal_line`;
  - `goal.clear_line_x` (2.0 for corridor, `null` elsewhere). A file without this key was scored under the old rule.
  - `success.relaxed_goal_line`;
  - `success.relaxed_goal_line_and_constraints`.
- **Per-variant summary.** New `success.relaxed_goal_line_rate` and `…_and_constraints_rate` sit beside the existing rates.
- **Module docstring:** one line added.

## Unchanged

- **The flights are unchanged.** No new early stop: a flight still ends at the 0.30 m goal sphere or at the budget. The
  control, the budget (396), the recorded paths, the violation counts, the strict `success` and the timing are all
  untouched. The UAV eval stays deterministic per configuration: re-flying a cell gives the same flights, and for
  corridor only the relaxed success flags can differ.
- **The DA batch** (`Data_Analysis/DA_UAV_v1`) needs no change. `success_relaxed` and `n_success_relaxed_and_constraints`
  read `success.relaxed*`, so they now carry the new rule. The old-rule keys are not exported as CSV columns yet; adding
  them takes two entries in `DA_UAV_v1/config.py` and is not done.
- **Existing corpora are not re-scored in their files.** For R33 the R45a DA gives exactly what this code would score.
  The DA reads x after each control step, while this code tests after every physics step, which could add a flight that
  touches x = 2.0 mid-step and falls back. None does: the only flight within 5 cm of the line (hump, diffusion,
  per-step r, trial 7) crosses during its last step and ends 16.7 mm past the line.
- **Not touched:**
  - `FM_v3_uav_test/eval_fm_uav.py` (the Gen11 sibling; no corridor run of record uses it);
  - the sbatch scripts (no new flag);
  - the R45c items (end a flight at its first crossing, seed torch per trial, store the end state, budget 792). Each of
    those changes the flights and needs its own go-ahead.

## Checks

- Local: `python3.14 -m py_compile mix_uav_test/eval_mix_uav.py` passes. A pure-python replay of the latch
  expression gives:
  - the flight ending 1.998785 → 2.0167 counts under the new rule and not under the old one;
  - x = 1.225 counts under neither;
  - a goal-plane crossing counts under both;
  - with no clear line, the new and old latches are equal.
- **Run on cluster (optional):** re-evaluate hump, diffusion K20, per-step `dpcc-r-bounds_free-pdes-tightened`, 10 flights.
  Expected:
  - `success.relaxed` 10/10, `relaxed_goal_line` 3/10, `relaxed_and_constraints` 10/10;
  - `obs_all` identical to the `p23cv3ah` corpus.
