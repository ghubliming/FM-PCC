# U_13 — CHANGELOG: deterministic episode length + DPCC-style goal/early-stop for UAV eval

**Date:** 2026-07-09. Implements the plan in
`INVESTIGATION_and_PLAN_deterministic_episode_length.md` (same folder). Makes UAV eval step
counts and goal/success outcomes **deterministic** by adopting the DPCC d3il-avoiding loop:
one FIXED step budget per scene + early-terminate on goal-reach.

## Problem (recap)

Episode length was a per-trial RANDOM draw the policy never sees:
`generator._build_traj_and_init` samples `dur = rng.uniform(...)` per scene (corridor
U(6,10)s), and `n_fm = round(dur*33)` set the step budget. The loop ran the full budget with
NO early stop, and `success` was checked only on the final step. Because the FM policy is
unconditioned on `dur`, it flies at one learned average speed — so short-budget trials got
cut off mid-route (scored a goal-miss despite flying fine) and long-budget trials overshot
after arriving. Step count and success varied trial-to-trial for reasons unrelated to policy
quality. (Full derivation in the Fix_12 report Q on step-count randomness.)

## Reference followed: DPCC d3il-avoiding

`aux_repo/dpcc/scripts/eval.py:203-268` + `config/avoiding-d3il.py:68`:
- FIXED `max_episode_length` (=200) for ALL trials — not a per-trial random duration.
- goal/success checked EVERY step; `break` on `success or terminated or budget-end`.
- `n_steps[i] = _` records the stop step → deterministic time-to-goal.

## What changed — `FM_v3_uav_test/eval_fm_uav.py`

1. **`SCENE_MAX_EPISODE_LENGTH`** (new module constant): fixed per-scene budget =
   `ceil(scene_max_expert_dur * 33 * 1.2)` — keyed to the SLOWEST expert of each scene with
   1.2× headroom (empty 504, corridor 396, pillars 634, s_curve 871). SAFETY kept at 1.2
   (not 1.5) so misses — which run the full budget — don't balloon compute on jobs that
   already brush the 24h SLURM limit (Fix_11).
2. **`--max-episode-length` CLI arg** (new): single-value override for all scenes, mirroring
   DPCC's `args.max_episode_length`. Also honours a yaml `max_episode_length`
   (scalar-for-all or per-scene dict). Precedence: CLI > yaml > per-scene default.
3. **`rollout_one`**:
   - new `max_episode_length` param; `n_fm` is now that FIXED budget instead of
     `round(dur*DATASET_HZ)` (old random path kept only as a defensive fallback when no
     budget is passed).
   - `dur` is STILL sampled but now used ONLY for the fixed goal endpoint + finish-line
     direction + initial pose — it no longer sets the step budget.
   - **goal-reach latch** (`goal_reached_latch`): set in the physics inner loop the instant
     `||p − goal|| < goal_radius` (goal-path scenes only; `empty` has no goal → never
     latches → always runs the full budget). Reuses the exact threshold the final
     `goal_reached` check uses.
   - **early termination** (DPCC pattern): after each FM step,
     `if goal_reached_latch or k == n_fm-1: steps_run = k+1; break`.
   - `goal_reached` is now the LATCH for goal-path scenes (reached at SOME step), not the
     final-position-only check — so a rollout that reaches then would-have-drifted is no
     longer scored a miss. `empty` keeps the final-position check (reported only).
   - result dict: `n_fm_steps` is now `steps_run` (actual executed steps = deterministic
     time-to-goal on success / full budget on miss); added `max_episode_length` (the budget).
4. **`_run_variant`**: resolves the budget (CLI/yaml/default), prints it, passes it to
   `rollout_one`. Summary gains a **`steps`** group: `mean` (all trials), `to_goal_mean`
   (reaching trials only — the true time-to-goal, since misses run the full budget and would
   otherwise dominate), `max_episode_length`. Stdout print shows `steps_to_goal=X/budget`.
5. Updated the U7 docstring/comments: episodes now DO early-stop on goal-reach;
   `success_relaxed`/`crossed_line` retained for the "grazed the line but stopped just
   outside goal_radius" case (`success ⇒ success_relaxed` still holds).

## `FM_v3_uav_test/eval_artifacts.py`

- `write_eval_log`: added a `steps_mean … (to_goal … / budget …)` line (DPCC "avg number of
  steps" analog).
- NPZ `n_steps` (via `save_npz`, unchanged code) now inherits the meaningful `n_fm_steps`.

## Resulting semantics (the user's model, confirmed)

- **Success (goal reached + safe):** early stop → `n_fm_steps` = deterministic time-to-goal.
- **Miss (never within goal_radius):** runs the full fixed budget → `n_fm_steps` = budget
  (identical for all such trials of a scene).
- **Hard crash early-terminate:** intentionally NOT added (kept the DPCC `terminated` branch
  out) so `contact_frac` statistics are unchanged; a crash just falls into the miss bucket.
- With fixed budget + per-step latch + deterministic `trial_seed = 10_000 + i`, `n_fm_steps`,
  `crossed_line`, `success_relaxed`, and `success` are now fully deterministic functions of
  (checkpoint, trial index, variant) — no dependence on a random `dur`.

## What did NOT change

- `generator._build_traj_and_init` `dur` sampling (still needed for goal/init).
- Constraint/projection logic; npz/json schema (only ADDED `steps` group + per-rollout
  `max_episode_length`; DA loaders read keys generically, so additive keys are safe).
- No sbatch change required (defaults apply); `--max-episode-length` is available if wanted.

## Verification

- `py_compile` clean on `eval_fm_uav.py` + `eval_artifacts.py`.
- No numpy/MuJoCo in Docker → **run on cluster**. Acceptance checks:
  1. Every trial of a scene logs the SAME `max_episode_length`; misses show `n_fm_steps` ==
     that budget; successes show a distance-proportional, **repeatable** `n_fm_steps`.
  2. Corridor rollouts previously "on track but cut short" now reach the goal.
  3. Two runs of the same seed produce identical `n_fm_steps`/`success`/`crossed_line`.
- Decisions taken (from the plan's open questions): SAFETY=1.2; hard-crash terminate
  deferred; `success_relaxed`/`crossed_line` kept; budget as code constant + CLI/yaml
  override (yaml supported for DPCC-config parity).

## Files touched

- `FM_v3_uav_test/eval_fm_uav.py` — `SCENE_MAX_EPISODE_LENGTH`, `--max-episode-length`,
  `rollout_one` (budget + latch + early break + metrics), `_run_variant` (budget resolve +
  `steps` summary + print).
- `FM_v3_uav_test/eval_artifacts.py` — `write_eval_log` steps line.
