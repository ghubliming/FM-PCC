# Visual-Aligning metrics — what each number actually is

**Date:** 2026-08-07 · scope: `mix_visual_aligning_test` (Gen14) / `fm_visual_aligning_test`
(Gen7) eval output, as read by `DA_VA_v2`. 65 columns land in
`per_rollout_detail.csv`; this is the short version, plus §7 — what to fix next.

---

## 0. The one that confuses everyone: `mean_distance`

**It is not a mean over time, and it is not in metres.**

```python
# d3il/environments/d3il/envs/gym_aligning_env/gym_aligning/envs/aligning.py:303-316
box_goal_pos_dist = np.linalg.norm(box_pos - target_pos)          # 3D, METRES
box_goal_rot_dist = rotation_distance(box_quat, target_quat)/np.pi # 0..1, UNITLESS
mean_distance     = 0.5 * (box_goal_pos_dist + box_goal_rot_dist)
```

So it is **half a distance in metres plus half a normalised angle** — a mixed-unit composite
score, evaluated at the **final step** of the rollout (`aligning_sim.py:156` stores the last
`info`). The word "mean" is d3il's, and it refers to averaging *across rollouts* when they
log it (`aligning_sim.py:247`), not across time.

Consequences:

* `mean_distance = 0.33` is **not** 33 cm.
* it cannot be compared against `pos_min_dist` (0.018) or against the INIT XY reference.
* a rollout that parks the box perfectly but 90° rotated scores ~0.5·0.5 = 0.25, the same as
  one that is 50 cm away with perfect rotation.

DA_VA_v2 exposes it twice, as `mean_dist_per_rollout` and `mean_distance` (identical, §6).

---

## 1. The distance family — pick one on purpose

| column | what | units | reference |
|---|---|---|---|
| `context_init_xy_dist` | box→target **XY** distance at rollout **start** | m | ~0.45 typical |
| `context_final_xy_dist` | box→target **XY** distance at rollout **end** | m | pass line **0.018** |
| `mean_dist_per_rollout` = `mean_distance` | env composite ½(3D pos + rot/π), final step | mixed | no clean threshold |
| `max_phys_error_per_rollout` | worst plan-vs-executed tracking error | m | diagnostic, not an outcome |

**Use `context_final_xy_dist`.** It is the only distance that is (a) in metres, (b) directly
comparable to `context_init_xy_dist`, and (c) the exact quantity `success_relaxed` thresholds.
Both `*_xy_dist` come from the eval, not the env
(`eval_mix_visual_aligning.py:1215-1226`) — XY only, z ignored.

⚠ heavy tail: `context_final_xy_dist` runs 0.002 – 4.15 m in a real batch. One rollout that
shoves the box off the table drags a whole column's mean. Read the median when ranking.

---

## 2. The success family — four different questions

| column | definition | source |
|---|---|---|
| `n_success` = `success_strict` | `pos_dist ≤ 0.018 m` **AND** `rot_dist ≤ 0.048` (≈8.6°) | env, `aligning.py:344` |
| `success_relaxed` | `final_xy_dist ≤ 0.018 m`, **angle ignored** | eval, `:1234` |
| `n_success_and_constraints` | `n_success × zero_violation` per rollout | derived |
| `n_success_relaxed_and_constraints` | `success_relaxed × zero_violation` per rollout | derived in the viewer (U3) |

Strict is brutal — 2.5 % in a real batch vs 7.1 % relaxed — because the rotation gate is
independent of the position gate. Report both; the gap *is* the finding (does the method
place the box, or place *and* orient it).

Both `*_and_constraints` are **per-rollout products**. `mean(a)·mean(b)` is a different,
wrong number.

---

## 3. The constraint family — one headline, the rest are forensics

Computed over the executed trajectory (`eval_mix_visual_aligning.py:698-740`), where a step
is violating if **any** family (bounds / halfspace / obstacles) is violated by > 1e-6 m.

| column | definition |
|---|---|
| `constraint_exec_sat_rate` | `1 − violated_steps / n_steps` — **the headline** |
| `collision_free_completed` = `constraint_exec_zero_violation` | the rollout never violated once |
| `n_violations` = `constraint_exec_n_violated_steps` | count of violating steps |
| `constraint_exec_max_{bounds,halfspace,obstacle}_*_m` | worst depth per family, metres |
| `constraint_exec_margin_mean_m` | mean clearance on non-violating steps |
| `constraint_exec_first_violation_step`, `..._longest_safe_streak` | when it broke, how long it held |
| `constraint_exec_dyn_err_{mean,max}` | ‖Δpos − action‖, sanity check that actions were executed |
| `constraint_plan_post_viol_rate_{mean,max}` | violations remaining in the **plan** after projection |

`sat_rate` is per-*step*, `zero_violation` is per-*rollout*. A method with sat_rate 0.99 can
still have zero_violation 0.0 — one bad step per rollout, every rollout.

---

## 4. Cost

| column | definition |
|---|---|
| `n_steps` | executed steps (early termination on success ⇒ **shorter = better only among successes**) |
| `avg_time` / `avg_time_ms` | wall time **per replan**, not per env step (`eval:1207`) |

---

## 5. Data quality — not results, gates on results

| column | definition |
|---|---|
| `frozen` | D1 box-obstacle conflict: the eval froze the rollout, **the model never ran**. Reports `sat_rate = 1.0`, so it inflates every constraint aggregate |
| `frozen_worst_overlap_m` | how bad the conflict was |
| `projection_cb_tripped`, `projection_cb_skipped_steps` | projector circuit breaker fired; those steps ran unprojected |
| `diagnostics_found` | the per-rollout JSON existed (0 ⇒ `frozen` is a guess, not a fact) |

Always check the **unfrozen** mask before believing a constraint number.

---

## 6. Aliases and dead columns (verified identical on a 1140-rollout batch)

| kept | duplicate of | max abs diff |
|---|---|---|
| `n_success` | `success_strict` | 0 |
| `n_violations` | `constraint_exec_n_violated_steps` | 0 |
| `collision_free_completed` | `constraint_exec_zero_violation` | 0 |
| `max_phys_error_per_rollout` | `outcome_max_physical_tracking_error` | 0 |
| `mean_dist_per_rollout` | `mean_distance` | 1e-7 (float32) |
| `avg_time_ms` | `avg_time` × 1000 | exact |

`total_violations` is **always NaN** on visual runs — the visual eval never accumulates a
running violation sum (it records per-family maxima). It exists only so the state-only
avoiding schema lines up.

That is 6 redundant columns out of 65, and one that is structurally empty.

---

## 7. Next steps — how to make this better

Ordered by value per hour. None of these are done.

1. **Rename `mean_distance` at the DA layer.** It is the single biggest source of
   misreading. Emit it as `env_score_composite` (or `d3il_mean_distance`) with a
   `METRIC_LABELS` entry that says *"env composite ½(pos_m + rot/π), final step — not
   metres"*. Keep the raw npz key untouched; rename only in `DA_VA_v2/data_loader.py` +
   `config.METRIC_LABELS`.
2. **Make `context_final_xy_dist` the headline distance.** Promote it in
   `config.PRIMARY_METRICS`, and swap the viewer's MIN_DIST matrix onto it so the
   INIT XY reference row becomes exact. (Currently that table is the composite, with a
   caption that wrongly says "[m]".)
3. **Add a median to the aggregation.** `aggregator._reduce` already computes mean/std/n;
   `median` and `p90` are one line each and would stop a single off-the-table rollout from
   deciding a column. Distance metrics need this more than success rates do.
4. **Emit `n_success_relaxed_and_constraints` in the pipeline.** The viewer derives it
   (U3), but CSV consumers — ranking table, Colab notebooks — cannot see it. ~5 lines in
   `data_loader._finalise_frame` + the `config` lists, then a re-run.
5. **Drop the alias columns from `per_rollout_detail.csv`** (§6) behind a
   `--keep-aliases` flag, or at minimum document them there. 65 columns is a wall; 58 with
   a legend is a table.
6. **Give every metric a direction.** A `HIGHER_IS_BETTER` / `LOWER_IS_BETTER` map in
   `config.py` would let the scorecard, the Pareto ranking and the viewer stop
   string-matching on `'success' in metric.lower()`.
7. **Record a `min_dist_over_rollout`.** `record_step_info` already accumulates
   `dist_to_target` per step but only the final value survives. The *closest the box ever
   got* separates "never approached" from "approached then got pushed away" — a real
   failure mode of the projected variants. Eval-side change, needs a re-run.
8. **Derive the rotation error.** Strict success fails mostly on rotation, but no column
   reports it: `context_final_box_angle_deg` is an absolute angle, not an error. Both
   halves are already in the CSV, so `wrap180(context_final_box_angle_deg −
   context_target_angle_deg)` is a DA-side derivation with no re-run — and it turns the
   strict-vs-relaxed gap from something you can see into something you can explain.
   *(Caveat: the env's gate uses the quaternion `rotation_distance/π`, so read the wrapped
   yaw error as a proxy, not as a reproduction of `rot_dist`.)*

Items 1–6 and 8 are DA-side and take effect on the batches already on disk. Only item 7
needs an eval change and a cluster re-run.
