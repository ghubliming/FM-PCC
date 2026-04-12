# DPCC MPC-Related Controls: What Exists Now (Similar Concepts Guide)

This document lists only current, implemented behavior in DPCC.
No new feature switch is assumed.

---

## Danger Note

There is no single built-in key to turn off only receding MPC planning.

Current truth:
1. Receding replanning is driven by eval loop structure.
2. Projection behavior is a separate mechanism.
3. Changing one does not automatically change the other.

---

## 1) Direct answer to your question

If you ask "is there a similar concept already in DPCC code?"

Yes, similar controls exist, but not a dedicated receding-MPC-off flag.

Existing controls are spread across:
1. Variant choice.
2. Projection enable/disable path.
3. Trajectory-selection strategy.
4. Horizon and rollout length settings.

---

## 2) Similar concepts that already exist

## 2.1 Projection variants (strongest existing planning switch)

In config projection_eval.yaml, projection_variants controls planning mode families.

Examples:
1. dpcc-r, dpcc-c, dpcc-t families.
2. diffuser baseline.
3. gradient, post_processing, model_free families.

What this changes:
1. Whether projector is active for a variant.
2. Constraint handling style and selection rule.

What this does not change:
1. Replan-every-step loop is still active in eval.

## 2.2 disable_projection path (implemented, but in code flow)

Policy call supports disable_projection argument.

What it changes:
1. Projector is bypassed in sampling.

What it does not change:
1. Receding replanning loop still runs each environment step.

## 2.3 trajectory_selection strategy

Current choices in policy logic:
1. random
2. temporal_consistency
3. minimum_projection_cost

What it changes:
1. Which sampled trajectory is chosen for action extraction.

What it does not change:
1. Replanning frequency.

## 2.4 Horizon and episode length

Current plan uses horizon (typically 8) and max_episode_length (typically 200).

What it changes:
1. Forward lookahead size per planning call.
2. Total rollout duration.

What it does not change by itself:
1. Whether replanning happens every step.

## 2.5 repeat_last support in diffusion sampling

Diffusion sampling supports repeat_last parameter in model code.

What it changes:
1. Extra denoising/projection behavior near the end of diffusion timesteps when used.

What it does not change:
1. Eval loop replanning schedule.

---

## 3) What does not exist now

Not implemented as a ready switch:
1. receding_mpc_enabled key consumed by eval.
2. replan_interval_steps key consumed by eval.
3. one-line turn_off_only_receding_mpc option.

---

## 4) Where receding behavior is currently injected

Receding behavior is injected by loop pattern:
1. In each env step, eval calls policy again.
2. It executes first action from new plan.

This is why it behaves like receding MPC in practice.

---

## 5) Training vs evaluation impact

Current training pipeline:
1. Uses model loss optimization over dataset batches.
2. Does not run the eval environment rollout loop.

So receding-MPC structure is mainly evaluation-time in current DPCC code.

But keep this warning:
1. If eval planning behavior is changed, train/eval mismatch can still increase.

---

## 6) Practical conclusion

For current DPCC code, the closest built-in similar controls are:
1. projection_variants
2. disable_projection path
3. trajectory_selection
4. horizon

But none of them alone is a true built-in off switch for only receding replanning.
