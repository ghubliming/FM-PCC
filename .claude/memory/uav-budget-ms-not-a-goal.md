---
name: uav-budget-ms-not-a-goal
description: UAV budget_ms/33 Hz and over_budget_frac are a data-rate artefact, never a project target; real-time is not an FM-PCC goal
metadata:
  type: project
---

`budget_ms = 30.3 ms` (33 Hz) in the UAV eval is `1000/DATASET_HZ` — the rate the expert
dataset happened to be recorded at (`mix_uav_test/behavior_logger.py:67`). **FM-PCC has never
set real-time control as a goal.** `over_budget_frac` / `real_time_safe` are logged as a
reference scale only, and the logger itself says the timing is *"cluster latency, NOT target
drone"* (`behavior_logger.py:184`) — shared-GPU wall-clock with an SLSQP solve in the loop.

**Why:** the metric's name and the code's `real_time_safe=YES/NO` print make it look like a
pass/fail gate. Reporting it that way invents a failure the project never signed up for — in
the first UAV DA draft it produced a headline "Is any UAV configuration real-time? → No" and a
"§4 real-time gate" section, both wrong.

**How to apply:** compare `ms` **between configurations** (ratios, spread, where the time goes —
the projection solve is 60–90 % of it). Never against 30.3 ms, never as deployability, never as
a gate or ranking axis. If a `× 30.3` column is useful for readability, label it a scale, not a
score. Same caution for any absolute-latency claim on cluster hardware. See
[[pareto-definition-of-good]] for the axes that *are* legitimate, and
[[da-target-is-best-baseline-variant]] for what counts as a real target.
