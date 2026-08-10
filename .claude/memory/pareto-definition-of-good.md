---
name: pareto-definition-of-good
description: "\"Good\" for an FM-PCC eval result means Pareto-dominant: at equal success+constraint satisfaction, fewer steps AND lower avg_time"
metadata:
  type: feedback
---

When judging whether an FM-PCC configuration is **good**, use Pareto dominance on this metric
triple, in this priority order:

1. **`n_success_and_constraints` (S&C)** — the gate. Never trade this away.
2. **`n_steps`** — fewer is better (reaches the goal sooner).
3. **`avg_time`** (s/step) — faster is better.

Dominance is always **against a specific opponent** — see [[benchmark-hierarchy-who-beats-whom]]
for which arm each generation is required to beat.

A config is **good** only if, *at the same or better S&C*, it has **both** fewer steps **and**
lower avg_time than the comparator — i.e. it strictly Pareto-dominates. If it wins on one and
loses on the other, say so plainly: it is a **trade-off, not a win**.

If nothing dominates it, the claim to make is **"Pareto-non-dominated"** — not "best". Only when a
config is beaten on *every* axis (strictly Pareto-worst) may you conclude it is simply bad.

**Scope:** the strict "both axes or it's only a trade-off" rule above is for crowning a config the
*best overall*. It is **not** the bar for beating the baseline: against the DPCC Target, S&C is the
only gate and a win on **either** `n_steps` or `avg_time` counts as a win — see
[[da-target-is-best-baseline-variant]].

**Why:** the user is building a low-NFE control argument. "Fast at K=2" is worthless if it costs
success or takes more steps to get there — any generator can be made cheap by discounting K (see
`logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md` L3). Dominance
is what makes low-K a *capability* rather than a discount.

**How to apply:**
- **Only compare `n_steps` between rows with equal S&C.** The eval script averages `n_steps` over
  *successful* trials only (`eval_flow_matching_v3_meanflow.py:518`), so a failing config posts a
  flatteringly short step count. It prints `0.00` when SR = 0, and `199` means the episode hit
  `max_episode_length` (timeout), not a real path length.
- **Use a tolerance on `avg_time`.** Wall-clock per step varies with GPU contention across jobs;
  differences under ~10–20 % are noise. Order-of-magnitude gaps (K=2 at ~0.03 s vs DPCC K10/K20 at
  0.3–0.6 s) are the claimable ones.
- **Check per-environment consistency, not just the pooled mean.** With `n_trials = 2` a pooled
  win can come from one env. "Wins in 3/3 halfspaces" is a real statement; "wins on the mean" is
  not. See [[fmpcc-dev-logs-navigation]] for where the per-env data lives.
- Old candidate results for cross-comparison live in the DA batch CSVs, e.g.
  `temp/2026-08-02/batch_avoiding_combined_20260802_092307/candidates_multidimensional_raw.csv`
  (per-seed rows — filter to the matching seed so trial counts line up).
