# HANDOVER → v2 — the result sentence Chapter 1 is waiting for

**From v3, 2026-09-15.** `thesis_v2.tex:332` holds

> `\hole{Result sentence for this contribution --- from Ch.~6, supplied by v3 as a "For v2:" line.}`

directly under the endpoint-projection contribution. Chapter 6 can now answer it. Take whichever of
these fits the sentence around it; both are supported by `sec:res:avoiding:projection` and
`tab:hf-ladder` as of v3.11.

**For v2 (one sentence):**
> On the benchmark the condition is not met at the budget the task is run at: with one or two network
> evaluations endpoint projection has no step on which to act before the last one, and it reduces to
> projecting a finished sample; it is ahead of per-step projection where the budget is large enough for
> it to run, and is never cheaper at an equal number of candidate plans.

**For v2 (shorter, if the list wants one clause):**
> Endpoint projection is the faster method wherever it runs, and at one and two network evaluations on
> the benchmark it does not run at all.

## What backs it, if a reviewer asks

- The guiding step count is zero at $\nfe = 1$ and $\nfe = 2$ for $\eta = 0.5$ — **measured**, not
  inferred: the evaluation records one program solved per plan at both, three at $\nfe=5$ and five at
  $\nfe=10$, matching the predicted count exactly.
- A run at $\nfe=2$ with both methods given the same number of candidate plans produced **bit-identical
  rollouts at 2.45× the cost**.
- Across the ladder ($\nfe \in \{1,2,5,10\}$, five training seeds × three geometries) endpoint projection
  is ahead of all three selection rules of per-step projection only at $\nfe=10$ untightened, and on
  tightened constraints only at $\nfe=1$, where it runs none of its own arithmetic.
- Evidence of record: `logs_in_develop/aggregated_hardflow_lowK/DA_20260824_does_HF_pay_when_it_actually_runs.md`
  §2–§4, batch `temp/1108/Revised_2/batch_avoiding_combined_20260811_221322`.

## Also: §4.5.4's forward reference is now satisfied

`thesis_v2.tex:1640` promises that "the endpoint-projection rows at small step budgets and $\eta = 0.5$
that fall into these two regimes are listed in `\autoref{sec:res:constraints:degenerate}`". Chapter 6
now lists them (`tab:hf-ladder` in `sec:res:avoiding:projection`, with the counts in
`sec:res:constraints:degenerate`). No change needed in Chapter 4 for that — but see
[`FROM_v3_20260915_ch4_naming.md`](FROM_v3_20260915_ch4_naming.md) for the two naming
changes that section does need.
