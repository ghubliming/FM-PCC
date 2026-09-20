# DA — is the truncated diffusion cell a symptom of wider data loss in the corpora?

**2026-09-20.** Triggered by the wave-3 fetch: the diffusion $\nfe=20$ 20-trials run turned out to hold
five of thirteen variants at seed 6 / `both-hard`, because the job died mid-cell. The question this
answers is whether that is one accident or the visible tip of a systematic problem.

**Short answer: it is job truncation, it is real, and it is rare and bounded. The CSVs are not wrong —
they faithfully report what is on disk. The defect is that nothing in the pipeline *says* a cell is
short.**

## Method

A cell is `(run folder, seed, geometry)`. A complete cell of the state-based avoiding corpus holds 13
variants. Counting raw "cells with fewer than 13" is misleading: 19 runs are *uniformly* short — every
one of their cells holds 7, or 8, or 3 — which is a configured variant subset, not a defect.

The defect signature is a cell short **relative to its own run's modal variant count**. That is what is
counted below.

## Result

| corpus | cells | runs | truncated cells |
| :-- | --: | --: | --: |
| `batch_avoiding_combined_20260919_132703` (of record) | 970 | 121 | **12** (1.2 %) |
| `batch_avoiding_combined_20260915_100757` (previous) | 910 | 117 | **12** |
| `batch_uav_20260919_111701` | 170 | 102 | **4** |

The count is **identical across the two avoiding corpora**, four days apart. Truncation is not
spreading, and re-running the aggregation does not introduce it. The four UAV cells are all in
`legacy(Bf_U8_debug)`, `Gen11` pillars (withdrawn) and `@u7hg` (dead) runs — none feeds the thesis.

## Why this is truncation and not a parsing or aggregation bug

Two things rule the software explanation out.

**Each truncated cell is a contiguous prefix of its run's variant sequence.** The two K20 20-trials
cells both stop at exactly the same place:

```
present : dpcc-r, dpcc-r-tightened, dpcc-c, dpcc-c-tightened, dpcc-t
absent  : dpcc-t-tightened, hardflow_new-{r,c,t}{,-tightened}, diffuser
```

A parser that dropped rows would drop them scattered, or drop one variant everywhere. A prefix is what a
process killed part-way through leaves behind.

**Every truncated thesis cell is at $\nfe \ge 10$** — the slowest evaluations in the corpus, and
therefore the ones that run into a wall-clock limit. Nothing at $\nfe = 1, 2, 5$ is affected.

## The four that touch the thesis

| run | cell | variants | what it costs |
| :-- | :-- | --: | :-- |
| Diffusion $\nfe=20$ 20-trials | seed 6, `both-hard` | 5/13 | 🔴 `both-hard` was this run's **only** seed there, so the whole geometry vanishes from the per-step-tightened cell → the `geos=2` row |
| FM $\nfe=20$ 20-trials | seed 8, `both-hard` | 5/13 | 🟠 a table row, but the other four seeds cover `both-hard`, so the cell averages 14 of 15 instead of 15 and stays `geos=3` |
| MeanFM $\nfe=20$ 20-trials | seed 8, `top-left-hard` | 1/13 | 🟡 not a table budget; feeds `fig:k-ladder` |
| MeanFM $\nfe=10$ 20-trials | seed 10, `both-hard` | 8/13 | 🟡 likewise; only the `hardflow_new-*` tail is missing |

The remaining eight are in `(old_no_mpc_traj)` and `Dfm_visual_avoiding` / `Ddiffuser_visual_avoiding`
runs, which are not thesis corpora.

## The actual defect: the pipeline is silent

No number here is known to be wrong. The aggregation averages each seed over the geometries that seed
covers, so a truncated cell quietly changes a denominator and reports a slightly different mean. **It
never raises anything.**

The only place incompleteness is visible today is the `seeds=` / `geos=` counters
`avoiding_table_spread.py` prints — and those catch a truncation only when it removes an *entire*
geometry from a cell, which of the four above is true for exactly one. The other three are invisible.

The UAV batch already writes a `data_quality.csv` carrying an `npz_complete` column. The avoiding batch
writes no equivalent.

## Recommendation

A completeness check belongs beside the other analysis modules: for every cell, compare its variant
count against its run's modal count, and print the shortfalls. It is about forty lines of stdlib and it
would have surfaced all twelve of these the day the corpus was built, rather than on the day a figure
happened to need one of them.

Not written — it is new code and needs a go-ahead.
