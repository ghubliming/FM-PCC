# Aggregate Analysis — D3IL visual-aligning eval

Tools for reading eval results that the main eval script couldn't summarize itself — chiefly
**time-limited runs** that died mid-rollout before writing `results_seed_{s}.json`.

## `aggregate_partial_results.py`

Reconstructs the two paper numbers (`success_rate`, behavior `entropy`) per seed and cross-seed
from the per-rollout `diagnostics/rollout_*_stats.json` files that *did* land, using the exact
paper entropy formula (copied from `eval_d3il_visual_aligning.py:compute_behavior_entropy`).

Torch-free (stdlib + numpy). Read-only on your results.

```bash
# from repo root — all seeds under the default logs root, paper scale (60 x 18)
python d3il_visual_aligning_baseline_test/aggregate_analysis/aggregate_partial_results.py

# specific seeds + emit per-seed results_seed_{s}.json (real eval schema, flagged "partial")
python d3il_visual_aligning_baseline_test/aggregate_analysis/aggregate_partial_results.py \
    --seeds 0 1 2 3 4 42 --write-seed-json

# diagnostics scp'd elsewhere
python .../aggregate_partial_results.py --logs-root /path/to/logs/d3il_visual_aligning_baseline
```

### What it reports
- Per seed: rollouts done / 1080, coverage %, contexts reached / fully-complete, `success_rate`,
  `entropy` (two variants), `score`, and a FULL/part completion flag.
- Cross-seed: `mean +/- std`, side-by-side with paper `0.278 +/- 0.071` / `0.139 +/- 0.054`.
- A **bias caveat** whenever any seed is partial: the eval loop is context-major, so missing
  rollouts are the high-index contexts → the aggregate leans on low-index initial states.

### Two entropy variants
- `H(reach)` — averaged over contexts with ≥1 rollout. **Headline** (and the `score` input).
- `H(compl)` — averaged over only fully-complete (18/18) contexts. Stricter, less per-context
  noise, but discards partly-done contexts. Use the spread between the two as a confidence band.

`success_rate` is the mean over all completed rollouts (robust to partial coverage; only the
low-index context bias applies).

### Outputs
- Prints the table.
- Writes `<agent_root>/aggregate_partial.json` (full machine-readable dump).
- With `--write-seed-json`: writes `seed_{s}/results_seed_{s}.json` in the real eval schema,
  flagged `"partial": true`, so the guide's step-4 snippet works unchanged.

## When to use the *real* numbers instead
This tool is a salvage path. For a paper-comparable result, finish the eval to completion: the
eval wall-clock is now 24h (was 4h — see `logs_in_develop/D3IL_Visual_Aligning_RUN/U2/CHANGELOG.md`),
which should clear all 1080 rollouts/seed and write the genuine `results_seed_{s}.json` +
`aggregate_results.json`.
