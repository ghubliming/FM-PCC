# TO v3 — D3IL-aligning: R2, R37 and R35 are ready; three *pending* blocks can be filled

**2026-09-23 · from the DA side.** Nothing in `v3/` touched. Analysis of record:
`Data_Analysis/DA_in_Paper/analysis/DA_20260923_R2fix_R37_aligning.md` (INDEX row added). Corpus
`temp/23-09/batch_va2_20260923_210100/per_rollout_detail.csv`; it reproduces every cell Tables 6.6, 6.7
and 6.8 already print, to the last digit, so no existing row moves. Conventions are `va_results.py`'s.

## What landed

| table | block | source | status |
| :-- | :-- | :-- | :-- |
| 6.8 `tab:va-projection-models` | diffusion per-step row, $\nfe=20$, $\eta=0.2$ | job 26113, tag `_msgR2fix` (26051 ran $\eta=0.5$ and is not used) | **R2 ✅** |
| 6.6 `tab:va-threshold` | $\nfe=100$, $\eta=0.1$ row, both projectors | job 26115, tag `_msgR37` | **R37 ✅** |
| 6.6 `tab:va-threshold` | $\nfe=2$ row | **Table 6.7's own $\nfe=2$ cell** (already printed there) | **R37 ✅** |
| 6.8 `tab:va-projection-models` | CI-MeanFM endpoint block | **already in the corpus** — no run needed | **R35 ✅** |

## The numbers (tightened, ten contexts, seed 6)

**Table 6.8 — diffusion, per-step** (violation-free · violating steps · moved · median · mean · ms/step):

| $r$ | 4/10 · 14.8 · 3/10 · 0.462 (−2 %) · 0.397 ± 0.160 · 809.2 ± 756.0 |
| :-- | :-- |
| $c$ | 1/10 · 35.0 · 0/10 · 0.456 (−1 %) · 0.453 ± 0.037 · 1069.7 ± 848.6 |
| $t$ | 1/10 · 41.1 · 1/10 · 0.462 (−2 %) · 0.456 ± 0.039 · 766.0 ± 284.5 |

**Table 6.8 — CI-MeanFM, endpoint** (same columns):

| $r$ | 8/10 · 2.7 · 7/10 · 0.425 (6 %) · 0.333 ± 0.183 · 294.2 ± 113.3 |
| :-- | :-- |
| $c$ | 6/10 · 10.1 · 4/10 · 0.456 (−1 %) · 0.440 ± 0.076 · 314.8 ± 119.6 |
| $t$ | 5/10 · 8.6 · 6/10 · 0.434 (4 %) · 0.368 ± 0.124 · 344.8 ± 164.6 |

**Table 6.6 — the two rows** (ms unproj. · per-step ms · median · clean · endpoint ms · median · clean):

| $\nfe=2$, $\eta=0.5$, guiding 0 | 24.6 · 46.2 · 0.275 · 8/10 · — no guiding step |
| :-- | :-- |
| $\nfe=100$, $\eta=0.1$, guiding 9 | 924.6 · 1112.0 · 0.248 · 8/10 · 1107.0 · 0.153 · 8/10 |

## Read before filling (details in the DA)

1. **R35's block comes from a second run of the same checkpoint** (`AFAFend0p2`, latest) as the printed
   CI-MeanFM per-step rows; the two runs agree on clean counts, moved and medians, and differ by under one
   violating step and 5 ms. Either add the endpoint block beside the printed rows and say so, or take both
   blocks from the untagged run (DA §9 has its per-step numbers). The author's call.
2. **The prose count changes:** endpoint keeps more contexts clean in **seven of nine** comparisons, not
   five of six; the new exception is CI-MeanFM under $t$ (6 against 5), by one context.
3. **Endpoint at $\nfe=100$ is 8/10, not 10/10** — the sentence "ten of ten contexts clean at both budgets"
   stays true for the two budgets it names only. Non-converged solves were logged; their count is not in
   the corpus, so do not attribute the two misses.
4. **The baseline's per-step rows:** MeanFM $r$ Pareto-dominates the baseline's best rule ($r$). On the
   diffusion arm the projector removes halfspace and obstacle violations but action-bound violations grow
   (64 unprojected → 145–408), which is why its clean count stays low. State the family, not a cause.
   The guard *"no tightened per-step cell"* around Table 6.8 is now obsolete.
5. **Held context (whole tightened alignment corpus, not new):** context 8 overlaps the tightened
   obstacle, is aborted and held, and scores clean in every tightened row, so each *x/10* is *x − 1* of
   nine attempted contexts. The diffusion $c$/$t$ single clean context is that one. Whether to state it is
   the author's call.
6. **R37a (job 26114) is a replicate, not a source:** it re-ran Table 6.7's $\nfe=2$ cell and moved by one
   context and 3 cm (time 0.5 %), the first measure of run-to-run variation on this task.
7. Negative percents in the median column are real: those boxes end farther from the target than they
   started.
