# DA — 2026-09-23 · the diffusion baseline at $\nfe=2$, five seeds (R26 closes)

**Preliminary.** Numbers are final and reproduce the corpus of record on every overlapping cell. The one
cost observation this DA flagged (§6) was explained on 23-09 in §29 of the ledger (R42): the $\nfe=2$
time is real, and it sits in the projection at t = 1.

**Ledger item:** R26, 🟡 in the NOW table of
[`PENDING_20260922_all_lacking_runs.md`](../../../logs_in_develop/Writing/Working_Space/data_status/PENDING_20260922_all_lacking_runs.md).
It fills the diffusion $\nfe=2$ row that `tab:avoiding-raw-models` (Table 6.1) and
`tab:avoiding-dpcc-protocol` (Table 6.2) print as *pending*.

**Corpus:** `temp/23-09/batch_avoiding_combined_20260923_110010(TenpK2D)`,
`candidates_multidimensional_raw.csv`.
**Runs:** trainings **26052–26055** (seeds 7–10, `n_diffusion_steps=2` patched in memory, the job-25965
pattern), evaluation **26112** (seeds 6–10, `afterok:26055`). Seed 6's checkpoint is the one job 25965
trained. Results tag `_msgdpccproto`, the same namespace as the flow-matching and CI-MeanFM protocol rows
of the 2026-09-17 wave.
**Protocol:** DPCC's own — 5 training seeds × 3 geometries × 2 episodes = 30 episodes per cell, four
candidate plans, tightened constraints.
**Aggregation:** verbatim the convention of
[`avoiding_unprojected_and_budget20.py`](avoiding_unprojected_and_budget20.py) — point estimate is the
mean per geometry over the seeds present, then the mean over the geometries; the spread is each seed
averaged over its geometries, then the sample standard deviation over the seeds.

---

## 1 · Coverage — complete, and it is a genuine five-seed row

| variant | seeds | geometries | missing |
| :-- | :-- | :-- | :-- |
| `diffuser` | 6, 7, 8, 9, 10 | all three | none |
| `dpcc-r-tightened` | 6, 7, 8, 9, 10 | all three | none |
| `dpcc-c-tightened` | 6, 7, 8, 9, 10 | all three | none |
| `dpcc-t-tightened` | 6, 7, 8, 9, 10 | all three | none |

Thirteen variants were evaluated; the four above are the ones the two tables use. `Missing_Seeds` is
empty for every one, and every seed–geometry pair is present, so this is 5 × 3 with no ragged cell —
unlike the $\nfe=20$ extended campaigns censused in §10 of the ledger.

🔴 **Do not pool with `_msgplanpanel63`.** A second $\nfe=2$ diffusion folder exists in the same corpus,
`H8_K2_T0.5_Dmodels.GaussianDiffusion_msgplanpanel63`, and it is **seed 6 only** (`Missing_Seeds =
[7, 8, 9, 10]`). It is the raw-plan figure panel from 2026-09-19, not a protocol cell. The two are
separate folders, so nothing mixes by accident, but the selection key must be `_msgdpccproto`.

## 2 · Validation — the new batch reproduces the corpus of record exactly

Every cell this batch shares with
`analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/batch_avoiding_combined_20260919_132703`
comes out identical to the last printed digit, including all three diffusion $\nfe=20$ rows
(0.967 / 74.7 / 573.2 · 1.000 / 70.1 / 553.4 · 1.000 / 76.1 / 563.0) and every MeanFM, CI-MeanFM and FM
cell at $\nfe=1,2$. The $\nfe=2$ diffusion row is therefore the only new content, and it enters a table
whose other rows are unchanged.

## 3 · The two rows the tables were waiting for

### Unprojected — `tab:avoiding-raw-models`

| model | $\nfe$ | success | S&C | violating steps | steps | ms/step |
| :-- | --: | --: | --: | --: | --: | --: |
| Diffusion | 1 | 0.933 ± 0.149 | 0.000 ± 0.000 | 19.0 ± 2.9 | 68.3 ± 16.1 | 9.4 ± 0.1 |
| **Diffusion** | **2** | **1.000 ± 0.000** | **0.000 ± 0.000** | **20.4 ± 2.4** | **61.7 ± 8.3** | **18.3 ± 0.1** |
| Diffusion | 10 | 0.967 ± 0.075 | 0.067 ± 0.149 | 19.1 ± 2.7 | 72.8 ± 8.1 | 88.8 ± 0.3 |
| Diffusion | 20 | 0.967 ± 0.075 | 0.100 ± 0.091 | 17.9 ± 2.5 | 67.8 ± 16.8 | 179.3 ± 1.2 |

### Projected, tightened — `tab:avoiding-dpcc-protocol`

| model | $\nfe$ | rule | S&C | violating steps | steps | ms/step |
| :-- | --: | :-- | --: | --: | --: | --: |
| **Diffusion** | **2** | $r$ | **1.000 ± 0.000** | 0.0 ± 0.0 | **65.1 ± 1.4** | **191.1 ± 26.5** |
| **Diffusion** | **2** | $c$ | **1.000 ± 0.000** | 0.0 ± 0.0 | **61.2 ± 1.2** | **211.4 ± 18.8** |
| **Diffusion** | **2** | $t$ | **0.933 ± 0.091** | 0.1 ± 0.1 | **60.9 ± 1.5** | **225.9 ± 48.5** |

Success (goal reached, ignoring constraints) is 1.000 ± 0.000 under all three rules.

## 4 · Reading it against the Target

The Target is the best projection variant of the pinned baseline, DPCC at $\nfe=20$ with `aw10` and
`GaussianDiffusion`: **$c$, S&C 1.000, 70.1 steps, 553.4 ms/step**.

| | S&C | steps | ms/step |
| :-- | --: | --: | --: |
| Target — Diffusion $\nfe=20$, $c$ | 1.000 | 70.1 | 553.4 |
| Diffusion $\nfe=2$, $c$ | 1.000 | 61.2 | 211.4 |

At equal S&C the $\nfe=2$ cell uses **fewer steps and 2.6× less time per step**, so it
**Pareto-dominates the Target**. The same holds for the $r$ rule (1.000, 65.1 steps, 191.1 ms, 2.9×).

🟠 **What this means for the thesis, and it cuts both ways.** The baseline's own best projected
configuration is not the one the chapter has been using. Two denoising steps plus the projector reach
the same S&C as twenty, at a third of the cost. The honest framing is that **the baseline is stronger
than its $\nfe=20$ row suggests**, and any claim of the form "our method is cheaper than DPCC" has to be
made against this row, not only against $\nfe=20$. It does not weaken the architecture-matched claim
below; it does raise the bar the claim clears.

## 5 · The architecture-matched comparison at the same budget

All four rows below are the **U-Net** backbone at $\nfe=2$, so the only difference is the generative
model. Best rule per model, S&C held at 1.000 where a model reaches it.

| model (U-Net) | $\nfe$ | rule | S&C | steps | ms/step | vs Diffusion $\nfe=2$ |
| :-- | --: | :-- | --: | --: | --: | --: |
| Diffusion | 2 | $c$ | 1.000 | 61.2 | 211.4 | — |
| FM | 2 | $c$ | 1.000 | 67.0 | 25.5 | **8.3× cheaper** |
| CI-MeanFM | 2 | $t$ | 1.000 | 60.1 | 27.0 | **7.8× cheaper** |
| MeanFM | 2 | $r$ / $t$ | 0.967 | 65.1 / 59.4 | 27.6 / 27.7 | does not hold S&C |

FM and CI-MeanFM hold S&C at 1.000 while spending roughly an eighth of the time per step. CI-MeanFM also
uses fewer steps than the baseline (60.1 against 61.2), so it Pareto-dominates it; FM takes 5.8 more
steps, so FM against the baseline is a **trade-off on steps and a win on time**, not a domination.
MeanFM does not reach 1.000 at this budget and is not in the comparison.

## 6 · 🟠 One open observation — the diffusion arm's projector cost is not on the flow arms' scale

> ✅ **RESOLVED 23-09 — see §29 of the ledger (R42, closed without a run).** The K2 time is real and
> reproducible: job 25966 (19-09, seed 6) measured 226 / 225 / 276 ms and job 26112 (22-09, five seeds)
> 211 / 191 / 226 ms, three days apart, so node load is ruled out. `post_processing-tightened` of the same
> cell — the final step only — costs 28.1 ms, which puts ≈185 ms on the projection at **t = 1**, a state one
> denoising step from pure noise. No larger budget hands the projector such a state, and SLSQP's cost is set
> by how infeasible the warm start is, not by the solve count. The numbers below stand; the recommendation
> that closes this section is superseded. The thesis now states the explanation and a 12× cost factor
> against the cheapest diffusion configuration.

Subtracting the unprojected cost from the projected ($c$ rule) gives what the projector adds per control
step:

| cell | unprojected | projected ($c$) | projector adds |
| :-- | --: | --: | --: |
| Diffusion $\nfe=1$ | 9.4 | 33.3 | 23.9 |
| Diffusion $\nfe=2$ | 18.3 | 211.4 | **193.1** |
| Diffusion $\nfe=10$ | 88.8 | 309.8 | 221.0 |
| Diffusion $\nfe=20$ | 179.3 | 553.4 | 374.1 |
| FM $\nfe=1$ | 9.0 | 17.3 | 8.3 |
| FM $\nfe=2$ | 17.8 | 25.5 | **7.7** |

The generation costs match within 3% at $\nfe=2$ (18.3 against 17.8 ms), but the projector adds 193 ms on
the diffusion arm and 8 ms on the flow arm, at the same budget, the same four candidates and the same
tightened constraint set. At $\nfe=20$ the two arms are on the same scale again (374 ms against FM's
~300 ms), so this is specific to the low-budget diffusion cells, and the jump from $\nfe=1$ to $\nfe=2$
(24 → 193 ms) is far steeper than the step count explains.

**Recommendation for the draft:** report the $\nfe=2$ diffusion times as measured, and do **not** yet
attribute the gap to the projector as a general property. Either the two arms project a different number
of sampling steps at the same $\nfe$, or the $\nfe=1$ diffusion cell is near-degenerate and the flow
cells at $\nfe\le2$ are too. Resolving it is a code reading, not a run: compare the projection schedule
of `scripts/eval.py` (gate `t <= T·n_timesteps`, t counting down) against the FM v3 sampler's
`int((1-T)·K)` window. Until then the cost ratio of §5 is a measurement, not a mechanism.

## 7 · What this closes, and what it changes

| | |
| :-- | :-- |
| **R26** | ✅ **data-complete.** Both pending diffusion $\nfe=2$ cells can be filled, as a genuine 5 seeds × 3 geometries × 2 episodes row |
| `tab:avoiding-raw-models` | add the unprojected row of §3 |
| `tab:avoiding-dpcc-protocol` | add the three projected rows of §3 |
| §6.1 prose | the baseline's best projected configuration is $\nfe=2$, not $\nfe=20$ (§4) — this needs a sentence, and any cost claim against DPCC should name this row |
| `fig:avoiding-tradeoff`, `fig:k-ladder` | both read the DPCC-protocol cells; the diffusion curve gains a point at $\nfe=2$ that sits well below $\nfe=10$ and $\nfe=20$ in cost at equal S&C |
| unchanged | every other cell of both tables; the 19-09 corpus reproduces exactly (§2) |

**Not claimed here.** Nothing about the extended (20-episode) protocol: that campaign stays quarantined
(§12 of the ledger) and this run is DPCC's protocol only. Nothing about MeanFM at $\nfe=2$ beyond the
S&C it already had.

## 8 · Reproducing these numbers

Selection keys, for whoever promotes this into the official scripts:

```
folder   H8_K2_T0.5_Dmodels.GaussianDiffusion_msgdpccproto
path     must contain /plans/diffusion/        (the naming trap: Dmodels.diffusion.GaussianDiffusion
                                                under flow_matching_v3_* is the FLOW model)
variants diffuser, dpcc-r-tightened, dpcc-c-tightened, dpcc-t-tightened
geos     top-left-hard, top-right-hard, both-hard
metrics  n_success, n_success_and_constraints, n_violations, n_steps, avg_time
```

Computed with a scratch script using the aggregation of
[`avoiding_unprojected_and_budget20.py`](avoiding_unprojected_and_budget20.py) unchanged. It is **not**
yet added to this folder: adding the cell to `UNPROJECTED` in that script and to the DPCC-protocol block
of [`avoiding_rules_by_protocol.py`](avoiding_rules_by_protocol.py) is the permanent home, and needs the
corpus of record to move to this batch (or the K2 folder to be merged into it) first. Say the word and I
will make that edit.
