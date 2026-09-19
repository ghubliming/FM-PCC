# DA — 2026-09-19 · the 09-17 and 09-18 waves, analysed

**Corpora of record for this analysis**
`Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/`
· `batch_avoiding_combined_20260919_132703` (supersedes the 15-09 avoiding batch)
· `batch_uav_20260919_111701` (supersedes the 15-09 UAV batch)
Visual aligning is unchanged — nothing ran — so `batch_va2_20260915_100754` stays the alignment corpus.

Recomputed with [`avoiding_rules_by_protocol.py`](avoiding_rules_by_protocol.py) and
[`uav_results.py`](uav_results.py), both repointed at the new batches. **Regression check:** every number
the 15-09 batch produced is reproduced exactly — DPCC $\nfe=20$ `dpcc-c-tightened` 0.983 / 69.0 steps /
563.5 ms and MeanFM $\nfe=1$ `dpcc-t-tightened` 0.993 / 61.0 / 18.1 ms, as in DA_20260827 §10.1. The new
rows are additions, not a re-derivation of the old ones.

Runbooks: [`SLURM_RUNBOOK_20260917`](../../../logs_in_develop/Writing/Working_Space/data_status/SLURM_RUNBOOK_20260917_dpcc_protocol_rows.md)
· [`SLURM_RUNBOOK_20260918`](../../../logs_in_develop/Writing/Working_Space/data_status/SLURM_RUNBOOK_20260918_pending_runs.md)

---

## 1 · D3IL-avoiding at DPCC's protocol — the table is complete (R8, R19)

Five training seeds × two episodes per geometry, three geometries, tightened constraints, four candidate
plans. Aggregation is per geometry first, then across geometries. **S&C = success and constraint
satisfaction.**

| model | $\nfe$ | rule | S&C | steps | ms/step |
| :-- | --: | :-- | --: | --: | --: |
| Diffusion (DPCC), the baseline | 20 | `dpcc-r` | 0.967 | 74.7 | 573.2 |
| | 20 | **`dpcc-c`** | **1.000** | **70.1** | **553.4** |
| | 20 | `dpcc-t` | 1.000 | 76.1 | 563.0 |
| Instantaneous-velocity matching (FM) | 1 | `dpcc-r` | 1.000 | 67.9 | 17.4 |
| | 1 | `dpcc-c` | 1.000 | 67.0 | 17.3 |
| | 1 | `dpcc-t` | 1.000 | 68.2 | 17.7 |
| | 2 | `dpcc-r` | 1.000 | 65.7 | 26.2 |
| | 2 | `dpcc-c` | 1.000 | 67.0 | 25.5 |
| | 2 | `dpcc-t` | 1.000 | 63.8 | 25.9 |
| Consistency-interpolated (CI-MeanFM, U-Net, floor 0.2) | 1 | `dpcc-r` | 1.000 | 61.5 | 17.9 |
| | 1 | `dpcc-c` | 0.967 | 68.4 | 18.0 |
| | 1 | **`dpcc-t`** | **1.000** | **59.2** | **18.1** |
| | 2 | `dpcc-r` | 0.900 | 63.3 | 27.0 |
| | 2 | `dpcc-c` | 0.933 | 89.7 | 26.6 |
| | 2 | `dpcc-t` | 1.000 | 60.1 | 27.0 |

New rows: the twelve flow-matching and CI-MeanFM cells (jobs 25878, 25880). The MeanFlow and diffusion rows
are unchanged from the 15-09 batch and are printed by the same script.

**The architecture-matched result, at the baseline's own protocol.** Against the pinned target — diffusion
$\nfe=20$ under its best rule, `dpcc-c`, at S&C 1.000 / 70.1 steps / 553.4 ms — both flow models at
$\nfe=1$ hold S&C at 1.000 **and** finish in fewer control steps **and** cost ≈ 31× less per step. That is
Pareto dominance on all three axes, from a 4.0 M-parameter U-Net against a 4.0 M-parameter U-Net:

* flow matching $\nfe=1$, `dpcc-c`: 1.000 · 67.0 steps · 17.3 ms — **32.0×** cheaper per step
* CI-MeanFM $\nfe=1$, `dpcc-t`: 1.000 · 59.2 steps · 18.1 ms — **30.6×** cheaper, and the fewest steps of any cell in the table

Flow matching holds S&C 1.000 under **all three** selection rules at $\nfe=1$ and $\nfe=2$ — the only model
in the table that does. CI-MeanFM is rule-sensitive: 1.000 under temporal consistency at both budgets, but
0.900 under random selection at $\nfe=2$, and its `dpcc-c` cell at $\nfe=2$ costs 89.7 steps — the same
selection-rule stall already reported for MeanFM's $\nfe=2$ `dpcc-c` (98.0 steps).

The baseline itself is not perfect at this protocol: `dpcc-r` costs it 0.967.

## 2 · UAV-corridor — endpoint projection completed across all four selection rules (R20)

Corridor v2 with the slide, tag `u17cv2`, seed 6, 12 flights per cell (four per route L/C/R). The 09-18
wave added `hardflow_sls-r` and `hardflow_sls-c` at $\nfe=3$ and $5$ for all three flow models; the table
now has every endpoint rule beside every per-step rule.

**S&C, full projected set (passed the goal on a collision-free flight):**

| model | $\nfe$ | per-step `dpcc-r` / `-c` / `-t` | endpoint single / `-r` / `-c` / `-t` |
| :-- | --: | :-- | :-- |
| MeanFM | 3 | 1.00 / 1.00 / 1.00 | 0.42 / **0.50** / **0.58** / 0.42 |
| MeanFM | 5 | 1.00 / 1.00 / 1.00 | 0.00 / **0.08** / **0.33** / 0.17 |
| CI-MeanFM | 3 | 1.00 / 1.00 / 1.00 | 0.25 / **0.33** / **0.42** / 0.58 |
| CI-MeanFM | 5 | 1.00 / 1.00 / 1.00 | 0.00 / **0.08** / **0.00** / 0.17 |
| FM | 3 | 0.00 / 0.00 / 0.00 | 0.00 / **0.00** / **0.00** / 0.00 |
| FM | 5 | 0.00 / 0.08 / 0.17 | 0.00 / **0.00** / **0.00** / 0.00 |

Bold = new this wave.

**The question R20 was opened to answer is settled: endpoint projection does not overtake the per-step
projector on this scene under any selection rule.** Its best cell is minimum-cost selection at $\nfe=3$
(MeanFM 0.58), against 1.00 for every per-step rule at the same budget — and the two rules added here do not
change the ordering, they fill it in. Endpoint projection is the cheaper arm (74 ms/step against 99 at
MeanFM $\nfe=3$, and 47 ms for the single-candidate form), so the trade it offers is roughly half the
projection cost for roughly half the constraint satisfaction. The bar endpoint projection has to clear —
beating the DPCC projector at a lower projection threshold — is **not** cleared on this scene.

Endpoint projection also gets **worse** as the budget rises (0.58 → 0.33 at MeanFM), while per-step stays at
1.00. Raising $\nfe$ lengthens the ODE the endpoint solve has to anticipate.

**A second, unplanned result from the guard cell.** The driver needed a non-HardFlow companion variant and
used `dpcc-c-bounds_free-pdes` — the published `dpcc-c` cell *without* the tightening margin. It scores
**0.00** at every budget for every model (48.1 violating steps at MeanFM $\nfe=3$), against 1.00 for the
tightened twin. On this scene the DPCC tightening margin is not an improvement to the projector; it is the
whole of its constraint satisfaction. That cell was free — it exists only because the eval refuses a
HardFlow-only variant set.

**Flow matching fails this scene** at every budget and under every rule (0.00, except 0.08/0.17 at
$\nfe=5$), while MeanFM and CI-MeanFM reach 1.00 with the same projector. Its unprojected plan has fewer
violating steps than theirs (42.6 against 54.2 at $\nfe=1$), so the difference appears under projection, not
before it.

## 3 · UAV-s-curve — the re-run is complete and the scene yields nothing (R10)

Tag `u18sc`, seed 6, 10 flights per cell, post switched-wall fix, six variants per model.

| model | $\nfe$ | S&C, all six variants | collision-free | flights aborted by the divergence guard |
| :-- | --: | --: | --: | :-- |
| FM | 20 | 0.00 | 0.00 | **34 / 60 (57 %)** |
| MeanFM | 10 | 0.00 | 0.00 | **55 / 60 (92 %)** |
| CI-MeanFM | 5 | 0.00 | 0.00 | **52 / 60 (87 %)** |

**Every cell is zero.** The pre-fix endpoint rows were withheld because of the switched-wall bug; the fixed
re-run does not rescue the scene, it confirms there is nothing to rescue. R10 closes as *run, all-zero*, not
as a result.

The divergence guard ends a flight when the drone inverts (`body z-axis · world z < 0`). It fires on the
**unprojected** plan as well — 4/10 for FM, 10/10 for MeanFM, 8/10 for CI-MeanFM — so this is the scene and
the tracking controller, not the projector and not a HardFlow effect. The guard is not new; it predates this
wave, so the earlier s-curve numbers were produced under it too.

s-curve should be reported as a **failure case with its abort counts beside the scores**. Quoting the S&C
column alone would read as a ranking of methods where the underlying event is that the aircraft flipped.

## 4 · UAV-pillars — withheld, not analysed

118 `pillars_hg` cells are in the batch (tags `u7hg` and `u7hga1`, including the $\nfe=1$ per-step and
$\nfe=2$ endpoint cells this wave produced). **None is reported.** `pillars_hg` enforces the constraint the
demonstration generator was built to satisfy, so its projected configurations measure how little of an
already-feasible plan a method disturbs — see
[`PENDING_20260918_pillars_geometry_redesign.md`](../../../logs_in_develop/Writing/Working_Space/data_status/PENDING_20260918_pillars_geometry_redesign.md).
`uav_results.py` gates the block behind `PILLARS_EXCLUDED = True` rather than deleting it, so the scene can
be restored once `pillars_xl` / `pillars_xxl` have been evaluated.

## 5 · Ledger effect

| row | was | now |
| :-- | :-- | :-- |
| R8, R19 | ⏳ four pending rows in `tab:avoiding-dpcc-protocol` | ✅ closed — §1 |
| R6 | ⏳ CI-MeanFM beyond seed 6 | ✅ at DPCC's protocol (seeds 6–10 trained and evaluated); the extended protocol still has seed 6 only |
| R20 | ⏳ endpoint `-r`/`-c` on corridor, and $\nfe=1$ | ✅ for $\nfe=3,5$ — §2. The $\nfe=1$ half is ❌ *not runnable*: no non-terminal step exists to guide |
| R10 | ⏳ s-curve endpoint re-run | ✅ run — §3 — all-zero; report as a failure case |
| R12, R13, R21 | ⏳ pillars | ⏸ superseded by the pillars redesign, not closed |
| R22 | ⏳ baseline under endpoint projection | ❌ *model limit* — the diffusion engine exposes no velocity field (`supports_hardflow=False`) |
| R18 | ⏳ baseline at 5 × 20 below its training budget | ⏳ still open; phase B is unblocked now the 09-17 queue is clear |

## 6 · Figures were not rebuilt, and did not need to be

`plotting/sources.py` still points every `Corpus` at its own `temp/` drop, not at these checkpoint batches,
and no published number changed, so every existing figure remains correct as built. Two figures will need
attention when the pillars replacement lands rather than now: `fig_uav_pillars_paths` and any panel drawing
on `pillars_hg`. The corridor and s-curve additions of this wave are table rows, not figure series.
