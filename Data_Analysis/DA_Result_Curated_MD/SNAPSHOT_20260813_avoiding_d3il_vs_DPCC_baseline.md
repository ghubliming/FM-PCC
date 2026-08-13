# `avoiding-d3il` — configurations beating the DPCC baseline

> **SNAPSHOT 2026-08-13.** Regenerated as new batches land; numbers change. Use the newest
> `SNAPSHOT_<date>_*` file in this folder.

**Batch:** `batch_avoiding_combined_20260813_102739` (`temp/1308/…/candidates_multidimensional_raw.csv`)
**Eval jobs:** 24515 (AlphaFlow), 24516 (MeanFlow-DiT), 24416/24496 (MeanFlow-UNet). All rows within-batch.
**Protocol:** 5 seeds {6–10} × 3 halfspace settings × 2 trials = 30 episodes per configuration.
**Statistics:** seed-clustered bootstrap, B = 20 000, 95 % percentile CI. `*` = CI excludes 0.

---

## 1. Definitions

**Target:** DPCC as published — `K=20`, `aw=10`, `GaussianDiffusion`, best projection variant
`dpcc-c-tightened` (C14): **S&C 1.000 · 70.13 steps · 0.5534 s/step · 38.53 s/episode**.

**Metrics.** `S&C` = fraction of episodes reaching goal ∧ satisfying all constraints.
`steps` = steps to goal, averaged over successful episodes only. `s/ep` = `steps × s/step`.

**Rule.** Beat = S&C ≥ 1.000 and improvement on `steps` or `s/ep`. Both = strict Pareto dominance.

**Architecture.** The baseline is a temporal UNet (`models.UNet1DTemporalCondModel`). Only the
`unet` arm is architecture-matched; `sit`/`dit` arms change network and objective simultaneously.

| engine | backbone | params | matched |
|---|---|---|---|
| MeanFlow-UNet | `unet`, `freq_dim=32` | 4.0 M | ✅ |
| MeanFlow-DiT | `mf_dit` | 10.1 M | ❌ |
| AlphaFlow-SiT | `sit` | 10.0 M | ❌ |

---

## 2. Results

All rows below clear S&C = 1.000. Δ is versus Target.

| engine | arch | config | S&C | steps | s/ep | Δ steps | Δ s/ep | ×faster |
|---|---|---|---|---|---|---|---|---|
| AlphaFlow-SiT | ❌ | K1 `dpcc-t-tight` | 1.000 | 66.10 | **0.92** | −4.03 `[−11.37,+2.73]` | **−37.61 `[−40.50,−34.65]`** `*` | **41.7** |
| MeanFlow-DiT | ❌ | K1 `dpcc-t-tight` | 1.000 | 76.90 | 2.25 | +6.77 `[−1.73,+15.57]` | **−36.29 `[−39.18,−33.27]`** `*` | 17.2 |
| **MeanFlow-UNet** | **✅** | K1 `hardflow-tight` | 1.000 | **63.77** | 2.64 | −6.37 `[−13.60,+0.23]` | **−35.89 `[−38.77,−32.91]`** `*` | 14.6 |
| AlphaFlow-SiT | ❌ | K10 `dpcc-c-tight` | 1.000 | **62.40** | 22.42 | **−7.73 `[−14.53,−1.60]`** `*` | **−16.11 `[−19.87,−12.13]`** `*` | 1.7 |
| DPCC K10 | ✅ | `dpcc-c-tight` | 1.000 | 70.33 | 21.66 | +0.20 `[−10.83,+11.17]` | **−16.87 `[−20.54,−13.06]`** `*` | 1.8 |
| naive FM | ✅ | K20 `dpcc-c-tight` | 1.000 | 63.23 | 29.65 | **−6.90 `[−13.73,−0.70]`** `*` | **−8.88 `[−12.01,−5.56]`** `*` | 1.3 |

**Architecture-matched result.** MeanFlow-UNet K1 `hardflow-tightened`, 4.0 M params: **14.6× lower
cost at S&C 1.000**, one-axis win. Lowest step count of any qualifying row (63.77); interval touches
zero, not claimed. Matched-group ordering: MeanFlow-UNet 2.64 < DPCC K10 21.66 < naive FM K20
29.65 < Target 38.53 s/ep.

**Cross-architecture result.** AlphaFlow-SiT K1 reaches 41.7×. AlphaFlow-SiT K10 is the only strict
Pareto domination in the study (both axes `*`). Both use a different, 2.5× larger network, so the
margin is not attributable to the objective. **No architecture-matched configuration strictly
dominates the Target.** Missing control: baseline and naive FM on a SiT/DiT backbone.

**K = 20 is worse than the baseline.** MeanFlow-DiT K20 **+24.62 s/ep** `[+19.77,+29.91]` `*`;
AlphaFlow-SiT K20 **+26.88** `[+18.42,+38.73]` `*`. The baseline's own K10 (21.66) beats its K20
(38.53). Every engine is cheapest at K = 1–2.

---

## 3. Failing configurations

MeanFlow-UNet post-hoc-projected arms never reach the gate, at any K:

| arm | best S&C (K) | shortfall |
|---|---|---|
| `dpcc-t-tightened` | 0.967 (K1, K2) | 1 episode / 30 |
| `dpcc-r-tightened` | 0.967 (K2) | 1 episode |
| `dpcc-c-tightened` | 0.933 (K1, K2) | 2 episodes |
| `hardflow-tightened`, K ≥ 2 | 0.933 (K2, K5) | 2 episodes |

S&C by K, best DPCC arm: 0.967 (K1) · 0.967 (K2) · 0.933 (K5) · 0.933 (K10). All losses occur on
`top-right-hard`; the K1 and K2 shortfall is the same (seed 7, `top-right-hard`) initial condition.

Slower than the baseline: MeanFlow-DiT K20 `dpcc-c-tight` 63.15 s/ep, AlphaFlow-SiT K20
`dpcc-t-tight` 65.41, MeanFlow-DiT K20 `dpcc-r-tight` 70.69.

---

## 4. Method comparisons

| comparison | result |
|---|---|
| Any engine, unprojected | S&C 0.000–0.267. All constraint satisfaction comes from the projection layer. |
| MeanFlow-UNet vs naive FM | 2.64 vs 29.65 s/ep = **11.2×** at equal S&C; steps tie (63.77 vs 63.23). |
| In-loop (HardFlow) vs post-hoc (DPCC), same checkpoint | K1: 1.000 vs 0.967 — in-loop clears the gate, at 2.5× per-step cost. K ≥ 2: post-hoc equal or better. |
| MeanFlow backbone, UNet vs DiT | UNet −7 to −18 steps at every K; DiT +0.033 S&C at K1/K5/K10, reaches 1.000 at K1/5/10/20 vs UNet at K1 only. Trade-off. |
| AlphaFlow backbone, SiT vs UNet | SiT 1.000 vs UNet 0.833 at K2. |

---

## 5. Limits

- 30 episodes per configuration; S&C resolves to 1/30. Bootstrap resamples 5 seed clusters — step CIs are wide.
- `steps` averages over successful episodes only; comparable only at equal S&C. All §2 rows are gate-matched at 1.000.
- `s/step` is wall-clock on shared GPUs, includes the constraint solve. Differences < 10–20 % are not resolvable.
- Architecture confound unresolved (§1). Baseline and naive FM on SiT/DiT not run.
- Adjacent-window train/test overlap at H = 8 affects all engines equally.
- Not run: MeanFlow-UNet K20; HardFlow activation-threshold sweep at 5 seeds (single-seed only).

---

## 6. Reproduction

Candidate IDs, this batch only — IDs do not transfer between CSVs.

| config | ID | config | ID |
|---|---|---|---|
| **Target** DPCC K20 `aw10` | **C14** | MeanFlow-UNet K1/2/5/10 | C128/C130/C134/C126 |
| DPCC K10 / K1 | C7 / C8 | MeanFlow-DiT K1/2/5/10/20 | C120/C122/C123/C119/C121 |
| naive FM K20 | C142 | AlphaFlow-SiT K1/2/5/10/20 | C39/C41/C42/C38/C40 |

```
filter: Candidate == <ID>, seed ∈ {6,7,8,9,10}, halfspace_variant ∈ {top-right,top-left,both}-hard
value : mean over the 15 (seed × halfspace) cells
        S&C -> n_success_and_constraints ; steps -> n_steps ; s/step -> avg_time
        s/ep -> mean of (n_steps × avg_time) per cell   [not mean(steps) × mean(s/step)]
bootstrap: resample the 5 seeds with replacement, B = 20000, percentile 95 % CI
```
