# `avoiding-d3il` — configurations beating the DPCC baseline

> **SNAPSHOT 2026-08-13.** Regenerated as new batches land; numbers change. Use the newest
> `SNAPSHOT_<date>_*` file in this folder.

**Batch:** `batch_avoiding_combined_20260813_102739` (`temp/1308/…/candidates_multidimensional_raw.csv`)
**Eval jobs:** 24515 (AlphaFlow), 24516 (MeanFlow-DiT), 24416/24496 (MeanFlow-UNet). All rows within-batch.
**Protocol:** 5 seeds {6–10} × 3 halfspace settings × 2 trials = 30 episodes per configuration.
**Statistics:** seed-clustered bootstrap, B = 20 000, 95 % percentile CI. `*` = CI excludes 0.

---

## 0. Metrics

| symbol | CSV field | meaning | direction |
|---|---|---|---|
| **`S&C`** | `n_success_and_constraints` | Fraction of episodes that reach the goal **and** violate no constraint at any step. The safety gate; a run that reaches the goal through an obstacle scores 0. Resolution 1/30. | higher better |
| **`steps`** | `n_steps` | Control steps taken to reach the goal — path length in decisions. Averaged over **successful episodes only**. | lower better |
| **`s/step`** | `avg_time` | Wall-clock seconds per control step: one trajectory generation (K network calls) **plus** the constraint projection solve. Per-decision compute. | lower better |
| **`s/ep`** | derived | **`steps × s/step` — wall-clock seconds to complete one episode.** The deployment-relevant cost: a policy can be cheap per step but need many steps. Computed per (seed × setting) cell, then averaged. | lower better |
| **`×faster`** | derived | `Target s/ep ÷ row s/ep`. | higher better |
| **`Δ`** | derived | Difference of means, row − Target. Negative = row is better on `steps` / `s/ep`. | — |
| **`[a, b]`** | derived | 95 % bootstrap CI on Δ. **`*`** = excludes 0, i.e. the difference resolves at this sample size. No `*` = not distinguishable from noise. | — |

**Configuration names.** `K` = sampling steps (network evaluations per plan). `dpcc-{r,c,t}` =
post-hoc projection with candidate-selection rule random / min-cost / temporal-consistency;
`-tightened` = 0.025 constraint margin; `hardflow-*` = projection solved **inside** the sampling
loop instead of after it; `diffuser` = no projection.

---

## 1. Definitions

**Target:** DPCC as published — `K=20`, `aw=10`, `GaussianDiffusion`, best projection variant
`dpcc-c-tightened` (C14): **S&C 1.000 · 70.13 steps · 0.5534 s/step · 38.53 s/episode**.

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

### 2.1 Matched-budget comparison (same K)

§2 compares our best row against the baseline at *its* K. This section fixes K on both sides. It
answers a different question: at an equal sampling budget, is the flow engine better?

**K = 1, 5 seeds — opponent: diffusion-DPCC K1 (C8)**

| engine | best arm | S&C | steps | s/ep | ΔS&C | Δ steps | Δ s/ep |
|---|---|---|---|---|---|---|---|
| DPCC K1 *(opponent)* | `dpcc-c-tight` | **0.667** | 72.07 | 2.68 | — | — | — |
| **MeanFlow-UNet K1** | `hardflow-tight` | **1.000** | 63.77 | 2.67 | **+0.333 `[+0.03,+0.63]`** `*` | −8.30 `[−23.8,+4.1]` | −0.01 `[−1.79,+1.05]` |
| MeanFlow-DiT K1 | `dpcc-t-tight` | **1.000** | 76.90 | 2.25 | **+0.333 `[+0.03,+0.63]`** `*` | +4.83 `[−11.3,+18.2]` | −0.44 `[−2.25,+0.72]` |
| AlphaFlow-SiT K1 | `dpcc-t-tight` | **1.000** | 66.10 | **0.92** | **+0.333 `[+0.03,+0.63]`** `*` | −5.97 `[−21.7,+6.4]` | **−1.76 `[−3.53,−0.73]`** `*` |

**At one network evaluation, diffusion-DPCC reaches S&C 0.667; all three flow engines reach 1.000,
CI excluding zero.** This is the only matched-budget axis on which the flow engines win
significantly. Cost at K = 1 is a tie for MeanFlow-UNet (2.67 vs 2.68 — its in-loop NLP consumes the
saving) and a significant win only for AlphaFlow-SiT.

**K = 10, 5 seeds — opponent: diffusion-DPCC K10 (C7)**

| engine | best arm | S&C | steps | s/ep | Δ steps | Δ s/ep |
|---|---|---|---|---|---|---|
| DPCC K10 *(opponent)* | `dpcc-t-tight` | 1.000 | 68.70 | **22.14** | — | — |
| MeanFlow-UNet K10 | `dpcc-t-tight` | **0.933** | 63.63 | 25.59 | −5.07 `[−10.7,+0.5]` | **+3.45 `[+0.75,+6.00]`** `*` |
| MeanFlow-DiT K10 | `dpcc-r-tight` | 1.000 | 68.27 | 26.71 | −0.43 `[−6.3,+5.3]` | **+4.57 `[+1.91,+7.24]`** `*` |
| AlphaFlow-SiT K10 | `dpcc-c-tight` | 1.000 | **62.40** | 22.42 | **−6.30 `[−12.0,−0.9]`** `*` | +0.28 `[−3.12,+3.66]` |

**At K = 10 no flow engine beats diffusion-DPCC on cost.** Two are significantly slower;
AlphaFlow ties on cost and wins on steps. MeanFlow-UNet also drops below the gate (0.933).

**K = 5, seed 6 only (6 episodes) — opponent: naive FM K5 (C143)**

| engine | best arm | S&C | steps | s/ep | Δ steps | Δ s/ep |
|---|---|---|---|---|---|---|
| naive FM K5 *(opponent)* | `dpcc-t-tight` | 1.000 | 62.67 | **7.07** | — | — |
| MeanFlow-UNet K5 | `dpcc-t-tight` | 1.000 | **59.33** | 13.07 | −3.33 | +6.00 |
| MeanFlow-DiT K5 | `dpcc-c-tight` | 1.000 | 67.17 | 14.10 | +4.50 | +7.03 |
| AlphaFlow-SiT K5 | `dpcc-t-tight` | 1.000 | 60.83 | 12.16 | −1.83 | +5.08 |

At equal K and equal S&C, naive FM is **1.7–1.8× cheaper per episode** than either few-step engine;
MeanFlow-UNet and AlphaFlow take fewer steps. Single seed — directional only.

**Reading.** The matched-budget result is narrow and specific:

- **Won:** reliability at K = 1. Diffusion-DPCC does not work at one step (0.667); the flow engines
  do (1.000, significant). Path length is shorter for MeanFlow-UNet and AlphaFlow at K = 1, 5 and
  10, though only AlphaFlow's K = 10 step margin resolves.
- **Not won:** wall-clock at equal K. At K = 5 and K = 10 the baselines are equal or cheaper per
  episode; our per-step cost is higher at the same budget.
- **Therefore** the 14.6×/41.7× in §2 does **not** come from being faster at equal compute. It comes
  from **being usable at K = 1 while the baseline requires K = 20** — the baseline cannot follow us
  down the ladder (S&C 0.667 at K1), and that gap is the result.

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
| MeanFlow-UNet vs naive FM | K20: 2.64 vs 29.65 s/ep = **11.2×** at equal S&C, steps tie (63.77 vs 63.23). Against naive FM at K5 the gap is **2.7×** — see §4.1. |
| In-loop (HardFlow) vs post-hoc (DPCC), same checkpoint | K1: 1.000 vs 0.967 — in-loop clears the gate, at 2.5× per-step cost. K ≥ 2: post-hoc equal or better. |
| MeanFlow backbone, UNet vs DiT | UNet −7 to −18 steps at every K; DiT +0.033 S&C at K1/K5/K10, reaches 1.000 at K1/5/10/20 vs UNet at K1 only. Trade-off. |
| AlphaFlow backbone, SiT vs UNet | SiT 1.000 vs UNet 0.833 at K2. |

### 4.1 naive FM (FMv3ODE) at lower K

Multi-seed data exists only at K20. Lower-K rows are **seed 6 only (6 episodes)** and are listed
separately for that reason. Protocol-matched rows require `T = 0.5`; the K10 rows in this batch are
`T = 0.05` / `T = 0.1` and are excluded.

| K | candidate | seeds | best arm | S&C | steps | s/step | s/ep |
|---|---|---|---|---|---|---|---|
| 5 | C143 | **6 only** | `dpcc-t-tightened` | 1.000 | 62.67 | 0.1154 | **7.07** |
| 20 | C142 | 6 only | `dpcc-c-tightened` | 1.000 | 62.17 | 0.4887 | 30.06 |
| 20 | C142 | 6–10 | `dpcc-c-tightened` | 1.000 | 63.23 | 0.4767 | 29.65 |
| *(K10)* | C138/C139 | 6 only | — | — | — | — | *excluded, `T=0.05`/`0.1`* |

Naive FM at K5 costs **4.3× less than its own K20** at S&C 1.000 (7.07 vs 30.06 s/ep, seed 6), so
its 1.3× margin over the Target in §2 is an artifact of pinning it at K20; at K5 it is ≈5.5×
(single seed). Seed 6 is not optimistic for this engine — its K20 seed-6 numbers match the 5-seed
values (1.000 / 62.17 vs 1.000 / 63.23).

**Consequence for §4:** MeanFlow-UNet's advantage over naive flow matching is **11.2× against
naive FM K20 but only 2.7× against naive FM K5** (2.64 vs 7.07 s/ep). The low-K benefit is not
exclusive to the few-step objective — naive FM captures much of it by simply reducing K. A
multi-seed naive-FM K1/K2 ladder is required before the few-step claim can be separated from the
low-K claim.

---

## 5. Limits

- 30 episodes per configuration; S&C resolves to 1/30. Bootstrap resamples 5 seed clusters — step CIs are wide.
- `steps` averages over successful episodes only; comparable only at equal S&C. All §2 rows are gate-matched at 1.000.
- `s/step` is wall-clock on shared GPUs, includes the constraint solve. Differences < 10–20 % are not resolvable.
- Architecture confound unresolved (§1). Baseline and naive FM on SiT/DiT not run.
- Adjacent-window train/test overlap at H = 8 affects all engines equally.
- Not run: MeanFlow-UNet K20; HardFlow activation-threshold sweep at 5 seeds (single-seed only);
  **naive FM below K20 at multi-seed** (K5 is seed-6 only, K1/K2 absent) — see §4.1.
- Matched-K coverage (§2.1) is limited by what the baselines have: DPCC exists at K1/K10/K20 only
  (no K2, K5); naive FM at `T=0.5` exists at K5 (seed 6) and K20 only. K = 1 and K = 10 are the only
  matched points with 5 seeds on both sides.

---

## 6. Reproduction

Candidate IDs, this batch only — IDs do not transfer between CSVs.

| config | ID | config | ID |
|---|---|---|---|
| **Target** DPCC K20 `aw10` | **C14** | MeanFlow-UNet K1/2/5/10 | C128/C130/C134/C126 |
| DPCC K10 / K1 | C7 / C8 | MeanFlow-DiT K1/2/5/10/20 | C120/C122/C123/C119/C121 |
| naive FM K20 / K5 *(seed 6)* | C142 / C143 | AlphaFlow-SiT K1/2/5/10/20 | C39/C41/C42/C38/C40 |

```
filter: Candidate == <ID>, seed ∈ {6,7,8,9,10}, halfspace_variant ∈ {top-right,top-left,both}-hard
value : mean over the 15 (seed × halfspace) cells
        S&C -> n_success_and_constraints ; steps -> n_steps ; s/step -> avg_time
        s/ep -> mean of (n_steps × avg_time) per cell   [not mean(steps) × mean(s/step)]
bootstrap: resample the 5 seeds with replacement, B = 20000, percentile 95 % CI
```
