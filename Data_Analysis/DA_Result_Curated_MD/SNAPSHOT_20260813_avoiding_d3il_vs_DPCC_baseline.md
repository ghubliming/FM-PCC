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

**At one network evaluation the published DPCC config (`aw10`) reaches S&C 0.667; all three flow
engines reach 1.000, CI excluding zero.** Cost at K = 1 is a tie for MeanFlow-UNet (2.67 vs 2.68 —
its in-loop NLP consumes the saving) and a significant win only for AlphaFlow-SiT.

🔴 **This does not generalise to all diffusion configs.** A second K = 1 diffusion row exists —
`ode_selectable / GaussianDiffusion, aw1, Euler, T0.5` (**C146**, 5 seeds) — and it **does** reach
S&C 1.000, at 1.17 s/ep. Against that opponent the matched-K = 1 picture is different:

| engine (K = 1) | S&C | steps | s/ep | ΔS&C | Δ steps | Δ s/ep |
|---|---|---|---|---|---|---|
| C146 diffusion-as-ODE `aw1` *(opponent)* | 1.000 | 67.50 | **1.17** | — | — | — |
| **MeanFlow-UNet** | 1.000 | **63.77** | 2.64 | +0.000 | **−3.73 `[−6.50,−1.07]`** `*` | **+1.47 `[+1.31,+1.63]`** `*` |
| AlphaFlow-SiT | 1.000 | 66.10 | **0.92** | +0.000 | −1.40 `[−4.27,+1.77]` | **−0.24 `[−0.29,−0.19]`** `*` |
| MeanFlow-DiT | 1.000 | 76.90 | 2.25 | +0.000 | **+9.40 `[+5.13,+16.13]`** `*` | **+1.08 `[+0.83,+1.43]`** `*` |

**MeanFlow-UNet wins the step axis significantly here (−3.73) and loses cost (2.3×).** AlphaFlow
wins cost (1.3×) with steps a tie. MeanFlow-DiT loses both. So "diffusion fails at K = 1" is a
property of the `aw10` DPCC configuration, not of diffusion — and the `aw1` variant is the harder
low-K opponent.

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

### 2.2 naive FM (FMv3ODE) and diffusion-as-ODE across K — reference ladders

Both live in `flow_matching_v3_ode_selectable`. 5-seed rows are marked; the rest are seed 6 only
(6 episodes). Best `dpcc-*-tightened` arm per row.

| engine | K | solver | `aw` | `T` | seeds | S&C | steps | s/step | s/ep | cand |
|---|---|---|---|---|---|---|---|---|---|---|
| **diffusion-as-ODE** | **1** | Euler | 1 | 0.5 | **5** | **1.000** | 67.50 | 0.0173 | **1.17** | C146 |
| diffusion-as-ODE | 5 | **midpoint** | 1 | — | **5** | 1.000 | 63.53 | 0.1538 | 9.74 | C148 |
| diffusion-as-ODE | 10 | — | 1 | — | **5** | 1.000 | 63.90 | 0.1818 | 11.65 | C144 |
| diffusion-as-ODE | 10 | Euler | 10 | — | **5** | 1.000 | 62.27 | 0.1917 | 11.92 | C149 |
| diffusion-as-ODE | 10 | Euler | 1 | **1.0** | **5** | 1.000 | 63.80 | 0.9400 | 59.72 | C145 |
| diffusion-as-ODE | 20 | Euler | 1 | — | **5** | 1.000 | 62.87 | 0.4785 | 29.60 | C147 |
| **naive FM (FMv3ODE)** | 5 | Euler | 10 | 0.5 | 1 | 1.000 | 62.67 | 0.1154 | 7.07 | C143 |
| naive FM (FMv3ODE) | 10 | Euler | 10 | 0.05 | 1 | 1.000 | 64.83 | 0.0943 | 6.11 | C138 |
| naive FM (FMv3ODE) | 10 | Euler | 10 | 0.1 | 1 | 1.000 | 64.83 | 0.0945 | 6.12 | C139 |
| naive FM (FMv3ODE) | 20 | Euler | 10 | 0.05 | 1 | 1.000 | 63.50 | 0.1797 | 11.41 | C140 |
| naive FM (FMv3ODE) | 20 | Euler | 10 | 0.1 | 1 | 1.000 | 62.33 | 0.1890 | 11.78 | C141 |
| **naive FM (FMv3ODE)** | **20** | Euler | 10 | 0.5 | **5** | 1.000 | 63.23 | 0.4767 | 29.65 | C142 |

**S&C is 1.000 for every row in this table**, at every K from 1 to 20. Neither reference engine has
a reliability problem on this task; they differ only in cost. Cost is governed by K and by the
projection threshold `T`, not by the engine: C145 (K10, `T=1.0`) costs 59.72 s/ep versus C144
(K10, `T` default) at 11.65 — a 5× spread at identical K and S&C.

#### FMv3ODE at K = 2 — the `flow_matching_v3_hardflow` family

A second set of FMv3ODE runs exists under a different leaf naming (`K{n}_thres{x}_mpc{n}_n{n}`),
including **K = 2**. Seed 6 only, best available arm:

| K | candidate | S&C | steps | s/step | s/ep |
|---|---|---|---|---|---|
| **2** | C52 `K2_thres0_mpc1_n2` | **1.000** | 73.33 | 0.0261 | **1.91** |
| 5 | C53 `K5_thres0_mpc1_n2` | 1.000 | 63.33 | 0.1101 | 6.94 |
| 10 | C48 `K10_thres0_mpc1_n2` | 1.000 | 63.17 | 0.1996 | 12.58 |
| 20 | C50 `K20_thres0_mpc1_n2` | 1.000 | 62.17 | 0.4745 | 29.18 |

The `thres` token is not a confound: at K = 20 this family gives 29.18 / 29.21 / 29.44 / 29.03 s/ep
at `thres` 0 / 0.5 / 0.05 / 0.1 — a 1 % spread. **K = 1 still does not exist for FMv3ODE.**

**Matched K = 2, seed 6 — does low-K naive FM beat MeanFlow? Depends on the arm rule.**

| rule | naive FM K2 | MeanFlow-UNet K2 | AlphaFlow-SiT K2 |
|---|---|---|---|
| **each engine's best arm** | 1.000 / 73.33 / **1.91** | **1.000 / 58.67 / 1.59** | **1.000 / 63.17 / 1.28** |
| **same arm** (`dpcc-c-tightened`) | **1.000 / 73.33 / 1.91** | 0.833 / 94.00 / 2.57 | **0.000** / 199.0 / 3.72 |

- **Best-arm rule: MeanFlow-UNet beats naive FM K2 on both axes** — **−14.66 steps and −0.32 s/ep**
  at equal S&C 1.000. AlphaFlow-SiT beats it too (−10.16 steps, −0.63 s/ep).
- **Same-arm rule: naive FM wins.** `dpcc-c-tightened` is the only tightened arm C52 carries, and it
  is precisely the arm on which the few-step engines collapse at K = 2 — MeanFlow-UNet drops to
  0.833/94 steps, MeanFlow-DiT and AlphaFlow-SiT time out entirely (199 steps, S&C 0.000).

Both readings are seed 6, 6 episodes. The honest statement is that **MeanFlow-UNet beats naive FM at
K = 2 when each engine uses its best projection rule, and loses when both are forced onto `-c`** —
and that the `-c` collapse at K = 2 is a known few-step pathology, not a property of the comparison.

⚠️ **Qualitative, not measured: trajectory quality of FMv3ODE at low K.** Visual inspection of the
plotted rollouts reports **trajectory explosion / non-smooth paths for FMv3ODE at low K**, which the
few-step engines do not show. **No metric in this DA measures smoothness**, so this observation is
neither supported nor refuted by the numbers above, and must be reported as a qualitative finding
with figures — not asserted as a quantitative result.

What the available metrics do say, for the record (seed 6, unprojected `diffuser` arm — the raw
field, before projection):

| K | naive FM: avg # viol | total violation mass | steps |
|---|---|---|---|
| 2 | 16.67 | 3.29 | **61.00** |
| 5 | 18.17 | 4.68 | 65.00 |
| 10 | 16.17 | 3.52 | 64.50 |
| 20 | 18.50 | 4.85 | 65.00 |

**Constraint violation of the raw field is flat in K** — K = 2 is not worse than K = 20 on either
count or mass. So if the low-K trajectories are rougher, that roughness is **not** expressing itself
as more constraint violation, and after projection the residual is 0.000000 at K = 2.

The one number consistent with rough low-K paths is **path length after projection: naive FM K2
takes 73.33 steps, versus 62–63 at K = 5–20** — the longest in its own ladder, while every other
row in that ladder sits within 1.2 steps. A wandering trajectory costs steps even when it is
projected to be safe. This is suggestive, single-seed, and not a smoothness measurement.

**To make this claim quantitatively**, the eval must log a smoothness statistic per rollout —
mean |Δaction| between consecutive steps, path curvature, or jerk. None is currently recorded.

**ODE solver (Euler / midpoint / RK4).** No matched solver A/B exists:

| solver | where it exists | matched Euler counterpart? |
|---|---|---|
| Euler | everywhere (default) | — |
| midpoint | C148 only — diffusion-as-ODE, K5, `aw1`, 5 seeds | ❌ no K5 Euler at same engine/`aw` |
| RK4 | iMeanFlow K2/K20, seed 6 only | ❌ K20 pair exists but seed 6 only |

C148 (midpoint, K5, 9.74 s/ep) sits between the K1 (1.17) and K10 (11.65) Euler rows of the same
parent, i.e. on the same cost-vs-K curve — no solver effect is separable from the K effect.
**A midpoint-vs-Euler claim requires a matched run and is not supported by current data.**

**Reading.** The matched-budget result is narrow and specific:

- **Won:** path length. MeanFlow-UNet takes significantly fewer steps than the working K = 1
  diffusion opponent (−3.73 `*`) and than DPCC K10 (−5.07, ns); AlphaFlow wins steps at K = 10
  (−6.30 `*`). Also won: reliability against the *published* `aw10` DPCC at K = 1 (1.000 vs 0.667 `*`).
- **Not won:** wall-clock at equal K. At K = 1 (vs C146), K = 5 and K = 10 the reference engines are
  equal or cheaper per episode; our per-step cost is higher at the same budget.
- **Therefore** the 14.6×/41.7× in §2 does **not** come from being faster at equal compute, and not
  from diffusion being unusable at low K in general — the `aw1` diffusion-as-ODE row reaches
  S&C 1.000 at 1.17 s/ep. It comes from **the published `aw10` DPCC configuration requiring K = 20**
  to stay reliable (0.667 at K = 1). The result is against that configuration, which is the paper's
  baseline, and should be stated that way rather than as a general claim about diffusion.

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

Naive FM at K5 costs **4.3× less than its own K20** at S&C 1.000 (7.07 vs 30.06 s/ep, seed 6), and
at K2 (§2.2, `hardflow` family) **1.91 s/ep — 15.7× less than its K20**. Its 1.3× margin over the
Target in §2 is therefore an artifact of pinning it at K20; at K2 it is ≈20× (single seed). Seed 6 is not optimistic for this engine — its K20 seed-6 numbers match the 5-seed
values (1.000 / 62.17 vs 1.000 / 63.23).

**Consequence for §4:** MeanFlow-UNet's advantage over naive flow matching collapses as naive FM is
allowed lower K — **11.2× vs naive FM K20, 2.7× vs K5, and at matched K = 2 the margin is 1.2× on
cost (1.59 vs 1.91) with −14.7 steps** (§2.2, seed 6). The low-K benefit is largely *not* exclusive
to the few-step objective: naive FM captures most of it by reducing K. What survives at matched K
is the **step advantage**, and the fact that naive FM at K = 2 needs the `-c` arm the few-step
engines fail on. A multi-seed naive-FM K1/K2 ladder is required to settle this.

---

## 5. Limits

- 30 episodes per configuration; S&C resolves to 1/30. Bootstrap resamples 5 seed clusters — step CIs are wide.
- `steps` averages over successful episodes only; comparable only at equal S&C. All §2 rows are gate-matched at 1.000.
- `s/step` is wall-clock on shared GPUs, includes the constraint solve. Differences < 10–20 % are not resolvable.
- Architecture confound unresolved (§1). Baseline and naive FM on SiT/DiT not run.
- Adjacent-window train/test overlap at H = 8 affects all engines equally.
- Not run: MeanFlow-UNet K20; HardFlow activation-threshold sweep at 5 seeds (single-seed only);
  **naive FM below K20 at multi-seed** (K2 and K5 are seed-6 only; **K1 absent entirely**) — §2.2, §4.1.
- **No smoothness metric is recorded.** Trajectory explosion / roughness reported visually for
  low-K FMv3ODE (§2.2) cannot be quantified from this data; adding mean |Δaction|, curvature or
  jerk to the eval would close this.
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
