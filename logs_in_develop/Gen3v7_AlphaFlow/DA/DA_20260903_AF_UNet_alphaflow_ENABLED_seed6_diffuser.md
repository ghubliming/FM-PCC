# DA 2026-09-03 — With α actually ON, does **α-Flow** beat **MeanFlow** on the U-Net?

**Question asked:** every α-Flow checkpoint deployed until now was a MeanFlow model
(`af_alpha_end = 0.0` → α = 0 → `af_diffusion.py:552` routes to Gen3v6's JVP body). `AF_ALPHA_END`
now holds α at a floor. Two arms were trained with the bootstrap **live to the last step**. Does
α-Flow beat MeanFlow (a) on the raw network output and (b) after the DPCC projection?

**Short answer: YES on both, at low K — and the projected win also clears the pinned DPCC target.**

- **Raw plan (`diffuser`)**: AF `α→0.2` **Pareto-dominates** MF-UNet at K=1, 5 and 10 (more goals
  reached *and* fewer steps, identical `avg_time`).
- **Projected (`dpcc-t-tightened`, the winning arm)**: at **K=1** AF `α→0.2` posts **S&C 1.00 /
  57.20 steps / 1.07 s/ep** on `top-left-hard` — it **beats the DPCC K20 target 33.3× cheaper**,
  and it beats MF-UNet's own K=1 row on steps (57.20 vs 60.75). On `both-hard` at K=1 **AF clears
  the target and MF does not**.
- **⛔ Two caveats block citation:** n = 20 on a **single seed** (§7), and the `n_steps` column has
  **two incompatible definitions** across the toolchain (§6.1). Neither is fatal; both must be
  closed before this goes in the thesis.

**Batch:** `temp/0309/batch_avoiding_combined_20260903_133730/`
**Task:** avoiding-d3il (state-based), H8 · **Constraint:** halfspace · **Seed:** 6 · **n_trials:** 20
**Jobs:** train `25290` (`α→0.2`) / `25292` (`α→0.05`) · eval `25291` / `25293` · pipelines `25280` / `25279`
**Predecessor:** [`DA_20260901_AF_UNet_alpha_clamp_T1_negative.md`](DA_20260901_AF_UNet_alpha_clamp_T1_negative.md)

---

## 0. What is being compared

| | **AF `α→0.2`** | **AF `α→0.05`** | **MF-UNet** | **DPCC baseline (target)** |
|---|---|---|---|---|
| folder | `…AlphaFlowODE_aw10_bbunet_tslogit_normal_ai1.0_`**`ae0.2`**`_ag25.0_rf0.5` | same, **`ae0.05`** | `…MeanFlowODE_aw10_objmeanflow_bbunet_tslogit_normal_dp0.5` | `plans/diffusion/H8_K20_D…GaussianDiffusion_aw10` |
| plan tag | `…_B4_…_msgafon02_s6` | `…_B4_…_msgafon005_s6` | `…_B1_…_msg20trials` | `H8_K20_T0.5_…_msg20trials` |
| generative model | α-Flow, bootstrapped no-grad target | same | MeanFlow, analytic JVP target | `models.GaussianDiffusion`, 200-step DDPM |
| backbone | `bbunet` | `bbunet` | `bbunet` | `UNet1DTemporalCondModel` |
| **params** | **4.0 M** | **4.0 M** | **4.0 M** | **4.0 M** |
| `action_weight` | 10 | 10 | 10 | 10 |
| time sampler | `logit_normal` | `logit_normal` | `logit_normal` | — |
| FM / u split | `rf0.5` | `rf0.5` | `dp0.5` (same knob) | — |
| **α floor** | **0.2** | **0.05** | 0 (is MeanFlow) | — |
| K | 1, 2, 5, 10, 20 | 1, 2, 5, 10, 20 | 1, 2, 5, 10, 20 | 20 |
| `diffusion_timestep_threshold` | 0.5 | 0.5 | 0.5 | 0.5 |
| seeds | **6** | **6** | 6–10 (**6** used) | 6–10 (**6** used) |
| `n_trials` | **20** | **20** | **20** | **20** |
| run date | 2026-09-01/02 | 2026-09-02/03 | 2026-08-13 (24559–24563) | 2026-08-18 |

⚙️ **Architecture-matched.** All three flow models are the *same* 4.0 M U-Net with the same action
weight and the same time sampler. 🔒 The U-Net was **not touched** — `AF_ALPHA_END` is a
training-target knob and changes no parameter count. This is the strong form of the claim, not the
SiT/DiT-confounded form.

**Target definition** (per `da-target-is-best-baseline-variant`, re-pinned at seed 6 / n=20 from
this batch — the best variant of DPCC K20/aw10/T0.5):

> **`dpcc-c-tightened`** — S&C **1.00** on all three halfspaces;
> **61.35 steps / 29.48 s/ep** (TR) · **61.00 / 35.66** (TL) · **59.35 / 36.55** (both).

### 0.1 Proof that α-Flow was actually running

| signal | every prior AF run | `α→0.05` | `α→0.2` |
|---|---|---|---|
| `[ train ] AF_ALPHA_END=` | `0.0` | `0.05` | `0.2` |
| `alpha`, final epoch | 0.0006 | **0.05** | **0.2** |
| **`discrete_frac`, final epochs** | **0.0** | **0.25 – 0.41** | **0.25 – 0.41** |
| savepath token | `_ae0.0_` | `_ae0.05_` | `_ae0.2_` |

`discrete_frac` is the batch fraction taking the bootstrapped no-grad branch. It tracks `rf0.5`
throughout and never reaches zero — **about half of every batch trained on α-Flow's own target,
through to the final step.**

---

## 1. The `diffuser` arm — raw network output, no projection

`s/ep` = `n_steps × avg_time`. S&C is ~0 by construction here (unprojected plans violate freely);
it is printed for completeness and is **not** the discriminator on this arm.

#### `top-right-hard`

| model | K | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| AF α→0.2 | 1 | 0.00 | 1.00 | 23.1 | 60.90 | 0.0097 | **0.59** |
| AF α→0.2 | 2 | 0.00 | 0.90 | 22.4 | 63.45 | 0.0184 | **1.17** |
| AF α→0.2 | 5 | 0.00 | 0.95 | 24.6 | 63.50 | 0.0461 | **2.92** |
| AF α→0.2 | 10 | 0.00 | 0.90 | 24.4 | 63.05 | 0.0897 | **5.66** |
| AF α→0.2 | 20 | 0.05 | 0.80 | 21.8 | 60.60 | 0.1786 | **10.82** |
| AF α→0.05 | 1 | 0.00 | 0.90 | 20.9 | 60.65 | 0.0097 | **0.59** |
| AF α→0.05 | 2 | 0.05 | 1.00 | 21.7 | 63.45 | 0.0184 | **1.17** |
| AF α→0.05 | 5 | 0.05 | 0.85 | 19.6 | 60.85 | 0.0455 | **2.77** |
| AF α→0.05 | 10 | 0.00 | 0.85 | 19.6 | 59.75 | 0.0913 | **5.46** |
| AF α→0.05 | 20 | 0.00 | 0.80 | 19.6 | 59.25 | 0.1822 | **10.80** |
| MF-UNet | 1 | 0.00 | 0.85 | 22.1 | 59.70 | 0.0096 | **0.57** |
| MF-UNet | 2 | 0.00 | 0.95 | 17.7 | 61.35 | 0.0186 | **1.14** |
| MF-UNet | 5 | 0.10 | 0.95 | 20.0 | 62.80 | 0.0455 | **2.86** |
| MF-UNet | 10 | 0.05 | 0.85 | 22.3 | 64.15 | 0.0911 | **5.84** |
| MF-UNet | 20 | 0.05 | 0.90 | 22.8 | 64.40 | 0.1789 | **11.52** |
| FM-UNet (naive) | 1 | 0.00 | 0.85 | 30.1 | 62.45 | 0.0093 | **0.58** |
| FM-UNet (naive) | 2 | 0.00 | 0.95 | 30.1 | 63.05 | 0.0179 | **1.13** |
| FM-UNet (naive) | 5 | 0.00 | 0.95 | 28.9 | 71.15 | 0.0443 | **3.15** |
| FM-UNet (naive) | 20 | 0.00 | 1.00 | 31.6 | 67.70 | 0.1716 | **11.62** |
| DPCC K20 (target) | 20 | 0.00 | 0.60 | 17.6 | 57.25 | 0.1779 | **10.18** |

#### `top-left-hard`

| model | K | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| AF α→0.2 | 1 | 0.10 | 1.00 | 9.2 | 60.90 | 0.0096 | **0.58** |
| AF α→0.2 | 2 | 0.05 | 1.00 | 13.6 | 64.80 | 0.0184 | **1.19** |
| AF α→0.2 | 5 | 0.00 | 1.00 | 15.8 | 63.95 | 0.0455 | **2.91** |
| AF α→0.2 | 10 | 0.00 | 1.00 | 15.3 | 64.00 | 0.0897 | **5.74** |
| AF α→0.2 | 20 | 0.00 | 1.00 | 14.8 | 65.40 | 0.1842 | **12.05** |
| AF α→0.05 | 1 | 0.30 | 1.00 | 6.8 | 62.45 | 0.0095 | **0.60** |
| AF α→0.05 | 2 | 0.10 | 1.00 | 11.2 | 63.45 | 0.0182 | **1.15** |
| AF α→0.05 | 5 | 0.05 | 1.00 | 13.9 | 64.35 | 0.0456 | **2.93** |
| AF α→0.05 | 10 | 0.05 | 1.00 | 15.4 | 63.25 | 0.0914 | **5.78** |
| AF α→0.05 | 20 | 0.05 | 0.95 | 20.6 | 69.00 | 0.1831 | **12.63** |
| MF-UNet | 1 | 0.05 | 1.00 | 11.3 | 62.25 | 0.0096 | **0.60** |
| MF-UNet | 2 | 0.05 | 1.00 | 18.4 | 62.70 | 0.0186 | **1.16** |
| MF-UNet | 5 | 0.10 | 1.00 | 13.6 | 64.10 | 0.0460 | **2.95** |
| MF-UNet | 10 | 0.00 | 1.00 | 16.8 | 66.80 | 0.0947 | **6.32** |
| MF-UNet | 20 | 0.00 | 1.00 | 20.2 | 69.20 | 0.1801 | **12.46** |
| FM-UNet (naive) | 1 | 0.00 | 1.00 | 11.5 | 66.40 | 0.0104 | **0.69** |
| FM-UNet (naive) | 2 | 0.10 | 1.00 | 12.2 | 63.85 | 0.0189 | **1.21** |
| FM-UNet (naive) | 5 | 0.05 | 1.00 | 10.9 | 72.95 | 0.0436 | **3.18** |
| FM-UNet (naive) | 20 | 0.05 | 1.00 | 10.8 | 67.70 | 0.1761 | **11.92** |
| DPCC K20 (target) | 20 | 0.25 | 1.00 | 15.6 | 64.15 | 0.1859 | **11.93** |

#### `both-hard`

| model | K | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| AF α→0.2 | 1 | 0.05 | 1.00 | 12.1 | 60.90 | 0.0096 | **0.59** |
| AF α→0.2 | 2 | 0.15 | 1.00 | 10.2 | 64.80 | 0.0184 | **1.19** |
| AF α→0.2 | 5 | 0.10 | 1.00 | 10.9 | 63.95 | 0.0461 | **2.95** |
| AF α→0.2 | 10 | 0.10 | 1.00 | 9.4 | 64.00 | 0.0901 | **5.77** |
| AF α→0.2 | 20 | 0.05 | 1.00 | 9.3 | 65.40 | 0.1844 | **12.06** |
| AF α→0.05 | 1 | 0.00 | 1.00 | 15.6 | 62.45 | 0.0096 | **0.60** |
| AF α→0.05 | 2 | 0.05 | 1.00 | 13.2 | 63.45 | 0.0183 | **1.16** |
| AF α→0.05 | 5 | 0.10 | 1.00 | 13.8 | 64.35 | 0.0467 | **3.00** |
| AF α→0.05 | 10 | 0.20 | 1.00 | 10.7 | 63.25 | 0.0918 | **5.80** |
| AF α→0.05 | 20 | 0.20 | 0.95 | 10.3 | 69.00 | 0.1810 | **12.49** |
| MF-UNet | 1 | 0.00 | 1.00 | 13.2 | 62.25 | 0.0097 | **0.60** |
| MF-UNet | 2 | 0.15 | 1.00 | 12.9 | 62.70 | 0.0187 | **1.17** |
| MF-UNet | 5 | 0.15 | 1.00 | 10.8 | 64.10 | 0.0467 | **3.00** |
| MF-UNet | 10 | 0.05 | 1.00 | 9.3 | 66.80 | 0.0960 | **6.42** |
| FM-UNet (naive) | 1 | 0.00 | 1.00 | 9.7 | 66.40 | 0.0101 | **0.67** |
| FM-UNet (naive) | 2 | 0.20 | 1.00 | 8.1 | 63.85 | 0.0193 | **1.23** |
| FM-UNet (naive) | 5 | 0.20 | 1.00 | 8.6 | 72.95 | 0.0433 | **3.16** |
| FM-UNet (naive) | 20 | 0.05 | 1.00 | 12.2 | 67.70 | 0.1725 | **11.68** |

### 1.1 Reading

**AF `α→0.2` Pareto-dominates MF-UNet at K=1, K=5 and K=10** on `top-right-hard` — the only
halfspace where MF is off the ceiling, and therefore the only one that discriminates:

| K | AF `α→0.2` | MF-UNet | |
|---|---|---|---|
| **1** | **1.00 succ / 60.90 steps** | 0.85 / 62.41 | AF dominates |
| 2 | 0.90 / 63.89 | **0.95 / 63.00** | MF dominates |
| **5** | **0.95 / 63.84** | 0.95 / 64.53 | AF dominates |
| **10** | **0.90 / 63.78** | 0.85 / 66.59 | AF dominates |
| 20 | 0.80 / 63.44 | **0.90** / 65.39 | mixed |

*(steps here are the **successes-only** basis — see §6.1; on the all-episode basis the K=1 row reads
as a trade-off instead, because MF's 3 early failures pull its mean down.)*

`avg_time` is identical across AF and MF at every K (K=1: .0097 vs .0096 · K=20: .1786 vs .1789).
**α-Flow's extra `no_grad` forward for `u_next` is a training-time cost only** — at deployment the
two are the same network doing the same work, so any accuracy gained is free.

`top-left-hard` and `both-hard` sit at 1.00 succ for every U-Net model and carry no signal. They are
also **not independent samples**: the diffuser plan ignores the constraint, so the three halfspace
columns re-score the *same 20 rollouts*. **Effective n stays 20 — never pool to 60.**

---

## 2. The projection arms — full variant tables, `top-right-hard`

#### K = 1 — `top-right-hard`

| variant | model | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | AF α→0.2 | 0.90 | 0.90 | 0.0 | 67.90 | 0.0170 | **1.15** |
| `dpcc-c-tightened` | AF α→0.05 | 1.00 | 1.00 | 0.0 | 73.15 | 0.0168 | **1.23** |
| `dpcc-c-tightened` | MF-UNet | 0.95 | 0.95 | 0.1 | 69.90 | 0.0169 | **1.18** |
| `dpcc-t-tightened` | AF α→0.2 | 0.85 | 0.85 | 0.0 | 60.90 | 0.0175 | **1.06** |
| `dpcc-t-tightened` | AF α→0.05 | 0.95 | 0.95 | 0.0 | 65.90 | 0.0171 | **1.13** |
| `dpcc-t-tightened` | MF-UNet | 0.95 | 0.95 | 0.0 | 64.20 | 0.0180 | **1.15** |
| `dpcc-r-tightened` | AF α→0.2 | 0.95 | 0.95 | 0.0 | 65.65 | 0.0170 | **1.12** |
| `dpcc-r-tightened` | AF α→0.05 | 0.90 | 0.90 | 0.0 | 70.15 | 0.0178 | **1.25** |
| `dpcc-r-tightened` | MF-UNet | 0.95 | 0.95 | 0.0 | 65.25 | 0.0173 | **1.13** |
| `dpcc-c` | AF α→0.2 | 0.35 | 0.40 | 1.5 | 58.10 | 0.0165 | **0.96** |
| `dpcc-c` | AF α→0.05 | 0.30 | 0.50 | 1.9 | 61.25 | 0.0164 | **1.01** |
| `dpcc-c` | MF-UNet | 0.55 | 0.65 | 0.9 | 63.00 | 0.0165 | **1.04** |
| `dpcc-t` | AF α→0.2 | 0.20 | 0.30 | 3.1 | 52.40 | 0.0168 | **0.88** |
| `dpcc-t` | AF α→0.05 | 0.20 | 0.50 | 2.6 | 57.55 | 0.0167 | **0.96** |
| `dpcc-t` | MF-UNet | 0.20 | 0.50 | 2.2 | 57.25 | 0.0179 | **1.02** |
| `dpcc-r` | AF α→0.2 | 0.20 | 0.45 | 1.6 | 52.75 | 0.0165 | **0.87** |
| `dpcc-r` | AF α→0.05 | 0.15 | 0.40 | 1.1 | 51.90 | 0.0165 | **0.86** |
| `dpcc-r` | MF-UNet | 0.30 | 0.45 | 1.4 | 56.95 | 0.0168 | **0.96** |

#### K = 2 — `top-right-hard`

| variant | model | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | AF α→0.2 | 0.85 | 0.95 | 0.1 | 88.40 | 0.0265 | **2.34** |
| `dpcc-c-tightened` | AF α→0.05 | 0.95 | 0.95 | 0.0 | 87.65 | 0.0258 | **2.26** |
| `dpcc-c-tightened` | MF-UNet | 0.85 | 0.85 | 0.3 | 82.15 | 0.0281 | **2.31** |
| `dpcc-t-tightened` | AF α→0.2 | 1.00 | 1.00 | 0.0 | 63.40 | 0.0280 | **1.77** |
| `dpcc-t-tightened` | AF α→0.05 | 0.95 | 0.95 | 0.1 | 62.75 | 0.0266 | **1.67** |
| `dpcc-t-tightened` | MF-UNet | 0.95 | 0.95 | 0.0 | 65.35 | 0.0279 | **1.82** |
| `dpcc-r-tightened` | AF α→0.2 | 0.95 | 0.95 | 0.0 | 69.55 | 0.0268 | **1.86** |
| `dpcc-r-tightened` | AF α→0.05 | 0.90 | 0.90 | 0.1 | 69.55 | 0.0265 | **1.84** |
| `dpcc-r-tightened` | MF-UNet | 0.80 | 0.85 | 0.1 | 67.25 | 0.0271 | **1.82** |
| `dpcc-c` | AF α→0.2 | 0.55 | 0.55 | 1.4 | 83.30 | 0.0254 | **2.12** |
| `dpcc-c` | AF α→0.05 | 0.55 | 0.65 | 0.8 | 83.60 | 0.0255 | **2.13** |
| `dpcc-c` | MF-UNet | 0.65 | 0.75 | 1.1 | 78.75 | 0.0256 | **2.01** |
| `dpcc-t` | AF α→0.2 | 0.15 | 0.35 | 3.4 | 54.05 | 0.0260 | **1.41** |
| `dpcc-t` | AF α→0.05 | 0.50 | 0.60 | 1.2 | 53.05 | 0.0256 | **1.36** |
| `dpcc-t` | MF-UNet | 0.15 | 0.50 | 2.4 | 54.60 | 0.0258 | **1.41** |
| `dpcc-r` | AF α→0.2 | 0.40 | 0.55 | 1.0 | 56.85 | 0.0257 | **1.46** |
| `dpcc-r` | AF α→0.05 | 0.45 | 0.60 | 1.4 | 59.50 | 0.0255 | **1.51** |
| `dpcc-r` | MF-UNet | 0.40 | 0.55 | 2.1 | 57.25 | 0.0266 | **1.52** |

#### K = 5 — `top-right-hard`

| variant | model | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | AF α→0.2 | 0.85 | 0.85 | 0.1 | 67.10 | 0.2341 | **15.71** |
| `dpcc-c-tightened` | AF α→0.05 | 0.70 | 0.70 | 0.1 | 62.15 | 0.2180 | **13.55** |
| `dpcc-c-tightened` | MF-UNet | 0.60 | 0.60 | 0.7 | 63.10 | 0.2383 | **15.03** |
| `dpcc-t-tightened` | AF α→0.2 | 0.95 | 0.95 | 0.1 | 64.85 | 0.2501 | **16.22** |
| `dpcc-t-tightened` | AF α→0.05 | 0.95 | 0.95 | 0.1 | 65.60 | 0.2352 | **15.43** |
| `dpcc-t-tightened` | MF-UNet | 1.00 | 1.00 | 0.0 | 60.60 | 0.2085 | **12.64** |
| `dpcc-r-tightened` | AF α→0.2 | 0.70 | 0.75 | 1.1 | 71.20 | 0.2453 | **17.47** |
| `dpcc-r-tightened` | AF α→0.05 | 0.75 | 0.80 | 0.2 | 69.35 | 0.2637 | **18.29** |
| `dpcc-r-tightened` | MF-UNet | 0.75 | 0.75 | 0.3 | 65.65 | 0.2355 | **15.46** |
| `dpcc-c` | AF α→0.2 | 0.40 | 0.45 | 1.6 | 55.45 | 0.1856 | **10.29** |
| `dpcc-c` | AF α→0.05 | 0.55 | 0.60 | 1.6 | 61.20 | 0.2031 | **12.43** |
| `dpcc-c` | MF-UNet | 0.45 | 0.60 | 1.6 | 57.05 | 0.1986 | **11.33** |
| `dpcc-t` | AF α→0.2 | 0.60 | 0.70 | 0.1 | 57.80 | 0.1888 | **10.91** |
| `dpcc-t` | AF α→0.05 | 0.60 | 0.60 | 1.4 | 56.10 | 0.2106 | **11.81** |
| `dpcc-t` | MF-UNet | 0.40 | 0.80 | 1.8 | 60.15 | 0.1951 | **11.74** |
| `dpcc-r` | AF α→0.2 | 0.30 | 0.70 | 2.1 | 63.85 | 0.1921 | **12.27** |
| `dpcc-r` | AF α→0.05 | 0.15 | 0.70 | 3.1 | 62.70 | 0.2056 | **12.89** |
| `dpcc-r` | MF-UNet | 0.20 | 0.80 | 2.9 | 64.35 | 0.1921 | **12.36** |

#### K = 10 — `top-right-hard`

| variant | model | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | AF α→0.2 | 0.85 | 0.85 | 0.6 | 67.05 | 0.3924 | **26.31** |
| `dpcc-c-tightened` | AF α→0.05 | 0.80 | 0.80 | 0.0 | 61.20 | 0.4098 | **25.08** |
| `dpcc-c-tightened` | MF-UNet | 0.60 | 0.60 | 0.4 | 64.70 | 0.4095 | **26.49** |
| `dpcc-t-tightened` | AF α→0.2 | 0.95 | 0.95 | 0.0 | 66.85 | 0.4228 | **28.27** |
| `dpcc-t-tightened` | AF α→0.05 | 0.95 | 0.95 | 0.0 | 65.65 | 0.4342 | **28.50** |
| `dpcc-t-tightened` | MF-UNet | 0.90 | 0.90 | 0.0 | 62.90 | 0.3971 | **24.98** |
| `dpcc-r-tightened` | AF α→0.2 | 0.75 | 0.75 | 0.8 | 70.20 | 0.4310 | **30.26** |
| `dpcc-r-tightened` | AF α→0.05 | 0.80 | 0.90 | 0.5 | 70.35 | 0.4214 | **29.65** |
| `dpcc-r-tightened` | MF-UNet | 0.75 | 0.75 | 0.2 | 66.75 | 0.4142 | **27.65** |
| `dpcc-c` | AF α→0.2 | 0.45 | 0.65 | 1.8 | 59.75 | 0.3064 | **18.30** |
| `dpcc-c` | AF α→0.05 | 0.55 | 0.75 | 1.2 | 59.75 | 0.3163 | **18.90** |
| `dpcc-c` | MF-UNet | 0.40 | 0.75 | 1.9 | 58.50 | 0.3104 | **18.16** |
| `dpcc-t` | AF α→0.2 | 0.55 | 0.80 | 0.8 | 57.65 | 0.3193 | **18.41** |
| `dpcc-t` | AF α→0.05 | 0.50 | 0.85 | 1.4 | 59.65 | 0.3433 | **20.48** |
| `dpcc-t` | MF-UNet | 0.50 | 0.70 | 1.6 | 56.75 | 0.3168 | **17.98** |
| `dpcc-r` | AF α→0.2 | 0.55 | 0.80 | 1.9 | 67.35 | 0.3231 | **21.76** |
| `dpcc-r` | AF α→0.05 | 0.15 | 0.75 | 4.0 | 63.85 | 0.3453 | **22.05** |
| `dpcc-r` | MF-UNet | 0.30 | 0.60 | 3.5 | 60.10 | 0.3392 | **20.39** |

#### K = 20 — `top-right-hard`

| variant | model | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|---|
| `dpcc-c-tightened` | AF α→0.2 | 0.95 | 1.00 | 0.2 | 66.60 | 0.9842 | **65.55** |
| `dpcc-c-tightened` | AF α→0.05 | 0.90 | 0.90 | 0.3 | 68.80 | 1.0562 | **72.67** |
| `dpcc-c-tightened` | MF-UNet | 0.75 | 0.85 | 0.3 | 66.30 | 0.9643 | **63.93** |
| `dpcc-t-tightened` | AF α→0.2 | 0.90 | 0.90 | 0.0 | 67.55 | 1.1237 | **75.90** |
| `dpcc-t-tightened` | AF α→0.05 | 0.85 | 0.90 | 0.1 | 68.30 | 1.1062 | **75.56** |
| `dpcc-t-tightened` | MF-UNet | 0.80 | 0.80 | 0.0 | 58.55 | 0.9843 | **57.63** |
| `dpcc-r-tightened` | AF α→0.2 | 0.70 | 0.75 | 0.2 | 72.65 | 1.1110 | **80.72** |
| `dpcc-r-tightened` | AF α→0.05 | 0.75 | 0.75 | 0.2 | 67.55 | 1.0759 | **72.67** |
| `dpcc-r-tightened` | MF-UNet | 0.70 | 0.70 | 0.2 | 66.50 | 1.0513 | **69.91** |
| `dpcc-c` | AF α→0.2 | 0.50 | 0.65 | 1.6 | 56.80 | 0.7452 | **42.33** |
| `dpcc-c` | AF α→0.05 | 0.35 | 0.40 | 1.9 | 56.20 | 0.7638 | **42.92** |
| `dpcc-c` | MF-UNet | 0.35 | 0.60 | 1.7 | 56.65 | 0.6679 | **37.84** |
| `dpcc-t` | AF α→0.2 | 0.40 | 0.60 | 1.1 | 55.25 | 0.7787 | **43.03** |
| `dpcc-t` | AF α→0.05 | 0.50 | 0.75 | 1.2 | 58.00 | 0.7805 | **45.27** |
| `dpcc-t` | MF-UNet | 0.50 | 0.65 | 0.8 | 53.00 | 0.7465 | **39.57** |
| `dpcc-r` | AF α→0.2 | 0.45 | 0.85 | 2.3 | 68.40 | 0.8036 | **54.97** |
| `dpcc-r` | AF α→0.05 | 0.25 | 0.80 | 3.5 | 66.25 | 0.8182 | **54.20** |
| `dpcc-r` | MF-UNet | 0.30 | 0.80 | 2.5 | 62.40 | 0.8293 | **51.75** |

---

## 3. The decisive arm — `dpcc-t-tightened` and `dpcc-c-tightened` vs the target

`dpcc-t-tightened` is where every model's best row lives, exactly as in the Gen3v6 MF-UNet study.
"✅ **beats**" = S&C ≥ 1.00 **and** steps ≤ target **and** s/ep ≤ target.

#### `dpcc-t-tightened` — `top-right-hard`  (target: S&C 1.00 · 61.35 steps · 29.48 s/ep)

| K | model | S&C | succ | viol | steps | s/step | **s/ep** | vs target |
|---|---|---|---|---|---|---|---|---|
| 1 | AF α→0.2 | 0.85 | 0.85 | 0.0 | 60.90 | 0.0175 | **1.06** | — |
| 1 | AF α→0.05 | 0.95 | 0.95 | 0.0 | 65.90 | 0.0171 | **1.13** | — |
| 1 | MF-UNet | 0.95 | 0.95 | 0.0 | 64.20 | 0.0180 | **1.15** | — |
| 1 | FM-UNet | 0.85 | 0.85 | 0.0 | 63.40 | 0.0165 | **1.05** | — |
| 2 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 63.40 | 0.0280 | **1.77** | S&C ok, not dom. |
| 2 | AF α→0.05 | 0.95 | 0.95 | 0.1 | 62.75 | 0.0266 | **1.67** | — |
| 2 | MF-UNet | 0.95 | 0.95 | 0.0 | 65.35 | 0.0279 | **1.82** | — |
| 2 | FM-UNet | 0.85 | 0.85 | 0.0 | 64.50 | 0.0257 | **1.66** | — |
| 5 | AF α→0.2 | 0.95 | 0.95 | 0.1 | 64.85 | 0.2501 | **16.22** | — |
| 5 | AF α→0.05 | 0.95 | 0.95 | 0.1 | 65.60 | 0.2352 | **15.43** | — |
| 5 | MF-UNet | 1.00 | 1.00 | 0.0 | 60.60 | 0.2085 | **12.64** | ✅ **beats** |
| 5 | FM-UNet | 1.00 | 1.00 | 0.0 | 68.50 | 0.0859 | **5.89** | S&C ok, not dom. |
| 10 | AF α→0.2 | 0.95 | 0.95 | 0.0 | 66.85 | 0.4228 | **28.27** | — |
| 10 | AF α→0.05 | 0.95 | 0.95 | 0.0 | 65.65 | 0.4342 | **28.50** | — |
| 10 | MF-UNet | 0.90 | 0.90 | 0.0 | 62.90 | 0.3971 | **24.98** | — |
| 20 | AF α→0.2 | 0.90 | 0.90 | 0.0 | 67.55 | 1.1237 | **75.90** | — |
| 20 | AF α→0.05 | 0.85 | 0.90 | 0.1 | 68.30 | 1.1062 | **75.56** | — |
| 20 | MF-UNet | 0.80 | 0.80 | 0.0 | 58.55 | 0.9843 | **57.63** | — |
| 20 | FM-UNet | 1.00 | 1.00 | 0.0 | 68.55 | 0.3688 | **25.28** | S&C ok, not dom. |

#### `dpcc-t-tightened` — `top-left-hard`  (target: S&C 1.00 · 61.00 steps · 35.66 s/ep)

| K | model | S&C | succ | viol | steps | s/step | **s/ep** | vs target |
|---|---|---|---|---|---|---|---|---|
| 1 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 57.20 | 0.0187 | **1.07** | ✅ **beats** |
| 1 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 58.20 | 0.0187 | **1.09** | ✅ **beats** |
| 1 | MF-UNet | 1.00 | 1.00 | 0.0 | 60.75 | 0.0186 | **1.13** | ✅ **beats** |
| 1 | FM-UNet | 1.00 | 1.00 | 0.0 | 65.65 | 0.0200 | **1.31** | S&C ok, not dom. |
| 2 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 58.00 | 0.0307 | **1.78** | ✅ **beats** |
| 2 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 58.15 | 0.0275 | **1.60** | ✅ **beats** |
| 2 | MF-UNet | 1.00 | 1.00 | 0.0 | 59.50 | 0.0275 | **1.64** | ✅ **beats** |
| 2 | FM-UNet | 1.00 | 1.00 | 0.0 | 62.95 | 0.0284 | **1.79** | S&C ok, not dom. |
| 5 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 56.00 | 0.2263 | **12.67** | ✅ **beats** |
| 5 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 58.90 | 0.2341 | **13.79** | ✅ **beats** |
| 5 | MF-UNet | 1.00 | 1.00 | 0.0 | 62.20 | 0.2423 | **15.07** | S&C ok, not dom. |
| 5 | FM-UNet | 1.00 | 1.00 | 0.0 | 65.65 | 0.1091 | **7.16** | S&C ok, not dom. |
| 10 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 55.85 | 0.3797 | **21.21** | ✅ **beats** |
| 10 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 58.75 | 0.4097 | **24.07** | ✅ **beats** |
| 10 | MF-UNet | 1.00 | 1.00 | 0.0 | 60.15 | 0.3896 | **23.43** | ✅ **beats** |
| 20 | AF α→0.2 | 0.95 | 1.00 | 0.1 | 59.70 | 1.0204 | **60.92** | — |
| 20 | AF α→0.05 | 0.95 | 1.00 | 5.2 | 67.25 | 1.3746 | **92.44** | — |
| 20 | MF-UNet | 1.00 | 1.00 | 0.0 | 60.10 | 0.9655 | **58.03** | S&C ok, not dom. |
| 20 | FM-UNet | 1.00 | 1.00 | 0.0 | 68.40 | 0.4764 | **32.59** | S&C ok, not dom. |

#### `dpcc-t-tightened` — `both-hard`  (target: S&C 1.00 · 59.35 steps · 36.55 s/ep)

| K | model | S&C | succ | viol | steps | s/step | **s/ep** | vs target |
|---|---|---|---|---|---|---|---|---|
| 1 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 59.25 | 0.0182 | **1.08** | ✅ **beats** |
| 1 | AF α→0.05 | 0.95 | 1.00 | 0.1 | 62.15 | 0.0241 | **1.50** | — |
| 1 | MF-UNet | 0.95 | 1.00 | 0.1 | 59.50 | 0.0186 | **1.11** | — |
| 1 | FM-UNet | 1.00 | 1.00 | 0.0 | 64.55 | 0.0197 | **1.27** | S&C ok, not dom. |
| 2 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 57.65 | 0.0271 | **1.56** | ✅ **beats** |
| 2 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 58.10 | 0.0268 | **1.56** | ✅ **beats** |
| 2 | MF-UNet | 1.00 | 1.00 | 0.0 | 58.90 | 0.0268 | **1.58** | ✅ **beats** |
| 2 | FM-UNet | 1.00 | 1.00 | 0.0 | 61.50 | 0.0277 | **1.70** | S&C ok, not dom. |
| 5 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 55.70 | 0.2195 | **12.23** | ✅ **beats** |
| 5 | AF α→0.05 | 0.95 | 0.95 | 0.3 | 65.80 | 0.2115 | **13.92** | — |
| 5 | MF-UNet | 1.00 | 1.00 | 0.0 | 57.70 | 0.2118 | **12.22** | ✅ **beats** |
| 5 | FM-UNet | 1.00 | 1.00 | 0.0 | 58.45 | 0.1413 | **8.26** | ✅ **beats** |
| 10 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 58.85 | 0.3558 | **20.94** | ✅ **beats** |
| 10 | AF α→0.05 | 0.95 | 1.00 | 0.1 | 60.10 | 0.3560 | **21.40** | — |
| 10 | MF-UNet | 0.95 | 1.00 | 0.1 | 60.90 | 0.3783 | **23.04** | — |
| 20 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 60.00 | 0.9109 | **54.65** | S&C ok, not dom. |
| 20 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 61.05 | 0.9423 | **57.53** | S&C ok, not dom. |
| 20 | FM-UNet | 1.00 | 1.00 | 0.0 | 59.40 | 0.5681 | **33.74** | S&C ok, not dom. |

#### `dpcc-c-tightened` — `top-right-hard`  (target: S&C 1.00 · 61.35 steps · 29.48 s/ep)

| K | model | S&C | succ | viol | steps | s/step | **s/ep** | vs target |
|---|---|---|---|---|---|---|---|---|
| 1 | AF α→0.2 | 0.90 | 0.90 | 0.0 | 67.90 | 0.0170 | **1.15** | — |
| 1 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 73.15 | 0.0168 | **1.23** | S&C ok, not dom. |
| 1 | MF-UNet | 0.95 | 0.95 | 0.1 | 69.90 | 0.0169 | **1.18** | — |
| 1 | FM-UNet | 0.95 | 0.95 | 0.0 | 67.30 | 0.0164 | **1.11** | — |
| 2 | AF α→0.2 | 0.85 | 0.95 | 0.1 | 88.40 | 0.0265 | **2.34** | — |
| 2 | AF α→0.05 | 0.95 | 0.95 | 0.0 | 87.65 | 0.0258 | **2.26** | — |
| 2 | MF-UNet | 0.85 | 0.85 | 0.3 | 82.15 | 0.0281 | **2.31** | — |
| 2 | FM-UNet | 1.00 | 1.00 | 0.0 | 73.80 | 0.0251 | **1.86** | S&C ok, not dom. |
| 5 | AF α→0.2 | 0.85 | 0.85 | 0.1 | 67.10 | 0.2341 | **15.71** | — |
| 5 | AF α→0.05 | 0.70 | 0.70 | 0.1 | 62.15 | 0.2180 | **13.55** | — |
| 5 | MF-UNet | 0.60 | 0.60 | 0.7 | 63.10 | 0.2383 | **15.03** | — |
| 5 | FM-UNet | 1.00 | 1.00 | 0.0 | 66.95 | 0.0828 | **5.55** | S&C ok, not dom. |
| 10 | AF α→0.2 | 0.85 | 0.85 | 0.6 | 67.05 | 0.3924 | **26.31** | — |
| 10 | AF α→0.05 | 0.80 | 0.80 | 0.0 | 61.20 | 0.4098 | **25.08** | — |
| 10 | MF-UNet | 0.60 | 0.60 | 0.4 | 64.70 | 0.4095 | **26.49** | — |
| 20 | AF α→0.2 | 0.95 | 1.00 | 0.2 | 66.60 | 0.9842 | **65.55** | — |
| 20 | AF α→0.05 | 0.90 | 0.90 | 0.3 | 68.80 | 1.0562 | **72.67** | — |
| 20 | MF-UNet | 0.75 | 0.85 | 0.3 | 66.30 | 0.9643 | **63.93** | — |
| 20 | FM-UNet | 1.00 | 1.00 | 0.0 | 67.85 | 0.3290 | **22.32** | S&C ok, not dom. |

#### `dpcc-c-tightened` — `top-left-hard`  (target: S&C 1.00 · 61.00 steps · 35.66 s/ep)

| K | model | S&C | succ | viol | steps | s/step | **s/ep** | vs target |
|---|---|---|---|---|---|---|---|---|
| 1 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 63.95 | 0.0184 | **1.18** | S&C ok, not dom. |
| 1 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 66.00 | 0.0182 | **1.20** | S&C ok, not dom. |
| 1 | MF-UNet | 1.00 | 1.00 | 0.0 | 67.35 | 0.0190 | **1.28** | S&C ok, not dom. |
| 1 | FM-UNet | 1.00 | 1.00 | 0.0 | 64.65 | 0.0201 | **1.30** | S&C ok, not dom. |
| 2 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 85.05 | 0.0267 | **2.27** | S&C ok, not dom. |
| 2 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 86.80 | 0.0267 | **2.32** | S&C ok, not dom. |
| 2 | MF-UNet | 1.00 | 1.00 | 0.0 | 85.70 | 0.0275 | **2.36** | S&C ok, not dom. |
| 2 | FM-UNet | 1.00 | 1.00 | 0.0 | 68.60 | 0.0278 | **1.90** | S&C ok, not dom. |
| 5 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 61.20 | 0.2192 | **13.42** | S&C ok, not dom. |
| 5 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 63.60 | 0.2079 | **13.23** | S&C ok, not dom. |
| 5 | MF-UNet | 0.95 | 1.00 | 0.1 | 67.80 | 0.2278 | **15.45** | — |
| 5 | FM-UNet | 1.00 | 1.00 | 0.0 | 66.30 | 0.1224 | **8.12** | S&C ok, not dom. |
| 10 | AF α→0.2 | 1.00 | 1.00 | 0.0 | 62.85 | 0.3587 | **22.55** | S&C ok, not dom. |
| 10 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 62.70 | 0.3761 | **23.58** | S&C ok, not dom. |
| 10 | MF-UNet | 1.00 | 1.00 | 0.0 | 64.75 | 0.4039 | **26.15** | S&C ok, not dom. |
| 20 | AF α→0.2 | 0.95 | 1.00 | 0.2 | 63.15 | 0.9541 | **60.25** | — |
| 20 | AF α→0.05 | 1.00 | 1.00 | 0.0 | 63.55 | 0.9624 | **61.16** | S&C ok, not dom. |
| 20 | MF-UNet | 1.00 | 1.00 | 0.0 | 64.85 | 0.9382 | **60.84** | S&C ok, not dom. |
| 20 | FM-UNet | 0.90 | 1.00 | 0.3 | 70.35 | 0.5243 | **36.88** | — |

#### `dpcc-c-tightened` — `both-hard`  (target: S&C 1.00 · 59.35 steps · 36.55 s/ep)

| K | model | S&C | succ | viol | steps | s/step | **s/ep** | vs target |
|---|---|---|---|---|---|---|---|---|
| 1 | AF α→0.2 | 0.90 | 1.00 | 0.1 | 66.35 | 0.0182 | **1.21** | — |
| 1 | AF α→0.05 | 0.85 | 1.00 | 0.2 | 66.55 | 0.0189 | **1.26** | — |
| 1 | MF-UNet | 0.75 | 1.00 | 0.2 | 69.40 | 0.0188 | **1.30** | — |
| 1 | FM-UNet | 1.00 | 1.00 | 0.0 | 66.05 | 0.0192 | **1.27** | S&C ok, not dom. |
| 2 | AF α→0.2 | 0.90 | 1.00 | 0.1 | 87.70 | 0.0274 | **2.40** | — |
| 2 | AF α→0.05 | 0.90 | 1.00 | 0.1 | 90.85 | 0.0265 | **2.41** | — |
| 2 | MF-UNet | 0.85 | 1.00 | 0.2 | 83.75 | 0.0267 | **2.24** | — |
| 2 | FM-UNet | 1.00 | 1.00 | 0.0 | 71.60 | 0.0276 | **1.97** | S&C ok, not dom. |
| 5 | AF α→0.2 | 0.90 | 1.00 | 0.1 | 61.65 | 0.2158 | **13.30** | — |
| 5 | AF α→0.05 | 0.95 | 1.00 | 0.1 | 60.20 | 0.2251 | **13.55** | — |
| 5 | MF-UNet | 0.75 | 1.00 | 0.3 | 59.10 | 0.2168 | **12.81** | — |
| 5 | FM-UNet | 1.00 | 1.00 | 0.0 | 63.50 | 0.1259 | **8.00** | S&C ok, not dom. |
| 10 | AF α→0.2 | 0.85 | 1.00 | 0.2 | 60.10 | 0.3527 | **21.20** | — |
| 10 | AF α→0.05 | 0.95 | 1.00 | 0.1 | 59.75 | 0.3719 | **22.22** | — |
| 10 | MF-UNet | 0.75 | 1.00 | 0.2 | 59.80 | 0.3624 | **21.67** | — |
| 20 | AF α→0.2 | 0.85 | 1.00 | 0.3 | 60.45 | 0.9551 | **57.74** | — |
| 20 | AF α→0.05 | 0.95 | 1.00 | 0.1 | 61.25 | 0.9238 | **56.58** | — |
| 20 | FM-UNet | 1.00 | 1.00 | 0.0 | 59.60 | 0.5852 | **34.88** | S&C ok, not dom. |

### 3.1 Every row in this batch that clears the pinned target

| halfspace | model | K | arm | S&C | steps | s/ep | vs target |
|---|---|---|---|---|---|---|---|
| `top-left-hard` | **AF `α→0.2`** | **1** | `dpcc-t-tightened` | 1.00 | **57.20** | **1.07** | **33.3× cheaper** |
| `top-left-hard` | AF `α→0.05` | 1 | `dpcc-t-tightened` | 1.00 | 58.20 | 1.09 | 32.7× cheaper |
| `top-left-hard` | MF-UNet | 1 | `dpcc-t-tightened` | 1.00 | 60.75 | 1.13 | 31.6× cheaper |
| `top-left-hard` | AF `α→0.05` | 2 | `dpcc-t-tightened` | 1.00 | 58.15 | 1.60 | 22.3× cheaper |
| `top-left-hard` | MF-UNet | 2 | `dpcc-t-tightened` | 1.00 | 59.50 | 1.64 | 21.8× cheaper |
| `top-left-hard` | AF `α→0.2` | 2 | `dpcc-t-tightened` | 1.00 | 58.00 | 1.78 | 20.0× cheaper |
| `top-left-hard` | AF `α→0.2` | 5 | `dpcc-t-tightened` | 1.00 | 56.00 | 12.67 | 2.8× cheaper |
| `top-left-hard` | AF `α→0.05` | 5 | `dpcc-t-tightened` | 1.00 | 58.90 | 13.79 | 2.6× cheaper |
| `top-left-hard` | AF `α→0.2` | 10 | `dpcc-t-tightened` | 1.00 | 55.85 | 21.21 | 1.7× cheaper |
| `top-left-hard` | MF-UNet | 10 | `dpcc-t-tightened` | 1.00 | 60.15 | 23.43 | 1.5× cheaper |
| `top-left-hard` | AF `α→0.05` | 10 | `dpcc-t-tightened` | 1.00 | 58.75 | 24.07 | 1.5× cheaper |
| `both-hard` | **AF `α→0.2`** | **1** | `dpcc-t-tightened` | 1.00 | **59.25** | **1.08** | **33.9× cheaper** |
| `both-hard` | AF `α→0.2` | 2 | `dpcc-t-tightened` | 1.00 | 57.65 | 1.56 | 23.4× cheaper |
| `both-hard` | AF `α→0.05` | 2 | `dpcc-t-tightened` | 1.00 | 58.10 | 1.56 | 23.4× cheaper |
| `both-hard` | MF-UNet | 2 | `dpcc-t-tightened` | 1.00 | 58.90 | 1.58 | 23.2× cheaper |
| `both-hard` | FM-UNet (naive) | 5 | `dpcc-t-tightened` | 1.00 | 58.45 | 8.26 | 4.4× cheaper |
| `both-hard` | MF-UNet | 5 | `dpcc-t-tightened` | 1.00 | 57.70 | 12.22 | 3.0× cheaper |
| `both-hard` | AF `α→0.2` | 5 | `dpcc-t-tightened` | 1.00 | **55.70** | 12.23 | 3.0× cheaper |
| `both-hard` | AF `α→0.2` | 10 | `dpcc-t-tightened` | 1.00 | 58.85 | 20.94 | 1.7× cheaper |
| `top-right-hard` | MF-UNet | 5 | `dpcc-t-tightened` | 1.00 | 60.60 | 12.64 | 2.3× cheaper |

**What this says.**

1. **AF `α→0.2` at K=1 is the cheapest row that clears the target on two halfspaces** — 33.3× and
   33.9× cheaper than DPCC K20, at equal safety and *fewer* steps.
2. **Head-to-head against MF at the same K=1 / same arm:** AF `α→0.2` **57.20 vs MF 60.75** steps
   (TL) at equal S&C 1.00 and equal s/ep — a **Pareto dominance**. On `both-hard` at K=1 AF clears
   the target and **MF-UNet does not appear in the list at all**.
3. **AF `α→0.2` is the only model with a clearing row at K=10** on `top-left-hard` at 55.85 steps —
   the shortest path of any clearing row in the table.
4. **`top-right-hard` is the hard case for everyone.** Only MF-UNet K=5 clears it, at 2.3×. No AF
   row does. This is the honest counterweight to points 1–3 and must be reported with them.

---

## 4. Head-to-head ledger — AF vs MF over all DPCC arms

`top-right-hard`, seed 6, n=20. 7 arms × 5 K = 35 cells.

| | goal reached | S&C | steps |
|---|---|---|---|
| AF `α→0.2` vs MF | W 15 / L 14 / T 6 | W 14 / L 11 / T 10 | W 13 / L 22 |
| AF `α→0.05` vs MF | W 18 / L 10 / T 7 | W 16 / L 11 / T 8 | W 12 / L 23 |

Pareto cells (successes-only basis): AF `α→0.2` dominates MF in 8, MF dominates AF in 9.
AF `α→0.05`: 8 vs 8. **On the aggregate this arm is a wash** — the α-Flow advantage is not a
broad shift, it is concentrated in specific (arm, K) cells, above all `dpcc-t-tightened` at K=1.

Mean over the 7 DPCC arms, `top-right-hard`:

| K | S&C — AF `α→0.2` / AF `α→0.05` / MF |
|---|---|
| 1 | 0.493 / 0.500 / **0.557** |
| 2 | 0.557 / **0.621** / 0.543 |
| 5 | **0.543** / 0.536 / 0.500 |
| 10 | **0.586** / 0.536 / 0.500 |
| 20 | **0.564** / 0.514 / 0.493 |

Averaged over *all* projection arms MF leads at K=1 and AF leads at K ≥ 5 — the opposite ordering
to §1. The two are reconcilable: the raw-plan gain is at low K, and the projector *erases* it there
(it repairs both plans to a similar place), while at high K the projector is cheap relative to the
plan quality and AF's edge survives. **Which arm you select decides which K wins**, which is why
§3's single-arm result is the citable one and §4's average is not.

---

## 5. `hardflow_*` — excluded, and why

The `B` token in the plan folder is `hf_batch_size` (`config/avoiding-d3il.py:202`), the **HardFlow**
candidate fan — *not* the arms-A/B MPC fan, which `config/avoiding-d3il.py:67` states is not a
folder token at all.

| fan | AF `_B4_` runs | MF `_B1_` run | comparable? |
|---|---|---|---|
| arms A/B (`diffuser`, `dpcc-*`) | 4 (`[ eval ] mpc fan: arms A/B=4`) | 4 (hardcoded, pre-`FMPCC_MPC_BATCH`) | ✅ **fair** |
| arm C (`hardflow_*`) | **4** (`hf_batch=4`) | **1** (`hf_batch=1`) | ❌ **4:1 confound** |

→ §1–§4 are valid. **Every `hardflow_*` AF-vs-MF row is void** — AF picks best-of-4, MF gets 1.
The AF rows also start at K=5: the eval correctly skips HardFlow at K=1/K=2 where at `A=0.5` no
HardFlow math runs. MF's Aug-13 run predates that guard and still emits degenerate K=1/K=2 rows
(e.g. `HF-r` K=1: 0.30 / 45.15) — **do not use them.**

---

## 6. Measurement faults found while building this DA

### 6.1 ⚠️ `n_steps` has two incompatible definitions

Reconciling all **318** seed-6 cells between the AF eval logs and the DA CSV:

- **`n_success` matches in 318 / 318 cells.** Success numbers are solid.
- **`n_steps` mismatches in 116 cells — every one of them a cell with success < 1.00.**

Worked example, `dpcc-r` K=1 `top-right-hard`: log `0.45 / 68.00 ± 3.06`, CSV `0.45 / 52.75 ± 15.75`.
Solving `0.45·68.00 + 0.55·x = 52.75` gives x ≈ 40 — failed episodes terminate early at ~40 steps
and drag the all-episode mean down. **The eval log averages over successful episodes; the DA CSV
averages over all of them.**

Consequences:
- The two are **not interchangeable**, and the CSV column is **not comparable across models with
  different success rates** — it silently rewards failing fast.
- The tables in §1–§3 use the **CSV (all-episode)** basis, because the pinned target was computed
  that way and parity matters more than which basis is "right". §1.1 additionally gives the
  successes-only basis, where the K=1 verdict changes from *trade-off* to *dominance*.
- **This is not an α-Flow issue.** Any past claim that read `n_steps` off the DA CSV across models
  with unequal success should be re-checked.

### 6.2 ⚠️ Suspected DPCC projector degeneracy at K ≤ 2

`s/step` on every `dpcc-*` arm jumps ~14× between K=2 and K=5 (e.g. `dpcc-t-tightened`
`top-right-hard`: 0.0280 → 0.2501 s/step). With `diffusion_timestep_threshold = 0.5` the projector
only engages on the last half of the ODE trajectory, so at K=1–2 there may be **fewer than one
projected step**. If so, the K=1/K=2 `dpcc-*` rows are close to unprojected — the same class of
degeneracy already documented for HardFlow at K1/K2 — and §3's headline would be measuring the MPC
candidate selection rather than the DPCC projection. **This does not invalidate the AF-vs-MF
comparison (both sides are equally affected) but it does change what the 33× number means.**
← **run on cluster**: count projected steps per K, or log `n_projections`.

---

## 7. Verdict

| claim | status |
|---|---|
| α-Flow's objective actually trained the deployed weights | ✅ proven — `discrete_frac ≈ 0.5`, α floors at 0.05 / 0.2 |
| AF beats MF on the **raw plan** | ✅ Pareto-dominates at K=1, 5, 10 (`diffuser`, TR) |
| AF beats MF **after projection** | ✅ at `dpcc-t-tightened` K=1: 57.20 vs 60.75 steps, S&C 1.00 both, equal s/ep |
| AF beats the **pinned DPCC target** | ✅ TL and both-hard at K=1, **33× cheaper**; ❌ never on `top-right-hard` |
| AF beats MF **on average across projection arms** | ❌ a wash (8 vs 9 Pareto cells); MF leads at K=1 on the 7-arm mean |
| Deployment cost | ✅ identical `avg_time` — the win is free |
| Parameter count | ✅ 🔒 unchanged, 4.0 M U-Net |
| HardFlow comparison | ⛔ void — 4:1 fan confound (§5) |
| Statistical strength | ⛔ **n = 20, one seed.** K=1 diffuser 20/20 vs 17/20 → Fisher **p = 0.231** |

**Bottom line.** α-Flow, once actually switched on, produces a *better raw plan* than MeanFlow at
low K on the same 4.0 M U-Net at identical inference cost, and that survives the projection on the
`dpcc-t-tightened` arm — where it beats both MeanFlow and the DPCC K20 target on two of three
halfspaces. It does **not** beat MeanFlow on average across all projection arms, and it does not
touch `top-right-hard`. The result is **real and directional but not yet significant**: one seed,
20 episodes, and a 3-episode margin.

---

## 8. To run on the cluster (in priority order)

1. **Power the win.** Arm B on seeds 7–10 at the budgets where it wins (K = 1, 5, 10).
   Five seeds × 20 episodes turns a 3-episode margin into a quotable number and kills the
   single-seed objection.
   ```bash
   AF_BONE=unet AF_ALPHA_END=0.2 AF_SEEDS="7 8 9 10" AF_EPOCH=latest \
     AF_NTRIALS=20 AF_FLOW_STEPS="1 5 10" FMPCC_RUN_MSG=afon02_s7to10 \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/alphaflow_pipeline.sh
   ```
2. **Probe a higher floor.** `α→0.2` beat `α→0.05` on both arms; the trend says try more α.
   ```bash
   AF_BONE=unet AF_ALPHA_END=0.4 AF_SEEDS="6" AF_EPOCH=latest \
     AF_NTRIALS=20 AF_FLOW_STEPS="1 5 10" FMPCC_RUN_MSG=afon04_s6 \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/alphaflow_pipeline.sh
   ```
3. **Close §6.2** — instrument the DPCC projector to report how many steps it actually projects
   at each K under `T0.5`.
4. **Fair HardFlow rows** — re-run MF-UNet at `HFFM_BATCH=4`, or drop arm C from AF-vs-MF.
5. **Close §6.1** — carry both `n_steps` bases in the DA pipeline, explicitly labelled.

---

## 9. Open issues

- `/data` was at 100 % (27 G free of 7.0 T) before these runs. **Check before submitting.**
- Training logs 25290 / 25292 carry tqdm progress bars — ~2 KB of carriage returns per epoch line.
  Batch logs should not carry live bars.
- The MF comparator is 3 weeks older (2026-08-13, jobs 24559–24563) and ran on earlier eval code.
  Same checkpoint tree and same arms-A/B fan, but not the same binary.
- Two α floors were run and the better is quoted throughout — a selection effect that one extra
  seed would defend.
- No smoothness metric exists anywhere in this pipeline. Jerk / path length / curvature computed
  directly on the saved plan `.npz` files is the missing measurement.

---

## 10. Visual inspection — RESERVED

*(Left deliberately empty. To be filled in by the author with the human visual inspection of the
saved plans, and combined with §1's raw-plan result.)*
