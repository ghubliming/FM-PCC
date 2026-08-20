# DA — `n_trials=20` cross-family: DPCC vs FM vs MeanFlow-UNet vs AlphaFlow-SiT

**Date:** 2026-08-19 · **Task:** avoiding-d3il · **Batch:** `batch_avoiding_combined_20260819_214620`
**New this round:** FMv3ODE (naive Flow Matching) K∈{1,2,5}, job **24698**, `Evaluation completed successfully`, 22 h wall.
**Predecessor DA:** `DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md` (baseline n=2 → n=20 shift)

All cells: 5 seeds (6–10) × 20 trials = **100 episodes**, `action_weight=10`, `T=0.5`, Euler.
`s/ep` = `n_steps × avg_time` = wall-clock seconds per episode. `SEM` = across-seed std / √5 → **differences below ~0.10 in S&C are not real.**

## 0. Roster and completeness

| model | code folder | backbone | K sweep | seeds | halfspaces |
|---|---|---|---|---|---|
| **DPCC** (GaussianDiffusion) | `diffusion/` | UNet | 20 | 6–10 | TL ✅ TR ✅ **BH 5/13 variants, seed 6 only** ⚠️ |
| **FM** (naive FlowMatchingODE) | `flow_matching_v3_ode_selectable/` | UNet | 1, 2, 5 | 6–10 | ✅ all three, 13/13 |
| **MeanFlow** | `flow_matching_v3_meanflow/` | UNet | 1, 2, 5, 10, 20 | 6–10 | ✅ except **K20 has no BH** ⚠️ |
| **AlphaFlow** | `flow_matching_v3_alphaflow/` | **SiT** ⚠️ | 1, 2 | 6–10 | ✅ all three, 13/13 |

**FMv3ODE ran clean.** 39 cells per (halfspace, seed) = 13 variants × 3 K, all 15 (halfspace, seed) combinations present, `_aw10` loadpath correct, `--flow-steps` override logged for each K. The `aw1`/`aw10` crash from job 24688 is resolved.

**Variant sets are not identical across families** — DPCC and FM ran `gradient*`/`post_processing*`/`model_free*`; MeanFlow and AlphaFlow ran `hardflow_new-*` instead. The **comparable core is 7 variants**: `dpcc-{r,c,t}[-tightened]` + `diffuser`. Everything in §1–§3 uses only that core.

⚠️ **Backbone confound:** AlphaFlow is SiT, everything else is UNet. Per the architecture-matched rule, AlphaFlow results are secondary and cannot be read as a like-for-like win.

---

## 1. Headline — naive FM Pareto-dominates the DPCC baseline

**FM K=2 with `dpcc-c-tightened`, against the pinned target DPCC K20 / `dpcc-c-tightened`, same UNet backbone:**

| | S&C TL | S&C TR | S&C BH | n_steps TL / TR | s/step | **s/ep TL / TR** |
|---|---|---|---|---|---|---|
| DPCC K20 (target) | 1.00 | 0.95 | 1.00† | 70.0 / 77.6 | 0.558 / 0.517 | 39.1 / 40.2 |
| **FM K2** | 1.00 | **1.00** | 1.00 | **65.5 / 71.6** | **0.028 / 0.026** | **1.8 / 1.9** |

† DPCC's `both-hard` cell is seed-6-only (see §0).

On both fully-measured halfspaces this is **strict Pareto dominance**: equal-or-higher S&C, *fewer* control steps, *and* lower time per step — all three axes, no trade-off. Wall-clock per episode is **21×** lower. FM K1 is nearly as good (worst-halfspace 0.99) at 1.3 s/ep, i.e. **30×**.

The S&C gap on top-right (1.00 vs 0.95, SEM 0.073) is inside seed noise on its own; the compute gap is not, and the *direction* is consistent across K=1, 2, 5. The defensible claim is **"matches the baseline's success at 20–30× less compute"**, not "beats it on success".

---

## 2. Full matrix on the comparable core

#### `dpcc-c-tightened` — the baseline’s target projector

| model | bb | K | S&C TL | S&C TR | S&C BH | **worst** | s/ep TL | s/ep TR | s/ep BH | viol TR |
|---|---|---|---|---|---|---|---|---|---|---|
| DPCC (diffusion) | UNet | 20 | 1.00 | 0.95 | 1.00 | **0.95** | 39.1 | 40.2 | 36.5 | 0.10 |
| FM (naive) | UNet | 1 | 1.00 | 0.99 | 1.00 | **0.99** | 1.3 | 1.2 | 1.3 | 0.00 |
| FM (naive) | UNet | 2 | 1.00 | 1.00 | 1.00 | **1.00** | 1.8 | 1.9 | 1.9 | 0.00 |
| FM (naive) | UNet | 5 | 1.00 | 1.00 | 1.00 | **1.00** | 7.1 | 6.3 | 8.4 | 0.00 |
| MeanFlow | UNet | 1 | 1.00 | 0.98 | 0.85 | **0.85** | 1.3 | 1.3 | 1.3 | 0.05 |
| MeanFlow | UNet | 2 | 0.99 | 0.93 | 0.86 | **0.86** | 2.6 | 2.7 | 2.7 | 0.13 |
| MeanFlow | UNet | 5 | 0.99 | 0.76 | 0.85 | **0.76** | 13.9 | 16.0 | 13.3 | 0.68 |
| MeanFlow | UNet | 10 | 1.00 | 0.80 | 0.86 | **0.80** | 23.8 | 27.2 | 22.1 | 0.59 |
| MeanFlow | UNet | 20 | 1.00 | 0.88 | — | **0.88*** | 61.4 | 66.3 | — | 0.22 |
| AlphaFlow | SiT | 1 | 0.85 | 0.86 | 0.86 | **0.85** | 1.7 | 1.7 | 1.7 | 0.00 |
| AlphaFlow | SiT | 2 | 0.16 | 0.16 | 0.16 | **0.16** | 3.4 | 3.4 | 3.5 | 0.00 |

#### `dpcc-t-tightened` — best projector for the flow models

| model | bb | K | S&C TL | S&C TR | S&C BH | **worst** | s/ep TL | s/ep TR | s/ep BH | viol TR |
|---|---|---|---|---|---|---|---|---|---|---|
| DPCC (diffusion) | UNet | 20 | 1.00 | 0.92 | — | **0.92*** | 43.8 | 49.4 | — | 0.21 |
| FM (naive) | UNet | 1 | 1.00 | 0.97 | 1.00 | **0.97** | 1.3 | 1.4 | 1.3 | 0.00 |
| FM (naive) | UNet | 2 | 0.98 | 0.97 | 1.00 | **0.97** | 1.9 | 1.8 | 1.8 | 0.00 |
| FM (naive) | UNet | 5 | 1.00 | 1.00 | 1.00 | **1.00** | 7.2 | 6.4 | 8.0 | 0.00 |
| MeanFlow | UNet | 1 | 1.00 | 0.99 | 0.99 | **0.99** | 1.1 | 1.2 | 1.1 | 0.00 |
| MeanFlow | UNet | 2 | 1.00 | 0.98 | 1.00 | **0.98** | 1.6 | 1.8 | 1.6 | 0.04 |
| MeanFlow | UNet | 5 | 1.00 | 0.95 | 0.99 | **0.95** | 13.5 | 15.1 | 12.5 | 0.19 |
| MeanFlow | UNet | 10 | 1.00 | 0.95 | 0.97 | **0.95** | 23.8 | 26.9 | 22.3 | 0.09 |
| MeanFlow | UNet | 20 | 1.00 | 0.91 | — | **0.91*** | 57.4 | 70.4 | — | 0.13 |
| AlphaFlow | SiT | 1 | 1.00 | 0.96 | 1.00 | **0.96** | 1.0 | 1.0 | 1.0 | 0.00 |
| AlphaFlow | SiT | 2 | 1.00 | 0.91 | 1.00 | **0.91** | 1.3 | 1.5 | 1.2 | 0.00 |

#### `dpcc-r-tightened` 

| model | bb | K | S&C TL | S&C TR | S&C BH | **worst** | s/ep TL | s/ep TR | s/ep BH | viol TR |
|---|---|---|---|---|---|---|---|---|---|---|
| DPCC (diffusion) | UNet | 20 | 1.00 | 0.95 | 0.95 | **0.95** | 43.8 | 46.5 | 38.0 | 0.08 |
| FM (naive) | UNet | 1 | 1.00 | 0.99 | 1.00 | **0.99** | 1.4 | 1.3 | 1.3 | 0.07 |
| FM (naive) | UNet | 2 | 0.99 | 1.00 | 1.00 | **0.99** | 1.9 | 1.9 | 1.8 | 0.00 |
| FM (naive) | UNet | 5 | 0.99 | 1.00 | 1.00 | **0.99** | 7.9 | 7.1 | 8.3 | 0.00 |
| MeanFlow | UNet | 1 | 1.00 | 0.98 | 0.81 | **0.81** | 1.2 | 1.2 | 1.2 | 0.04 |
| MeanFlow | UNet | 2 | 1.00 | 0.95 | 0.85 | **0.85** | 1.8 | 1.9 | 1.8 | 0.04 |
| MeanFlow | UNet | 5 | 1.00 | 0.89 | 0.89 | **0.89** | 15.8 | 16.9 | 13.5 | 0.28 |
| MeanFlow | UNet | 10 | 0.99 | 0.89 | 0.84 | **0.84** | 29.2 | 29.8 | 24.4 | 0.21 |
| MeanFlow | UNet | 20 | 1.00 | 0.86 | — | **0.86*** | 73.4 | 76.2 | — | 0.17 |
| AlphaFlow | SiT | 1 | 1.00 | 1.00 | 1.00 | **1.00** | 1.1 | 1.0 | 1.0 | 0.00 |
| AlphaFlow | SiT | 2 | 1.00 | 0.95 | 1.00 | **0.95** | 1.5 | 1.5 | 1.3 | 0.00 |

`*` = worst taken over fewer than 3 halfspaces (missing cell). DPCC's `both-hard` column is seed-6-only throughout.

#### Best common-core projector per model (worst-halfspace gate, then cheapest)

| model | bb | K | best projector | S&C TL | S&C TR | S&C BH | **worst** | mean s/ep | steps TR |
|---|---|---|---|---|---|---|---|---|---|
| DPCC (diffusion) | UNet | 20 | `dpcc-c-tightened` | 1.00 | 0.95 | 1.00 | **0.95** | 38.6 | 77.6 |
| FM (naive) | UNet | 1 | `dpcc-c-tightened` | 1.00 | 0.99 | 1.00 | **0.99** | 1.3 | 69.3 |
| FM (naive) | UNet | 2 | `dpcc-c-tightened` | 1.00 | 1.00 | 1.00 | **1.00** | 1.9 | 71.6 |
| FM (naive) | UNet | 5 | `dpcc-t-tightened` | 1.00 | 1.00 | 1.00 | **1.00** | 7.2 | 70.5 |
| MeanFlow | UNet | 1 | `dpcc-t-tightened` | 1.00 | 0.99 | 0.99 | **0.99** | 1.1 | 65.3 |
| MeanFlow | UNet | 2 | `dpcc-t-tightened` | 1.00 | 0.98 | 1.00 | **0.98** | 1.6 | 64.6 |
| MeanFlow | UNet | 5 | `dpcc-t-tightened` | 1.00 | 0.95 | 0.99 | **0.95** | 13.7 | 65.9 |
| MeanFlow | UNet | 10 | `dpcc-t-tightened` | 1.00 | 0.95 | 0.97 | **0.95** | 24.3 | 64.6 |
| MeanFlow | UNet | 20 | `dpcc-t-tightened` | 1.00 | 0.91 | — | **0.91** | 63.9 | 67.6 |
| AlphaFlow | SiT | 1 | `dpcc-r-tightened` | 1.00 | 1.00 | 1.00 | **1.00** | 1.0 | 69.4 |
| AlphaFlow | SiT | 2 | `dpcc-r-tightened` | 1.00 | 0.95 | 1.00 | **0.95** | 1.4 | 73.3 |

#### Cost decomposition — `top-right-hard`, generation vs projection

| model | bb | K | s/step `diffuser` (gen only) | per-NFE gen cost | s/step `dpcc-c-tightened` | projection overhead | projected steps* |
|---|---|---|---|---|---|---|---|
| DPCC (diffusion) | UNet | 20 | 0.190 | 0.0095 | 0.517 | 0.327 | ~10 |
| FM (naive) | UNet | 1 | 0.010 | 0.0102 | 0.018 | 0.008 | ~1 |
| FM (naive) | UNet | 2 | 0.019 | 0.0093 | 0.026 | 0.007 | ~1 |
| FM (naive) | UNet | 5 | 0.044 | 0.0089 | 0.089 | 0.044 | ~3 |
| MeanFlow | UNet | 1 | 0.010 | 0.0096 | 0.017 | 0.008 | ~1 |
| MeanFlow | UNet | 2 | 0.019 | 0.0093 | 0.027 | 0.008 | ~1 |
| MeanFlow | UNet | 5 | 0.046 | 0.0092 | 0.233 | 0.187 | ~3 |
| MeanFlow | UNet | 10 | 0.094 | 0.0094 | 0.402 | 0.308 | ~5 |
| MeanFlow | UNet | 20 | 0.180 | 0.0090 | 0.965 | 0.784 | ~10 |
| AlphaFlow | SiT | 1 | 0.006 | 0.0063 | 0.014 | 0.008 | ~1 |
| AlphaFlow | SiT | 2 | 0.012 | 0.0059 | 0.019 | 0.007 | ~1 |

`*` projected steps ≈ `K − int(0.5·K)`, i.e. the DPCC-style in-loop projection fires only after the `T=0.5` threshold.

---

## 3. Per-family read

### FM (naive) — the winner, and it does not need K
Worst-halfspace S&C is **0.99 / 1.00 / 1.00** at K = 1 / 2 / 5, with essentially **zero** constraint violations (0.00 everywhere on `dpcc-c-tightened` and `dpcc-t-tightened`; a single 0.07 on `dpcc-r-tightened` K1/TR). Raising K buys nothing on quality and costs linearly: 1.3 → 1.9 → 7.2 s/ep. **K=2 is the operating point** (K=1 loses 0.01 on top-right, within noise, so K=1 is defensible too if NFE is the headline).

### MeanFlow-UNet — does not beat naive FM, and gets *worse* with K
Its best is K=1–2 with `dpcc-t-tightened` (0.99 / 0.98 worst-halfspace), which is ≈ FM but never above it, and it is strictly worse under the baseline's own `dpcc-c-tightened` projector (0.85–0.86 on `both-hard` vs FM's 1.00). Then the trend inverts the usual expectation:

| K | 1 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| S&C TR (`dpcc-c-tightened`) | 0.98 | 0.93 | 0.76 | 0.80 | 0.88 |
| violations TR | 0.05 | 0.13 | 0.68 | 0.59 | 0.22 |

**More NFE makes MeanFlow worse.** That is consistent with MeanFlow's training objective, which targets the 1-NFE average-velocity solution; iterating the learned mean-velocity field is not a convergent ODE integration. It also means the K10/K20 runs are wasted compute for this model — MF K20 costs 64 s/ep, *more than the DPCC baseline*, for a worse score.

Per the benchmark hierarchy, **MeanFlow must beat naive FM to justify itself. It does not, at any K, on any halfspace.**

### AlphaFlow-SiT — cheapest per NFE, but unstable, and backbone-confounded
With `dpcc-r-tightened` / `dpcc-t-tightened` it looks strong (worst-halfspace 1.00 at K=1, 0.95 at K=2) and it is the cheapest cell in the whole study at **1.0 s/ep** — but that is partly the SiT backbone: per-NFE generation cost is 0.006 s vs 0.009 s for every UNet (§ cost table). Not a like-for-like win.

⚠️ **AlphaFlow K=2 collapses under the `c` projector:** `dpcc-c` 0.12 / `dpcc-c-tightened` 0.16 on top-right, with `n_steps` blowing up to **184** (vs ~70 normal) — the episodes are running to the step cap. The same checkpoint scores 0.91–1.00 with `t`/`r` projectors, and AF K=1 is fine at 0.86. This is projector-specific divergence, not a weak model: **treat AF K2 + `dpcc-c*` as a bug to investigate, not a result to report.**

### DPCC — see the predecessor DA
Target unchanged: `dpcc-c-tightened`, 1.00 (TL) / 0.95 (TR), ~39 s/ep.

---

## 4. Where the time actually goes

Generation cost is **exactly linear in K** and identical across UNet models — 0.0089–0.0102 s per NFE for DPCC, FM and MeanFlow alike, 0.0059–0.0063 for SiT. This confirms K is a real NFE budget, not a folder label, and that the backbones are cost-comparable.

Projection cost also scales with K, because the DPCC-style projector fires once per sampling step past the `T=0.5` threshold (≈ K/2 solves per control step). Per-solve cost is *not* model-independent:

| model | projection overhead ÷ projected steps |
|---|---|
| FM K1–K2 / MeanFlow K1–K2 / AlphaFlow | ~0.008 s |
| FM K5 | ~0.015 s |
| DPCC K20 | ~0.033 s |
| MeanFlow K5–K20 | **0.062–0.078 s** |

MeanFlow's NLP solves are 4–10× more expensive than FM's at comparable K. Combined with its higher violation counts, the reading is that MeanFlow hands the projector trajectories that are further from feasible, so the solver works harder *and* still lands worse.

**So the 20–30× speedup is not one effect but two multiplied:** K drops 20 → 2 (10× fewer NFE *and* 10× fewer NLP solves), and FM's per-solve cost is ~4× lower than DPCC's.

---

## 5. Projector notes

**`hardflow_new-r`, `-c`, `-t` are numerically identical to each other** in every MeanFlow and AlphaFlow row, on all three halfspaces — likewise the three tightened ones. The constraint-type flag is a no-op for this projector, so the "6 hardflow variants" are really **2** (plain / tightened). Either the `r/c/t` argument is not threaded through, or it is ignored by design; worth confirming before any table reports them as distinct. HardFlow-tightened peaks at 1.00 (TL) / 0.99 (TR) / 0.86 (BH) for MF K1 — it does **not** clear the DPCC projector on `both-hard`.

**`post_processing` ≡ `dpcc-r` at K ≤ 2 is expected, not a bug.** With `T=0.5` the in-loop projection starts at step `int(0.5·K)` — step 0 for K=1, step 1 for K=2 — which is the *last* step either way, so projecting inside the loop and projecting the finished sample are the same operation. Verified: FM K1 and K2 give 12/14 metrics exactly equal (only the two timing metrics differ), while FM K5 gives 0–2/14.

This **sharpens** the anomaly flagged in the predecessor DA: the n=2 DPCC run showed the same identity at **K=20**, where `int(0.5·20)=10` means ten projected steps and the identity cannot hold — and indeed the n=20 DPCC run gives 0/14 at the same K. So that earlier finding stands as a genuine defect in the n=2 data, distinct from this benign K≤2 identity.

---

## 6. Verdict against the benchmark hierarchy

| claim | status |
|---|---|
| FM beats diffusion-DPCC (architecture-matched, UNet) | ✅ **yes** — Pareto-dominant at K=2 on both complete halfspaces, 21× less wall clock |
| MeanFlow beats naive FM | ❌ **no** — never, at any K, on any halfspace |
| AlphaFlow beats naive FM | ⚠️ **not established** — comparable S&C at lower cost, but SiT backbone and a hard failure at K2+`c` |
| HardFlow beats the DPCC projector | ❌ **no** — ties on TL/TR, loses badly on BH (0.86 vs 1.00) |

The headline result of this round is **FM, not the flow-family extensions.** The extensions currently cost credibility rather than add it.

---

## 7. Gaps

- [ ] **DPCC `both-hard`** — 5/13 variants, seed 6 only. The whole baseline comparison rests on TL/TR until this lands. Resubmit with the halfspace list narrowed to `['both-hard']` and the same `FMPCC_RUN_MSG=20trials`; ~12 h.
- [ ] **MeanFlow K20 `both-hard`** — missing.
- [ ] **AlphaFlow K5+** — only K1/K2 exist; and AF needs a UNet run before any claim is architecture-matched.
- [ ] **Investigate AF K2 + `dpcc-c*`** (S&C 0.16, n_steps 184).
- [ ] **Confirm `hardflow_new` r/c/t degeneracy** is intended.
- [ ] Supersede `SNAPSHOT_20260813_avoiding_d3il_vs_DPCC_baseline.md` — its headline row (`hardflow` K1) does not survive at n=20.

---
---

# Part II — Was `n_trials=20` worth it? Is the residual spread real?

*Added 2026-08-19 after Part I. Part I above is unchanged. Revised the same day — see the correction note in §9.*

Two questions:
1. Did 20 trials actually tighten the error bars, or widen them?
2. Whatever spread is left — is it **statistically real**, or is it pure randomness (in which case the eval is measuring noise and more compute is wasted)?

## 8. Method

Everything below is computed from **per-seed values** (`candidates_multidimensional_raw.csv`), not from aggregated columns.

Matched-pair design: every cell existing at both trial counts for the same checkpoint, K, projection variant and halfspace, with all 5 seeds — **255 matched cells** across 7 model×K configurations. Overdispersion tests (§10–12) additionally use all **385** five-seed n=20 cells, which includes FM.

| available as a matched pair | not available |
|---|---|
| DPCC K20, MeanFlow-UNet K1/K2/K5/K10, AlphaFlow-SiT K1/K2 | **FM (naive)** — K1/K2 have no n=2 predecessor, K5's is seed-6-only |
| | MeanFlow-UNet K20 — its n=2 sibling is the **DiT** checkpoint |

**n=2 identification is verified, not assumed:** all 285 candidate n=2 cells have S&C that is an exact multiple of 0.1 — the arithmetic signature of 5 seeds × 2 trials. The n=20 cells are not.

**A note on `seed` in this repo:** seeds 6–10 index *separately trained checkpoints* (`.../H8_.../6/`, `/7/`, …), not just eval RNG. So "between-seed variance" here means **training-run-to-training-run variation**, which is why §11 matters.

## 9. Answer 1 — the error bar **halved**

> ⚠️ **Correction.** The first version of this Part reported that the spread went *up* (0.111 → 0.275). That was wrong. It read `n_success_and_constraints_std` from the aggregated CSV as the across-seed spread; it is not — it is the **per-episode Bernoulli std** (≈ √(p(1−p))), averaged over seeds. Verified: for DPCC `top-left`/`dpcc-c`, per-seed rates are [0.85, 0.65, 0.50, 0.80, 0.70]; √(p(1−p)) averages to 0.438, which is exactly the reported "std", while the actual spread of those five numbers is 0.122. Every figure below is recomputed from per-seed values. The conclusion reverses.

| model × K | matched cells | across-seed sd n=2 | across-seed sd n=20 | change | SEM n=2 | SEM n=20 |
|---|---|---|---|---|---|---|
| DPCC K20 (UNet) | 26 | 0.153 | **0.100** | −35% | 0.069 | **0.045** |
| MeanFlow K1 (UNet) | 39 | 0.148 | **0.094** | −36% | 0.066 | **0.042** |
| MeanFlow K2 (UNet) | 39 | 0.191 | **0.079** | −59% | 0.085 | **0.035** |
| MeanFlow K5 (UNet) | 39 | 0.204 | **0.071** | −65% | 0.091 | **0.032** |
| MeanFlow K10 (UNet) | 34 | 0.189 | **0.073** | −61% | 0.085 | **0.033** |
| AlphaFlow K1 (SiT) | 39 | 0.151 | **0.100** | −34% | 0.068 | **0.045** |
| AlphaFlow K2 (SiT) | 39 | 0.172 | **0.062** | −64% | 0.077 | **0.028** |
| **all** | **255** | **0.173** | **0.082** | **−53%** | **0.077** | **0.037** |

Across-seed sd fell in **61%** of cells and by **53%** on average; SEM on the 5-seed mean went **0.077 → 0.037**. Every configuration improved. **n=20 was not wasted — it did exactly what it was supposed to do.**

> **This also corrects Part I.** Part I's header says `SEM = across-seed std / √5 → differences below ~0.10 are not real`, and its `±SEM` columns come from the same misread field. The true SEMs are ~2× smaller: for DPCC K20 `dpcc-c-tightened` @ top-right, Part I prints ±0.073; the real value is **±0.020**. Part I's numbers are otherwise correct — only the ± column and the ±0.10 floor are overstated, which makes Part I's claims *conservative*, not wrong. See §14 for what the headline comparison looks like with the correct error bars.

## 10. Answer 2 — the residual spread is **not** pure randomness

If the five checkpoints were interchangeable, each seed's 20 trials would be Binomial(20, p) with a shared p, and the spread across the five seed rates would be exactly √(p(1−p)/20). Test that directly: for each cell, X² = Σᵢ (xᵢ − 20p̂)² / (20 p̂(1−p̂)), which is χ²₄ under that null.

**Pooled over 303 non-degenerate cells: X² = 3041 on 1212 df — a dispersion ratio of 2.5×, p < 10⁻³⁰⁰.** Cells rejecting seed-homogeneity: **32% at p<0.05** and **21% at p<0.01**, against the 5% and 1% expected by chance.

Per configuration (all 5-seed n=20 cells, FM included):

| model × K | cells | pooled X² / df | dispersion | seeds interchangeable? | **effective episodes** (of 100) |
|---|---|---|---|---|---|
| DPCC K20 (UNet) | 21 | 305 / 84 | **3.6×** | **no** (z = +17) | **28** |
| FM K1 (UNet) | 31 | 308 / 124 | **2.5×** | **no** (z = +12) | 40 |
| FM K2 (UNet) | 26 | 578 / 104 | **5.6×** | **no** (z = +33) | **18** |
| FM K5 (UNet) | 29 | 364 / 116 | **3.1×** | **no** (z = +16) | 32 |
| MeanFlow K1 | 33 | 287 / 132 | 2.2× | **no** (z = +10) | 46 |
| MeanFlow K2 | 33 | 211 / 132 | 1.6× | **no** (z = +5) | 63 |
| MeanFlow K5 | 34 | 150 / 136 | 1.1× | **yes** (z = +0.8) | 91 |
| MeanFlow K10 | 29 | 150 / 116 | 1.3× | marginal (z = +2.2) | 77 |
| MeanFlow K20 | 13 | 53 / 52 | 1.0× | **yes** (z = +0.1) | 98 |
| AlphaFlow K1 (SiT) | 25 | 455 / 100 | **4.5×** | **no** (z = +25) | 22 |
| AlphaFlow K2 (SiT) | 29 | 180 / 116 | 1.6× | **no** (z = +4) | 64 |

**So: not randomness — but it costs you sample size.** The nominal "100 episodes" behind each cell is worth **18–98 independent episodes** depending on the model. DPCC (28), AlphaFlow K1 (22) and FM K2 (18) are the ones running on far less evidence than the headline count suggests.

Note the inverse pattern for MeanFlow: dispersion falls monotonically with K (2.2 → 1.6 → 1.1 → 1.3 → 1.0). At K≥5 its five checkpoints are statistically indistinguishable — but Part I already showed that is where MeanFlow's *mean* performance is worst. **It is consistently mediocre, not unstable.**

## 11. What kind of seed effect is it?

Two candidates: (a) a globally hard seed — e.g. one seed's eval initial states are harder for everyone; (b) checkpoint-specific training instability. These have opposite implications, and they are distinguishable.

**Kendall's W across cells within each configuration** — do the seeds keep the same rank order from variant to variant?

| model × K | mean rank by seed (6 → 10) | Kendall W | Friedman χ²(4) | consistent? |
|---|---|---|---|---|
| DPCC K20 | 3.5  **1.9**  3.6  3.2  2.8 | 0.18 | 15.3 | **yes** |
| FM K1 | 2.5  3.3  3.9  **2.2**  3.1 | 0.17 | 21.6 | **yes** |
| FM K2 | 3.5  3.0  2.6  3.1  2.8 | 0.04 | 4.6 | no |
| FM K5 | 2.8  3.2  2.5  2.7  3.8 | 0.10 | 11.8 | marginal |
| MeanFlow K1 | **1.7**  3.5  3.0  3.4  3.5 | 0.24 | 32.2 | **yes** |
| MeanFlow K2 | **2.1**  2.5  3.5  3.4  3.5 | 0.18 | 23.2 | **yes** |
| MeanFlow K5 | **1.8**  2.8  3.5  3.8  3.3 | 0.25 | 33.7 | **yes** |
| MeanFlow K10 | **2.1**  2.8  3.5  3.6  3.0 | 0.15 | 16.8 | **yes** |
| AlphaFlow K1 | 3.7  2.8  3.4  **2.3**  2.8 | 0.11 | 11.3 | marginal |
| AlphaFlow K2 | **1.9**  2.6  3.9  3.0  3.6 | 0.27 | 30.8 | **yes** |

Significant in 7 of 10, so there **is** a systematic per-seed effect — but W is only 0.04–0.27, meaning it explains a small slice of the ordering. Most of the overdispersion is seed × variant *interaction*, not a uniformly good or bad seed.

**And it is checkpoint-specific, not environmental.** Correlating the per-seed profiles between models gives mean r = **+0.07** (median +0.09; 12 of 45 pairs above +0.5, 8 below −0.5 — exactly what noise on 5 points looks like). A seed that is weak for MeanFlow is not weak for DPCC. Within a checkpoint family it *is* consistent: **MeanFlow's seed 6 is the worst at K=1, 2, 5 and 10** (mean S&C 0.590 vs ~0.69 for the others), and **DPCC's seed 7 is its worst** (0.362 vs ~0.47) at every variant.

**Conclusion: hypothesis (b).** The spread is training-run instability — retraining the same recipe with a different seed gives a measurably different policy. That is a model/training property, not an eval artifact, and no amount of eval compute removes it.

## 12. Is this "a more severe problem"?

Partly yes, and it is worth being precise about which part.

- **Not severe:** the eval protocol. n=20 halved the error bar as intended, and the residual is a real property of the models.
- **Severe:** any result in this repo's history based on **a single seed** is measuring one draw from a distribution whose sd is ~0.08–0.10 in S&C — and up to 0.35 for individual variants. Seed-6-only rows (of which there are many in the older logs, and DPCC's `both-hard` in Part I) should be treated as unusable for ranking.
- **Severe for the baseline specifically:** DPCC K20 has dispersion 3.6× — the *least* stable of the UNet configurations. The baseline everything is measured against is itself seed-sensitive.

## 13. Did n=2 mislead? (unchanged — these use means, not the misread column)

| | |
|---|---|
| mean Δ (n=20 − n=2) | **−0.018** — essentially zero |
| median Δ | 0.000 |
| \|Δ\| > 0.10 | **33% of cells** |
| \|Δ\| > 0.20 | **13% of cells** |
| worst case | **−0.40 … +0.36** |

**n=2 was unbiased but imprecise** — it scattered every cell by ±0.13 in a random direction rather than favouring anything. Of 77 cells reading exactly 1.00 at n=2, only **49 (64%)** are still 1.00 at n=20; the worst has fallen to 0.76. But of 1002 pairwise "A beats B" verdicts with a ≥0.10 gap at n=2, only **70 (7%)** are overturned — n=2 was adequate for coarse ranking, not for quoting a number.

The one systematic casualty remains MeanFlow's `hardflow_new-*` on `top-left-hard`, falling 0.90 → 0.50–0.56 at K=1, 2 **and** 5 independently. Three runs do not agree by accident: a real ~0.35 overestimate, not sampling noise.

## 14. The headline comparison with correct error bars

Part I's central claim, now with per-seed data:

| config | halfspace | per-seed S&C (6, 7, 8, 9, 10) | mean | sd | SEM |
|---|---|---|---|---|---|
| DPCC K20 `dpcc-c-tightened` | top-left | 1.00 1.00 1.00 1.00 1.00 | 1.000 | 0.000 | 0.000 |
| DPCC K20 `dpcc-c-tightened` | top-right | **1.00 0.90 1.00 0.90 0.95** | 0.950 | 0.045 | **0.020** |
| **FM K2** `dpcc-c-tightened` | top-left | 1.00 1.00 1.00 1.00 1.00 | 1.000 | 0.000 | 0.000 |
| **FM K2** `dpcc-c-tightened` | top-right | **1.00 1.00 1.00 1.00 1.00** | 1.000 | 0.000 | 0.000 |
| **FM K2** `dpcc-c-tightened` | both | 1.00 1.00 1.00 1.00 1.00 | 1.000 | 0.000 | 0.000 |

**FM K2 under `dpcc-c-tightened` is perfect on all 5 seeds × 3 halfspaces × 20 trials — 300/300 episodes, zero spread.** Paired by seed against DPCC on top-right, the differences are [0.00, +0.10, 0.00, +0.10, +0.05]: mean +0.050, t = 2.24 on 4 df, **p ≈ 0.09** — short of 0.05 only because DPCC ties it on 2 of 5 seeds.

So the honest statement is stronger than Part I's, in one direction and weaker in another: FM K2 is **not statistically proven better** on success (p≈0.09), but it is **provably not worse** and it is perfectly stable across seeds where the baseline is not — while costing 21× less wall-clock, which is not in doubt at all.

## 15. What to spend the next compute on

Method-of-moments decomposition over the 204 non-saturated matched cells: between-seed sd **0.077**, mean p(1−p) **0.163** → 42% of a seed-mean's variance is between-seed, 58% within-seed at n=20.

| seeds × trials | episodes (cost) | SEM |
|---|---|---|
| 5 × 2 (old) | 10 | 0.132 |
| **5 × 20 (current)** | **100** | **0.053** |
| 5 × 100 | 500 | 0.039 |
| 20 × 5 | 100 | 0.044 |
| 10 × 10 | 100 | 0.047 |
| 10 × 20 | 200 | 0.038 |
| 20 × 20 | 400 | 0.027 |

Both levers still work at n=20 — this is *not* the "trials are useless now" picture the first version of this Part claimed. But seeds are the better buy: **10 × 10 and 20 × 5 both cost the same 100 episodes as today and give a 11–17% tighter SEM**, and 10 × 20 costs 2× for a 28% improvement. Given §11 showed the residual is training instability, more seeds also *characterises* the thing that is actually varying, which more trials never will.

**Recommendation: keep n=20, go to 10 seeds.** Do not go to n=50.

## 16. Bottom line

1. **Was n=20 worth it? Yes.** Across-seed sd fell 53% (0.173 → 0.082), SEM 0.077 → 0.037, improving in every configuration. The earlier "variance went up" reading was a misread column and is retracted (§9).
2. **Is the leftover spread pure randomness? No.** Pooled X² = 3041 on 1212 df, dispersion 2.5×, p < 10⁻³⁰⁰; 32% of cells individually reject seed-homogeneity at p<0.05.
3. **What is it? Training instability.** Consistent within a checkpoint family (MeanFlow's seed 6, DPCC's seed 7 are reliably worst) but uncorrelated across models (r ≈ +0.07) — so it is the training run, not a hard environment seed.
4. **The severe consequence** is not that n=20 was wasted, but that each cell carries **18–98 effective episodes instead of 100**, and that **single-seed results are worthless** for ranking. DPCC, the baseline, is among the least stable (3.6×).
5. **FM K2 is the exception:** 300/300 episodes under `dpcc-c-tightened`, zero seed spread. Its Part I claim survives the correct error bars.
