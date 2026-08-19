# DA 2026-08-19 — Does naive **FMv3ODE** also have a variant that beats the DA target?

**Question asked:** we plan to claim *MeanFlow-UNet* and *AlphaFlow-SiT* beat the DPCC baseline.
Does plain **Flow Matching (FMv3ODE, `models.diffusion.FlowMatchingODE`)** also clear the bar under
the new DA guidelines — i.e. is there **one specific variant** that beats the target metric?

**Short answer: YES against the pinned target — but it does NOT survive the cross-K check.**
`FMv3ODE K20 / T0.5 / dpcc-c-tightened` **Pareto-dominates** the target on `top-left-hard` and
`top-right-hard` at matched `n_trials = 2`, and is a **non-dominated trade-off** (net cheaper) on
`both-hard`. **Two caveats block citation:** (1) the FM run only exists at `n_trials = 2`, and
trial-count parity against the n=20 target is mandatory; (2) **the baseline's own `K10` run is
cheaper than FMv3ODE K20 at identical S&C** — see the **Addendum (§9)**, which is the decisive
section and was added after the first draft.

> ⚠️ **Read §9 before citing anything in §2–§6.** §2's "Pareto-dominates the target" is correct as
> stated (the target is pinned at K20), but the conservative cross-K check in §9 shows FMv3ODE K20
> **loses to DPCC K10** by 0.64–0.85×. The MeanFlow low-K conclusion is **unaffected and in fact
> strengthened** by §9.

**Batch:** `temp/1808/batch_avoiding_combined_20260818_152911/`
**Task:** avoiding-d3il (state-based) · **Constraint:** halfspace

---

## 0. What is being compared

| | **FMv3ODE (candidate)** | **DPCC baseline (target)** |
|---|---|---|
| candidate # | **C156** | **C15** (n=2) / **C10** (n=20) |
| folder | `flow_matching_v3_ode_selectable/H8_D…FlowMatchingODE_a1.5_b1.0_aw10/H8_K20_Meuler_T0.5_D…FlowMatchingODE` | `plans/diffusion/H8_K20_D…GaussianDiffusion_aw10_thres0.5` |
| generative model | `models.diffusion.FlowMatchingODE` (deterministic ODE, Euler) | `models.GaussianDiffusion` (200-step DDPM head, K=20 sampling) |
| backbone | `models.Flow_matcher_U_Net_v2` | `models.UNet1DTemporalCondModel` |
| backbone config | dim 32, mults (1,2,4,8), hidden 256, no attention | **identical** |
| `action_weight` | **10** | **10** |
| K (`flow_steps_v3` / `n_diffusion_steps`) | **20** | **20** |
| `diffusion_timestep_threshold` | **0.5** | **0.5** |
| horizon | 8 | 8 |
| seeds | 6–10 (5) | 6–10 (5) |
| `n_trials` | **2** (10 episodes/cell) | 2 → C15 · **20** (100 episodes/cell) → C10 |
| run date | 2026-07-03 | 2026-05-04…09 (n=2) · 2026-08-18, job 24639 (n=20) |

⚙️ **The two backbones are the same network.** `diff` of `flow_matcher_v3_ode_selectable/models/unet1d_temporal_cond.py`
against `diffuser/models/unet1d_temporal_cond.py` shows an identical `ResidualTemporalBlock` and an
identical UNet body; the only delta is an optional FiLM `cond_mlp` branch in the diffuser copy that is
**disabled** (`use_cond_projection=False`) on the state-based pipeline. Same width, same depth, same
parameter count. This is the **architecture-matched** comparison — the strong form of the claim, not
the SiT/DiT-confounded form. *(Param count should still be printed once on the cluster to close it out.)*

**Target definition** (per `da-target-is-best-baseline-variant`, re-pinned in
`Data_Analysis/DA_Result_Curated_MD/DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md` §4):

> **DPCC K20 / aw10 / T0.5 / `dpcc-c-tightened` — S&C 1.00 (TL), 0.95 (TR); 39.1 / 40.2 s/ep.**

---

## 1. Full FMv3ODE K20 result table (C156, 5 seeds × 2 trials)

`s/ep` = `n_steps × avg_time` = wall-clock seconds per episode.

#### `top-left-hard`

| variant | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|
| **`dpcc-c-tightened`** | **1.00** | 1.00 | 0.0 | **65.9** | **0.403** | **26.6** |
| `dpcc-t-tightened` | 1.00 | 1.00 | 0.0 | 65.1 | 0.461 | 30.0 |
| `dpcc-c` | 0.80 | 1.00 | 0.7 | 67.5 | 0.434 | 29.3 |
| `dpcc-t` | 0.70 | 1.00 | 0.8 | 62.6 | 0.451 | 28.2 |
| `dpcc-r-tightened` ⚠️ | 0.60 | 1.00 | 11.3 | 89.3 | 1.539 | 137.5 |
| `post_processing-tightened` ⚠️ | 0.60 | 1.00 | 11.3 | 89.3 | 1.539 | 137.4 |
| `dpcc-r` ⚠️ | 0.60 | 1.00 | 0.8 | 75.1 | 0.529 | 39.7 |
| `post_processing` ⚠️ | 0.60 | 1.00 | 0.8 | 75.1 | 0.526 | 39.5 |
| `gradient` | 0.30 | 1.00 | 7.8 | 69.4 | 0.199 | 13.8 |
| `model_free-tightened` | 0.20 | 1.00 | 9.6 | 71.9 | 0.267 | 19.2 |
| `model_free` | 0.20 | 1.00 | 9.8 | 72.1 | 0.256 | 18.5 |
| `gradient-tightened` | 0.20 | 1.00 | 10.0 | 71.9 | 0.198 | 14.3 |
| `diffuser` | 0.20 | 1.00 | 7.4 | 70.2 | 0.184 | 12.9 |

#### `top-right-hard`

| variant | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|
| `post_processing-tightened` ⚠️ | 1.00 | 1.00 | 0.0 | 71.5 | 0.377 | 27.0 |
| `dpcc-t-tightened` | 1.00 | 1.00 | 0.0 | 66.5 | 0.391 | 26.0 |
| `dpcc-r-tightened` ⚠️ | 1.00 | 1.00 | 0.0 | 71.5 | 0.378 | 27.0 |
| **`dpcc-c-tightened`** | **1.00** | 1.00 | 0.0 | **67.0** | **0.400** | **26.8** |
| `dpcc-c` | 0.70 | 0.90 | 1.9 | 65.1 | 0.330 | 21.5 |
| `post_processing` / `dpcc-r` ⚠️ | 0.10 | 0.80 | 12.3 | 78.2 | 0.361 | 28.2 |
| `model_free-tightened` | 0.00 | 1.00 | 32.5 | 75.1 | 0.249 | 18.7 |
| `model_free` | 0.00 | 1.00 | 32.1 | 73.7 | 0.243 | 17.9 |
| `gradient-tightened` | 0.00 | 1.00 | 34.8 | 72.8 | 0.202 | 14.7 |
| `gradient` | 0.00 | 1.00 | 32.0 | 69.8 | 0.202 | 14.1 |
| `dpcc-t` | 0.00 | 0.80 | 9.6 | 75.5 | 0.368 | 27.8 |
| `diffuser` | 0.00 | 1.00 | 31.1 | 70.2 | 0.183 | 12.9 |

#### `both-hard`

| variant | S&C | succ | viol | steps | s/step | **s/ep** |
|---|---|---|---|---|---|---|
| `post_processing-tightened` ⚠️ | 1.00 | 1.00 | 0.0 | 59.0 | 0.634 | 37.4 |
| `dpcc-t-tightened` | 1.00 | 1.00 | 0.0 | 59.4 | 0.607 | 36.1 |
| `dpcc-r-tightened` ⚠️ | 1.00 | 1.00 | 0.0 | 59.0 | 0.641 | 37.8 |
| **`dpcc-c-tightened`** | **1.00** | 1.00 | 0.0 | **56.8** | 0.627 | **35.6** |
| `dpcc-t` | 0.30 | 1.00 | 8.4 | 69.8 | 0.512 | 35.7 |
| *(rest ≤ 0.30 S&C — noise floor, do not rank)* | | | | | | |

⚠️ **The same `post_processing` ↔ `dpcc-r` aliasing bug reported for the n=2 baseline is present in
this FM run too.** `post_processing` vs `dpcc-r` and `post_processing-tightened` vs `dpcc-r-tightened`
are **12/14 metrics exactly equal** on all three halfspaces (only `avg_time`/`avg_time_std` differ, in
the 3rd decimal). Those four rows report the same projection under two names — **do not use them.**
The `dpcc-r-tightened` row at `top-left-hard` (11.3 violations *and* S&C 0.60 *and* 1.539 s/step) is
the visible symptom. **Usable projected arms for FMv3ODE at n=2: `dpcc-c*` and `dpcc-t*` only.**

---

## 2. The head-to-head — `dpcc-c-tightened`, FMv3ODE K20 vs DPCC K20/aw10, matched `n=2`

| halfspace | S&C FM / DPCC | viol FM / DPCC | steps FM / DPCC | s/step FM / DPCC | **s/ep FM / DPCC** | verdict |
|---|---|---|---|---|---|---|
| `top-left-hard` | **1.00 / 1.00** | 0.0 / 0.0 | **65.9 / 68.4** | **0.403 / 0.565** | **26.6 / 38.7** | ✅ **Pareto-dominant** |
| `top-right-hard` | **1.00 / 1.00** | 0.0 / 0.0 | **67.0 / 77.2** | **0.400 / 0.504** | **26.8 / 38.9** | ✅ **Pareto-dominant** |
| `both-hard` | **1.00 / 1.00** | 0.0 / 0.0 | **56.8 / 64.8** | 0.627 / 0.590 ✗ | **35.6 / 38.3** | ◐ non-dominated (net cheaper) |

Pareto-dominance is claimed strictly per `pareto-definition-of-good`: **S&C tied, violations tied,
fewer control steps AND lower per-step time.** On `both-hard` the per-step time is 6 % *higher*, so
that cell is a **trade-off**, not a dominance — FM gets there in 8 fewer steps and 2.7 s less per
episode, but each step costs more.

### 2.1 Per-seed paired speedup (DPCC s/ep ÷ FM s/ep)

5-seed cluster bootstrap, 20 000 resamples:

| variant | halfspace | speedup | 95 % CI | per-seed (6,7,8,9,10) |
|---|---|---|---|---|
| `dpcc-c-tightened` | `top-left` | **1.52×** | `[1.14, 1.93]` | 1.10 · 2.19 · 1.14 · 1.24 · 1.96 |
| `dpcc-c-tightened` | `top-right` | **1.48×** | `[1.24, 1.77]` | 1.58 · 1.19 · 1.21 · 2.03 · 1.38 |
| `dpcc-c-tightened` | `both-hard` | 1.08× | `[1.01, 1.18]` | 1.02 · 0.98 · 1.14 · 1.03 · 1.24 |
| `dpcc-c-tightened` | TL+TR pooled | **1.50×** | `[1.31, 1.67]` | 1.34 · 1.69 · 1.17 · 1.63 · 1.67 |
| `dpcc-t-tightened` | `top-left` | 1.53× | `[1.03, 2.13]` | 1.03 · 1.74 · 0.83 · 1.41 · 2.63 |
| `dpcc-t-tightened` | `top-right` | **2.02×** | `[1.60, 2.64]` | 1.86 · 1.64 · 1.45 · 3.22 · 1.90 |
| `dpcc-t-tightened` | `both-hard` | 0.94× | `[0.79, 1.08]` | 1.07 · 0.80 · 0.70 · 1.03 · 1.11 |

**FM is faster in 14 of 15 seed × halfspace cells for `dpcc-c-tightened`** (the exception is seed 7 on
`both-hard`, 0.98×), with S&C = 1.00 in **all 30** cells on both sides. The CI excludes 1.0 on TL, TR
and pooled. `dpcc-t-tightened` shows the same picture on TL/TR but is **slower than the baseline on
`both-hard`** — another reason `dpcc-c-tightened` is the variant to claim.

### 2.2 Is the cross-job wall-clock comparison trustworthy? — Yes, and here is the calibration

The two runs are 2 months apart on the same cluster, so raw seconds could be hardware noise. The
**unprojected `diffuser` arm is a per-job hardware constant** and it is essentially identical across
every job in the batch:

| candidate | job date | K | `diffuser` s/step (TL / TR / BH) | **s/step per NFE** |
|---|---|---|---|---|
| C156 FMv3ODE | 2026-07-03 | 20 | 0.1842 / 0.1833 / 0.1831 | 0.0092 |
| C15 DPCC n=2 | 2026-05-04…09 | 20 | 0.1788 / 0.1786 / 0.1807 | 0.0089 |
| C10 DPCC n=20 | 2026-08-18 | 20 | 0.1837 / 0.1900 / — | 0.0092 |
| C154 FMv3ODE T0.05 | 2026-08-04 | 20 | 0.1702 / 0.1700 / 0.1699 | 0.0085 |
| C147 MF-UNet | 2026-08-11 | 5 | 0.0462 | 0.0092 |
| C139 MF-UNet | 2026-08-13 | 1 | 0.0096 | 0.0096 |

Per-NFE cost is **0.0085–0.0096 s across four months and five model families** — under ±6 %. The
FM job was, if anything, on **3 % slower** hardware than the baseline job, so the measured 1.5× is a
mild *under*-estimate, not a hardware artefact.

### 2.3 Where the speedup comes from — it is **not** fewer NFE

FM K20 and DPCC K20 have **identical generation cost** (0.184 vs 0.179 s/step on the `diffuser` arm).
The entire gain lives in the **projection stage**. Normalising each run by its own `diffuser` time:

| halfspace | variant | FMv3ODE s/step ÷ own diffuser | DPCC s/step ÷ own diffuser |
|---|---|---|---|
| `top-left` | `dpcc-c-tightened` | **2.19×** | 3.16× |
| `top-left` | `dpcc-t-tightened` | **2.50×** | 3.12× |
| `top-right` | `dpcc-c-tightened` | **2.19×** | 2.82× |
| `top-right` | `dpcc-t-tightened` | **2.13×** | 3.24× |
| `both-hard` | `dpcc-c-tightened` | 3.42× | 3.27× ✗ |
| `both-hard` | `dpcc-t-tightened` | 3.32× | 3.06× ✗ |

**Reading:** DPCC's per-step projection overhead on top of generation is ~2.1–2.2× on the FM reference
trajectory versus ~2.8–3.2× on the diffusion one — i.e. the **deterministic FM trajectory is ~30–35 %
cheaper to project** on the single-halfspace tasks. That is consistent with the NLP needing fewer
iterations to converge on a straighter, lower-variance reference. **On `both-hard` the effect reverses**
— the two-constraint problem is harder for the FM trajectory, and FM only stays ahead on `s/ep`
because it needs fewer control steps.

⚠️ This is an *inference*, not a measurement: `nlp_solves` / `nlp_failures` / `nfe` are **all `None`**
in these two runs, so iteration counts were never logged. To state the mechanism in a paper, re-run
with NLP-iteration logging enabled.

---

## 3. Against the **n=20** target — what survives and what does not

| | S&C TL | S&C TR | s/ep TL | s/ep TR |
|---|---|---|---|---|
| **Target** — DPCC `dpcc-c-tightened`, **n=20** | 1.00 | **0.95** | 39.1 | 40.2 |
| DPCC `dpcc-c-tightened`, n=2 | 1.00 | 1.00 | 38.7 | 38.9 |
| **FMv3ODE K20 `dpcc-c-tightened`, n=2** | **1.00** | **1.00** | **26.6** | **26.8** |

**The cost claim survives the trial-count gap; the success claim does not.**

- ✅ **Cost.** Per `DA_20260815_ntrials20_stability_MF_UNet.md` §1.5, `n_steps` and `avg_time` are
  *stable* between n=2 and n=20 (median relative error 3.9 %, p90 ~14 %). The baseline itself moved
  only 38.7 → 39.1 and 38.9 → 40.2 s/ep. **FM's 26.6 / 26.8 s/ep vs the target's 39.1 / 40.2 is a
  1.47 / 1.50× win that a 20-trial rerun will not erase** — it is ~10× the observed n-sensitivity.
- ❌ **S&C.** FM's `1.00` values are 10-episode numbers with 0.10 resolution. The baseline's own
  `top-right` cell fell **1.00 → 0.95** when re-measured at n=20, and §1(a) of the baseline DA says
  *any* n=2 `1.00` should be read as "≥0.90 ± 0.10". FM's `1.00` is therefore **unverified**, not
  proven equal. The precondition "S&C held" is established **only at n=2**.

**So the honest claim today is:**

> At a matched 2-trial protocol, matched backbone, matched `aw10`, matched K = 20 NFE and matched
> `T = 0.5`, **naive FMv3ODE + `dpcc-c-tightened` Pareto-dominates DPCC + `dpcc-c-tightened` on both
> single-halfspace tasks** (S&C 1.00 vs 1.00, 0 violations both, fewer steps, 1.5× less wall-clock),
> and is net-cheaper-but-non-dominated on `both-hard`.

**Do not write it as a beat of the n=20 target until C156 is re-run at `n_trials = 20`.**

---

## 4. Consequences for the MeanFlow / AlphaFlow story

Per `benchmark-hierarchy-who-beats-whom`, MF/AF must beat **both** diffusion-DPCC **and** naive FM.
This DA says naive FM is **not** a weak intermediate — it already clears the diffusion baseline at
matched budget. Three things follow.

**(a) The MF/AF headline has to be re-anchored.** Right now:

| method | backbone | K | S&C TL / TR / BH | s/ep TL / TR / BH | n |
|---|---|---|---|---|---|
| DPCC baseline `dpcc-c-tightened` (**target**) | UNet 32 | 20 | 1.00 / 0.95 / — | 39.1 / 40.2 / — | 20 |
| **FMv3ODE `dpcc-c-tightened`** | **UNet 32** | 20 | 1.00 / 1.00 / 1.00 | **26.6 / 26.8 / 35.6** | **2** ⚠️ |
| MF-UNet K1 `dpcc-t-tightened` | UNet 32 | **1** | 1.00 / 0.99 / 0.99 | **1.08 / 1.15 / 1.07** | 20 |
| MF-UNet K2 `dpcc-t-tightened` | UNet 32 | 2 | 1.00 / 0.98 / 1.00 | 1.60 / 1.76 / 1.56 | 20 |
| MF-UNet K5 `dpcc-t-tightened` | UNet 32 | 5 | 1.00 / 0.95 / 0.99 | 13.52 / 15.09 / 12.50 | 20 |
| AF-SiT K1 `dpcc-t-tightened` | SiT | 1 | 1.00 / 0.96 / 1.00 | 1.00 / 0.97 / 0.96 | 20 |
| AF-SiT K2 `dpcc-c-tightened` | SiT | 2 | 0.16 / 0.16 / 0.16 | 3.44 / 3.42 / 3.47 | 20 |

MF-UNet K1 is still **~25× cheaper than naive FM K20** at equal S&C, and it is measured at n=20. The
MF claim is intact — but it is a claim about **collapsing K to 1**, not about being intrinsically
friendlier to the projector. Frame it that way.

**(b) There is no naive-FM low-K run with 5 seeds, so the controlled "MF beats FM" comparison does not
exist yet.** All naive-FM candidates other than C156 (K20) are **single-seed**: C157 (K5), C152/C153
(K10, T0.05/T0.1), C154/C155 (K20, T0.05/T0.1).

**(c) ⚠️ The uncomfortable indication.** The single-seed naive-FM K5 run (C157) reports
`dpcc-c-tightened` at S&C 1.00 with **8.70 / 5.15 / 7.63 s/ep**, versus MF-UNet K5 at n=2
(**13.66 / 12.53 / 12.52**) and n=20 (**13.87 / 15.96 / 13.29**). Both runs have the same `diffuser`
cost (0.0456 vs 0.0462 s/step), so the difference is projection cost: naive FM 0.131 s/step vs
MF-UNet 0.218 s/step at `top-left`. **At equal K, naive FM may be cheaper to project than MeanFlow.**
C157 is 1 seed × 2 trials = **2 episodes**, so this is a hypothesis, not a finding — but it is exactly
the comparison a reviewer will ask for, and it should be settled before the MF write-up.

---

## 5. Second lead: the FM `T` (threshold) knob works and the diffusion one does not

`diffusion_timestep_threshold` controls the fraction of the sampling trajectory over which DPCC
projects, so it is a direct compute dial. Single-seed sweeps in this batch:

| candidate | model | K | T | `dpcc-c-tightened` S&C (TL/TR) | s/ep (TL/TR) |
|---|---|---|---|---|---|
| C156 | FMv3ODE | 20 | 0.5 | 1.00 / 1.00 (5 seeds) | 26.6 / 26.8 |
| C155 | FMv3ODE | 20 | 0.1 | 1.0 / 1.0 | **11.9 / 12.7** |
| C154 | FMv3ODE | 20 | 0.05 | 1.0 / 1.0 | 15.9 / 16.6 |
| C153 | FMv3ODE | 10 | 0.1 | 1.0 / 1.0 | **10.8 / 10.8** |
| C152 | FMv3ODE | 10 | 0.05 | 1.0 / 1.0 | 10.8 / 10.8 |
| C13 `X_` | DPCC | 20 | 0.1 | 1.0 / 1.0 | 33.2 / 33.5 |
| C12 `X_` | DPCC | 20 | 0.05 | 1.0 / 1.0 | 33.5 / 33.8 |
| C14 `X_` | DPCC | 20 | 1.0 | 1.0 / 1.0 | 33.7 / 34.0 |

The DPCC rows are **flat in T** (0.532–0.581 s/step for T ∈ {0.05, 0.1, 1.0}) — the knob had no
effect on the diffusion path, which is presumably why those three folders carry the `X_` (excluded)
prefix. On the FM path the knob **does** bite: 0.400 → 0.187 s/step going T 0.5 → 0.1 at K20.

If T0.1 holds at 5 seeds × 20 trials, `FMv3ODE K20 T0.1` lands at **~12 s/ep against the target's
~40 s/ep — a 3.3× win**, and `K10 T0.1` at ~10.8 s/ep. **All of these are 1 seed × 2 trials = 2
episodes. Treat as a lead to test, not a result.**

---

## 6. Verdict

| question | answer |
|---|---|
| Does FMv3ODE have a variant that beats the **pinned K20 target**? | **Yes — `K20 / T0.5 / dpcc-c-tightened`.** |
| Does it beat the baseline's **best K** (K10)? | **No — see §9.** DPCC K10 is 0.64–0.85× the cost at the same S&C 1.00. |
| On what axis? | **Wall-clock**: 1.50× (TL), 1.48× (TR) at tied S&C and tied 0 violations, plus fewer control steps. |
| Pareto-dominant or trade-off? | **Pareto-dominant** on `top-left-hard` and `top-right-hard`. **Non-dominated trade-off** on `both-hard` (fewer steps, higher s/step, net cheaper). |
| Architecture-matched? | **Yes** — same UNet, same width/depth, same `aw10`, same K, same T. This is the strong form. |
| Citable today? | **No.** n=2 only, *and* it loses the cross-K check (§9). Cost claim vs K20 will survive n=20; the S&C 1.00 will not survive as-is. |
| Does it weaken the MF/AF claim? | Not the K-reduction claim. It **does** mean naive FM must appear in every table as a real competitor, and it raises an open question at matched K (§4c). |

---

## 7. To run on the cluster (in priority order)

1. **`FMv3ODE K20, T0.5, n_trials=20`, seeds 6–10** — the one run that converts §2 into a citable
   result. `config/projection_eval.yaml`: `n_trials: 2 → 20`; `FMPCC_RUN_MSG=20trials`;
   `FMV3_FLOW_STEPS="20"`; `Slurm_Codes/sbatch/eval_fmv3_ode_job.sh`.
   ⏱ Budget from the baseline's rate: job 24639 bought ~2.1 halfspaces in 24 h at K20 — **submit one
   job per halfspace**, or `--time` will kill it exactly as it killed the baseline's `both-hard`.
2. **`FMv3ODE K1 / K2 / K5, T0.5, n_trials=20`, 5 seeds** — closes §4(b)/(c). Same protocol as the
   MF-UNet ladder so the K-matched MF-vs-FM comparison finally exists. Cheap: K≤5 finishes fast.
3. **`FMv3ODE K20 / K10 at T ∈ {0.1, 0.05}`, 5 seeds × 20 trials** — tests the §5 lead (potential 3×).
4. **Fix / re-verify the `post_processing` path** before any FM table is published — the aliasing bug
   (§1 ⚠️) is present in C156 exactly as in the n=2 baseline, and the n=20 baseline shows it was
   fixed somewhere between May and August. Determine when, and whether C156 predates the fix.
5. **Enable NLP-iteration logging** (`nlp_solves`, `nlp_failures`, `nfe`) so §2.3's mechanism can be
   measured rather than inferred.

## 8. Open issues

- [ ] `both-hard` for the n=20 baseline is still missing (only seed 6, 5/13 variants) — the FM
      `both-hard` comparison has no valid n=20 reference either way.
- [ ] Print and record the parameter count of `Flow_matcher_U_Net_v2` vs `UNet1DTemporalCondModel`
      at dim 32 / mults (1,2,4,8) to close the architecture-match argument numerically.
- [ ] Any table that pairs an n=2 method row with an n=20 baseline row is not a comparison — this
      document's §2 is n=2/n=2 and §3 is explicitly flagged. Keep that discipline elsewhere.

---
---

# 9. ADDENDUM — the K-ladder check: does FMv3ODE change the MeanFlow low-K conclusion?

**Added after the first draft, in response to: "does this change the conclusion that MF-UNet at
lower K beats the baseline? check higher/lower K situations."**

**Answers up front:**

1. **The MeanFlow-UNet low-K conclusion is NOT changed — it is strengthened.** MF-UNet K1/K2 beat
   *every* rung of the baseline's own K ladder, by 11–23×, and naive FMv3ODE gets nowhere near them.
2. **The FMv3ODE conclusion IS changed at high K.** §2's win is against the *pinned K20 target only*.
   Against the **baseline's own K10 run** — same code, same `aw10`, same `T0.5`, same 5 seeds, same
   n=2 — FMv3ODE K20 **loses**, at 0.64–0.85× the cost ratio. §2 stands as written; this is a
   cross-K check §2 did not perform.
3. **The mid-K band (K5–K10) is the genuinely unsettled region** for both methods.

## 9.1 What I had missed: the baseline has its own 5-seed K ladder

The first draft compared only against `H8_K20…aw10_thres0.5`. The batch also contains **two more
fully-seeded DPCC rungs at the same `aw10`/`T0.5` protocol**:

| candidate | folder | K | seeds | n_trials |
|---|---|---|---|---|
| **C8** | `plans/diffusion/H8_K1_D…GaussianDiffusion_aw10/H8_K1_T0.5_D…` | **1** | 6–10 | 2 |
| **C7** | `plans/diffusion/H8_K10_D…GaussianDiffusion_aw10_thres0.5` | **10** | 6–10 | 2 |
| C15 | `plans/diffusion/H8_K20_D…GaussianDiffusion_aw10_thres0.5` | 20 | 6–10 | 2 |

Note K is a **training** parameter for diffusion (`n_diffusion_steps`; the loadpath carries it), so
C8 and C7 are *separately trained* models, not the K20 checkpoint sampled with fewer steps. For
FM/MeanFlow, K is inference-only.

## 9.2 The full ladder, `dpcc-c-tightened`, matched `n=2`, 5 seeds

| run | TL S&C / viol / s/ep | TR S&C / viol / s/ep | BH S&C / viol / s/ep |
|---|---|---|---|
| **DPCC K1** | **0.60** / 1.3 / 3.20 | **0.50** / 5.5 / 2.15 | 0.90 / 0.1 / 1.91 |
| **DPCC K10** | 1.00 / 0.0 / **21.98** | 1.00 / 0.0 / **21.03** | 1.00 / 0.0 / **22.25** |
| DPCC K20 *(target rung)* | 1.00 / 0.0 / 38.67 | 1.00 / 0.0 / 38.95 | 1.00 / 0.0 / 38.25 |
| FMv3ODE K5 *(1 seed!)* | 1.00 / 0.0 / 8.70 | 1.00 / 0.0 / 5.15 | 1.00 / 0.0 / 7.63 |
| **FMv3ODE K20** | 1.00 / 0.0 / **26.57** | 1.00 / 0.0 / **26.80** | 1.00 / 0.0 / **35.60** |
| MF-UNet K1 | 1.00 / 0.0 / **1.26** | 1.00 / 0.0 / **1.25** | 0.80 / 0.3 / 1.46 |
| MF-UNet K2 | 1.00 / 0.0 / 2.61 | 0.90 / 0.0 / 2.62 | 0.90 / 0.2 / 2.69 |
| MF-UNet K5 | 1.00 / 0.0 / 13.66 | 0.90 / 0.0 / 12.53 | 0.60 / 0.6 / 12.52 |
| MF-UNet K10 | 1.00 / 0.0 / 23.10 | 0.90 / 0.0 / 21.99 | 0.60 / 0.9 / 21.39 |

Per-NFE cost is constant across the whole ladder — `diffuser` arm gives 0.0089 (DPCC K1), 0.0089
(K10), 0.0089 (K20), 0.0092 (FM K20), 0.0098 (MF K1), 0.0092 (MF K5), 0.0092 (MF K10) s/NFE — so
every s/ep number above is on the same clock and the cross-run comparison is sound.

## 9.3 Paired per-seed cost ratios (5-seed bootstrap, 20 000 resamples)

Ratio > 1 means **the row's method is cheaper** than the comparator at the stated S&C.

| comparison | variant | top-left | top-right | both-hard |
|---|---|---|---|---|
| FMv3ODE K20 vs **DPCC K20** | `dpcc-c-t` | **1.52×** `[1.15,1.93]` | **1.48×** `[1.23,1.77]` | 1.08× `[1.01,1.18]` |
| **FMv3ODE K20 vs DPCC K10** | `dpcc-c-t` | **0.85×** `[0.70,1.02]` ❌ | **0.80×** `[0.66,0.93]` ❌ | **0.64×** `[0.50,0.78]` ❌ |
| **FMv3ODE K20 vs DPCC K10** | `dpcc-t-t` | 0.89× `[0.64,1.25]` | 0.90× `[0.70,1.16]` | **0.52×** `[0.44,0.60]` ❌ |
| MF-UNet K1 vs **DPCC K1** | `dpcc-c-t` | 2.96× `[1.20,5.54]` | 2.28× `[0.68,4.82]` | 1.31× `[1.21,1.42]` |
| **MF-UNet K1 vs DPCC K10** | `dpcc-c-t` | **17.5×** `[15.7,19.3]` | **17.1×** `[14.2,20.3]` | **15.5×** `[12.5,18.4]` |
| **MF-UNet K1 vs DPCC K10** | `dpcc-t-t` | **23.2×** `[18.3,29.0]` | **21.6×** `[19.5,23.5]` | **17.2×** `[15.3,18.5]` |

S&C in every one of those cells: FM K20 = DPCC K10 = **1.00 vs 1.00** (tie, so the cost ratio is the
whole story). MF K1 vs DPCC K10 = **1.00/1.00/0.80 vs 1.00/1.00/1.00** on `dpcc-c-t`, and
**1.00/0.90/1.00 vs 1.00/1.00/1.00** on `dpcc-t-t`.

## 9.4 Q1 — does this change "MF-UNet at low K beats the baseline"? **No. It strengthens it.**

Three independent reasons:

**(a) MF-UNet K1/K2 beat the baseline's *cheapest working rung*, not just its most expensive one.**
DPCC K10 at 21.0–22.3 s/ep is the strongest cross-K attack a reviewer can mount, and MF-UNet K1
still wins it by **17–23×** at tied S&C on both single halfspaces. Naive FMv3ODE K20 fails this
exact test. **MF's advantage is robust to the choice of baseline rung; FM's is not.**

**(b) At K = 1, diffusion-DPCC collapses and MeanFlow does not.** This is the sharpest result in the
addendum and it was invisible while only K20 was in view:

| | TL S&C | TR S&C | BH S&C | violations TL/TR |
|---|---|---|---|---|
| DPCC K1 `dpcc-c-tightened` | **0.60** | **0.50** | 0.90 | 1.3 / 5.5 |
| DPCC K1 `dpcc-t-tightened` | **0.60** | **0.20** | 1.00 | 3.3 / 7.7 |
| MF-UNet K1 `dpcc-c-tightened` | **1.00** | **1.00** | 0.80 | 0.0 / 0.0 |
| MF-UNet K1 `dpcc-t-tightened` | **1.00** | 0.90 | 1.00 | 0.0 / 0.0 |

MeanFlow's claim upgrades from *"cheaper at the same quality"* to **"MeanFlow retains the constraint
guarantee at a 1-NFE budget where the diffusion baseline loses it"** — a mechanism claim, which is
much harder to dismiss as a wall-clock artefact.
⚠️ Anticipate the obvious rebuttal: a K=1 DDPM is a degenerate model *by construction*, so a reviewer
will call C8 a strawman. The defensible framing is **"diffusion cannot reduce NFE without retraining,
and retraining to K=1 does not recover the performance; MeanFlow is designed for it."** Report the
DPCC K10 comparison as the primary number and DPCC K1 as the supporting one, not the reverse.

**(c) Naive FMv3ODE never enters MF's operating region.** FM's cheapest **5-seed** point is 26.6 s/ep
(K20). MF-UNet K1 is at 1.26 s/ep — **21× cheaper**. Even the single-seed FM K5 (5.15–8.70 s/ep) is
4–7× more expensive than MF K1. There is no naive-FM run anywhere near 1–3 s/ep.

## 9.5 Q2 — where FMv3ODE *does* change conclusions

### High K (K10–K20): FMv3ODE's §2 win does not survive the cross-K check — but MF-UNet's mostly does not either

| method @ high K | passes the target S&C gate? | vs DPCC K20 | **vs DPCC K10** |
|---|---|---|---|
| FMv3ODE K20 `dpcc-c-t` | tied at n=2 (unverified) | ✅ 1.48–1.52× | ❌ **0.80–0.85×** |
| MF-UNet K10 `dpcc-t-t` (n=20) | ✅ 1.00 / 0.95 | ✅ 1.84× | ⚠️ **1.07× — a tie** |
| MF-UNet K10 `dpcc-c-t` (n=20) | ❌ TR 0.80 | — | — |
| MF-UNet K20 `dpcc-t-t` (n=20) | ❌ TR 0.91, and incomplete (2 seeds TL) | ❌ 0.76× | ❌ 0.44× |

**Nothing at K ≥ 10 is worth claiming.** FMv3ODE K20 loses to DPCC K10; MF-UNet K10 merely ties it;
MF-UNet K20 loses outright and is incomplete besides. **Drop K20 from the MF headline** — its only
role is as an ablation showing MeanFlow does not benefit from extra NFE.

### Mid K (K5): the one place naive FM might actually beat MeanFlow

At matched K = 5 the `diffuser` arms are identical (FM 0.0456 vs MF 0.0460 s/step), so any gap is
pure projection cost:

| | TL s/step | TL s/ep | TR s/ep | BH s/ep | TL/TR/BH S&C |
|---|---|---|---|---|---|
| FMv3ODE K5 `dpcc-c-t` **(1 seed, 2 episodes)** | **0.131** | **8.70** | **5.15** | **7.63** | 1.0 / 1.0 / 1.0 |
| MF-UNet K5 `dpcc-c-t` (5 seeds, n=2) | 0.218 | 13.66 | 12.53 | 12.52 | 1.00 / 0.90 / 0.60 |
| MF-UNet K5 `dpcc-c-t` (5 seeds, n=20) | — | 13.87 | 15.96 | 13.29 | 0.99 / **0.76** / 0.85 |

MF-UNet K5 on `dpcc-c-tightened` **fails the target gate at n=20** (TR 0.76 ±0.06 vs the target's
0.95) and is 1.6–2.4× more expensive than the 1-seed naive-FM K5 point. On `dpcc-t-tightened` MF K5
does pass (0.95 TR) at 13.52 s/ep — still ~2× the FM K5 indication. **If FM K5 holds up at 5 seeds ×
20 trials, naive FM beats MeanFlow in the mid-K band.** That is the single most dangerous open
result for the MF story, and it rests on **2 episodes**. Settle it.

### Low K (K1–K2): unchanged, because naive FM has never been run there

**There is no naive-FMv3ODE run at K1 or K2 in this batch at any seed count.** The K-matched
"MeanFlow beats naive Flow Matching at K=1" comparison — the one that justifies the MeanFlow
objective at all — **does not exist yet.** Everything currently supporting it is inference from
higher-K behaviour. Given that FM at K5 already looks strong, a naive FM K1 run is not guaranteed to
fail, and if it succeeds the entire MeanFlow rationale narrows sharply. **This is the highest-value
missing experiment in the generation.**

## 9.6 Which rows actually pass the target gate at n=20 (reference table)

Gate = TL S&C ≥ 1.00 and TR S&C ≥ 0.95, both within seed SEM, per §4 of the baseline DA.

| run | variant | TL S&C (±SEM) | TR S&C (±SEM) | TL / TR s/ep | gate |
|---|---|---|---|---|---|
| DPCC K20 **(target)** | `dpcc-c-t` | 1.00 ±0.00 | 0.95 ±0.02 | 39.05 / 40.15 | — |
| MF-UNet K1 | `dpcc-t-t` | 1.00 ±0.00 | **0.99** ±0.01 | **1.08 / 1.15** | ✅ **PASS**, 36–43× cheaper |
| MF-UNet K2 | `dpcc-t-t` | 1.00 ±0.00 | **0.98** ±0.01 | 1.60 / 1.76 | ✅ PASS, 24–28× |
| MF-UNet K1 | `dpcc-c-t` | 1.00 ±0.00 | 0.98 ±0.01 | 1.26 / 1.28 | ✅ PASS, 31× |
| MF-UNet K2 | `dpcc-c-t` | 0.99 ±0.01 | 0.93 ±0.03 | 2.56 / 2.67 | ✅ PASS (within SEM) |
| MF-UNet K5 | `dpcc-t-t` | 1.00 ±0.00 | 0.95 ±0.03 | 13.52 / 15.09 | ✅ PASS, 2.9–3.3× |
| MF-UNet K10 | `dpcc-t-t` | 1.00 ±0.00 | 0.95 ±0.02 | 23.75 / 26.94 | ✅ PASS, 1.6–1.8× |
| AF-SiT K1 | `dpcc-t-t` | 1.00 ±0.00 | 0.96 ±0.03 | 1.00 / 0.97 | ✅ PASS, 39–51× *(SiT — confounded)* |
| AF-SiT K2 | `dpcc-t-t` | 1.00 ±0.00 | 0.91 ±0.07 | 1.33 / 1.54 | ✅ PASS (within SEM) |
| MF-UNet K5 | `dpcc-c-t` | 0.99 ±0.01 | **0.76** ±0.06 | 13.87 / 15.96 | ❌ FAIL |
| MF-UNet K10 | `dpcc-c-t` | 1.00 ±0.00 | **0.80** ±0.07 | 23.77 / 27.21 | ❌ FAIL |
| MF-UNet K20 | `dpcc-t-t` | 1.00 ±0.00 | 0.91 ±0.04 | 57.36 / 70.42 | ❌ FAIL + incomplete |
| AF-SiT K1 | `dpcc-c-t` | 0.85 ±0.14 | 0.86 ±0.14 | 1.68 / 1.68 | ❌ FAIL |
| AF-SiT K2 | `dpcc-c-t` | 0.16 ±0.06 | 0.16 ±0.06 | 3.44 / 3.42 | ❌ FAIL (collapsed) |
| FMv3ODE K20 | `dpcc-c-t` | 1.00 *(n=2)* | 1.00 *(n=2)* | 26.57 / 26.80 | ⚠️ n=2 only; loses to DPCC K10 |

**Note the variant split:** MeanFlow passes the gate on **`dpcc-t-tightened`** across the whole
ladder but only at K1–K2 on `dpcc-c-tightened`, whereas the *baseline's* best variant is
`dpcc-c-tightened`. Comparing our `dpcc-t-t` against the baseline's `dpcc-c-t` is legitimate
(each method gets its best projector) but **must be stated explicitly in any table**, or it reads as
cherry-picking. AlphaFlow-SiT is entirely dependent on `dpcc-t-tightened`: its `dpcc-c-tightened`
arm is 0.85 at K1 and **collapses to 0.16 at K2**.

## 9.7 Revised bottom line

| claim | status after the K-ladder check |
|---|---|
| MF-UNet **K1** beats the baseline | ✅ **Unchanged and strengthened.** Beats DPCC K20 by 36–43× and DPCC K10 by 17–23×, at n=20, on a matched UNet. Plus: DPCC at K1 collapses (0.50–0.60 S&C) and MeanFlow does not. |
| MF-UNet **K2** beats the baseline | ✅ Holds (24–28× vs K20, 11–14× vs K10). |
| MF-UNet **K5** beats the baseline | ◐ Holds on `dpcc-t-tightened` only; **fails** on `dpcc-c-tightened` (TR 0.76). Threatened by the 1-seed FM K5 point. |
| MF-UNet **K10/K20** beat the baseline | ❌ Drop. K10 only ties DPCC K10; K20 loses and is incomplete. |
| AF-SiT **K1** beats the baseline | ✅ On `dpcc-t-tightened`, but backbone-confounded (SiT ≠ UNet) — keep as secondary per `architecture-matched-beat-is-the-strong-claim`. |
| **FMv3ODE K20** beats the baseline | ◐ **Only against the pinned K20 rung.** Loses to DPCC K10 by 0.64–0.85×. Report as "naive FM improves the projector's efficiency at matched K", **not** as "naive FM beats DPCC". |

## 9.8 Added to the run queue (supersedes §7 priorities 1–2)

1. **`FMv3ODE K1, K2, K5 @ T0.5, 5 seeds × 20 trials`** — now the **top** priority, above the K20
   rerun. It simultaneously (a) settles whether naive FM beats MeanFlow in the mid-K band (§9.5),
   (b) supplies the missing K-matched control for the MeanFlow objective at K=1 (§9.5), and (c) is
   cheap to run. `FMV3_FLOW_STEPS="1 2 5"`, `FMPCC_RUN_MSG=20trials`.
2. **`DPCC K10 @ aw10, T0.5, 5 seeds × 20 trials`** — the cross-K comparator currently only exists at
   n=2, so every §9.3 ratio against it inherits the n=2 caveat. Cheaper than the K20 rerun (~half
   the sampling cost) and it is the number a reviewer will demand.
3. `FMv3ODE K20 @ T0.5, n=20` — demoted; it now only confirms a comparison that loses on cross-K.
4. `FMv3ODE K20/K10 @ T ∈ {0.1, 0.05}`, 5 seeds × 20 trials — §5 lead, unchanged.

## 9.9 Caveats specific to this addendum

- **DPCC K1 (C8) and DPCC K10 (C7) carry the same `post_processing` ↔ `dpcc-r` aliasing bug**
  (12–16 of 14–18 metrics exactly equal on every halfspace). Their `post_processing*` and `dpcc-r*`
  rows are unusable; §9 uses only `dpcc-c*` / `dpcc-t*`, which are unaffected.
- **DPCC K10 and K1 exist only at `n_trials = 2`.** Cost ratios against them are reliable (cost
  metrics are n-stable to ~4 %); their S&C 1.00 values carry the same 0.10-resolution caveat as
  every other n=2 number.
- **The FMv3ODE K5 row is 1 seed × 2 trials = 2 episodes.** It is an indication, not a result, and
  §9.5 treats it as such.
- MF-UNet K20 at n=20 (C140) has only 2 usable seeds on `top-left-hard` — the 24 h wall again.
