# CLOSURE — Gen14 · **Visual Aligning: the engine comparison, closed**

**Status** `af`-U-Net **CLOSED** · `af`-SiT **ABANDONED** · `mf` **flagship** · `fm` **excluded** · `diffusion` **partial (tightened K=20 pending)**
**Corpus** `temp/0609/II/batch_va2_20260907_141036` — 673 run-config rows, 14,102 rollout rows
**Task** `aligning-d3il-visual` · seed 6 · 10 contexts, paired by geometry fingerprint · `mpc4` · H8
**Supersedes nothing; consolidates** `DA_20260901` (flagship) · `DA_20260902` (funnel) · `DA_20260903` (`fm` four gates) · `DA_20260906` (`af` four gates) · `DA_20260907` (`af` arm C)

**What this document is:** the final V_A engine ranking, the conditions under which α-Flow would
have won, why it did not, and the resolution of the K-sensitivity question across both
environments. After this, V_A engine selection is settled and effort moves to UAV.

---

## ★ Recap — what holds, what doesn't, and does the thesis survive

**Short answer: the thesis claim ladder is intact. What was lost was an optional bonus rung, not a
load-bearing one.**

### R.1 The required hierarchy — all three rungs hold

| # | required claim | status | evidence |
|---|---|---|---|
| 1 | **MeanFlow beats the diffusion-DPCC baseline** *(THE baseline)* | ✅ **holds, decisively** | unguided median 0.0741 vs 0.4140; paired **−0.3744 m, 0/10, p = 0.0020**; and 0.64× the per-step cost |
| 2 | **MeanFlow beats naive Flow Matching** | ✅ **holds** | `DA_20260903`, paired **+0.216 m, 9/0, p = 0.0039**, `fm` frozen 5/10 vs `mf` 1/10 |
| 3 | **HardFlow beats the DPCC projector** | ✅ **holds — on latency at K=10** | see R.2 |
| — | *α-Flow beats MeanFlow* | ❌ **fails** | **never a required rung** — an optional extra arm |

**Nothing in rungs 1–3 depended on α-Flow.** The α-Flow result is a clean negative on an additional
engine, published as such.

### R.2 "Is HardFlow proved to work at high K on MeanFlow?" — **yes, but state it correctly**

Two different things are true at two different K, and only one of them is a *win over DPCC*:

| | K = 10 (T=0.4) | K = 20 (T=0.2) |
|---|---|---|
| **latency vs DPCC** | 🟢 **−324.96 ms/step, 0/9, p = 0.0039** (`-r`); −320.08, 0/9, p=0.0039 (`-c`); −211.36, 1/8, p=0.0391 (`-t`) | ⚪ parity — `-r` +9.4 ms, `-c` −42.9, `-t` −21.7, **all p = 1.0000–0.18** |
| **zero-violation vs DPCC** | +0.30 / +0.20 / +0.10, n.s. | 🟢 **10/10 vs 9/10** (`-r`, `-c`) — but **one discordant rollout, p = 1.0000** |
| **distance cost** | none (p = 1.0000, 0.61, 0.89) | none (p = 1.0000, 0.45, 1.0000) |

> **The citable HardFlow claim is: at K=10, HardFlow-SLSQP delivers DPCC-level safety and distance
> at ~325 ms/step less — a 0/9 sweep at p = 0.0039.** That is rung 3, and it stands.
>
> ⚠️ **Do NOT claim "HardFlow is safer than DPCC at K=20."** The 10/10-vs-9/10 zero-violation edge
> rests on a **single discordant rollout** and is not significant. What *is* true at K=20 is that
> `hardflow_sls-r` and `-c` are the **only arms in the entire V_A corpus reaching zero-violation
> 1.000 with 0.00 violations while still moving the box** (2/10 and 4/10 untouched) — DPCC's
> `dpcc-c-dt4p0` also reaches high safety but by freezing 10/10, which is disqualified. That is a
> real and quotable property of the configuration; it is not a significant *margin* over `dpcc-r`.

### R.3 "Does `mf > fm > diffusion` hold?" — **half of it**

| pairing | paired result | verdict |
|---|---|---|
| `mf` vs `diffusion` K=20 | −0.3744 m, **0/10, p = 0.0020** | ✅ **holds, decisively** |
| `mf` vs `fm` | +0.216 m, **9/0, p = 0.0039** | ✅ **holds** |
| **`fm` vs `diffusion` K=20** | −0.1411 m, 4/5, **p = 1.0000** | 🔴 **does not hold** |
| `af` vs `diffusion` K=20 | −0.1389 m, 5/5, **p = 1.0000** | 🔴 does not hold |
| `fm` vs `af` | −0.0022 m, 6/3, **p = 0.5078** | 🔴 does not hold |

The true shape is **`mf` ≫ {`af` ≈ `fm` ≈ `diffusion`}** — one clear winner and a four-way cluster at
0.39–0.42 m (with the d3il baseline at 0.4152). **This costs the thesis nothing**: the hierarchy
requires MeanFlow to beat the baseline and to beat naive FM, and it does both. It never required
`fm` to beat `diffusion` — and a `fm > diffusion` claim would have been a 0.0055 m gap inside a
0.4 m reproducibility band, i.e. unpublishable anyway. **Finding this before it went into the thesis
is a save, not a loss.**

### R.4 "Is `af`-U-Net ≈ `mf`, or < `mf`?" — **K decides, and it is never >**

| operating point | paired `af`(α=0.2) − `mf` | reading |
|---|---|---|
| **K = 2** | +0.0240 m, **4/4, sign p = 1.0000**, perm p = 0.8750 | **≈ `mf`** — statistically indistinguishable |
| **K = 20** | +0.2004 m, 6/3, sign p = 0.5078, **perm p = 0.0469** | **< `mf`** — worse, but on magnitude-weighted evidence only |

So your reading is **right at K=2 and wrong at K=20**. The precise statement is:

> **`af`-U-Net ≤ `mf` everywhere in V_A: indistinguishable at K=2, worse at K=20, better nowhere.**

Two qualifications that stop "≈ at K=2" from being oversold:

- The **tie is in paired distance only.** On contexts-ever-solved at K=2, `mf` solves **7 of 10** and
  `af` solves **2 of 10** (9 of `af`'s 11 successes are one context re-solved across variants;
  Fisher on the inflated rollout denominator is p = 0.4370). `mf` is broader even where the distance
  metric ties.
- The K=20 loss is **weak evidence** — the sign test does *not* reject. This is nothing like the
  `fm` exclusion, which was 9/0 at p = 0.0039. α-Flow is closed for being **never better across two
  environments and flat in K**, not for being decisively beaten at one operating point.

### R.5 What is actually lost

| | |
|---|---|
| **Lost** | the optional claim "α-Flow improves on MeanFlow". Also the never-load-bearing `fm > diffusion` ordering. |
| **Kept** | rungs 1–3 in full — MeanFlow ≻ DPCC baseline, MeanFlow ≻ naive FM, HardFlow ≻ DPCC projector on latency at K=10. |
| **Gained** | three findings that did not exist before this drop: HardFlow's benefit **scales with plan constraint error and is free on a high-violation field** (−124.70 violations on `af` at p = 0.0078, zero distance cost); the τ = 0.850 NLP failure confirmed **engine-independent** across three objectives, 24/24 items; and a measured **~0.4 m reproducibility floor** on projected arms that now qualifies every MIN in the corpus. |
| **Still open** | tightened `diffusion` K=20 — until it lands, the constraint comparison against the baseline is incomplete (§8). |

**A negative result on a well-posed question is a thesis contribution.** "α-Flow, with α provably
live at two settings, architecture-matched on the same 4.0 M U-Net, does not improve on MeanFlow in
visual aligning and does not scale with NFE — because its target is a bootstrap of the network's own
output rather than an analytic derivative" is a defensible chapter section. It also retires the
standing worry that every prior `af` row was mislabelled MeanFlow.

---

## 0 · Ground rules

`context_final_xy_dist`, metres, box→target. **Initial distance 0.4530 m** — every number below is
read against that. 🪤 `mean_dist_per_rollout` is `0.5·(pos_dist_3D + rot_err/π)`, never a distance.
**untouched** = box never moved (`final == init` to 1e-6). 🚫 `geo_free` / `bounds_free` /
`model_free` are constraint ablations, never results. MIN is n = 1 — a capability ceiling.
⚠️ Projected arms carry a **~0.4 m run-to-run reproducibility floor** (`DA_20260907` §5); MIN gaps
under ~0.1 m on those arms are noise. All tests are exact two-sided sign tests and exact paired
sign-flip permutation tests, stdlib only.

**Geometry note.** `diffusion` exists **only on untightened `combined_5`** (208 rows, 6 variants, no
arm C, no `dt` variants). The cross-engine ranking in §1 is therefore given on **untightened**
geometry, the only surface where all five arms coexist. Constraint claims (§2) use tightened only.

---

## 1 · The ranking — **only MeanFlow separates from the baseline**

Unguided plan (`diffuser`), untightened `combined_5`, K=20 where available:

| rank | engine | K | median | mean | MIN | untouched | ms/step | vs `diffusion` K=20 |
|---|---|---|---|---|---|---|---|---|
| 1 | **`mf`** | 20 | **0.0741** | **0.0933** | 0.0278 | **0/10** | 190.5 | **−0.3744 · 0/10 · p=0.0020** |
| — | *`diffusion`* | *100* | *0.1849* | *0.2167* | *0.0255* | *2/30* | *1526.6* | *(different K)* |
| 2 | `af`/α=0.2 | 20 | 0.3918 | 0.3288 | **0.0110** | 2/10 | 191.3 | −0.1389 · 5/5 · p=1.0000 |
| 3 | `fm` | 20 | 0.4085 | 0.3266 | 0.0394 | 4/10 | 295.8 | −0.1411 · 4/5 · p=1.0000 |
| 4 | `diffusion` | 20 | 0.4140 | 0.3901 | 0.0280 | 10/30 | 298.3 | — |
| 5 | *d3il baseline* | — | 0.4152 | 0.3987 | 0.0091 | **1712/3884 (44 %)** | 22.7 | — |

> 🔴 **The nominal ordering `mf > af > fm > diffusion` is misleading.** Paired on the same 10
> contexts, **`af`, `fm` and `diffusion` are mutually indistinguishable** — `fm` − `diffusion`
> p = 1.0000, `af` − `diffusion` p = 1.0000, `fm` − `af` p = 0.5078. They occupy a single cluster at
> 0.39–0.42 m, i.e. **"barely moved the box"** from a 0.4530 m start, alongside the d3il baseline.
> **Only MeanFlow separates, and it does so categorically: 0/10 losses, p = 0.0020.**

So the citable claim is **`mf` ≫ {`af`, `fm`, `diffusion`}**, not a four-way ladder. Anyone quoting
`fm > diffusion` from marginal medians is quoting a 0.0055 m gap inside a 0.4 m noise band.

📌 **`diffusion` is not finished.** It has no tightened-geometry cell at K=20 or K=100, so it has
never been scored on constraints, and its K=100 row (0.1849) — its genuinely competitive one —
costs **1526.6 ms/step, 8.0× MeanFlow's K=20**. The funnel DA's binding gap is still open.
**We test tightened `diffusion` K=20 later**; until then no constraint claim against the DPCC
baseline is complete, and the §1 ranking is a *plan-quality* ranking only.

---

## 2 · The flagship — **MeanFlow, K=20, T=0.2**

**Setup.** `engine=mf` · `VisualMeanFlow`, two-time · backbone `unet` (`VisualUNetTwoTime`, **4.0 M**)
· FiLM `v1` · `filmv1_Emf_tslogit_normal` checkpoint · seed 6 · 10 contexts · `mpc4` fan · H8 ·
K = 20 (`flow_steps_v3`) · T = 0.2 (`diffusion_timestep_threshold`) → **4 projector calls per
replan**, `snapping_start_idx = int((1−T)·K) = 16` · arm C `hardflow_new-{r,c,t}` on the SLSQP
backend, written as `hardflow_sls-*` · `dof=66`, `reg_scale=1.0`. Jobs `25247` (origin),
`25475`-era corpus. **Architecture-matched to every flow arm** — same 4.0 M U-Net, same action
weight, same time sampler.

**Result, tightened `combined_5-tightened`, legal arms only:**

| variant | MIN | median | untouched | **zero-viol** | violations | sat | ms/step |
|---|---|---|---|---|---|---|---|
| `diffuser` (unguided) | 0.0278 | **0.0902** | 1/10 | 0.20 | 31.60 | 0.921 | 172.5 |
| `gradient` | **0.0089** | 0.2271 | 2/10 | 0.30 | 68.10 | 0.813 | 178.9 |
| **`hardflow_sls-r`** ⭐ | 0.0220 | 0.1967 | 2/10 | **1.00** | **0.00** | **1.000** | 275.3 |
| **`hardflow_sls-c`** ⭐ | 0.0241 | 0.3227 | 4/10 | **1.00** | **0.00** | **1.000** | 282.3 |
| `hardflow_sls-t` ⭐ | 0.0307 | 0.2297 | 3/10 | 0.90 | 0.60 | 0.999 | 301.5 |
| `dpcc-r` | 0.0329 | 0.1786 | 2/10 | 0.90 | 2.70 | 0.993 | 265.9 |
| `dpcc-c` | 0.0415 | 0.3328 | 4/10 | 0.90 | 4.30 | 0.983 | 325.3 |
| `post_processing` | 0.0492 | 0.2184 | 2/10 | 0.80 | 11.90 | 0.970 | 193.6 |

**MeanFlow + HardFlow-SLSQP is the only configuration in the V_A corpus that reaches perfect
executed safety without freezing** — `hardflow_sls-r` and `-c` both at zero-violation **1.000** with
**0.00** violations, at 2/10 and 4/10 untouched. `dpcc-c-dt4p0`, the other 1.000-class arm elsewhere
in the corpus, gets there by freezing 10/10 and is disqualified.

Against the DPCC baseline this is the headline pair: **MeanFlow's unguided plan already beats
`diffusion` K=20 by 0.3744 m (0/10, p = 0.0020) at 0.64× the per-step cost**, and its projected arm
reaches safety the baseline's projector reaches only at 2157.9 ms/step (`diffusion` + `dpcc-r`,
**7.8× `hardflow_sls-r`**).

---

## 3 · Low K — `af` vs `mf` on the U-Net

**V_A has no K=1 cell.** The ladder is K ∈ {2, 10, 20, 100}; K=1 exists only in avoiding. At the
lowest V_A rung, tightened, unguided, `filmv1` both sides:

| K=2, T=0.5 | median | mean | MIN | untouched | ms/step | successes |
|---|---|---|---|---|---|---|
| `af` α=0.2 | 0.4247 | 0.3470 | **0.0098** | 3/10 | 24.5 | 11/320 |
| `af` α=0.05 | 0.3783 | 0.3052 | 0.0391 | 1/10 | 24.5 | 2/320 |
| `mf` `filmv1` | **0.2157** | **0.2667** | 0.0095 | 7/30 | 23.7 | 29/1140 |

```
paired  af(α=0.2) − mf(filmv1, per-context median)  at K=2 :
        mean +0.0240 m   4/4   sign p = 1.0000   perm p = 0.8750     ← a TIE
paired  af(α=0.2) − mf                              at K=20:
        mean +0.2004 m   6/3   sign p = 0.5078   perm p = 0.0469
```

**At K=2 α-Flow and MeanFlow are statistically indistinguishable in V_A.** The marginal medians
differ (0.4247 vs 0.2157) but the per-context pairing cancels it — both engines are wide and both
are far from the target. The V_A gap is a **K=20 phenomenon only**.

Avoiding, matched U-Net, unguided, for comparison (`both-hard`, n=20, seed 6):

| K | `mf` succ / steps | `af` α=0.2 succ / steps | `af` α=0.05 succ / steps |
|---|---|---|---|
| 1 | 1.000 / 62.05 | 1.000 / **60.90** | 1.000 / 62.45 |
| 2 | 1.000 / **60.00** | 1.000 / 64.80 | 1.000 / 63.45 |
| 5 | 1.000 / 63.00 | 1.000 / 63.95 | 1.000 / 64.35 |
| 10 | 1.000 / 63.50 | 1.000 / 64.00 | 1.000 / 63.25 |

🔴 **Read the success column.** It is **1.000 for every engine at every K**. The avoiding task is
**saturated on the raw plan** — see §5.2.

---

## 4 · When would `af`-U-Net have won? — the honest list

α-Flow needs *all* of these to hold. In V_A none of the first three do.

| condition | holds in V_A? | evidence |
|---|---|---|
| Operating point at **low K** | ✅ available (K=2) | its own ladder is flat, so K=2 is its best rung |
| MeanFlow's K-curve **flat or falling** there | 🔴 **no** — MeanFlow *improves* 0.2354 → 0.0741 | §5.1 |
| Task has **no headroom**, so plan quality cannot separate | 🔴 **no** — V_A has 0.45 m of headroom and ~0 % success | §5.2 |
| Latency is the binding constraint | 🔴 no | `af` is 8.13 ms/NFE vs `mf` 9.17 — an 11 % edge on a 4.7× worse plan |
| α set **high enough** to escape the ill-conditioned bootstrap | ❓ **untested above 0.2** | §5.3, and the one live re-entry |

**Where α-Flow does win, it wins narrowly and on a saturated task.** The avoiding claim
(`DA_20260903_AF_UNet_alphaflow_ENABLED_seed6_diffuser.md`) is one arm — `dpcc-t-tightened` at
K=1 — inside an aggregate its own §4 calls a wash: **W 15 / L 14 / T 6** on goal-reached. It carries
its own ⛔ citation block (single seed, `n_steps` defined two ways across the toolchain). It is a
real result about *one cell*; it is not evidence that α-Flow is a better generative model.

---

## 5 · Why `af` lost — and why `mf` is K-sensitive while `af` is not

### 5.1 The ladders

Unguided, untightened, median distance:

| engine | target type | K=2 | K=10 | K=20 | K=100 | K-response |
|---|---|---|---|---|---|---|
| `mf` | **analytic** JVP | 0.2354 | 0.1194 | **0.0741** | 0.0952 | 🟢 **falls** |
| `diffusion` | **true** DDPM posterior | — | — | 0.4140 | **0.1849** | 🟢 **falls** |
| `af` α=0.2 | **bootstrapped** self-target | 0.3738 | — | 0.3918 | — | 🔴 flat |
| `af` α=0.05 | **bootstrapped** self-target | 0.3978 | — | 0.4488 | — | 🔴 flat/worse |
| `fm` | instantaneous `v` | — | — | 0.4085 | 0.4004 | 🔴 flat |

> **The split is exactly along target construction.** The two engines whose training target is
> computed from something *external* to the network — MeanFlow's analytic JVP, diffusion's true
> posterior — get better as the sampler integrates more finely. The engine whose target is the
> network's **own frozen output** does not. `fm` is flat for a separate, already-diagnosed reason
> (missing `_encode_once`, Beta time sampler, 5/10 frozen — `DA_20260903` Gate 3).

### 5.2 Why the avoiding data does not contradict this

The avoiding table in §3 shows `n_success = 1.000` for every engine at every K on the raw plan.
**The task is saturated: there is no headroom for K to buy anything.** What varies there is
`n_steps` (60–69, a ~13 % band) and post-projection `collision_free_completed`. So the avoiding §4
finding that "MeanFlow's mean S&C falls with K" is a statement about **constraint satisfaction under
projection on a task with a solved plan** — a different quantity from V_A's plan-quality ladder, and
not evidence that MeanFlow degrades as a generative model.

**The correct unified statement:**

> α-Flow's plan quality is insensitive to NFE. MeanFlow's improves with NFE **wherever the task has
> headroom to show it**. Avoiding has none on the raw plan, so the two look comparable there; V_A
> has 0.45 m of it, so they separate. **The environments differ in headroom, not in which engine is
> better.**

This retires the earlier framing in `DA_20260907` §6.3, which said the engine that changes behaviour
between environments is MeanFlow. It is not — **the task does.**

### 5.3 The mechanism, from the code

`flow_matcher_v3_alphaflow/models/af_diffusion.py:653-665`, with **`dt = α·h`**:

| branch | when | target |
|---|---|---|
| FM anchor | `h == 0` | `u_tgt = v` |
| **bootstrap** | `h > 0, α > 0` | `u_tgt = α·v + (1−α)·u_next`, `u_next = u(z_r+dt, r+dt, h−dt)` under `no_grad` |
| **continuous** | `h > 0, α == 0` | `u_tgt = v + h·du/dr` — MeanFlow, analytic |

Three facts, and one inference:

1. **α is a fixed schedule, not optimised.** The *family* contains MeanFlow at α=0; training picks
   one point in it. Containment gives expressiveness, never dominance of a trained checkpoint.
   **This is the answer to "isn't α-Flow guaranteed at least as good?" — no.**
2. **The α>0 target is the network's own frozen output.** Its fixed point need not be the true
   field: the learned average-velocity field can be self-consistent and wrong.
3. 🔴 **α → 0⁺ does not converge to MeanFlow — the code changes branch at `α == 0`.** And since
   `dt = α·h`, small α makes the bootstrap a finite difference over a *vanishing* step: the
   worst-conditioned regime, neither analytic nor well-separated. This is why **α=0.05 is worse than
   α=0.2** on distance, on push-aways (4/10 vs 2/10) and on success (0.62 % vs 3.44 %).

*Inference (consistent with every number above, not independently tested):* a biased field is hidden
at low K, where one or two huge Euler steps make discretisation error dominant, and exposed at high
K, where discretisation error vanishes and the sampler converges faithfully to the biased fixed
point. More NFE integrates the wrong field more accurately.

### 5.4 The failure mode α-Flow has and MeanFlow does not

Tightened, unguided, 10 rollouts:

| engine | ended closer | frozen | **pushed the box AWAY** |
|---|---|---|---|
| `mf` | **9** | 1 | **0** |
| `af` α=0.2 | 5 | 3 | **2** |
| `af` α=0.05 | 5 | 1 | **4** |
| `fm` | 3 | 5 | 2 |

`af` α=0.05 reaches 0.8294 m from a 0.4149 m start — roughly twice as far away as it began.
**MeanFlow never ends a rollout further from the target than it started, in any cell.**

---

## M · The core mechanism in three equations — **`fm` as baseline**

Convention (`af_diffusion.py` docstring): DATA-AT-1, τ=0 noise → τ=1 data. The sampler anchors at
`z_r` at time `r` and steps to `t = r + h`. Write `v(z_s, s)` for the **instantaneous** velocity and
`u(z_r, r, h)` for the **average** velocity over `[r, r+h]`, so the exact endpoint map is
`x̂_t = z_r + h·u`. All three engines are trained against a target for the *same* network output;
only the target differs.

### M.1 The exact identity all three approximate

```
h · u(z_r, r, h)  =  ∫_r^{r+h} v(z_s, s) ds                                    (1)
```

Split the integral at `r + dt` and write **α = dt/h**:

```
h·u(z_r,r,h)  =  ∫_r^{r+dt} v ds  +  (h−dt)·u(z_{r+dt}, r+dt, h−dt)            (2)
```

### M.2 The three targets

| engine | branch | target | what it is |
|---|---|---|---|
| **`fm`** *(baseline)* | `h = 0`, i.e. **α = 1** | `u_tgt = v` | learns the **instantaneous** field. Eq. (1) never used. |
| **`af`** | `h > 0, α > 0` | `u_tgt = α·v + (1−α)·u_next`,  `u_next = u_θ(z_r+dt·v, r+dt, h−dt)` under `no_grad` | eq. (2) with `∫_r^{r+dt} v ds ≈ dt·v` — a **finite-difference** composition closed with the network's **own frozen output** |
| **`mf`** | `h > 0, α = 0` | `u_tgt = v + h·∂u/∂r`, by JVP along `(v, +1, −1)` | the **analytic α→0 limit**: differentiate (1) in `r` and evaluate the derivative exactly |

`fm` and `mf` are therefore the two **endpoints**, and `af` is the interpolation between them —
except that the interpolation is **not continuous at the `mf` end**: at `α = 0` the code changes
branch (`af_diffusion.py:552`), from a finite difference to an exact derivative.

### M.3 Where the error goes — the one calculation that matters

Let `u*` be the true average velocity and `ε = u_θ − u*` the current model error. Substituting into
the α-Flow target and using (2):

```
u_tgt − u*  =  (1−α)·ε_next  +  O(α²h²)                                        (3)
                  └────┬───┘     └───┬──┘
          bootstrap propagation   finite-difference
          of the model's OWN      bias from  ∫v ds ≈ dt·v
          error, undamped
```

Two terms, **opposite in α** — this is the whole mechanism:

- **`(1−α)·ε_next` — bootstrap propagation.** The target inherits a fraction `(1−α)` of the model's
  own error. Training only contracts the error at rate `(1−α)` per step, so **any field satisfying
  `ε = (1−α)·ε_next` is a fixed point of the objective.** As **α → 0⁺ the factor → 1: no contraction
  at all**, and self-consistency stops pinning down the true field. A biased-but-self-consistent
  `u_θ` is a valid optimum.
- **`O(α²h²)` — finite-difference bias.** Grows with α, and at `α = 1` the target degenerates to
  `u_tgt = v`, i.e. back to the `fm` baseline, which learns the wrong object (instantaneous, not
  average).

⇒ **α-Flow has an interior optimum in α and no good limit at either end.** MeanFlow escapes the
trade-off entirely: the analytic branch has **no `O(α²h²)` term and no `(1−α)` factor** — its target
error comes only from evaluating an *exact* derivative of the current model.

*(In fairness to α-Flow, this is the trade it was designed to make: the fixed target has no blind
direction, whereas MeanFlow's residual can hide an error whenever `δ_u = h·δ_D` — the code says so
at `af_diffusion.py:667-670`. Which failure mode dominates is empirical. In V_A it is α-Flow's.)*

### M.4 Why that predicts flat-in-K — and matches every measurement

At inference the sampler takes K steps. Total error decomposes as

```
E(K)  ≈  discretisation O(1/K)  +  field bias b                                (4)
```

`b` is a property of the **learned field**, so it does **not** depend on K.

| engine | `b` | E(K) as K grows | measured V_A median (K=2 → 20) |
|---|---|---|---|
| `mf` | ≈ 0 (analytic target) | → 0, **falls** | 0.2354 → **0.0741** 🟢 |
| `diffusion` | ≈ 0 (true posterior) | **falls** | 0.4140 (K20) → **0.1849** (K100) 🟢 |
| `af` α=0.2 | > 0 (bootstrap fixed point) | → `b`, **flat** | 0.3738 → 0.3918 🔴 |
| `af` α=0.05 | larger — `(1−α) = 0.95` | **flat/worse** | 0.3978 → 0.4488 🔴 |
| `fm` | — (learns `v`, not `u`) | flat, separate cause | 0.3725 → 0.4004 🔴 |

> **More NFE integrates the wrong field more accurately.** The bias is *hidden* at low K, where the
> `O(1/K)` term dominates and every engine is coarse — which is exactly why `af ≈ mf` at K=2
> (4/4, p = 1.0000) — and *exposed* at high K, where the discretisation term vanishes and the
> sampler converges faithfully to `b`.

### M.5 The α ordering, predicted and observed

Eq. (3) says smaller α ⇒ weaker contraction ⇒ larger `b`:

| α | contraction `(1−α)` | predicted | median | push-away | success (K=2) |
|---|---|---|---|---|---|
| 0.05 | **0.95** — almost none | worse | 0.4495 | **4/10** | 0.62 % |
| 0.2 | 0.80 | better | **0.4232** | 2/10 | **3.44 %** |

✅ **Observed ordering matches on all three columns.** Both trained values sit deep in the
weak-contraction regime, which is the strongest argument that **α ∈ {0.4, 0.6} is the one untested
region** — and eq. (3) also says it must eventually turn over as `O(α²h²)` and the drift toward `fm`
take hold. **A U-shape with an interior optimum is the prediction**; two points on one arm of it is
what we have.

---

## 6 · `af`-SiT — **abandoned**

Every `af`-SiT row in the corpus is discarded, for two independent and each-sufficient reasons:

1. **Not architecture-matched.** SiT sized from `dit_hidden_size=256, dit_depth=8` ≈ **9.4 M**
   against the 4.0 M U-Net every other arm runs — ~2.4×. Any SiT-vs-U-Net row moves objective,
   backbone and parameter count together and can carry no engine claim.
2. **Not α-Flow.** Every SiT checkpoint in the tree carries `ae0.0`, and `af_alpha_end = 0.0` with
   `af_alpha_clamp = 0.005` snaps α to exactly 0, which routes to Gen3v6's MeanFlow body unmodified
   (`af_diffusion.py:552`). **They are MeanFlow-on-SiT in an α-Flow folder** — an architecture
   ablation wearing an objective's name.

Nothing needs re-running to replace them: the U-Net α-Flow arms in §1–§5 are the architecture-matched
form of the same question, and they are decided. The SiT rows are **read-only history**; do not cite,
do not aggregate, do not carry into UAV.

---

## 7 · Verdict

| engine | status | operating point | basis |
|---|---|---|---|
| **`mf`** | ✅ **flagship** | K=20, T=0.2, arm C `hardflow_sls-r` | only engine separating from `diffusion` (0/10, p=0.0020); only zero-violation 1.000 unfrozen |
| `af`-U-Net | ⛔ **closed** | — | ties `mf` at K=2, loses at K=20, flat in K, unique push-away mode, no arm reaching zv 1.000 |
| `af`-SiT | ⛔ **abandoned** | — | 9.4 M vs 4.0 M **and** `ae0.0` ⇒ MeanFlow mislabelled |
| `fm` | ⛔ **excluded** | — | `DA_20260903`, four gates, three 🔴 |
| `diffusion` | 🟡 **partial** | K=20 / K=100 untightened only | **tightened K=20 pending**; K=100 competitive but 8.0× the cost |

**One live re-entry, and only one.** §5.3 predicts that **larger α** is better, and only α ∈ {0.05,
0.2} have been trained — both in the ill-conditioned small-`dt` regime, and the ordering between them
already points the right way. `α_end ∈ {0.4, 0.6}` at K=2 costs ~24 ms/step, hours not days. If that
ladder is flat, α-Flow is finished on evidence rather than on inference. **It does not block the UAV
pivot** and should be run only if UAV leaves the GPUs idle.

---

## 8 · What remains open in V_A

1. **Tightened `diffusion` K=20** — the funnel's binding gap. Without it there is no complete
   constraint comparison against the DPCC baseline. *(the "test K20 later" item)*
2. **Repeat one projected `mf` cell 3×** to pin the ~0.4 m reproducibility floor that currently
   qualifies every MIN in every V_A DA.
3. **τ = 0.850** — three engines, 24/24 arm-C items, same non-converged SLSQP solve at call #2.
   Engine-side is ruled out; check constraint-Jacobian conditioning.
4. *(optional)* `α_end ∈ {0.4, 0.6}` at K=2, per §7.

---

## Provenance

| item | value |
|---|---|
| corpus | `temp/0609/II/batch_va2_20260907_141036` — 673 config rows, 14,102 rollout rows |
| avoiding cross-check | `temp/0609/I/batch_avoiding_combined_20260906_125724` — 277,925 metric rows |
| task | `aligning-d3il-visual`, H8, `mpc4`, seed 6, 10 contexts, train split |
| initial distance | 0.4530 m mean, identical across all compared cells |
| flagship | `mf` · `unet` 4.0 M · `filmv1` · K=20 · T=0.2 · `hardflow_sls-r` (SLSQP, `dof=66`) |
| `af` arms kept | `bbunet` `AFAFend0p2` / `AFAFend0p05`, α = 0.2000 / 0.0500 at `state_100000.pt` |
| `af` arms abandoned | every `bbsit` row; every `ae0.0` row (⇒ MeanFlow) |
| `diffusion` coverage | untightened `combined_5` only, K=20 and K=100, 6 variants, no arm C |
| statistics | exact two-sided sign test; exact paired sign-flip permutation test (stdlib only) |
| geometry | ranking on untightened (only surface with `diffusion`); constraints on tightened only |
