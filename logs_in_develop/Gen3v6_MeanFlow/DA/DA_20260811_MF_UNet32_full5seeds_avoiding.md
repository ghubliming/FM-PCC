# DA — MeanFlow UNet@32 K-ladder on `avoiding-d3il`, full 5 seeds

**Date:** 2026-08-11 · **Type:** data analysis / batch report · **Status:** Target reached (time axis, significant); low-K claim confirmed
**Batch:** `temp/1108/Revised_2/batch_avoiding_combined_20260811_221322/` — all rows below are **within-batch**
**Runs:** eval **24416** (seeds 7–10, 4 h 17 m) · **24496** (seed 6, 1 h 02 m, git `ff4e3fb`) · train 24317 lineage (`backbone=unet freq_dim=32 params=4.0M`)
**Scope:** 5 seeds {6,7,8,9,10} × 3 halfspaces × `n_trials=2` = **30 episodes per arm**; K ∈ {1,2,5,10}; 13 variants
**Convention:** [[da-target-is-best-baseline-variant]] · [[pareto-definition-of-good]] · [[benchmark-hierarchy-who-beats-whom]]

---

## 0. TL;DR

**Target reached.** The Target is `DPCC K10 / dpcc-t-tightened` — **S&C 1.000, 68.70 steps, 0.3217 s/step**
(30 episodes). Our beating row is **MeanFlow UNet@32, K = 1, `hardflow_new-*-tightened`**:

| | S&C | steps | s/step | s/episode |
|---|---|---|---|---|
| **Target** — DPCC K10 `dpcc-t-tightened` | 1.000 | 68.70 | 0.3217 | 22.14 |
| **MF UNet@32 K1 `hardflow_new-tightened`** | **1.000** | **63.77** | **0.0417** | **2.64** |

**Target reached: MF UNet@32 K1 HardFlow-tightened beats the Target on `avg_time`** — 7.7× per step,
**8.4× per episode**, seed-cluster bootstrap `−19.50 s/ep, 95 % CI [−21.83, −17.14]`, at **equal
S&C (1.000, and 5/5 seeds perfect on both sides)**. Steps are also lower (−4.93) but the CI
straddles 0, so that axis is *not* claimed. Per the win rule this is a **win on one axis**, not a
Pareto domination.

**Three further results, all new at 5 seeds:**

1. 🔴 **The seed-6 "K2 dpcc-t-tightened = 1.000" does not survive.** At 5 seeds it is **0.967**
   (29/30) and therefore **fails the S&C gate** against the Target — despite being −9.27 steps
   `CI[−14.97, −3.67]` and 11.6× faster. One episode costs it the Target claim (§2.2).
2. ✅ **Low K is a capability, not a discount — confirmed monotonically.** MF `dpcc-t-tightened`
   S&C runs **0.967 (K1) → 0.967 (K2) → 0.933 (K5) → 0.933 (K10)** and steps **58.57 → 59.43 →
   60.60 → 63.63**. More NFE is *worse* on both axes (§3).
3. ✅ **UNet@32 beats `mf_dit` decisively at matched K = 2 (§4.1)** — **fewer steps in 13 of 13
   arms**, better S&C in 8 (tied 2, behind 3 by ≤ 0.167). The mechanism is one failure mode:
   `mf_dit` **times out on 12 of 15 `-c` cells** (S&C 0.100) where the UNet times out on **0 of 15**
   (S&C 0.933). On the matched `-t-tightened` arm it is −9.00 steps `CI[−16.03, −3.70]` at equal
   S&C. The one-seed 1/6-permutation ceiling from the Fix_8 guide is gone.

🔴 **But we are not the best config in the batch — `AlphaFlow bbsit K2` is** (§4.4). At equal
S&C 1.000 it runs **1.9× cheaper per episode** than our Target-beating row, with the CI excluding 0,
while our only lead (−3.83 steps) is not significant. The Target belongs to us; the frontier does
not. Note also that **backbone preference is family-dependent** — MeanFlow prefers the UNet,
AlphaFlow clearly prefers SiT (`af_sit` 1.000 vs `af_unet` 0.833) — so §4.1 must not be generalised.

**Per-environment (§2.3):** the Target beat **holds in all three halfspaces** — S&C 1.00 everywhere
on both sides, time win significant in each (8.3× / 10.0× / 6.4×). But the step advantage is a
`top-left-hard` effect alone (−11.0 there, −3.5 and **−0.3** elsewhere), and every S&C loss in this
DA sits on **`top-right-hard`** — including the §2.2 gate miss, which is the *same* (seed 7,
top-right) cell at both K = 1 and K = 2.

⚠️ The unprojected `diffuser` arm scores **S&C 0.000–0.167** at every K. All safety comes from
projection; the MeanFlow field alone is not a controller (§5).

---

## 1. The Target

Per [[da-target-is-best-baseline-variant]] the Target is the **best baseline diffusion-DPCC row in
this batch**, gated on S&C then Pareto-read on `(n_steps, avg_time)`. Top baseline rows, 5 seeds:

| baseline row | S&C | steps | s/step | s/ep |
|---|---|---|---|---|
| **`DPCC K10` / `dpcc-t-tightened`** ← **TARGET** | **1.000** | **68.70** | 0.3217 | 22.14 |
| `DPCC K10` / `dpcc-c-tightened` | 1.000 | 70.33 | **0.3098** | 21.66 |
| `DPCC K20` / `dpcc-c-tightened` | 1.000 | 70.13 | 0.5534 | 38.53 |
| `DPCC K20` / `dpcc-t-tightened` | 1.000 | 76.13 | 0.5630 | 42.54 |
| `DPCC K20 T0.5` / `dpcc-r-tightened` | 1.000 | 78.67 | 0.5973 | 46.70 |

**Target = `DPCC K10 / dpcc-t-tightened`** — fewest steps of every S&C = 1.000 baseline row, at a
time within 3.8 % of the cheapest. Its only non-dominated sibling is `DPCC K10 / dpcc-c-tightened`
(−3.7 % time, +1.6 steps); picking that one instead changes no verdict below, since our margin is
7–8×, not percent-level.

📌 **Note the baseline's own K:** the best diffusion-DPCC row is **K10, not K20**. Every K20 row is
dominated by its own K10 sibling. Framing anything as "we beat DPCC K20" understates the bar —
K10 is the real one.

---

## 2. Target check — which of our rows beat it

### 2.1 The beat: K = 1, HardFlow-tightened

| axis | Target | MF K1 HF-tightened | Δ (seed-cluster bootstrap, B = 20000) | verdict |
|---|---|---|---|---|
| **S&C** (gate) | 1.000 | **1.000** | +0.000 `[0, 0]` | ✅ gate cleared, exactly tied |
| **avg_time** | 0.3217 | **0.0417** | **−0.280 `[−0.293, −0.271]`** | ✅ **BEAT — 7.7×**, CI excludes 0 |
| **s/episode** | 22.14 | **2.64** | **−19.50 `[−21.83, −17.14]`** | ✅ **BEAT — 8.4×**, CI excludes 0 |
| n_steps | 68.70 | 63.77 | −4.93 `[−11.07, +1.07]` | 🟡 favourable, not significant |

Per-seed S&C is **1.000 on all five seeds** for both rows — the gate is not a rounding artifact.
Per-seed steps: ours `[64.8, 60.2, 66.0, 67.5, 60.3]` vs Target `[61.5, 62.2, 69.5, 72.2, 78.2]`;
we win 3/5 seeds, which is why the step CI straddles 0 despite the −4.93 mean.

> **Target reached: `MF UNet@32 K1 hardflow_new-tightened` beats the Target on `avg_time`
> (7.7× per step, 8.4× per episode) at equal S&C.** Steps favour us but are not claimable.
> This is a **one-axis win**, not a domination.

`-r` / `-c` / `-t` are numerically identical here — at `hardflow.batch_size: 1` all three selection
rules collapse to index 0, exactly as the yaml documents. Report them as one arm.

### 2.2 The near-miss: K = 1 and K = 2 DPCC-projected

| row | S&C | steps | Δsteps vs Target | Δtime | gate |
|---|---|---|---|---|---|
| MF K1 `dpcc-t-tightened` | **0.967** | 58.57 | **−10.13 `[−15.93, −4.53]`** | −0.303 `[−0.316, −0.294]` | ❌ 0.967 < 1.000 |
| MF K2 `dpcc-t-tightened` | **0.967** | 59.43 | **−9.27 `[−14.97, −3.67]`** | −0.294 `[−0.306, −0.285]` | ❌ 0.967 < 1.000 |

Both would be **strict Pareto dominations of the Target** — significantly fewer steps *and* ~11.6×
less time — but **S&C is the only gate, and they miss it.** 0.967 = **29 of 30 episodes**; the
single failure is seed 7 (`S&C 0.83` on one of its three halfspaces). The ΔS&C CI is
`[−0.100, +0.000]`, i.e. not significantly worse than the Target — but "not significantly worse"
does not clear a gate defined on the point estimate.

🔴 **This is the correction to the earlier seed-6-only reading.** On seed 6 alone `dpcc-t-tightened`
scored S&C 1.000 at both K1 and K2 and looked like a clean Target beat. Four more seeds turn it into
a near-miss. The Fix_8 guide's §3 headline
([`../Fix_8_Unet/GUIDE_bootstrap_UNet32K2_vs_FMv3K20_DPCCK20.md`](../Fix_8_Unet/GUIDE_bootstrap_UNet32K2_vs_FMv3K20_DPCCK20.md))
should be read as superseded on that point — its own §5 said exactly this would happen.

### 2.3 Per-environment breakdown — does the Target beat hold in all three?

The pooled figure is the headline (paper convention), but the three halfspaces are genuinely
different problems and must be inspected separately. Cells are **S&C / steps / s-per-episode**,
10 episodes each (5 seeds × 2 trials).

| row | `top-right-hard` | `top-left-hard` | `both-hard` |
|---|---|---|---|
| **Target** DPCC K10 `-t-tight` | 1.00 / 67.7 / 22.37 | 1.00 / 69.3 / 25.51 | 1.00 / 69.1 / 18.53 |
| **MF K1 HF-tightened** | **1.00** / 67.4 / **2.68** | **1.00** / **58.3** / **2.33** | **1.00** / 65.6 / **2.91** |
| MF K1 `dpcc-t-tightened` | **0.90** / 60.1 / 1.03 | 1.00 / 58.6 / 1.10 | 1.00 / 57.0 / 1.08 |
| MF K2 `dpcc-t-tightened` | **0.90** / 62.2 / 1.76 | 1.00 / 58.9 / 1.63 | 1.00 / 57.2 / 1.57 |

Per-env seed-cluster bootstrap, **MF K1 HF-tightened − Target**:

| env | ΔS&C | Δsteps | Δ s/episode |
|---|---|---|---|
| `top-right-hard` | +0.00 `[0, 0]` | −0.30 `[−7.50, +7.20]` | **−19.69 `[−22.81, −17.29]`** `*` |
| `top-left-hard` | +0.00 `[0, 0]` | **−11.00 `[−23.40, −2.50]`** `*` | **−23.18 `[−29.34, −17.69]`** `*` |
| `both-hard` | +0.00 `[0, 0]` | −3.50 `[−9.90, +3.00]` | **−15.62 `[−17.03, −13.60]`** `*` |

**Two things this changes:**

1. ✅ **The Target beat is robust — it holds in all three environments.** S&C is exactly 1.00 in
   every env on both sides, and the time win is significant in every env (8.3×, 10.0×, 6.4×).
   The pooled claim is not carried by one easy environment.
2. 🔴 **The step advantage is a `top-left-hard` effect only.** The pooled −4.93 steps decomposes
   into −11.0 (significant, top-left), −3.5 (ns, both-hard) and **−0.30 (a dead tie, top-right)**.
   This is the concrete reason §2.1 does not claim the step axis — per-env inspection shows *why*
   the pooled CI straddles 0, and the honest phrasing is *"faster everywhere, shorter paths only on
   `top-left-hard`"*.

**Failure localisation — the §2.2 gate miss is one cell, and it is reproducible:**

```
K1  seed 7  top-right-hard : S&C = 0.5   (1 of 2 trials)
K2  seed 7  top-right-hard : S&C = 0.5   (1 of 2 trials)
```

The *same seed in the same environment* fails at both K = 1 and K = 2, and nowhere else. So 0.967 is
not "one unlucky episode out of thirty" — it is **a specific (seed 7, top-right-hard) initial
condition this checkpoint family cannot solve under DPCC projection**, which the HardFlow arm *does*
solve (1.00 there). That is a far more actionable statement than a pooled rate, and it makes §7.3 a
single-episode debugging task rather than a statistics problem.

📌 **`top-right-hard` is the hard environment for the whole MeanFlow family**, consistent with the
history: it was the width-256 run's catastrophic cell (RESULTS §4). Every MF row that loses S&C in
this DA loses it there.

---

## 3. The K-ladder — low K is a capability, not a discount

MF UNet@32, `dpcc-t-tightened` (its best DPCC arm at every K), 30 episodes each:

| K | S&C | steps | s/step | s/episode |
|---|---|---|---|---|
| **1** | **0.967** | **58.57** | **0.0183** | **1.07** |
| **2** | **0.967** | 59.43 | 0.0277 | 1.65 |
| 5 | 0.933 | 60.60 | 0.2238 | 13.56 |
| 10 | 0.933 | 63.63 | 0.4023 | 25.59 |

**Monotone in the wrong direction for K.** Extra NFE buys nothing: S&C falls, steps rise, and time
rises ~22×. The same shape holds on `hardflow_new-tightened` (1.000 → 0.933 → 0.933 → 0.833).

**Per-environment** (S&C / steps / s-per-episode), same rows:

| K | `top-right-hard` | `top-left-hard` | `both-hard` |
|---|---|---|---|
| **1** | 0.90 / 60.1 / 1.03 | 1.00 / 58.6 / 1.10 | 1.00 / 57.0 / 1.08 |
| **2** | 0.90 / 62.2 / 1.76 | 1.00 / 58.9 / 1.63 | 1.00 / 57.2 / 1.57 |
| 5 | **0.80** / 65.7 / 14.78 | 1.00 / 59.6 / 13.79 | 1.00 / 56.5 / 12.09 |
| 10 | 0.90 / 68.6 / 28.43 | 1.00 / 62.4 / 24.76 | **0.90** / 59.9 / 23.59 |

**The K-monotonicity is real but env-localised.** `top-left-hard` and `both-hard` sit at S&C 1.00
for K ∈ {1, 2, 5} and only `both-hard` slips at K = 10; the entire pooled S&C decline comes from
`top-right-hard` (0.90 → 0.90 → **0.80** → 0.90). **The step trend, by contrast, is monotone in
every environment** (top-right 60.1→68.6, top-left 58.6→62.4, both-hard 57.0→59.9) — that is the
sturdier half of the low-K claim.

⚠️ The HardFlow ladder degrades differently and worse: `hardflow_new-tightened` on `both-hard` goes
**1.00 (K1) → 0.90 → 0.90 → 0.50 (K10)**. Its K = 1 dominance (§4.3) is not a mild edge; at K = 10
the arm half-fails on that env.

This is the ladder [`RESULTS_Fix_8…md`](../Fix_8_Unet/RESULTS_Fix_8_unet_width32_rerun_24317_24334.md) §8.7
asked for, and it strengthens the L3 leg of
[`DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md`](./DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md),
which previously rested on AlphaFlow alone: **a second K = 1–2 family now dominates its own K = 10.**

⚠️ Do not read the K5/K10 drop as "MeanFlow degrades with integration". These are the *same
checkpoint* sampled at different NFE; a two-time model trained for few-step generation has no
reason to improve with more Euler steps, and the projector sees a different (slower) trajectory
distribution. The claim is "K = 1–2 is sufficient", not "K = 10 is broken".

---

## 4. Hierarchy obligations ([[benchmark-hierarchy-who-beats-whom]])

### 4.1 MF must beat `mf_dit` — ✅ at matched K = 2

| row (K = 2) | S&C | steps | s/step | s/ep |
|---|---|---|---|---|
| **MF UNet@32 `dpcc-t-tightened`** | 0.967 | **59.43** | 0.0277 | **1.65** |
| MF `mf_dit` `dpcc-t-tightened` | 0.967 | 68.43 | 0.0253 | 1.73 |

Equal S&C, **−9.00 steps**, marginally slower per step (+9 %, within contention noise) but lower
per episode. Per-seed S&C shows the two fail on *different* seeds (ours seed 7, `mf_dit` seed 10).
**The UNet is the better MeanFlow backbone at matched K**, now on 5 real seeds — the Fix_8 guide's
§4 could only assert this against a 1/6 permutation floor with one UNet seed.

**Per-env — the win is almost entirely `top-right-hard`:**

| row (K = 2) | `top-right-hard` | `top-left-hard` | `both-hard` |
|---|---|---|---|
| **MF UNet@32 `-t-tight`** | 0.90 / **62.2** | 1.00 / **58.9** | 1.00 / **57.2** |
| `mf_dit` `-t-tight` | 0.90 / **83.7** | 1.00 / 61.0 | 1.00 / 60.6 |

−21.5 steps on `top-right-hard`, versus −2.1 and −3.4 elsewhere. Both backbones lose the same
0.10 of S&C on that env. So on the *matched headline arm* the reading is **"the UNet is far better
where the MeanFlow family is weak, and roughly equal where it is strong"**.

But `dpcc-t-tightened` is the arm that flatters the DiT most. Across the full arm set the picture is
one-sided.

#### 4.1.1 All 13 arms at K = 2 — the UNet has fewer steps in **13 / 13**

| arm | UNet@32 S&C / steps | `mf_dit` S&C / steps | ΔS&C | Δsteps |
|---|---|---|---|---|
| `diffuser` | 0.033 / 62.70 | 0.067 / 68.07 | −0.033 | −5.37 |
| `dpcc-r` | 0.467 / 63.03 | 0.633 / 79.50 | −0.167 | −16.47 |
| **`dpcc-c`** | **0.800 / 97.30** | **0.067 / 185.93** | **+0.733** | **−88.63** |
| `dpcc-t` | 0.467 / 58.77 | 0.433 / 68.37 | +0.033 | −9.60 |
| `dpcc-r-tightened` | 0.967 / 65.07 | 0.967 / 70.80 | +0.000 | −5.73 |
| **`dpcc-c-tightened`** | **0.933 / 97.20** | **0.100 / 186.13** | **+0.833** | **−88.93** |
| `dpcc-t-tightened` | 0.967 / 59.43 | 0.967 / 68.43 | +0.000 | −9.00 |
| `hardflow_new-r` | 0.567 / 59.37 | 0.533 / 75.87 | +0.033 | −16.50 |
| `hardflow_new-c` | 0.567 / 59.37 | 0.400 / 102.93 | +0.167 | −43.57 |
| `hardflow_new-t` | 0.567 / 59.37 | 0.467 / 75.90 | +0.100 | −16.53 |
| `hardflow_new-r-tightened` | 0.933 / 64.80 | 0.900 / 69.77 | +0.033 | −4.97 |
| `hardflow_new-c-tightened` | 0.933 / 64.80 | 0.767 / 96.93 | +0.167 | −32.13 |
| `hardflow_new-t-tightened` | 0.933 / 64.80 | 0.967 / 70.23 | −0.033 | −5.43 |

**Δsteps is negative in every single arm — 13 of 13.** On S&C the UNet is ahead in 8 arms, tied in
2, behind in 3 (`diffuser`, `dpcc-r`, `hardflow_new-t-tightened`), and every one of those three
losses is ≤ 0.167 while its two biggest wins are **+0.733 and +0.833**.

#### 4.1.2 The mechanism: `mf_dit` times out on `-c`, the UNet never does

Almost all of the aggregate difference is one failure mode. Counting cells whose `n_steps ≥ 198`
(i.e. `max_episode_length − 1`, a **timeout**), out of 15 seed×env cells:

| config | `dpcc-c` | `dpcc-c-tightened` |
|---|---|---|
| **`mf_dit`** | **12 / 15 timeouts** | **12 / 15 timeouts** |
| **UNet@32** | **0 / 15** | **0 / 15** |

Per-env, `dpcc-c-tightened` (S&C / steps):

| config | `top-right-hard` | `top-left-hard` | `both-hard` |
|---|---|---|---|
| **UNet@32** | 0.90 / 96.3 | 1.00 / 96.0 | 0.90 / 99.3 |
| `mf_dit` | **0.10 / 187.1** | **0.10 / 185.0** | **0.10 / 186.3** |

This is the U3 "crushed to a point" collapse
([`../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md)):
at K = 2 the DiT emits plans that barely move, `-c` locks onto them because a motionless plan is
trivially the cheapest to leave alone, and the agent never departs. **It reproduces on 4 of 5 seeds
and uniformly across all three environments** — a property of the backbone, not of a seed or a map.

**The UNet does not have this failure mode at all.** It is slow on `-c` (97.20 steps vs 59.43 on
`-t`, §7.4) but it always moves and it almost always arrives. That single difference is worth
+0.833 S&C and −89 steps, and it is why the *pooled* backbone verdict is decisive even though the
matched `-t-tightened` arm looks close.

#### 4.1.3 On the matched arm, the win is steps — and it is significant

Seed-cluster bootstrap, `dpcc-t-tightened`, **UNet − `mf_dit`**:

| axis | Δ | verdict |
|---|---|---|
| S&C | +0.000 `[−0.067, +0.067]` | tied at 0.967 |
| **n_steps** | **−9.00 `[−16.03, −3.70]`** `*` | **UNet wins** |
| avg_time | **+0.002 `[+0.002, +0.003]`** `*` | DiT wins, but +8 % on a 0.027 s base |
| s/episode | −0.078 `[−0.243, +0.062]` | tie |

Per-seed `S&C / steps` (seeds 6–10):

| config | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|
| **UNet@32** | 1.00 / **58.7** | **0.83** / **60.5** | 1.00 / 62.3 | 1.00 / **58.5** | 1.00 / **57.2** |
| `mf_dit` | 1.00 / 65.5 | 1.00 / 66.7 | 1.00 / **61.2** | 1.00 / 67.0 | **0.83** / 81.8 |

**UNet has fewer steps on 4 of 5 seeds.** The two backbones drop their single S&C failure on
*different* seeds (ours 7, the DiT's 10), which is why ΔS&C is a clean tie rather than either
winning. The DiT's per-step time edge is real but tiny and does not survive into `s/episode`,
where the UNet's shorter paths cancel it.

#### 4.1.4 What it costs — and the limits of this comparison

The UNet is also the **cheaper model**: 4.0 M parameters vs the DiT's ~10 M, and 8 h 07 m to train
vs ~11 h (RESULTS §1). So it wins on steps, ties on safety and per-episode time, avoids a
catastrophic failure mode, and costs less to train.

⚠️ **Limits.** `mf_dit` has 5 seeds only at **K = 2**, so this is a single-K comparison — there is
no DiT K-ladder to set against §3, and "the UNet is better" is established at K = 2 and nowhere
else. ⚠️ And per §4.4, this result is **MeanFlow-specific**: AlphaFlow prefers SiT decisively.

### 4.2 MF must beat naive FM — ✅ on time, tie on steps

| row | S&C | steps | s/step | s/ep |
|---|---|---|---|---|
| **MF UNet@32 K1 HF-tightened** | 1.000 | 63.77 | **0.0417** | **2.64** |
| FMv3 K20 `dpcc-c-tightened` (best naive-FM row) | 1.000 | **63.23** | 0.4767 | 29.65 |

Equal S&C, steps a statistical tie (+0.54), **time 11.4× better, wall 11.2× better.** Naive FM is
cleared on the compute axis, not on path length.

### 4.3 HardFlow must beat the DPCC projector — ✅ only at K = 1

Same checkpoint, `hardflow_new-tightened` vs `dpcc-t-tightened`:

| K | HF-tightened S&C / steps | DPCC-projected S&C / steps | verdict |
|---|---|---|---|
| **1** | **1.000** / 63.77 | 0.967 / 58.57 | ✅ HF wins the gate (+0.033), loses steps (+5.2) |
| 2 | 0.933 / 64.80 | **0.967** / 59.43 | ❌ HF loses both |
| 5 | 0.933 / 66.57 | 0.933 / **60.60** | ❌ tie on gate, HF loses steps |
| 10 | 0.833 / 66.37 | **0.933** / 63.63 | ❌ HF loses both |

**HardFlow's advantage exists only at K = 1** — and that is precisely where the Target beat lives.
The in-loop NLP buys the last 1/30 of safety exactly when there is almost no trajectory to correct;
by K = 2 post-hoc projection is already better.

🔴 **But measure what the arm is actually doing before calling it "in-loop".** NLP solves per
environment step, at the `A = 0.5` every run here used:

| K | solves / env step | fraction of integration steps projected |
|---|---|---|
| **1** | **1.02** | **1 of 1 — 100 %** |
| 2 | 1.02 | 1 of 2 — 50 % |
| 5 | 3.05 | 3 of 5 |
| 10 | 5.08 | 5 of 10 |

**At K = 1 the HardFlow arm is terminal-only projection.** There is one integration step, the final
solve always fires, so exactly one NLP is solved — the same count post-hoc projection would use.
The §2.1 Target beat is therefore *not* evidence that in-loop constrained sampling works; it is
evidence that **the cheapest possible configuration of arm C works**, and that its NLP (which
carries the flow-dynamics constraint) is a better single projection than DPCC's.

⚠️ **Consequence for the threshold sweep this obligation demands: it is a no-op at K = 1 and K = 2.**
`activation_threshold` cannot push below one solve, and at A = 0.5 both K values are already there.
Only K = 5 and K = 10 have intermediate solves (2 and 4) that a lower threshold can remove — those
are the only settings where the sweep can measure anything (§7.1).

### 4.4 vs AlphaFlow — the one family we do **not** beat

`af_sit` also has all 5 seeds in this batch, so it gets a full comparison rather than a footnote.

| row (5 seeds) | S&C | steps | s/step | s/ep |
|---|---|---|---|---|
| **AF `bbsit` K2 `dpcc-r-tightened`** | **1.000** | 67.60 | **0.0202** | **1.36** |
| **MF `bbunet` K1 `hardflow_new-tightened`** (our Target beat) | **1.000** | **63.77** | 0.0417 | 2.64 |
| MF `bbunet` K2 `dpcc-t-tightened` | 0.967 | 59.43 | 0.0277 | 1.65 |
| MF `bbmf_dit` K2 `dpcc-t-tightened` | 0.967 | 68.43 | 0.0253 | 1.73 |
| AF `bbunet` K2 `dpcc-t-tightened` | 0.833 | 61.77 | 0.0732 | 5.08 |

Head-to-head, **MF UNet K1 HF-tightened − AF sit K2 `-r-tightened`** (seed-cluster bootstrap):

| axis | Δ | verdict |
|---|---|---|
| S&C | +0.000 `[0, 0]` | tied at 1.000 |
| n_steps | −3.83 `[−12.23, +2.07]` | favours us, **not significant** |
| avg_time | **+0.021 `[+0.020, +0.022]`** `*` | **AF wins, 2.1×** |
| s/episode | **+1.28 `[+1.06, +1.47]`** `*` | **AF wins, 1.9×** |

🔴 **AlphaFlow-sit is the strongest row in this batch, and we do not beat it.** At equal, perfect
S&C our only lead (steps) is inside the noise, while its time advantage is significant on both
time axes. Applying the same win rule that granted us the Target in §2.1, **AF beats us on
`avg_time`** — the verdict has to run both ways. Per-env confirms it is not a pooling artifact:
AF sit is **1.00 / 1.00 / 1.00** across the three halfspaces, the same clean sweep our K1 arm has.

#### 4.4.1 How it beats us — it never has to pay for the expensive arm

This is *not* a case of AlphaFlow having a faster model. Per-arm cost, side by side:

| config | arm | S&C | s/step | s/ep | |
|---|---|---|---|---|---|
| **MF unet K1** | `dpcc-t-tightened` | 0.967 | **0.0183** | **1.07** | ← **our cheapest arm, cheaper than anything AF has** |
| MF unet K1 | `dpcc-c-tightened` | 0.933 | 0.0183 | 1.32 | |
| **MF unet K1** | `hardflow_new-tightened` | **1.000** | 0.0413 | 2.64 | ← the only MF arm that reaches 1.000 |
| **AF sit K2** | `dpcc-r-tightened` | **1.000** | 0.0202 | **1.36** | ← **reaches 1.000 on cheap post-hoc projection** |
| AF sit K2 | `dpcc-t-tightened` | 0.933 | 0.0228 | 1.61 | |
| AF sit K2 | `hardflow_new-tightened` | 1.000 | 0.0686 | 4.58 | AF's HF arm is *more* expensive than ours — and unnecessary |

**Read the first and fourth rows together.** Our cheapest arm is **21 % cheaper per episode than
AlphaFlow's winner** (1.07 vs 1.36 s/ep) — MeanFlow at K = 1 is genuinely the fastest generative
engine in this batch. But it stalls at S&C 0.967. To buy the last episode we must switch to the
HardFlow arm, which costs **2.3× more per step** (0.0413 vs 0.0183) and lands at 2.64 s/ep.

> **AlphaFlow's entire advantage is that it never needs the expensive arm.** It gets perfect safety
> out of ordinary post-hoc DPCC projection; we have to buy ours with an in-loop NLP.

#### 4.4.2 Why — its unprojected field is 8× more often constraint-clean

The `diffuser` arm (no projection at all) explains the whole thing:

| config | unprojected S&C | avg # violations | total violation mass |
|---|---|---|---|
| **AF `bbsit` K2** | **0.267** | 15.27 | 2.811 |
| AF `bbunet` K2 | 0.067 | 14.30 | 1.747 |
| MF `bbmf_dit` K2 | 0.067 | 17.57 | 3.274 |
| MF `bbunet` K2 | 0.033 | 15.47 | 2.526 |
| MF `bbunet` K1 | 0.033 | 15.50 | 2.102 |

**AF sit produces a fully constraint-satisfying trajectory 8× more often than MF UNet** (0.267 vs
0.033) — roughly 1 episode in 4 versus 1 in 30. Note this is *not* visible in the average violation
count (15.27 vs 15.50, essentially identical) or the violation mass (2.81 vs 2.10, where MF is
actually **better**). The difference is in the *tail*: AF's field puts a meaningful fraction of its
mass entirely inside the feasible set, so the projector starts from a solvable place far more often.
MF's field is on average no worse — it is just never *clean*.

That is the mechanism to attack if MeanFlow is to close this gap, and it is a **training-side**
problem, not a projection-side one.

#### 4.4.3 Our step "lead" is one seed

Per-seed, `S&C / steps / s-per-episode` (seeds 6, 7, 8, 9, 10):

| config | s6 | s7 | s8 | s9 | s10 |
|---|---|---|---|---|---|
| AF sit K2 `-r-tight` | 1.00/64.3/**1.28** | 1.00/**83.3**/**1.65** | 1.00/62.8/**1.28** | 1.00/64.7/**1.31** | 1.00/62.8/**1.28** |
| MF unet K1 HF-tight | 1.00/64.8/2.85 | 1.00/**60.2**/2.40 | 1.00/66.0/2.77 | 1.00/67.5/2.73 | 1.00/60.3/2.46 |

- **Time: AF wins 5/5 seeds**, every one by ~2×. That is why the CI is tight and the verdict is safe.
- **Steps: AF wins 3/5** (seeds 6, 8, 9); we win seeds 7 and 10. Our −3.83 pooled lead comes almost
  entirely from **seed 7, where AF takes 83.3 steps** — its own worst seed by 19 steps. Drop seed 7
  and AlphaFlow leads on steps as well.

🔴 **So the step axis should not be reported as a MeanFlow strength at all.** It is a single-seed
artifact of one bad AlphaFlow seed, and §4.4's table already marks it non-significant. The honest
summary of this head-to-head is: **AlphaFlow-sit is better, on the axis that resolves, on every
seed.**

So the honest standing of this DA's headline: *we reached the Target, and a sibling generation
reached it harder.* [[benchmark-hierarchy-who-beats-whom]] puts diffusion-DPCC as the bar we must
clear — which we did — but AlphaFlow is the internal state of the art on `avoiding` and any
"best config" language belongs to it, not to MeanFlow.

**⚠️ Backbone preference is family-dependent — do not generalise §4.1.**

| family | DiT/SiT backbone | UNet backbone | winner |
|---|---|---|---|
| **MeanFlow** | `mf_dit` 0.967 / 68.43 | **`bbunet` 0.967 / 59.43** | **UNet** |
| **AlphaFlow** | **`bbsit` 1.000 / 67.60** | `bbunet` 0.833 / 61.77 | **SiT** |

AlphaFlow + UNet is markedly worse (per-env S&C **0.80 / 0.80 / 0.60**) while MeanFlow + UNet is
better. §4.1's "the UNet is the better backbone" is a **MeanFlow-specific** result; transplanting it
to AlphaFlow would be a mistake, and `fix_1`'s original error was exactly this kind of
over-generalisation from one arm.

📌 **One thing UNet-MeanFlow alone gets right:** the `dpcc-c-tightened` collapse. `mf_dit` scores
**0.100 / 186.13 steps** and `af_sit` **0.200 / 181.00** there — both timing out — while MF `bbunet`
holds **0.933 / 97.20**. It is slow on `-c` (§7.4) but it is the only one of the three that does not
fail on it.

---

## 5. The unprojected field

`diffuser` (no projection), S&C by K: **0.033 (K1) · 0.033 (K2) · 0.000 (K5) · 0.167 (K10)**, with
14–16 average violations throughout. Goal-reaching is fine; constraint satisfaction is absent.

**The DPCC design intent holds:** the generative brain plans, the projector supplies safety
(0.03 → 1.00 at K1). It also means **none of our numbers are a property of the MeanFlow field
alone** — every S&C ≥ 0.9 row in this DA is a projector result on a MeanFlow prior.

---

## 6. Caveats

- 🔴 **`n_steps` averages over goal-successful trials only** (`eval_flow_matching_v3_meanflow.py:518`),
  so step counts are only comparable between rows at equal S&C. The §2.1 and §4.1 comparisons are
  gate-matched; the §2.2 rows are *not* (0.967 vs 1.000) and their flattering step counts carry
  that bias.
- 🔴 **30 episodes, `n_trials = 2` per cell.** Per-seed S&C resolves to 1/6; the pooled figure to
  1/30. The bootstrap resamples **seeds** (5 clusters) — the honest unit — which is why the step CIs
  are wide. `n_trials ≥ 10` would resolve §2.2's one-episode miss.
- ⚠️ **`avg_time` is wall-clock on shared GPUs** and includes the NLP solve. Differences under
  ~10–20 % are noise; only the order-of-magnitude gaps (0.018–0.042 vs 0.31–0.60 s) are claimed.
- ⚠️ **No activation-threshold sweep** — HardFlow ran `A0.5_B1` only (§4.3).
- ⚠️ **`-r`/`-c`/`-t` are identical for the HardFlow arms** at `batch_size: 1` by construction, not
  by coincidence. Do not read them as three independent results.
- ⚠️ **Window-level train/test split leak** (inherited, POST_U10_III §4.2) affects all arms equally.
- ✅ **Provenance is clean this time:** seeds 6–10 all ran `batch_size: 1`, `activation_threshold:
  0.5`, `T = 0.5`, `n_trials = 2`, with the eval yaml snapshotted per seed (Fix_9). Seed 6's
  earlier `HFFM_BATCH=4` run is **superseded and must not be pooled** with these.
- ✅ **Reproducibility spot-check:** the accidental 24470 re-run of seeds 7–10 reproduced
  **624/624 cells identically** on S&C and `n_steps`, confirming the eval is deterministic on the
  outcome axes.

---

## 7. Next

1. **Activation-threshold sweep for the HardFlow arm — at K = 5 and K = 10, _not_ K = 1.**
   §4.3's solve-density table shows the threshold has no reachable effect at K ∈ {1, 2}: the
   terminal solve always fires and A = 0.5 already yields exactly one solve, so the knob cannot go
   lower. Sweep where intermediate solves exist:

   ```bash
   for K in 5 10; do for A in 0.0 0.1 0.25; do
     HFFM_ACT_THRESHOLD=$A HFFM_FLOW_STEPS=$K \
       ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
   done; done
   ```

   Six jobs, ~15 min each; Fix_9's `A` token keeps the results directories distinct. Hold
   `DPCC_THRESHOLD` at 0.5 as the fixed arm-B reference — the two knobs are independent by design
   (`eval_flow_matching_v3_meanflow.py:75`), and moving both makes the result unreadable.
   `A = 0.0` is the terminal-only floor and answers the question the K = 1 result raises: **do the
   intermediate solves buy anything at all?**
2. **`n_trials ≥ 10`** to resolve the §2.2 one-episode gate miss. At 150 episodes/arm the K1/K2
   DPCC-projected rows either clear 1.000 and become strict Target dominations, or they do not —
   currently undecidable.
3. **Probe the `(seed 7, top-right-hard)` cell** on MF K1/K2 `dpcc-t-tightened` (§2.3). It is the
   *same* initial condition failing at both K, and it is the entire difference between a one-axis
   win and a Pareto domination of the Target. The HardFlow arm solves it — diffing the two
   trajectories on that one episode is the cheapest available explanation of when in-loop
   projection actually matters. Start from the saved `.npz` in
   `.../H8_K2_…_A0.5_B1_…/7/results/halfspace_top-right-hard/dpcc-t-tightened.npz`.
4. **Re-check the `-c` arm.** MF `dpcc-c-tightened` posts 97.20 steps at K = 2 (vs 59.43 for `-t`) —
   the dawdling documented in RESULTS §4.2 reproduces across all 5 seeds and is still unexplained.
5. **Close the AlphaFlow gap on the training side, not the projection side (§4.4.2).** The gap is
   *not* that MeanFlow's field violates more — average violations (15.50 vs 15.27) and violation
   mass (2.10 vs 2.81) are equal-or-better than AF sit. It is that AF's field is **fully clean 8×
   more often** (0.267 vs 0.033 unprojected S&C). Chasing lower mean violation will not help;
   what is needed is more probability mass entirely inside the feasible set. Worth checking whether
   AlphaFlow's bootstrapped target / α-annealing ([[meanflow-family-upstreams]]) is what produces
   that tail, since that is a portable training change rather than an architecture one.
6. **Fold this into `MASTER_TEST_HISTORY.md`** — Gen3v6 row. **Not edited here** (standing
   convention: never self-edit the master index).

## 8. One-line verdict

**At K = 1, MeanFlow UNet@32 with HardFlow-tightened projection matches the best diffusion-DPCC (§9 lists the two sweeps this DA is missing)
configuration's perfect safety (S&C 1.000, 5/5 seeds, all three halfspaces) while running 8.4×
faster per episode** — Target reached on the time axis, the step axis favourable but not
significant, the DPCC-projected K1/K2 rows missing the same Target by one reproducible
`(seed 7, top-right-hard)` episode — **and AlphaFlow-sit reaching that same Target 1.9× cheaper
still, so the frontier on `avoiding` remains Gen3v7's, not ours.**

---

## 9. ⚠️ Known gaps — two sweeps this DA does **not** have (deferred to 2026-08-12)

Recorded explicitly so nothing in §0–§8 is read as more complete than it is. **Both gaps are
comparison-side, not data-side:** every number above is real, matched and 5-seed. What is missing is
*coverage of the configuration space around* those numbers.

### 9.1 Gap A — the K study is one-sided

Only our own arm has a full multi-seed ladder. The two families we compare against do not:

| family | K1 | K2 | K5 | K10 | K20 |
|---|---|---|---|---|---|
| **MF `bbunet` (ours)** | **5 seeds** | **5** | **5** | **5** | — |
| **MF `mf_dit`** | seed 6 only | **5** | seed 6 only | seed 6 only | seed 6 only |
| **AF `bbsit`** | 7–10 (4) | **5** | 7–10 (4) | 7–10 (4) | — |

**Consequences for what is claimed above:**

- §4.1's "the UNet is the better MeanFlow backbone" is established **at K = 2 and nowhere else**.
  There is no `mf_dit` ladder, so we cannot say whether the DiT's `-c` collapse (§4.1.2) is a
  K = 2 pathology or holds at every K — and we cannot rule out a K at which the DiT is the better
  backbone.
- §3's low-K claim is **internal to our own arm**. "K = 1–2 is sufficient" is demonstrated for
  `bbunet`; that it generalises to the MeanFlow objective is an inference, not a measurement.
- §4.4's AlphaFlow verdict is at K = 2 only. AF's own ladder is 4-seed and unmerged with seed 6.

🔴 **Additional finding — AlphaFlow's HardFlow arms are provenance-contaminated and must not be
quoted.** AF's runs predate Fix_9 (`808cb1a4`, 2026-08-07): their folders carry no `A`/`B` token, so
**seed 6 (`hf_batch=4`, `hf_act_threshold=0.5`) and seeds 7–10 (`hf_batch=1`, threshold 1.0) wrote
into the same directory.** The pooled AF `hardflow_new-*` rows therefore mix two configurations —
including the 4.58 s/ep figure in §4.4.1, which should be treated as unusable. **AF's `dpcc-*` arms
are unaffected** (`T0.5` throughout, and the HF knobs do not touch arm B), so §4.4's headline —
which rests on `dpcc-r-tightened` — **stands**. When AF is re-run, run all five seeds, not just
seed 6: post-Fix_9 output lands in a new `A…_B…` folder and will not merge with the old one anyway.

### 9.2 Gap B — no HardFlow activation-threshold study

Every arm in this DA ran at a single `activation_threshold = 0.5` (`A0.5_B1`). The hierarchy
obligation ([[benchmark-hierarchy-who-beats-whom]]) asks for arm C to be swept over its threshold
before any HardFlow claim is complete, and that sweep does not exist yet.

📌 **One structural fact makes this gap narrower than it looks — and one makes it wider.**

*Narrower:* per §4.3, the threshold is **saturated at K = 1 and K = 2** — the terminal solve always
fires, `A = 0.5` already yields exactly one NLP solve per step, and no lower value can reduce it.
**Our winning row (K = 1 HF-tightened) therefore cannot improve under this sweep.** Its Target beat
in §2.1 is threshold-independent and is not at risk.

*Wider:* the corollary is that **only the losers can move.** Every configuration currently ranked
below the winner sits at K ≥ 5, where 2–4 intermediate solves exist and a lower threshold would make
them cheaper. So the sweep can only ever *promote* a currently-bad arm — never demote the winner.

### 9.3 The "best horse" decision, and its one real risk

**Decision (deliberate, taken 2026-08-11):** sweep the threshold on the configuration that already
wins, rather than sweeping the full K × A grid, and accept the cost in coverage.

That is the right call for throughput — the full grid is 4 K × 4 A × 2 families ≈ 32 jobs, versus 6
for the targeted sweep in §7.1. But it carries a specific, named failure mode:

> 🔴 **Selection-on-the-winner.** We picked the best horse *at `A = 0.5`*. A configuration that
> looks bad at `A = 0.5` can be the best at a different `A`, and sweeping only the current winner
> will never find it — especially since §9.2 shows the winner is the one arm that *cannot* move.

Where that risk is concretely live in this batch — the arms worth a look even though they lost:

| arm | at `A = 0.5` | why it could move |
|---|---|---|
| MF `bbunet` **K10** `hardflow_new-tightened` | S&C 0.833, 16.35 s/ep | 5 solves/step; at low `A` it drops toward 1 and gets ~5× cheaper. If S&C holds it becomes a serious row. |
| MF `bbunet` **K5** `hardflow_new-tightened` | S&C 0.933, 9.11 s/ep | 3 solves/step, same logic, and it is already only 0.067 off the gate. |
| MF `bbunet` **K10** on `both-hard` | S&C **0.50** | The worst cell in the DA (§3). If this is *over*-projection rather than a weak field, a lower `A` fixes it — and that would be the "magical" case. |
| `mf_dit` HF arms at K ≠ 2 | unmeasured | Gap A: no data at all. |

⚠️ **Note the `both-hard` K10 row specifically.** It is the one place where more projection plausibly
*hurts* — an arm that half-fails at high solve density and would be expected to *improve* as the
threshold drops. If that happens it inverts the §4.3 story ("HardFlow only works at K = 1") into
"HardFlow works at the extremes of solve density and fails in the middle", which is a different and
more interesting claim.

### 9.4 What to run tomorrow

Ordered so the cheapest thing that could overturn a conclusion runs first.

```bash
# 1) HF threshold sweep — only K=5,10 have headroom (§4.3). 6 jobs, ~15 min each.
cd ~/FMPCC/FM-PCC
for K in 5 10; do for A in 0.0 0.1 0.25; do
  HFFM_ACT_THRESHOLD=$A HFFM_FLOW_STEPS=$K \
    ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
done; done

# 2) mf_dit K-ladder at seeds 7-10 — eval only, checkpoints exist. ~4h17m.
git checkout config/avoiding-d3il.py            # restores imf_backbone: 'mf_dit' (lines 714 AND 1364)
grep -n "'imf_backbone':" config/avoiding-d3il.py
# meanflow_projection_eval.yaml -> seeds: [7, 8, 9, 10]   (verify ONE seeds key)
grep -n '^seeds:' config/meanflow_projection_eval.yaml
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow.sh
# 🔴 flip imf_backbone back to 'unet' afterwards, before any further UNet run

# 3) AlphaFlow ladder, all 5 seeds, post-Fix_9 provenance. ~5h.
# alphaflow_projection_eval.yaml -> seeds: [6, 7, 8, 9, 10]
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/eval_alphaflow.sh
```

**Pre-registration, so tomorrow's results cannot be read post-hoc:** the §2.1 Target beat is
**not** revisable by any of these runs (§9.2 — the winner is threshold-saturated). What *is* open is
the **best-config** question: if any K ≥ 5 arm reaches S&C ≥ 1.000 at a lower `A` and a lower
s/episode than 2.64, it replaces K = 1 HF-tightened as our headline row, and §0 must be rewritten.
Decide that rule now, before seeing the numbers.

⚠️ Two hazards that already cost a day each this week — check both before submitting:
duplicate `seeds:` keys in a yaml are silently legal and **the last one wins** (this is what made
24416 and 24470 evaluate the wrong seeds), and `imf_backbone` is a config edit with **no env
override**, so a stale working copy silently evaluates the wrong backbone.

### 9.5 Expected impact — why §0 will most likely survive both sweeps

Recorded *before* running them (2026-08-11), so tomorrow's outcome can be checked against a
prediction rather than rationalised after the fact.

**Prediction: neither sweep changes any conclusion in §0.** Grounds:

1. **The Target beat cannot move.** §9.2 — K = 1 is threshold-saturated at one NLP solve. No value
   of `A` reduces it further, so §2.1 is immune by construction.
2. **Most of the DA does not involve arm C at all.** §2.2, §3 and §4.4's headline all rest on
   `dpcc-*` rows, which the HardFlow threshold does not govern.
3. **The best-config ranking is immune on arithmetic, not just on structure.** To displace K = 1
   HF-tightened a challenger must beat **2.64 s/ep**. But the K ≥ 5 arms are expensive *before* any
   NLP: their cheapest rows are **0.2238 (K5)** and **0.4023 (K10)** s/step against K = 1's
   **0.0183** — 12× and 22×. K5 HF-tightened is 9.11 s/ep at `A = 0.5`; going from 3 solves to 1
   might reach ~6 s/ep, still **more than double** the incumbent. K10 is further away again.
   **A lower threshold cannot close a 12–22× base-cost gap.**

**What could genuinely move, in order of value:**

| candidate | can it change a claim? |
|---|---|
| **`mf_dit` K-ladder (§9.4 step 2)** | **Yes** — §4.1's backbone verdict is currently K = 2-only. This is the higher-value job of the two and the only one with a live claim attached. |
| K10 `both-hard` S&C 0.50 at low `A` | Scientifically yes, for §4.3's story (over-projection vs weak field); **not** for the ranking — the arm is 16.35 s/ep either way. |
| Everything else in the threshold sweep | Discharges a formal hierarchy obligation. Expect no ranking change. |

> **Bottom line: run the sweeps to close the obligations, not because the result is in doubt.**
> If §0 *does* change, the most likely cause is the `mf_dit` ladder, not the threshold sweep —
> and if the threshold sweep alone overturns something, treat that as a signal to re-examine the
> cost model above, since it would contradict a straightforward arithmetic argument.
