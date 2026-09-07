# DA 2026-08-17 — AlphaFlow-SiT at `n_trials = 20`, K1 / K2

**Batch:** `temp/1708/batch_avoiding_combined_20260817_092728/`
**Protocol:** 5 seeds {6–10} × 3 halfspaces × **20 trials = 300 episodes** per configuration.
**Statistics:** seed-clustered bootstrap, B = 20 000, 95 % percentile CI. `*` = CI excludes 0.
**Follows:** [`../../Gen3v6_MeanFlow/DA/DA_20260815_ntrials20_stability_MF_UNet.md`](../../Gen3v6_MeanFlow/DA/DA_20260815_ntrials20_stability_MF_UNet.md)

## 0. Data integrity — ✅ ready

| candidate | engine | K | trials | seeds | coverage |
|---|---|---|---|---|---|
| **C40** | AlphaFlow-SiT `_msg20trials` | 1 | **20** | 6–10 | ✅ 3/3 halfspaces every seed |
| **C43** | AlphaFlow-SiT `_msg20trials` | 2 | **20** | 6–10 | ✅ 3/3 |
| C39 / C42 | AlphaFlow-SiT (n=2 originals) | 1 / 2 | 2 | 6–10 | ✅ **intact, not overwritten** |
| C135/C139/C144/C132 | MeanFlow-UNet `_msg20trials` | 1/2/5/10 | 20 | 6–10 | ✅ |
| C136 | MeanFlow-UNet `_msg20trials` | 20 | 20 | 6–10 | 🔴 still truncated (1 halfspace on some seeds) — unusable |

`FMPCC_RUN_MSG=20trials` worked: the 20-trial runs landed in `_msg20trials` directories and the
2-trial AlphaFlow data survives for the paired comparison. Trial count verified by value
granularity (n=20 → multiples of 0.05).

---

## 1. Did AlphaFlow's `S&C = 1.000` survive 300 episodes? — **Partly, and the part that survived is clean**

The 08-15 DA predicted AlphaFlow's 2-trial 1.000 readings "would likely fall to ~0.95–0.99".
**That prediction was wrong for two arms and right for the rest.**

**K = 1:**

| arm | S&C n=2 | S&C n=20 | steps n=2 → n=20 | s/ep n=2 → n=20 |
|---|---|---|---|---|
| **`dpcc-r-tightened`** | 1.000 | **1.000** ✅ held | 76.53 → 73.37 | 1.06 → **1.03** |
| **`hardflow-{r,c,t}-tightened`** | 1.000 | **1.000** ✅ held | 77.27 → 73.67 | 2.30 → 2.23 |
| `dpcc-t-tightened` | 1.000 | 0.987 🔴 lost | 66.10 → 67.73 | 0.92 → 0.98 |
| `dpcc-c-tightened` | 0.900 | 0.857 | 129.53 → 121.61 | 1.76 → 1.67 |
| `diffuser` | 0.067 | 0.137 | 70.90 → 67.31 | 0.44 → 0.42 |

**K = 2:** every arm that read 1.000 at n=2 lost it — `dpcc-r-tightened` 1.000 → 0.983,
`hardflow-{r,c,t}-tightened` 1.000 → 0.987. `dpcc-c-tightened` remains collapsed (0.200 → 0.160,
184 steps).

### 1.1 `AF-SiT K1 dpcc-r-tightened` is 300/300

| | seed 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|
| `top-right-hard` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `top-left-hard` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| `both-hard` | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

**Not a single failure in 300 episodes**, across all seeds and all three constraint settings. This
is the first configuration in the study whose `S&C = 1.000` is verified rather than assumed.

### 1.2 The arm ranking moved for AlphaFlow too

At n=2 AlphaFlow's best K = 1 row looked like `dpcc-t-tightened` (0.92 s/ep at S&C 1.000). At n=20
`dpcc-t` drops to 0.987 while **`dpcc-r-tightened` holds 1.000** and is essentially as cheap
(1.03 s/ep). Same phenomenon as MeanFlow-UNet on 08-15: at 2 trials the near-ceiling arms cannot be
ordered, and the apparent winner changes once they can.

---

## 2. Grand comparison at `n = 20`

All rows 300 episodes. Target is still `DPCC K20 aw10 dpcc-c-tightened` (**C14, n = 2**):
S&C 1.000 · 70.13 steps · 38.53 s/ep.

| engine | K | arm | S&C | steps | s/ep | Δ steps vs Target | Δ s/ep vs Target | ×faster |
|---|---|---|---|---|---|---|---|---|
| **AlphaFlow-SiT** | 1 | **`dpcc-r-tight`** | **1.000** | 73.37 | **1.03** | +3.24 `[−4.04,+9.95]` | **−37.51 `[−40.41,−34.54]`** `*` | **37.5** |
| AlphaFlow-SiT | 1 | `dpcc-t-tight` | 0.987 | 67.73 | **0.98** | −2.41 `[−9.75,+4.33]` | **−37.56 `[−40.46,−34.59]`** `*` | 39.4 |
| **MeanFlow-UNet** | 1 | `dpcc-t-tight` | 0.993 | **60.99** | 1.10 | **−9.15 `[−15.76,−3.15]`** `*` | **−37.43 `[−40.33,−34.46]`** `*` | 35.0 |
| MeanFlow-UNet | 2 | `dpcc-t-tight` | 0.993 | **60.41** | 1.64 | **−9.72 `[−16.33,−3.63]`** `*` | **−36.89 `[−39.79,−33.92]`** `*` | 23.5 |
| AlphaFlow-SiT | 1 | `hardflow-r-tight` | **1.000** | 73.67 | 2.23 | +3.54 `[−3.68,+10.21]` | **−36.31 `[−39.20,−33.33]`** `*` | 17.3 |
| AlphaFlow-SiT | 2 | `hardflow-r-tight` | 0.987 | 72.72 | 2.73 | +2.58 `[−4.68,+9.38]` | **−35.81 `[−38.71,−32.83]`** `*` | 14.1 |

**Only one row clears the S&C gate at 300 episodes: `AlphaFlow-SiT K1 dpcc-r-tightened`.**
It beats the Target on cost by **37.5×** with the CI excluding zero, at exactly equal S&C. Its step
count is nominally higher (+3.24) but not significant, so this is a **one-axis win, verified**.

### 2.1 Head-to-head, both at n = 20, same arm

MeanFlow-UNet K1 − AlphaFlow-SiT K1, `dpcc-t-tightened` both sides:

| axis | Δ | verdict |
|---|---|---|
| S&C | +0.007 `[−0.013, +0.030]` | tie |
| steps | **−6.74 `[−9.99, −4.39]`** `*` | **MeanFlow-UNet wins** |
| s/ep | **+0.125 `[+0.052, +0.174]`** `*` | **AlphaFlow-SiT wins** |

**A genuine two-sided trade-off at 300 episodes each: MeanFlow-UNet produces paths ~6.7 steps
shorter; AlphaFlow-SiT is ~0.13 s/episode cheaper. Neither dominates.** Comparing each engine's
*best* arm instead, AlphaFlow additionally clears the gate (1.000 vs 0.993) while MeanFlow-UNet
keeps a 12.4-step advantage (60.99 vs 73.37).

### 2.2 Architecture caveat unchanged

AlphaFlow-SiT is **10.0 M parameters on a SiT backbone**; the DPCC baseline and MeanFlow-UNet are
UNets (4.0 M for MeanFlow-UNet). AlphaFlow's win over the Target therefore changes network *and*
objective — it is the **larger, confounded** claim. **MeanFlow-UNet remains the only
architecture-matched engine**, and its result is the 35.0× / −9.15 steps row at S&C 0.993.
See [[architecture-matched-beat-is-the-strong-claim]].

---

## 3. What changes

| claim | status |
|---|---|
| AlphaFlow reaches S&C 1.000 | ✅ **Verified at 300 episodes** — but only `K1 dpcc-r-tightened` and `K1 hardflow-*-tightened`, not the arm that looked best at n=2 |
| AlphaFlow is the strongest configuration overall | ✅ **Confirmed** — sole gate-clearing row, 37.5× vs Target |
| AlphaFlow beats MeanFlow-UNet | 🟡 **Trade-off, not domination** — cheaper per episode `*`, but ~6.7 more steps `*` at matched arm |
| MeanFlow-UNet has the shortest paths | ✅ **Confirmed at 300 episodes** — 60.99 steps, lowest of any engine at any K |
| MeanFlow-UNet clears the gate | 🔴 **No** — 0.993, short by 2 episodes in 300; `ΔS&C` CI does include 0 |
| The 08-15 prediction that AF's 1.000 would fall | 🟡 **Half wrong** — held exactly for 2 of 4 K1 arms; fell for all K2 arms |

## 4. Was `n_trials = 20` worth the 10× compute?

Two questions, both answerable from the paired AlphaFlow data.

### 4.1 Did it change any decision? — **Yes, in both engines**

**The ordering barely moved.** Spearman ρ between the n=2 and n=20 arm ranking:
**0.952 (K1), 1.000 (K2)** — at K2 the ordering is *identical*. n=2 was right about the structure.

**But the shipped configuration changes.** At n=2, four K1 arms all read exactly 1.000
(`dpcc-r`, `dpcc-t`, `hardflow-r/c/t`), so you would pick the cheapest of them — `dpcc-t-tightened`
at 0.92 s/ep. At n=20, `dpcc-t` is **0.987** and does not clear the gate; the arms that hold 1.000
are `dpcc-r-tightened` (1.03 s/ep) and `hardflow-*` (2.23).

| | chosen at n=2 | chosen at n=20 |
|---|---|---|
| AlphaFlow-SiT K1 | `dpcc-t-tightened` (0.92 s/ep, "1.000") | **`dpcc-r-tightened`** (1.03 s/ep, 1.000 verified) |
| MeanFlow-UNet K1 | `hardflow-tightened` (2.64 s/ep, "1.000") | **`dpcc-t-tightened`** (1.10 s/ep, 0.993) |

**In both engines the arm you would deploy is different**, and in MeanFlow's case the n=2 choice was
2.4× more expensive than the n=20 choice. The change is not cosmetic — it is the recommendation.

### 4.2 Does it reduce variance, or just move numbers? — **It reveals variance n=2 could not see**

There are two regimes, and AlphaFlow shows the second one clearly:

| regime | example | effect of n=20 on the CI |
|---|---|---|
| Arm **away** from the ceiling | MeanFlow-UNet `dpcc-t` (0.93–0.97) | width **shrinks 5–8×** — classic variance reduction |
| Arm **at** the ceiling | AlphaFlow K1 `dpcc-t` | n=2 width **0.000** → n=20 width **0.033** — the interval gets *wider* |

At n=2 an arm whose 30 episodes all succeed reports mean 1.000 with a **zero-width** bootstrap CI.
That is not low variance; it is **no information**. n=20 either confirms the value on 300 samples
(`dpcc-r-tightened`, still zero width — now credible) or exposes the true rate with a real interval
(`dpcc-t`, 0.987 ± 0.017).

**Per-cell accuracy improves exactly where it matters.** AlphaFlow, 300 cells by true level:

| true S&C of cell | cells | mean \|n=2 error\| | exactly correct |
|---|---|---|---|
| ≥ 0.95 | 159 | **0.006** | **94 %** |
| 0.60–0.95 | 47 | 0.238 | 0 % |
| 0.20–0.60 | 57 | 0.309 | 2 % |
| < 0.20 | 37 | 0.081 | 49 % |

So n=2 is *accurate* near the ceiling (94 % of cells exactly right) but **cannot detect the rare
failure that decides the gate.**

### 4.3 The decisive number: detection power

Probability that a 2-trial protocol (30 episodes) observes **at least one** failure, i.e. has any
chance of noticing the arm is not perfect:

| true S&C | P(detect) at 30 episodes | P(detect) at 300 episodes |
|---|---|---|
| 0.993 | **19 %** | 88 % |
| 0.987 | **33 %** | 98 % |
| 0.980 | 46 % | 99.8 % |
| 0.950 | 79 % | 100 % |

**An arm at true S&C 0.987 looks perfect two times out of three at n=2.** That is precisely what
happened to `AF-SiT K1 dpcc-t-tightened`. To have a 95 % chance of catching a 0.987 arm you need
**229 episodes**; for a 0.993 arm, **427**. At n_trials = 20 we have 300 — enough for 0.987, still
short for 0.993.

> **Verdict: worth it.** n=2 gets the ranking right (ρ ≈ 0.95–1.00) and the near-ceiling cells right
> (94 % exact), but has only a **19–33 % chance of detecting the failures that decide the S&C gate**.
> Since the gate is the entire acceptance criterion, n=2 cannot answer the question the study is
> asking. 10× the compute buys the one thing that was missing.
>
> ⚠️ **Corollary: 300 episodes is still not enough to certify S&C ≥ 0.993.** Claims at that level
> (MeanFlow-UNet K1/K2) remain provisional; `n_trials = 30` (450 episodes) would settle them.

---

## 5. Next

1. **Re-run the Target (C14) and DPCC K10/K1 at `n_trials = 20`.** Every Δ in §2 compares a
   300-episode row against a 30-episode baseline. Per the 08-15 analysis the baseline's own 1.000 is
   itself unverified, and it is the reference for every claim in this DA.
2. **Re-run MeanFlow-UNet K20 at n = 20** — C136 is still truncated.
3. **MeanFlow-DiT at n = 20, K1/K2** — the third engine is still 2-trial only.
4. Consider `AF-SiT K1 dpcc-r-tightened` the reference configuration for `avoiding-d3il` until the
   baselines are re-measured.
