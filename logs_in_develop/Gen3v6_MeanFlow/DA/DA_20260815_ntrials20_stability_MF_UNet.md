# DA 2026-08-15 — `n_trials = 20` vs `n_trials = 2`: is the larger sample actually more stable?

**Batch:** `temp/1508/batch_avoiding_combined_20260815_135634/`
**New runs:** MeanFlow-UNet (`bbunet`, 4.0 M), one job per K — **24559** (K1, 5 h 49 m) · **24560**
(K2, 6 h 56 m) · **24561** (K5, 15 h 40 m) · **24562** (K10, 24 h 00 m) · **24563** (K20, 24 h 00 m).
All `hf_batch=1 · A=0.5 · dpcc_threshold=0.5`, 5 seeds {6–10}, `n_trials = 20`.
**Identification:** the 20-trial rows carry a `_msg20trials` suffix in the exp_name, so they sit in
separate candidates from the 2-trial ladder and nothing was overwritten.

| K | n=2 candidate | n=20 candidate | n=20 status |
|---|---|---|---|
| 1 | C129 | **C130** | ✅ complete — 300 episodes |
| 2 | C133 | **C134** | ✅ complete — 300 episodes |
| 5 | C138 | **C139** | ✅ complete — 300 episodes |
| 10 | C126 | C127 | ⚠️ seed 10 missing 5 of 39 cells (all `hardflow-*`) |
| 20 | *(none)* | C131 | 🔴 **unusable** — 13–26 of 39 cells per seed, 1–2 halfspaces only |

🔴 **Jobs 24562 and 24563 hit the 24 h Slurm wall exactly** (`JOB END` at +24 h 00 m) and were killed
mid-sweep. K20 at `n=20` does not exist in usable form; K10 is usable for the `dpcc-*` arms only.
**Raise `--time` or split by halfspace before re-running K ≥ 10 at 20 trials.**

---

## 1. Is `n_trials = 20` more stable? — Yes, decisively

### 1.1 Uncertainty shrinks 5–8×

Seed-clustered bootstrap 95 % CI on S&C, `dpcc-t-tightened`:

| K | n=2 mean [CI] | width | n=20 mean [CI] | width | width ratio |
|---|---|---|---|---|---|
| 1 | 0.967 `[0.900, 1.000]` | 0.100 | 0.993 `[0.980, 1.000]` | 0.020 | **5.0×** |
| 2 | 0.967 `[0.900, 1.000]` | 0.100 | 0.993 `[0.987, 1.000]` | 0.013 | **7.5×** |
| 5 | 0.933 `[0.800, 1.000]` | 0.200 | 0.980 `[0.963, 0.997]` | 0.033 | **6.0×** |
| 10 | 0.933 `[0.800, 1.000]` | 0.200 | 0.973 `[0.957, 0.990]` | 0.033 | **6.0×** |

### 1.2 Most of the apparent *seed* variance was trial noise

Between-seed SD of S&C — the quantity the cluster bootstrap is built on:

| K | arm | SD (n=2) | SD (n=20) | reduction |
|---|---|---|---|---|
| 1 | `dpcc-t-tightened` | 0.0667 | 0.0133 | 5.0× |
| 2 | `dpcc-t-tightened` | 0.0667 | 0.0082 | **8.2×** |
| 5 | `dpcc-t-tightened` | 0.1333 | 0.0194 | 6.9× |
| 10 | `dpcc-t-tightened` | 0.1333 | 0.0200 | 6.7× |
| 5 | `hardflow-tightened` | 0.1333 | 0.0221 | 6.0× |

At `n=2` the per-seed estimate is itself a 6-episode average, so "between-seed spread" was mostly
sampling noise, not genuine seed-to-seed difference. **The seeds agree far more than the 2-trial
data suggested.**

### 1.3 How wrong individual cells were

Per-cell (seed × halfspace) comparison, `dpcc-t-tightened`, 60 cells:

| K | worst cell | n=2 said | n=20 says | error |
|---|---|---|---|---|
| 1 | seed 7, `top-right-hard` | 0.50 | 1.00 | 0.50 |
| 2 | seed 7, `top-right-hard` | 0.50 | 1.00 | 0.50 |
| **5** | **seed 7, `top-right-hard`** | **0.00** | **0.85** | **0.85** |
| 10 | seed 8, `top-right-hard` | 0.50 | 0.95 | 0.45 |

Mean absolute per-cell error of `n=2`: **0.055**. At K = 5 a cell that read as *total failure*
(0.00) is in fact 0.85.

### 1.4 The worst failure mode: `n=2` produced false certainty

`hardflow-tightened`, K = 1:

| | mean | 95 % CI | width |
|---|---|---|---|
| n=2 | **1.000** | `[1.000, 1.000]` | **0.000** |
| n=20 | **0.950** | `[0.910, 0.987]` | 0.077 |

All 5 seeds × 2 trials happened to succeed, so the bootstrap saw zero variance and reported a
**zero-width interval around a value that is wrong by 0.05**. This is not imprecision — a
resampling CI cannot represent uncertainty the sampling never exposed. **Any `n=2` row reading
exactly 1.000 should be treated as unverified, not as perfect.**

### 1.5 Scope of the deviation — which cells, and which metrics

Per-cell (seed × halfspace) comparison, all four K pooled, 60 cells per arm:

| arm | identical | small (<0.2) | **big (≥0.2)** | n=2 too high | n=2 too low | mean shift |
|---|---|---|---|---|---|---|
| `dpcc-t-tightened` | **45 (75 %)** | 10 | **5 (8 %)** | 10 | 5 | +0.035 |
| `dpcc-r-tightened` | 27 (45 %) | 21 | 12 (20 %) | 22 | 11 | +0.033 |
| `dpcc-c-tightened` | 24 (40 %) | 21 | 15 (25 %) | 24 | 12 | +0.023 |
| `hardflow-tightened` | 26 (44 %) | 23 | 10 (17 %) | **27** | 6 | **−0.042** |

**Two different phenomena, not one:**

- **`dpcc-t-tightened` (the best arm): a few cells, large corrections.** 75 % of cells are *exactly*
  identical; the aggregate +0.03…+0.05 shift is carried by **5 cells out of 60**, each corrected by
  ≥ 0.2 (the largest, 0.00 → 0.85). Not broad drift — a handful of badly-sampled cells.
- **`hardflow-tightened`: broad, one-directional drift.** n=2 read *too high* in 27 cells versus too
  low in 6 — a 4.5 : 1 bias — and **51 % of the cells that read exactly 1.000 at n=2 are below
  1.000 at n=20** (27 of 53). For `dpcc-t-tightened` the same figure is **18 %** (10 of 55).

**The instability is confined to S&C. The cost metrics were fine at n = 2:**

| metric | median \|rel. error\| | p90 | max |
|---|---|---|---|
| `n_steps` | **3.89 %** | 13.77 % | 17.90 % |
| `avg_time` | **3.86 %** | 14.62 % | 28.99 % |

S&C is a rare-event rate — at 2 trials it can only take the values {0, 0.5, 1}, so a single unlucky
episode moves a cell by 0.5. Steps and time are continuous averages over ~60 control steps and are
already well estimated at 2 trials. **This is why every step/time conclusion in the earlier DAs
survives and every S&C conclusion had to be re-checked.**

### 1.6 The deviation is structured, not random — and n=2 was mostly right

**Rank preservation across arms.** Spearman ρ between the n=2 and n=20 arm ordering (10 arms):

| K | ρ |
|---|---|
| 1 | **0.927** |
| 2 | **0.927** |
| 5 | 0.818 |
| 10 | **0.929** |

At K = 1 the ordering below the top four is **preserved exactly** — `dpcc-c-tightened` #5→#5,
`dpcc-r-tightened` #6→#6, `dpcc-c` #7→#7, `dpcc-t` #8→#8, `dpcc-r` #9→#9, `diffuser` #10→#10. Only
the top four reshuffled, and those four sit within 0.933–1.000 of each other.

**Why: the error follows binomial sampling, so it is largest mid-range and near-zero at the ceiling.**
Grouping all 597 cells by their *true* (n=20) success level:

| true S&C of the cell | cells | mean \|n=2 error\| | exactly correct | √(p(1−p)/2) |
|---|---|---|---|---|
| ≥ 0.95 | 233 | **0.027** | **75 %** | 0.110 |
| 0.80–0.95 | 87 | 0.202 | 0 % | 0.234 |
| 0.60–0.80 | 97 | 0.278 | 0 % | 0.324 |
| 0.40–0.60 | 90 | 0.295 | 12 % | 0.354 |
| 0.20–0.40 | 43 | 0.308 | 0 % | 0.324 |
| < 0.20 | 47 | 0.102 | 21 % | 0.212 |

The observed error tracks the theoretical 2-trial binomial SD across the whole range. **This is not
randomness in the pejorative sense — it is exactly the sampling error a 2-trial estimator must
have, and it is smallest precisely where our best configurations live (≥ 0.95: 0.027 mean error,
three-quarters of cells exactly right).**

**Per-seed, the error is uniform — no rogue seed.** Mean \|error\| over all arms × halfspaces:

| seed | K1 | K2 | K5 | K10 | mean |
|---|---|---|---|---|---|
| 6 | 0.173 | 0.170 | 0.175 | 0.172 | 0.172 |
| **7** | 0.155 | 0.192 | **0.243** | 0.227 | **0.204** |
| 8 | 0.125 | 0.135 | 0.118 | 0.162 | **0.135** |
| 9 | 0.118 | 0.153 | 0.142 | 0.188 | 0.150 |
| 10 | 0.117 | 0.138 | 0.142 | 0.148 | 0.136 |

Seed 7 is the noisiest and seed 8 the calmest, but the spread is only 1.5× (0.135–0.204) and every
seed sits in the same band at every K. **The n=2 unreliability is not concentrated in one seed** —
which also means adding seeds would not have fixed it; only adding trials does.

> **So the manual reading is correct:** n=2 preserved the broad structure (ρ ≈ 0.93), deviated
> little on the rows that matter, and showed no seed-specific pathology. **What it could not do is
> discriminate between arms bunched at 0.93–1.00** — a 0.027 mean error is enough to reorder them,
> and that is exactly the discrimination the S&C gate requires. n=2 is adequate for *screening*,
> not for *ranking near the ceiling* or for claiming 1.000.

> **Verdict: `n_trials = 20` is not a marginal improvement.** It narrows intervals 5–8×, removes
> most of the apparent seed variance, corrects individual cells by up to 0.85, and eliminates a
> false-certainty mode that `n=2` cannot detect from within.

---

## 2. What the corrected numbers change

### 2.1 The best MeanFlow-UNet arm flips at every K

Full `n=20` table, 5 seeds × 20 trials (K1/K2/K5 = 300 episodes; K10 `dpcc-*` only):

| K | arm | S&C | steps | s/step | s/ep |
|---|---|---|---|---|---|
| **1** | **`dpcc-t-tightened`** | **0.993** | **60.99** | 0.0181 | **1.10** |
| 1 | `hardflow-tightened` | 0.950 | 63.40 | 0.0419 | 2.66 |
| 1 | `dpcc-c-tightened` | 0.943 | 72.01 | 0.0180 | 1.29 |
| 1 | `dpcc-r-tightened` | 0.930 | 64.65 | 0.0184 | 1.19 |
| 1 | `diffuser` | 0.120 | 61.87 | 0.0096 | 0.60 |
| **2** | **`dpcc-t-tightened`** | **0.993** | 60.41 | 0.0271 | 1.64 |
| 2 | `hardflow-tightened` | 0.900 | 67.06 | 0.0503 | 3.37 |
| 2 | `dpcc-c-tightened` | 0.927 | 98.00 | 0.0268 | 2.62 |
| **5** | **`dpcc-t-tightened`** | **0.980** | 60.84 | 0.2250 | 13.71 |
| 5 | `hardflow-tightened` | 0.897 | 67.31 | 0.1408 | 9.48 |
| **10** | **`dpcc-t-tightened`** | **0.973** | 61.01 | 0.3982 | 24.33 |

🔴 **`hardflow-tightened` is no longer the best arm anywhere.** At `n=2` it was the only arm reaching
S&C 1.000 at K = 1; at `n=20` it scores **0.950** and is beaten by `dpcc-t-tightened` on **all three
axes** — higher S&C (0.993), fewer steps (60.99 vs 63.40) and **2.4× cheaper** (1.10 vs 2.66 s/ep).

The in-loop NLP was never buying safety; it was buying a 2-trial artifact.

### 2.2 Shift table, n=2 → n=20

| K | arm | S&C n=2 → n=20 | |
|---|---|---|---|
| 1 | `dpcc-t-tightened` | 0.967 → **0.993** | +0.027 |
| 1 | `hardflow-tightened` | **1.000 → 0.950** | −0.050 🔴 gate lost |
| 2 | `dpcc-t-tightened` | 0.967 → **0.993** | +0.027 |
| 2 | `hardflow-tightened` | 0.933 → 0.900 | −0.033 |
| 5 | `dpcc-t-tightened` | 0.933 → **0.980** | +0.047 |
| 5 | `hardflow-tightened` | 0.933 → 0.897 | −0.037 |
| 10 | `dpcc-t-tightened` | 0.933 → **0.973** | +0.040 |

**Systematic pattern: the DPCC-projected arms were *underestimated* at n=2 (+0.03 to +0.05); the
HardFlow arms were *overestimated* (−0.03 to −0.05).** The two errors pointed in opposite
directions, which is exactly why the arm ranking inverted.

### 2.3 The localized failure was itself an artifact

The 2026-08-11 DA localized MeanFlow-UNet's shortfall to a single reproducible cell,
*(seed 7, `top-right-hard`)*, at both K = 1 and K = 2. **At n = 20 that cell scores 1.000.** The
real residual failures at K = 1 are elsewhere and smaller:

```
seed 6  top-right-hard : 0.95   (1 of 20 trials)
seed 6  both-hard      : 0.95   (1 of 20 trials)
all other 13 cells     : 1.00
```

Per-seed S&C at K = 1: `0.967 · 1.000 · 1.000 · 1.000 · 1.000`. **Two failures in 300 episodes,
both on seed 6.** The earlier seed-7 investigation was chasing noise.

### 2.4 Does this overturn "MeanFlow-UNet is superior"?

Separating what was measured from what was claimed:

| claim | status after 300 episodes |
|---|---|
| **Cost advantage over the DPCC baseline** | ✅ **Strengthened.** 14.6× → **35.0×**, and `s/ep` was never the noisy metric (§1.5: 3.9 % median error at n=2). |
| **Step advantage over the baseline** | ✅ **Strengthened and now significant.** −6.37 (ns) → **−9.15 `[−15.76,−3.15]` `*`**. |
| **Safety parity with the baseline** | 🟡 **Measured, not perfect.** 0.993 over 300 episodes; `ΔS&C = −0.01 [−0.02, +0.00]` includes zero, so not distinguishable from the Target at 5 seeds — but the point estimate is below 1.000. |
| **"S&C = 1.000" / gate cleared** | 🔴 **Withdrawn.** That reading came from the `hardflow` arm at n=2 with a zero-width CI (§1.4). |
| **In-loop (HardFlow) projection is what makes it work** | 🔴 **Withdrawn.** `hardflow` is now dominated on all three axes by plain `dpcc-t-tightened` (§2.1). |
| **Superior to AlphaFlow-SiT / MeanFlow-DiT** | ⛔ **Unresolved and untestable right now.** Those rows are n=2. Given that 51 % of `hardflow` cells reading 1.000 at n=2 fell below 1.000 at n=20, their 1.000 values are likely inflated by a similar amount. |

**Net:** the result did not weaken — it *relocated*. The engine's advantage over the published
baseline is on **cost and path length**, both of which are now larger and both of which rest on the
metrics that were already reliable at n = 2. What was never actually established — and still is not
— is a safety *advantage*; the honest statement is safety **parity within measurement error** at
35× lower cost. The engine-vs-engine ranking needs the baselines re-run at n = 20 before anything
is claimed.

---

## 3. Grand table — vs the Target

⚠️ **Trial counts are unequal.** MeanFlow-UNet rows are `n=20` (300 episodes); every baseline and
every other engine is still `n=2` (30 episodes). Per §1, the `n=2` rows carry ±0.05-scale error and
any of them reading exactly 1.000 is unverified. **Δ and CI are computed seed-clustered on 5 seeds
in both cases.**

Target = **DPCC K20 `aw10` `dpcc-c-tightened` (C14, n=2): S&C 1.000 · 70.13 steps · 38.53 s/ep.**

| engine | trials | arch | config | S&C | steps | s/ep | Δ steps | Δ s/ep | ×faster |
|---|---|---|---|---|---|---|---|---|---|
| **MeanFlow-UNet** | **20** | ✅ | **K1 `dpcc-t-tight`** | **0.993** | **60.99** | **1.10** | **−9.15 `[−15.76,−3.15]`** `*` | **−37.43 `[−40.33,−34.46]`** `*` | **35.0** |
| MeanFlow-UNet | 20 | ✅ | K2 `dpcc-t-tight` | 0.993 | 60.41 | 1.64 | **−9.72 `[−16.33,−3.63]`** `*` | **−36.89 `[−39.79,−33.92]`** `*` | 23.5 |
| MeanFlow-UNet | 20 | ✅ | K1 `hardflow-tight` | 0.950 | 63.40 | 2.66 | **−6.73 `[−13.60,−0.53]`** `*` | **−35.88 `[−38.76,−32.91]`** `*` | 14.5 |
| MeanFlow-UNet | 20 | ✅ | K5 `dpcc-t-tight` | 0.980 | 60.84 | 13.71 | **−9.30 `[−15.87,−3.27]`** `*` | **−24.82 `[−27.72,−21.84]`** `*` | 2.8 |
| AlphaFlow-SiT | 2 | ❌ | K1 `dpcc-t-tight` | 1.000 | 66.10 | 0.92 | −4.03 | **−37.61** `*` | 41.7 |
| MeanFlow-DiT | 2 | ❌ | K1 `dpcc-t-tight` | 1.000 | 76.90 | 2.25 | +6.77 | **−36.29** `*` | 17.2 |
| naive FM | 2 | ✅ | K20 `dpcc-c-tight` | 1.000 | 63.23 | 29.65 | **−6.90** `*` | **−8.88** `*` | 1.3 |

**Gate status of the best MeanFlow-UNet row.** S&C 0.993 < 1.000, so it **fails the gate on the
point estimate** — by 2 episodes in 300. But `ΔS&C = −0.01 [−0.02, +0.00]` **includes zero**: at
5 seeds it is *not* statistically distinguishable from the Target on safety, while beating it
significantly on **both** other axes (−9.15 steps, −37.43 s/ep).

This is a **better position than the 2026-08-11 headline**, which was a one-axis win by an arm
(`hardflow` K1, 2.64 s/ep) that the 20-trial data now scores at 0.950:

| | old headline (n=2) | new best (n=20) |
|---|---|---|
| arm | K1 `hardflow-tightened` | K1 `dpcc-t-tightened` |
| S&C | 1.000 *(unverified — §1.4)* | 0.993 (300 episodes) |
| s/ep | 2.64 → **14.6×** | **1.10 → 35.0×** |
| steps | −6.37, ns | **−9.15 `*`** |
| axes won | 1 (time) | **2 (time + steps)** |

🔴 **Do not compare the S&C column across trial counts.** `AlphaFlow-SiT K1 = 1.000` and
`MeanFlow-UNet K1 = 0.993` are not a ranking — the first is 30 episodes, the second 300. Given §1.4,
AlphaFlow's 1.000 would likely fall to ~0.95–0.99 at 20 trials. **The comparison is unresolved until
the baselines are re-run at n = 20.**

---

## 4. Actions

1. **Re-run the baselines and the other engines at `n_trials = 20`.** Until then the grand table
   mixes 300-episode and 30-episode rows and no S&C ranking is defensible. Priority: Target (C14),
   AlphaFlow-SiT K1, MeanFlow-DiT K1.
2. **Re-run K10 and K20 at n = 20 with a longer wall-clock or split by halfspace** — both hit the
   24 h limit (§0). K20 has no usable 20-trial data at all.
3. **Retire the HardFlow K1 headline.** §2.1 shows `dpcc-t-tightened` dominates it on all three
   axes at 300 episodes. The arm-C claim needs restating from scratch.
4. **Update the curated snapshot** (`Data_Analysis/DA_Result_Curated_MD/`) — its headline row is the
   `hardflow` K1 result that no longer holds. Supersede rather than edit.
5. **Treat every `n=2` S&C of exactly 1.000 in prior DAs as unverified** (§1.4), including the
   Target's own.
