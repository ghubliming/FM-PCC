# Gen3v7 U2 — first `sit` run analysis (α-Flow on its own SiT backbone), `temp/2807`

**Run:** `temp/2807/2807`, seed 6, n=2 trials × 3 halfspaces. Two arms trained 100 k steps + eval:
- **SIT (Gen3v7 α-Flow)** — `bbsit`, α annealed 1→0 (`ai1.0_ae0.0_ag25.0_rf0.5`), jobs 23929 (train) / 23930 (eval). Eval swept **K ∈ {1,2,5,10}**.
- **MF_DIT (Gen3v6 MeanFlow)** — `bbmf_dit`, jobs 23926 / 23927. Eval **K=2 only** (comparison arm).

Both are the U2 defaults just set (`imf_backbone='sit'` / `'mf_dit'`), i.e. each objective now runs on **its own paper's network**.

## TL;DR

1. **α-Flow's own SiT trains and works.** `per_dim_rms_u` **0.356** (val), goal-reach SR ≈ 1.0 almost everywhere, DPCC-projected safety airtight. The U2 SiT port is functional — this is the first α-Flow result on a faithful backbone.
2. **The α-homotopy delivers its promised stability.** Against Gen3v6 pure-MeanFlow (`mf_dit`) trained under identical settings, **SIT's training is dramatically more controlled**: peak `raw_mse_u` **84 vs 1900**, peak α=0 loss **1.5 vs 70**, final grad-norm **60 vs 527**. Annealing from FM (α=1) tames the MeanFlow blind-direction blow-up — exactly the α-Flow thesis.
3. **But the raw generative field is still not good enough alone**, and the **few-NFE bucket is still unstable** (b3 spikes to 374 at the α→0 endpoint; eval K-scaling is non-monotonic). DPCC does the heavy lifting, as in every prior generation.
4. **Matched-budget (K=2) head-to-head:** SIT ≥ MF_DIT on the safety-critical *tightened* arms and the raw arms — but the margins are **within seed-6/n=2 noise** (0.17 granularity). The trustworthy signal here is the training curves, not the eval deltas.

## 1. Training — the decisive, trustworthy signal (full 100 k, both arms)

| metric (val, final) | **SIT (α-Flow)** | MF_DIT (MeanFlow) | note |
|---|---|---|---|
| `per_dim_rms_u` (last) | **0.356** | 0.447 | SIT cleaner final field |
| `per_dim_rms_u` (**max**) | **1.11** | **1.73** | MF overshoots > data scale |
| `raw_mse_u` (**max**) | **84** | **1900** | MF transiently diverges ~23× |
| `h_mse_b1` (max) | **69** | **4530** | |
| `h_mse_b2` (max) | **390** | **5710** | |
| `h_mse_b3` (max) | 374 | 12500 | MF's worst transient is catastrophic |
| α=0 branch loss (max) | **1.53** | **69.8** | the MeanFlow objective alone blows up |
| grad-norm (final / mean-last-10) | **60 / 127** | 527 / 216 | MF never settles |

**Read:** both objectives *end* near the same `per_dim_rms_u` (0.36 vs 0.45), but the **path** is completely different. Pure MeanFlow (`mf_dit`) throws its raw field to `raw_mse ≈ 1900` and its α=0 loss to ~70 mid-training and only claws back by step 99 k; α-Flow's schedule keeps every quantity **1–2 orders of magnitude smaller** throughout. This is the clearest evidence yet that the α-anneal is doing real work, not cosmetics.

**The one place SIT is worse: the hardest few-NFE bucket at the very end.** `h_mse_b3` (h∈[0.6,1.0], i.e. 1–2 step jumps) **spikes to 374 exactly at the final step** — the α→0 endpoint reintroduces the MeanFlow blind-direction instability in the largest-interval bucket (MF's b3 had recovered to ~20 by then). This is the training-side fingerprint of the eval K-instability below. (Eval loads `diffusion_epoch='best'`, so the deployed checkpoint may predate the spike — but the instability is real and recurring, same defect flagged in the `init/` insight.)

## 2. Eval — SIT across K (seed 6, SC = success **and** constraints, mean/3 halfspaces)

| variant | K1 | K2 | K5 | K10 | viol (K2) | time ms (K2) |
|---|---|---|---|---|---|---|
| **dpcc-r-tightened** | **1.00** | **1.00** | **1.00** | **1.00** | 0 | 20 |
| **post_processing-tightened** | **1.00** | **1.00** | **1.00** | **1.00** | 0 | 20 |
| dpcc-c-tightened | 1.00 | **0.00**† | 1.00 | 1.00 | 0 | 19 |
| dpcc-t-tightened | 1.00 | 0.83 | 1.00 | 0.83 | 0 | 21 |
| dpcc-t | 1.00 | 0.33 | 0.83 | 0.83 | 0.03 | 20 |
| dpcc-c | 0.83 | **0.00**† | 0.67 | 1.00 | 0 | 20 |
| dpcc-r / post_processing | 0.50 | 0.50 | 0.33 | 0.00 | 0.06 | 81 |
| model_free (raw) | 0.17 | 0.50 | 0.17 | 0.50 | 1.6 | 19 |
| diffuser (raw ODE) | 0.17 | 0.33 | 0.33 | 0.33 | 1.5 | 12 |

**Goal-only SR ≈ 1.00 across the board** (the model reaches the target; it's *constraints* that need projection).

Three takeaways:
- **DPCC's "physical brakes" are airtight with SIT.** `dpcc-r-tightened` and `post_processing-tightened` are **SC 1.00 at every K with zero violation.** Safety does not depend on raw quality.
- **The raw field is weak on its own.** `model_free`/`diffuser` sit at SC 0.17–0.50 with violations of 1.5–5.6 — the generative brain alone does not satisfy constraints (unchanged story from Gen3v4/v6).
- **K-scaling is non-monotonic and sometimes *inverted*** — `dpcc-r`/`post_processing` **degrade** 0.50→0.00 as K goes 1→10, and `model_free` bounces 0.17↔0.50. More flow steps do not help and can hurt, precisely the few-NFE `b3` instability from §1.

† **`dpcc-c` collapses to 0.00 at K=2 for *both* SIT and MF_DIT** (1.00 at K=1/5/10). Because it is backbone-independent, this is a **K=2 / `dpcc-c` eval artifact, not a Gen3v7 defect** — flag for the projection code, not this generation.

## 3. Matched-budget head-to-head @ K=2 (SIT vs MF_DIT, SC)

| variant | **SIT** | MF_DIT | | variant | **SIT** | MF_DIT |
|---|---|---|---|---|---|---|
| dpcc-r-tightened | **1.00** | 0.83 | | post_processing-tightened | **1.00** | 0.83 |
| dpcc-t-tightened | 0.83 | 0.83 | | dpcc-c-tightened | 0.00† | 0.00† |
| model_free | **0.50** | 0.33 | | diffuser | **0.33** | 0.17 |
| dpcc-r | 0.50 | **0.83** | | post_processing | 0.50 | **0.83** |
| dpcc-t | 0.33 | **0.50** | | | | |

SIT wins the **safety-critical tightened headline** (1.00 vs 0.83) and the **raw arms** (its cleaner field showing through, consistent with §1); MF wins some untightened DPCC arms. **All deltas are ≤ 2 episodes (0.33)** at n=2 — directionally consistent with the training story but **not individually significant.**

## 4. Caveats

- **seed 6, n=2** → 6 episodes/cell, **0.17 SC granularity**; eval differences under ~0.33 are noise. The **training curves (full 100 k)** are the reliable comparison; the eval merely fails to contradict them.
- **MF_DIT evaluated at K=2 only** — the head-to-head is single-budget; SIT's own K-sweep shows how sensitive these numbers are to K.
- Compute: SIT `dpcc-*-tightened` is cheap at K=1/2 (~14–20 ms) but jumps to **~190–400 ms at K=5/10** — the projection cost, not the network.

## 5. Verdict & next steps

**Verdict:** Gen3v7 U2 is a **success at the level it can claim**: α-Flow now runs on its own SiT, it trains, and it demonstrably **inherits the α-homotopy's stability advantage** over pure MeanFlow. It is **not** yet a planning win — the raw field is still projection-dependent, and the few-NFE `b3` instability persists at the α→0 endpoint.

**Next:**
1. **Fix the b3/α→0 endpoint spike** — cap the schedule short of α=0 (e.g. `ae` = 0.05–0.1) or slow the final anneal, so the FM anchor never fully vanishes in the hardest bucket; re-check `h_mse_b3` stays bounded to the last step.
2. **Scale to 5 seeds (6–10)** at **K=2 fixed** for a real matched-budget SIT-vs-MF_DIT number — the current edges need n≥30 episodes to survive.
3. **Run the endpoint-error diagnostic** (`endpoint_error_alphaflow.py`) on the `sit` checkpoint to separate interval-prediction error from terminal error, and confirm whether the K-inversion is generation or projection.
4. **Report `dpcc-c` K=2=0.00 to the projection code** as a backbone-independent artifact.
