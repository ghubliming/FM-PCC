# DA — HardFlow (arm C) vs the DPCC projector (arm B) on visual aligning

**Question:** across every candidate in the visual-aligning tree, how does the HardFlow in-loop
sampler behave against the DPCC projector — better or worse?

**Date:** 2026-08-23 · **Batch:** `batch_va2_20260823_135156` (DA_VA_v2, 18 candidates / 321 units,
full scan of `logs/aligning-d3il-visual/plans` + 2 baseline bridges)
**Companion:** [`Gen14/U8/DA_20260823_Gen14_U8_mf_dit_visual_aligning.md`](../U8/DA_20260823_Gen14_U8_mf_dit_visual_aligning.md)

**Short answer: neither a game changer nor clearly better — mostly indistinguishable from noise at
this sample size.** Direction is consistent (HardFlow wins 9 of 12 matched pairings on violations,
across two independent engines) but **0 of those 12 clear p < 0.05** on a paired test. The single
result that survives multiple-comparison correction is a *distance* improvement, not a constraint
one: `mf` tightened with min-cost selection, −0.067 m at dz = −0.69, p = 1e-4. Cost is worse by a
flat ~3.3× on every pairing, and the one arm-B configuration that matters most — `dpcc-t` plus
tightening — reaches **zero** violations at a third of the price, which HardFlow never matches.
**As configured, HardFlow does not earn its cost on visual aligning; the comparison the benchmark
hierarchy actually asks for (a lower projection threshold) has never been run.**

---

## 1. Coverage: arm C exists on 2 of 18 candidates

"All the candidates" is a short list. Scanning every rollout in the batch:

| candidate | model | arm C? |
|---|---|---|
| **6** | `af` U-Net FiLM v1, K=2 | ✅ `hardflow_new-{c,r,t}` |
| **14** | `mf` U-Net FiLM v1, K=2 | ✅ `hardflow_new-{c,r,t}` |
| 12 | `mf` **DiT** (`_Bdit_`, 80 k) | ❌ |
| 15 | `mf` U-Net FiLM v2 | ❌ |
| 5, 7, 13 | `af`/`mf` other K, other FiLM | ❌ |
| 1–4, 10, 11 | `fm` (Gen7 + Gen14) | ❌ |
| 8, 9, 16 | `diffusion` / Gen9 visual DPCC | ❌ — refused by design¹ |
| 17, 18 | d3il baselines | ❌ n/a |

¹ `ENGINE_INIT_NOISE = {'fm': 0.5, 'mf': 1.0, 'af': 1.0}` — `diffusion` is deliberately absent
(`hardflow_projection.py:501-506`); a DDPM reverse chain has no velocity field to integrate, so
`resolve_engine_hf` refuses it. That exclusion is correct, not a gap.

The real gap is **`fm` and the DiT**: both are legal HardFlow hosts and neither has a single arm-C
cell. Arm C is opt-in and off by default —

```yaml
# config/visual_aligning_eval.yaml:433
hardflow_variants: []
# hardflow_variants: ['hardflow_new-r', 'hardflow_new-c', 'hardflow_new-t']   # ← line 434
```

— read via `HFFM_VARIANTS` first (`eval_mix_visual_aligning.py:2455-2458`), so filling either gap is
eval-only, no retraining:

```bash
HFFM_VARIANTS="hardflow_new-r hardflow_new-c hardflow_new-t" \
  sbatch Slurm_Codes/sbatch/mix_visual_aligning/eval_mix_visual_aligning.sh mf 6
```

**Everything below therefore rests on two candidates.** That they are different engines (`mf` and
`af`) sharing one bone is what makes the agreement between them meaningful.

## 2. 🔴 Parity check first — this batch is B4, not B1

The failure mode that invalidated the state-MeanFlow arm-C comparison was **candidate-fan
mismatch**: arm C running at fan 1 while arms A/B ran at fan 4, making arm C look ~5× cheaper
because it was doing ~5× less work. Checked before anything else here, from `run_config.csv`:

| candidate | arm A | arm B | **arm C** |
|---|---|---|---|
| 6 (`af`) | `mpc_batch_size` 4.0 | 4.0 | **4.0** |
| 14 (`mf`) | 4.0 | 4.0 | **4.0** |

**Parity holds. Arm C ran at fan 4 on both candidates, matching arms A and B.** This is consistent
with the code — `resolve_hf_batch_size` returns `max(1, configured_batch)` for `-r`/`-c`/`-t` and
pins only the *bare* `hardflow_new` name to 1 (`hardflow_projection.py:486-488`) — and the bare
variant was never run here, so no batch-1 cell exists in this batch at all.

Both arms also ran at the same run-level `diffusion_timestep_threshold = 0.5`. **That matters for
§8.**

The cost numbers below are therefore like-for-like and can be compared directly.

## 3. `mf` (candidate 14) — matched selection rule, arm B → arm C

Δ is C − B; **negative = HardFlow better** for distance/violations/depth, positive = better for
sat-rate/collision-free. n = 30 per cell.

### `combined_5` (untightened)

| metric | B-c | **C-c** | Δ | B-r | **C-r** | Δ | B-t | **C-t** | Δ |
|---|---|---|---|---|---|---|---|---|---|
| dist (task) | 0.4094 | **0.3249** | **−0.0845** | 0.3423 | **0.3064** | −0.0359 | **0.2867** | 0.3074 | +0.0206 |
| dist (xy) | 0.4332 | **0.2773** | **−0.1559** | 0.2492 | 0.2472 | −0.0021 | 0.2512 | **0.2317** | −0.0195 |
| violations | 69.5 | **60.3** | −9.2 | 58.4 | **49.6** | −8.7 | 66.2 | **40.2** | **−26.1** |
| sat rate | 0.826 | **0.849** | +0.023 | 0.854 | **0.876** | +0.022 | 0.834 | **0.900** | **+0.065** |
| zero-viol / collision-free | 0.233 | **0.267** | +0.033 | **0.367** | 0.267 | −0.100 | 0.267 | **0.500** | **+0.233** |
| max viol depth (m) | 0.0610 | **0.0209** | −0.0402 | **0.0177** | 0.0361 | +0.0184 | 0.0494 | **0.0400** | −0.0094 |
| **avg ms** | **56.9** | 194.0 | **+137.0** | **55.5** | 181.9 | +126.3 | **52.8** | 174.9 | +122.1 |

### `combined_5-tightened`

| metric | B-c | **C-c** | Δ | B-r | **C-r** | Δ | B-t | **C-t** | Δ |
|---|---|---|---|---|---|---|---|---|---|
| dist (task) | 0.3626 | **0.2959** | **−0.0667** | 0.3145 | **0.3117** | −0.0027 | **0.3066** | 0.3112 | +0.0045 |
| violations | 12.6 | **2.2** | **−10.4** | 4.5 | **3.4** | −1.0 | **0.0** | 0.3 | +0.3 |
| sat rate | 0.9685 | **0.9945** | +0.026 | 0.9888 | **0.9914** | +0.003 | **1.0000** | 0.9992 | −0.001 |
| zero-viol / collision-free | 0.733 | **0.867** | +0.133 | 0.900 | **0.933** | +0.033 | **1.000** | 0.967 | −0.033 |
| max viol depth (m) | 0.0411 | **0.0097** | −0.0314 | 0.0097 | 0.0097 | −0.0000 | **0.0000** | 0.0020 | +0.0020 |
| **avg ms** | **54.9** | 148.3 | +93.4 | **42.3** | 147.2 | +105.0 | **42.3** | 145.6 | +103.2 |

## 4. `af` (candidate 6) — the independent replication

### `combined_5` (untightened)

| metric | B-c | **C-c** | Δ | B-r | **C-r** | Δ | B-t | **C-t** | Δ |
|---|---|---|---|---|---|---|---|---|---|
| dist (task) | **0.3501** | 0.3553 | +0.0052 | 0.4220 | **0.3316** | **−0.0904** | **0.3378** | 0.3434 | +0.0056 |
| violations | 85.6 | **57.2** | **−28.4** | 34.7 | **31.4** | −3.3 | **52.6** | 53.8 | +1.2 |
| sat rate | 0.786 | **0.857** | **+0.071** | 0.913 | **0.922** | +0.009 | **0.869** | 0.866 | −0.003 |
| zero-viol / collision-free | 0.167 | **0.333** | +0.167 | 0.367 | **0.533** | +0.167 | **0.400** | 0.300 | −0.100 |
| **avg ms** | **55.4** | 192.0 | +136.6 | **52.7** | 177.5 | +124.8 | **53.4** | 185.7 | +132.4 |

### `combined_5-tightened`

| metric | B-c | **C-c** | Δ | B-r | **C-r** | Δ | B-t | **C-t** | Δ |
|---|---|---|---|---|---|---|---|---|---|
| dist (task) | 0.3597 | **0.3128** | −0.0469 | 0.3866 | **0.3288** | −0.0578 | **0.3486** | 0.3649 | +0.0163 |
| violations | 18.6 | **4.5** | **−14.1** | 14.9 | **3.1** | **−11.8** | **1.1** | 4.9 | +3.9 |
| sat rate | 0.953 | **0.989** | +0.035 | 0.963 | **0.992** | +0.030 | **0.997** | 0.988 | −0.010 |
| max viol depth (m) | 0.0110 | **0.0021** | −0.0089 | 0.0161 | **0.0015** | −0.0146 | **0.0008** | 0.0335 | +0.0327 |
| **avg ms** | **49.1** | 158.5 | +109.4 | **48.6** | 154.4 | +105.9 | **42.6** | 156.4 | +113.8 |

## 5. The pattern: HardFlow rescues weak selection and loses to strong selection

Tallying all **12 matched pairings** (2 candidates × 2 geometries × 3 selection rules):

| metric | HardFlow wins | HardFlow loses | where the losses are |
|---|---|---|---|
| violations | **9 / 12** | 3 | **all three are `-t`** |
| exec sat rate | **9 / 12** | 3 | **all three are `-t`** |
| distance (task) | 7 / 12 | 5 | **all four `-t`**, plus `af`-`c` untightened |
| avg ms | **0 / 12** | 12 | everywhere, 2.7–3.7× |
| success / S&C | — | — | pure noise (0–2 episodes of 30) |

Broken out by selection rule, the structure is unmistakable:

| rule | what it does | HardFlow's effect |
|---|---|---|
| `-c` min-projection-cost | pick candidate needing least correction | **largest gains** — dist −0.085/−0.047, viol −9 to −28 |
| `-r` random (index 0) | no selection at all | **consistent gains** — dist 4/4, viol 4/4 |
| `-t` temporal consistency | pick candidate closest to previous plan | **loses on every metric, both engines, both geometries** |

**HardFlow's advantage is inversely proportional to how good arm B's candidate selection already
is.** Where DPCC picks badly (`-c` is actively harmful for the U-Net — see the companion DA §3.4) or
not at all (`-r`), enforcing constraints *during* integration recovers most of the loss. Where DPCC
already picks well (`-t`), in-loop enforcement adds nothing and costs accuracy.

Two independent engines agreeing on this — `mf` and `af`, trained separately — is the strongest
claim in this document. Everything else here rests on n = 30 single-seed cells; this pattern holds
across 24 of them.

## 6. Best-of-arm: what you would actually deploy

Nobody ships a selection rule they know is worse. Comparing the *best* arm-B cell against the *best*
arm-C cell:

| engine | geo | metric | best arm B | best arm C | winner |
|---|---|---|---|---|---|
| `mf` | `combined_5` | distance | **0.2867** (`-t`) | 0.3064 (`-r`) | **B** |
| `mf` | `combined_5` | violations | 58.4 (`-r`) | **40.2** (`-t`) | **C** |
| `mf` | `-tightened` | distance | 0.3066 (`-t`) | **0.2959** (`-c`) | **C** |
| `mf` | `-tightened` | violations | **0.0000** (`-t`) | 0.3333 (`-t`) | **B** |
| `af` | `combined_5` | distance | 0.3378 (`-t`) | **0.3316** (`-r`) | **C** |
| `af` | `combined_5` | violations | 34.7 (`-r`) | **31.4** (`-r`) | **C** |
| `af` | `-tightened` | distance | 0.3486 (`-t`) | **0.3128** (`-c`) | **C** |
| `af` | `-tightened` | violations | **1.0667** (`-t`) | 3.1333 (`-r`) | **B** |
| both | all | **avg ms** | **42–53 ms** | 146–178 ms | **B, 4/4** |

Best-of-arm C wins 3/4 on distance and 2/4 on violations — **but the margins are 0.006–0.036 m and
3–18 violations, against a flat 3.3× cost penalty.**

🔴 **The decisive row: `dpcc-t` + tightening reaches 0.0000 violations on `mf` and 1.07 on `af`.**
HardFlow cannot beat zero, and it does not get there — it lands at 0.33 and 3.13 respectively, for
3.4× the compute. **Tightening plus a good selection rule is the cheaper solution to the problem
HardFlow was brought in to solve.**

## 7. Cost

Every arm-C pairing is slower, with no exceptions:

| engine / geo | arm B range | arm C range | multiplier |
|---|---|---|---|
| `mf` `combined_5` | 52.8–56.9 ms | 174.9–194.0 ms | 3.28–3.41× |
| `mf` `-tightened` | 42.3–54.9 ms | 145.6–148.3 ms | 2.70–3.48× |
| `af` `combined_5` | 52.7–55.4 ms | 177.5–192.0 ms | 3.37–3.48× |
| `af` `-tightened` | 42.6–49.1 ms | 154.4–158.5 ms | 3.18–3.67× |

**Consistently ~3.3×, engine-independent and geometry-independent.** Tightening reduces arm C's
absolute cost (~180 → ~150 ms) but not the ratio.

⚠️ **No NFE instrumentation in this batch.** Unlike the Gen16 avoiding runs — which log `nfe_total`,
`nlp_solves` and `nlp_failures` — the aligning tree carries none of it. So the 3.3× is a wall-clock
observation with **no mechanistic breakdown**: we cannot say how much is extra NFE, how much is NLP
solves, or how many solves failed and fell back. That instrumentation should be ported from Gen16
before arm C is tuned on this task.

## 8. The claim the hierarchy asks for has not been tested

Per the benchmark hierarchy, HardFlow must beat the DPCC projector **at a lower projection
threshold** — that is the specific claim that would justify an in-loop constrained sampler. §2
established that both arms ran at the same run-level `diffusion_timestep_threshold = 0.5`.

So what this DA measures is *"HardFlow vs DPCC, threshold held equal"* — a cost-for-constraint-quality
trade — and **not** the hierarchy's question. Same verdict Gen16 reached on avoiding. The companion
DA's §3.5 is directly relevant: arm B's threshold response has a clear optimum near `dt = 0.5` and
collapses above 1.0, and the useful range below 1.0 currently has only two samples in it. **Sweeping
`dt` on both arms together is the experiment that would settle this**, and it is eval-only.

## 9. Gaps and confounds

1. **Two candidates.** Both U-Net FiLM v1 at K=2. Two engines is a genuine replication; two bones
   would be better, and there is only one.
2. **`fm` and the DiT have no arm C** (§1) — both are legal hosts, both are eval-only to add. The DiT
   case is the more interesting: `-c` is its *best* arm-B rule (companion DA §3.4) and `-c` is where
   HardFlow's gains are largest (§5), so the DiT is the single most likely place for arm C to look
   good, and it has never been run there.
3. **No NFE / NLP-solve instrumentation** (§7).
4. **One seed (6), n = 30/cell, train split only.** Success is at the noise floor throughout and
   carries no information; distance and violations are the only usable axes.
5. **Threshold held equal** (§8) — the untested claim.
6. The `-t` losses are small in absolute terms on tightened geometries (0.0 → 0.33 violations) but
   they are consistent in *sign* across both engines and both geometries, which is what makes them
   worth reporting rather than dismissing as noise.

## 10. How big is it really? Significance testing

The tallies in §5 count *directions*, not effects. Rollouts are paired by context — `rollout_idx`
maps to the same starting configuration across every variant (verified: initial box→target distance
matches to 1e-9 across all 30 shared indices) — so a paired sign-flip permutation test on the
per-context differences is available, and it is far more powerful than comparing cell means.

50 000 permutations per pairing, two-sided. `dz` = mean difference / sd of differences.
Negative = HardFlow better. `*` p < 0.05, `.` p < 0.10.

### Violations

| engine / geo | rule | B | C | Δ | dz | p |
|---|---|---|---|---|---|---|
| `mf` untightened | `-c` | 69.50 | 60.30 | −9.20 | −0.14 | 0.470 |
| `mf` untightened | `-r` | 58.37 | 49.63 | −8.73 | −0.16 | 0.394 |
| `mf` untightened | `-t` | 66.23 | 40.17 | **−26.07** | −0.37 | 0.057 `.` |
| `mf` tightened | `-c` | 12.60 | 2.20 | −10.40 | −0.21 | 0.281 |
| `mf` tightened | `-r` | 4.47 | 3.43 | −1.03 | −0.04 | 0.814 |
| `mf` tightened | `-t` | **0.00** | 0.33 | +0.33 | +0.19 | 1.000 |
| `af` untightened | `-c` | 85.63 | 57.23 | **−28.40** | −0.36 | 0.062 `.` |
| `af` untightened | `-r` | 34.70 | 31.37 | −3.33 | −0.06 | 0.751 |
| `af` untightened | `-t` | 52.60 | 53.80 | +1.20 | +0.02 | 0.935 |
| `af` tightened | `-c` | 18.63 | 4.50 | −14.13 | −0.29 | 0.097 `.` |
| `af` tightened | `-r` | 14.93 | 3.13 | −11.80 | −0.27 | 0.140 |
| `af` tightened | `-t` | 1.07 | 4.93 | +3.87 | +0.22 | 0.344 |

**0 of 12 significant at p < 0.05.** Three reach p < 0.10, all in HardFlow's favour. Every effect
size is small (|dz| ≤ 0.37).

### Distance (task metric)

| engine / geo | rule | B | C | Δ | dz | p |
|---|---|---|---|---|---|---|
| `mf` untightened | `-c` | 0.409 | 0.325 | −0.084 | −0.22 | 0.296 |
| `mf` untightened | `-r` | 0.342 | 0.306 | −0.036 | −0.18 | 0.349 |
| `mf` untightened | `-t` | 0.287 | 0.307 | +0.021 | +0.15 | 0.439 |
| **`mf` tightened** | **`-c`** | **0.363** | **0.296** | **−0.067** | **−0.69** | **0.0001** `*` |
| `mf` tightened | `-r` | 0.314 | 0.312 | −0.003 | −0.02 | 0.923 |
| `mf` tightened | `-t` | 0.307 | 0.311 | +0.005 | +0.03 | 0.862 |
| `af` untightened | `-c` | 0.350 | 0.355 | +0.005 | +0.04 | 0.815 |
| **`af` untightened** | **`-r`** | **0.422** | **0.332** | **−0.090** | **−0.43** | **0.020** `*` |
| `af` untightened | `-t` | 0.338 | 0.343 | +0.006 | +0.04 | 0.851 |
| `af` tightened | `-c` | 0.360 | 0.313 | −0.047 | −0.37 | 0.055 `.` |
| `af` tightened | `-r` | 0.387 | 0.329 | −0.058 | −0.33 | 0.082 `.` |
| `af` tightened | `-t` | 0.349 | 0.365 | +0.016 | +0.08 | 0.731 |

**2 of 12 significant.** Both favour HardFlow, both on `-c`/`-r`, and two more reach p < 0.10 (also
`-c`/`-r`). With 24 tests total, ~1.2 false positives are expected at α = 0.05 — but **`mf` tightened
`-c` (dz = −0.69, p = 0.0001) survives Bonferroni correction** (α/24 = 0.002) and is the only result
in this document that does.

### What this does to §5

The 9/12 and 7/12 tallies remain the correct description of *direction*, and direction consistency is
not nothing: the `-t`-loses / `-c`-and-`-r`-win structure reproduces across two independently trained
engines. But a sign test over 12 non-independent pairings gives p ≈ 0.15, so **the pattern is
suggestive, not established.**

Two corrections to the emphasis this DA carried before this section existed:

1. **The constraint advantage — which §11 originally called "the real result" — does not clear
   significance anywhere.** The eye-catching relative drops (`-c` tightened: 12.6 → 2.2 = −83 %,
   18.6 → 4.5 = −76 %) are large in percentage terms and small in effect size, because violation
   counts are heavily skewed with variance far exceeding the shift. They are real directions with
   weak evidence, not demonstrated wins.
2. **The distance axis — originally dismissed as "a wash" — carries the only statistically solid
   finding in the document.** HardFlow with min-cost selection on tightened `mf` geometry improves
   final distance by 0.067 m at dz = −0.69, p = 1e-4. That is a medium effect and it is not fragile.

The corrected reading: **HardFlow's demonstrated benefit on visual aligning is a
distance improvement under weak candidate selection, not the constraint improvement it was adopted
for.**

## 11. Verdict — better or worse?

**Better at what it is for; worse at what it costs; and beaten by the cheap alternative.**

* **Constraint satisfaction — directionally better, statistically unproven.** 9/12 on violations,
  9/12 on sat rate, with the largest drops on `-c` (69.5 → 60.3 and 85.6 → 57.2 untightened;
  12.6 → 2.2 and 18.6 → 4.5 tightened) and max violation depth falling with them (0.061 → 0.021 m,
  0.041 → 0.010 m). But **0 of 12 clear p < 0.05 paired** (§10), and every effect size is small.
  Consistent direction across two independent engines; not a demonstrated win.
* **Task distance — the only solid finding, and it is HardFlow's.** `mf` tightened `-c` improves by
  0.067 m at **dz = −0.69, p = 0.0001**, the one result here that survives Bonferroni over all 24
  tests. `af` untightened `-r` is also significant (dz = −0.43, p = 0.020). Both are `-c`/`-r`, both
  favour HardFlow. Elsewhere on distance it is genuinely a wash.
* **Cost — worse, unambiguously.** 3.3× on all 12 pairings.
* **Against the configuration that matters — worse.** `dpcc-t` + tightening hits **zero** violations
  on `mf` at 42 ms. HardFlow hits 0.33 at 146 ms. There is nothing left for it to win.

**Recommendation: do not adopt arm C for visual aligning as currently configured, and do not report
it as a win.** Its measured benefit is concentrated on selection rules nobody should deploy, and the
one deployable arm-B configuration already saturates the constraint metric at a third of the cost.

That verdict is conditional on §8: it applies to *this* threshold. The reason HardFlow exists is the
hypothesis that it holds constraints at thresholds where the DPCC projector cannot, and that
hypothesis remains untested on this task.

## 12. Next, in order

1. **Joint `dt` sweep across arms B and C** (0.1 / 0.25 / 0.5 / 0.75 / 1.0). Eval-only. This is the
   only experiment that can convert §10 from "not worth it here" into an actual answer to the
   hierarchy's question (§8).
2. **Run arm C on the DiT** (candidate 12). Eval-only, one command (§1). §9.2 argues it is the most
   likely place for arm C to show an advantage, and it is currently blank.
3. **Port the Gen16 NFE/NLP instrumentation** into the aligning eval so the 3.3× can be attributed
   (§7).
4. **Run arm C on `fm`** — the third legal host, also blank.
5. **Seeds 7–10.** §10 makes this structural rather than cosmetic: at n = 30 single-seed, effects of
   |dz| ≈ 0.3 are undetectable, and that is the size of most of what is being argued about here.
   Four more seeds is the cheapest way to turn the §5 direction pattern into evidence — or to
   retire it.
6. A test split, once 1–3 have narrowed what is worth measuring.

## 13. One-line summary

**Across the only two visual-aligning candidates that have it (`mf` and `af`, both U-Net FiLM v1
K=2, fan-parity verified at 4), HardFlow wins 9 of 12 matched pairings on constraint violations and
9 of 12 on satisfaction rate — every loss on `-t`, every large win on `-c`, so its benefit scales
inversely with how well arm B already selects candidates — but paired significance testing puts
**0 of those 12 below p < 0.05**, leaving one Bonferroni-surviving result in the whole document and
it is on distance rather than constraints (`mf` tightened `-c`, dz = −0.69, p = 1e-4); at a flat
3.3× cost, never reaching the zero violations `dpcc-t` plus tightening achieves at 42 ms, it is not
a game changer and not worth adopting on this task as configured — and the threshold sweep the
benchmark hierarchy actually asks for has never been run.**
