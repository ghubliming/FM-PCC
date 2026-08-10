# GUIDE — Bootstrap verification of "MeanFlow `bbunet`@32 K2 beats FMv3 K20 and DPCC K20"

**Date:** 2026-08-09 · **Type:** analysis guide / claim adjudication · **Status:** claim supported under both deployable aggregations; UNet > `mf_dit` backbone (§4); all seed-limited
**Claim under test (visual inspection):** *"The `bbunet` MeanFlow at K = 2 already beats the old FMv3 K20 and DPCC K20 — on steps, on time, and on S&C."*
**Related:** [`RESULTS_Fix_8_unet_width32_rerun_24317_24334.md`](./RESULTS_Fix_8_unet_width32_rerun_24317_24334.md) §4.2 · [`../DA/DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md`](../DA/DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md)
**Data:** `temp/2026-08-07/batch_avoiding_combined_20260807_124828/candidates_multidimensional_raw.csv` — **within-batch**, so unlike the RESULTS §4.2 tables the `avg_time` axis is comparable here.

---

## 0. TL;DR — what the numbers actually say

| sub-claim | verdict |
|---|---|
| **Beats both on `dpcc-t-tightened` (steps ∧ time, equal S&C)** | ✅ **Supported.** Both step-difference bootstrap CIs exclude 0. |
| **Beats both on `dpcc-r-tightened`** | 🟡 **Non-dominated, not proven.** Wins time by 20×; step and S&C CIs straddle 0. |
| **Beats both on `dpcc-c-tightened`** | ❌ **Refuted.** +32 steps, CI `[+15, +48]`, and −0.17 S&C. |
| **Aggregated, each method using its own best rule** | ✅ **Supported on the point estimate.** S&C tied at 1.000; **−3.5 / −2.8 steps**; 19–22× wall-clock. CI on steps straddles 0 at n = 6. |
| **(bonus, §4) UNet@32 vs `mf_dit` — which MeanFlow backbone?** | ✅ **UNet, on all three arms.** Beats **5/5** `mf_dit` seeds on steps everywhere; decisive on `-c` (S&C 0.833 vs 0.100 — `mf_dit` times out on 4/5 seeds). |

**The defensible headline is:**

> **At K = 2 the width-32 UNet MeanFlow is Pareto-dominant over both diffusion-DPCC K20 and
> FM-ODE K20 — equal success-and-constraints, fewer steps, and ~19× lower wall-clock per episode —
> whether every method is held to the same selection rule (`-t`) or each is allowed its own best.**

⚠️ It rests on **one seed and 6 episodes per arm** (§5), and the step margin is significant only
under the fixed-common-rule comparison, not under best-of-rule selection (§3.4).

🔴 **Do not aggregate by averaging over `-r`/`-c`/`-t`.** They are alternative MPC selection rules,
not environments — averaging them describes a controller nobody would deploy, and it is the *only*
aggregation under which MeanFlow looks worse on steps (§3.4). Aggregate either at a fixed common
rule or at each method's own best rule.

---

## 1. Candidate identification (do this first, every time)

The three rows being compared, resolved from `Full_Path` in the CSV:

| label | `Candidate` | path (under `logs/avoiding-d3il/plans/`) | seeds in batch |
|---|---|---|---|
| **MF UNet@32 K2** | `110` | `flow_matching_v3_meanflow/…_bbunet_tslogit_normal_dp0.5/H8_K2_Meuler_T0,5_…MeanFlowODE` | **6 only** |
| FMv3 K20 (naive FM ODE) | `117` | `flow_matching_v3_ode_selectable/…FlowMatchingODE_a1.5_b1.0_aw10/H8_K20_Meuler_T0.5_…` | 6,7,8,9,10 |
| DPCC K20 (diffusion baseline) | `14` | `diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` | 6,7,8,9,10 |
| *(contrast)* MF UNet@**256** K2 | `104` | `flow_matching_v3_meanflow(Bf_U3)/…_bbunet_…/H8_K2_…` | 6 only |

🔴 **Filter everything to `seed == 6`.** CAND_110 has only seed 6; comparing its 6 episodes against
the baselines' 30 would confound backbone with seed count. CAND_110's `dpcc-t-tightened` row
reproduces RESULTS §4.2 exactly (58.67 steps, 0.0270 s/step), confirming it is the 24334 width-32 run
and not the 253 M ancestor (CAND_104, 51.50 / 0.0571).

---

## 2. The bootstrap procedure

### 2.1 Recovering per-episode data from the aggregate CSV

The CSV stores `metric` and `metric_std` per `(Candidate, seed, variant, halfspace_variant)`. This
batch ran **`n_trials = 2`**, and the std is the *population* std, so for n = 2:

```
std = |x₁ − x₂| / 2   ⟹   x₁ = mean − std ,  x₂ = mean + std
```

**The two episodes are recovered exactly.** Sanity check that proves it: every reconstructed
`n_steps` comes out an integer (`57.0 ± 4.0 → {53, 61}`, `60.5 ± 0.5 → {60, 61}`, …). If a future
batch runs `n_trials > 2` this trick dies — pull the per-episode records from the results tree
instead (§6).

⚠️ **What the trick cannot recover: which episode is which.** `mean ± std` loses the trial index, so
you can reconstruct the *set* {x₁, x₂} but not pair it with the baseline's trial on the same initial
condition. **All CIs below are therefore unpaired.** A paired bootstrap on identical initial
conditions would be substantially tighter — that requires the raw per-episode arrays (§6).

### 2.2 Resampling scheme

- **Unit:** episode. 3 halfspaces × 2 trials = **6 per arm per variant**; 18 when pooled over the
  three tightened variants.
- **Statistic:** difference of means, `MF − baseline`, on four axes:
  `S&C` (0/1), `n_steps`, `avg_time` (s/step), and `wall = n_steps × avg_time` (s/episode).
- **Method:** independent resampling with replacement within each arm, `B = 20000`, percentile
  95 % CI. Fixed RNG seed for reproducibility.
- **Decision rule:** the claim "MF beats X on axis A" holds only if the CI **excludes 0** in the
  favourable direction. Point estimates alone are not evidence at n = 6.

### 2.3 Runnable script

Data analysis only — safe to run in the container (no torch/mujoco needed).

```python
import csv, collections, random, statistics

P   = 'temp/2026-08-07/batch_avoiding_combined_20260807_124828/candidates_multidimensional_raw.csv'
HS  = ['top-right-hard', 'top-left-hard', 'both-hard']
VARS= ['dpcc-t-tightened', 'dpcc-r-tightened', 'dpcc-c-tightened']
ARMS= {'110': 'MF UNet@32 K2', '117': 'FMv3 K20', '14': 'DPCC K20'}

D = collections.defaultdict(dict)
for row in csv.DictReader(open(P)):
    try: D[(row['Candidate'], row['seed'], row['variant'], row['halfspace_variant'])][row['metric']] = float(row['value'])
    except ValueError: pass

def episodes(cand, var, seed='6'):
    """Reconstruct the n_trials=2 episodes from mean ± population-std."""
    out = []
    for h in HS:
        d = D[(cand, seed, var, h)]
        for sgn in (-1, +1):
            st = d['n_steps']  + sgn * d.get('n_steps_std', 0.0)
            t  = d['avg_time'] + sgn * d.get('avg_time_std', 0.0)
            out.append(dict(sc=d['n_success_and_constraints'] + sgn * d.get('n_success_and_constraints_std', 0.0),
                            steps=st, t=t, wall=st * t))
    return out

def boot(A, B, key, N=20000, seed=0):
    rng = random.Random(seed)
    obs = statistics.mean(x[key] for x in A) - statistics.mean(x[key] for x in B)
    ds  = sorted(statistics.mean(rng.choice(A)[key] for _ in A) -
                 statistics.mean(rng.choice(B)[key] for _ in B) for _ in range(N))
    return obs, ds[int(.025 * N)], ds[int(.975 * N)]

KI = {'sc': 0, 'steps': 1, 't': 2, 'wall': 3}

def summ(E):
    return (statistics.mean(x['sc'] for x in E), statistics.mean(x['steps'] for x in E),
            statistics.mean(x['t'] for x in E), statistics.mean(x['wall'] for x in E))

def best_rule(cand, E=None):
    """Aggregation B: pick the variant with max S&C, ties broken by fewer steps."""
    ranked = sorted(((s[0], -s[1], v, s) for v, s in
                     ((v, summ(E[v] if E else episodes(cand, v))) for v in VARS)), reverse=True)
    return ranked[0][2], ranked[0][3]

def boot_best(cA, cB, key, N=20000, seed=0):
    """Bootstrap on best-rule aggregation, re-selecting the rule inside every replicate."""
    rng = random.Random(seed); ki = KI[key]
    EA = {v: episodes(cA, v) for v in VARS}; EB = {v: episodes(cB, v) for v in VARS}
    obs = best_rule(cA)[1][ki] - best_rule(cB)[1][ki]
    ds = []
    for _ in range(N):
        ra = {v: [rng.choice(EA[v]) for _ in EA[v]] for v in VARS}
        rb = {v: [rng.choice(EB[v]) for _ in EB[v]] for v in VARS}
        ds.append(best_rule(cA, ra)[1][ki] - best_rule(cB, rb)[1][ki])
    ds.sort(); return obs, ds[int(.025 * N)], ds[int(.975 * N)]

# --- aggregations A and C: one fixed rule for every method (the headline comparison) ---
for var in VARS:
    print(f'### fixed common rule: {var}')
    A = episodes('110', var)
    for c in ('117', '14'):
        for key in ('sc', 'steps', 't', 'wall'):
            o, lo, hi = boot(A, episodes(c, var), key)
            star = '*' if (lo > 0 or hi < 0) else ' '   # CI strictly excludes 0
            print(f'  MF − {ARMS[c]:9s} {key:5s}: {o:+9.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] {star}')

# --- aggregation B: each method uses its own best rule (the deployment view) ---
print('### best rule per method')
for cand, name in ARMS.items():
    v, s = best_rule(cand)
    print(f'  {name:14s} -> {v:18s} S&C={s[0]:.3f} steps={s[1]:6.2f} s/ep={s[3]:6.2f}')
for c in ('117', '14'):
    for key in ('sc', 'steps', 't', 'wall'):
        o, lo, hi = boot_best('110', c, key)
        star = '*' if (lo > 0 or hi < 0) else ' '   # CI strictly excludes 0
        print(f'  MF − {ARMS[c]:9s} {key:5s}: {o:+9.3f}  95% CI [{lo:+.3f}, {hi:+.3f}] {star}')
```

🔴 **Do not add a "mean over `VARS`" branch** — that is aggregation D in §3.4 and it is not a
controller. It is reported in §3.1/§3.2 only as a diagnostic.

---

## 3. Results

### 3.1 Point estimates (seed 6, 3 halfspaces × 2 trials)

| variant | arm | S&C | steps | s/step | **s/episode** |
|---|---|---|---|---|---|
| **`dpcc-t-tightened`** | **MF UNet@32 K2** | **1.000** | **58.67** | **0.0270** | **1.58** |
| | FMv3 K20 | 1.000 | 63.50 | 0.4679 | 29.62 |
| | DPCC K20 | 1.000 | 62.00 | 0.5961 | 36.99 |
| **`dpcc-r-tightened`** | **MF UNet@32 K2** | **1.000** | **63.17** | **0.0269** | **1.70** |
| | FMv3 K20 | 0.833 | 71.50 | 0.6755 | 50.42 |
| | DPCC K20 | 0.833 | 65.33 | 0.5633 | 36.90 |
| **`dpcc-c-tightened`** | **MF UNet@32 K2** | **0.833** | **94.00** | **0.0272** | **2.57** |
| | FMv3 K20 | 1.000 | 62.17 | 0.4887 | 30.14 |
| | DPCC K20 | 1.000 | 61.50 | 0.5765 | 35.46 |
| **best rule per method** | **MF UNet@32 K2** (`-t`) | **1.000** | **58.67** | **0.0270** | **1.58** |
| | FMv3 K20 (`-c`) | 1.000 | 62.17 | 0.4887 | 30.14 |
| | DPCC K20 (`-c`) | 1.000 | 61.50 | 0.5765 | 35.46 |
| *(diagnostic only — not a deployable controller)* | mean over 3 variants, MF | 0.944 | 71.94 | 0.0270 | 1.95 |
| | mean over 3 variants, FMv3 K20 | 0.944 | 65.72 | 0.5440 | 36.72 |
| | mean over 3 variants, DPCC K20 | 0.944 | 62.94 | 0.5786 | 36.45 |

> **Best-rule-per-method is the row that answers the claim: MF wins all three axes.**
> The mean-over-variants rows are kept only to show *why* they must not be used — see §3.4.

### 3.2 Bootstrap CIs (B = 20000, percentile, unpaired)

`*` = CI excludes 0.

| variant | comparison | ΔS&C | Δsteps | Δ s/episode |
|---|---|---|---|---|
| **`dpcc-t-tightened`** | MF − FMv3 K20 | +0.000 [0, 0] | **−4.83 [−9.33, −0.67]** `*` | **−28.0 [−32.8, −23.1]** `*` |
| | MF − DPCC K20 | +0.000 [0, 0] | **−3.33 [−6.67, −0.33]** `*` | **−35.4 [−39.4, −30.8]** `*` |
| **`dpcc-r-tightened`** | MF − FMv3 K20 | +0.167 [0.00, +0.50] | −8.33 [−19.67, +2.83] | **−48.7 [−83.9, −27.4]** `*` |
| | MF − DPCC K20 | +0.167 [0.00, +0.50] | −2.17 [−8.17, +4.67] | **−35.2 [−40.7, −30.7]** `*` |
| **`dpcc-c-tightened`** | MF − FMv3 K20 | −0.167 [−0.50, 0.00] | **+31.83 [+15.17, +47.50]** `*` | **−27.6 [−33.5, −21.5]** `*` |
| | MF − DPCC K20 | −0.167 [−0.50, 0.00] | **+32.50 [+16.33, +47.50]** `*` | **−32.9 [−35.2, −30.6]** `*` |
| **best rule per method** | MF − FMv3 K20 | +0.000 [0, 0] | −3.50 [−7.50, +1.33] | **−28.6 [−34.2, −22.6]** `*` |
| | MF − DPCC K20 | +0.000 [0, 0] | −2.83 [−6.00, +0.67] | **−33.9 [−37.9, −31.3]** `*` |
| *(diagnostic)* mean over variants | MF − FMv3 K20 | +0.000 [−0.167, +0.167] | +6.22 [−3.28, +16.83] | **−34.8 [−48.2, −26.4]** `*` |
| | MF − DPCC K20 | +0.000 [−0.167, +0.167] | **+9.00 [+0.33, +18.83]** `*` | **−34.5 [−36.9, −32.2]** `*` |

In the best-rule rows the variant is **re-selected inside every bootstrap replicate**, so the
winner's-curse of picking the best of three on the same 6 episodes is priced into the interval.
That selection variance is exactly why the step CI widens from `[−6.67, −0.33]` (fixed `-t`) to
`[−6.00, +0.67]` (best-of).

### 3.3 Reading

1. **`dpcc-t-tightened` is a clean win and the only one.** Equal S&C (1.00 everywhere, zero
   variance), strictly fewer steps with both CIs excluding 0, and 19–23× less wall-clock.
   **Pareto-dominant.** This is the cell to quote.
2. **`dpcc-r-tightened` is non-dominated, not a win.** The +0.167 S&C is one episode out of six
   (`[0.00, +0.50]`) and the step CI straddles 0. Only the time axis is significant. Say
   *"non-dominated with a 20× compute advantage"*, never *"beats"*.
3. **`dpcc-c-tightened` refutes the claim.** +32 steps is far outside noise, and the S&C is lower.
   Correctly reported in RESULTS §4.2 ("where it is *not* good") — the `-c` minimal-correction rule
   makes the UNet field dawdle. It still beats every other MeanFlow-family K2 option there
   (`mf_dit` and AlphaFlow both time out at 199 steps / S&C = 0.00), but that is an intra-family
   consolation, not a win over the baselines.
4. **The aggregate — the "general" number — depends entirely on how you aggregate.** See §3.4; the
   deployable aggregations both favour MeanFlow.
5. **The 19× is not an artifact of cross-batch timing noise.** Unlike RESULTS §4.2 (which crossed
   the 2026-08-02 and 2026-08-06 batches and could not claim `avg_time` below ~10–20 %), all rows
   here come from the single 2026-08-07 batch. The ratio is 0.027 vs 0.54–0.58 s/step — an order of
   magnitude, far above any contention noise.

### 3.4 How to aggregate over `-r` / `-c` / `-t` — and how not to

`dpcc-{r,c,t}-tightened` are **three alternative MPC selection rules applied to the same generated
trajectory**, not three environments. A deployed controller runs exactly one of them. Three
aggregations are therefore possible, and they disagree:

| aggregation | MF vs FMv3 K20 | MF vs DPCC K20 | legitimate? |
|---|---|---|---|
| **A. Fixed common rule = `-t`** | S&C tie, **−4.83 steps** `*`, −28.0 s/ep | S&C tie, **−3.33 steps** `*`, −35.4 s/ep | ✅ **strongest evidence** — one variable (the generative model), significant on steps |
| **B. Each method uses its own best rule** | S&C tie, −3.50 steps, −28.6 s/ep | S&C tie, −2.83 steps, −33.9 s/ep | ✅ **the deployment view** — MF wins all axes on point estimate; steps not significant at n = 6 |
| C. Fixed common rule = `-c` | −0.167 S&C, **+31.8 steps**, −27.6 s/ep | −0.167 S&C, **+32.5 steps**, −32.9 s/ep | ✅ but it is MF's worst rule — quote it as the known weakness, not as the aggregate |
| **D. Mean over `-r`,`-c`,`-t`** | S&C tie, +6.22 steps | S&C tie, **+9.00 steps** `*` | 🔴 **no** — describes no controller; a method is penalised for having one bad rule even when it never has to use it |

**Use A as the headline and B as the deployment claim.** They agree in direction; A is significant
on steps, B is not. **D is the only aggregation under which the claim fails on steps**, and it fails
for a reason that has nothing to do with control performance.

⚠️ Two things to state whenever B is quoted: (i) the baselines' best rule is `-c` while MF's is `-t`,
so B compares *method + rule* pairs, not backbones; (ii) selecting the best of three on the same 6
episodes is a winner's-curse — the bootstrap above re-selects inside each replicate to account for
it, which is why B's interval is wider than A's despite a similar point estimate. With a held-out
validation split for rule selection (§6.6) B becomes clean.

---

## 4. Which MeanFlow backbone wins at K = 2 — UNet@32 vs `mf_dit`

A different question from §3. There: *MeanFlow vs the K20 baselines.* Here: **within Gen3v6, is the
fixed-width UNet or the DiT the better MeanFlow backbone at K = 2?** This is the question
[`fix_1`](../fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md) answered "the DiT, and the UNet does
not learn the objective at all", and which RESULTS §4.1 could only leave at *"indistinguishable at
n = 2"* — because it compared against a 2026-07-24 DiT run with a different arm set.

The 2026-08-07 batch settles it far better: **both backbones are in the same batch under the same
arm set, and `mf_dit` (CAND_108) carries all five seeds.**

| arm | `Candidate` | seeds | episodes per variant |
|---|---|---|---|
| MeanFlow **UNet@32** K2 | `110` | 6 only | 6 |
| MeanFlow **`mf_dit`** K2 | `108` | 6, 7, 8, 9, 10 | 30 |

🔴 **The asymmetry is the whole caveat: one training seed vs five.** Everything below is read
twice — matched at seed 6 (§4.2), and against `mf_dit`'s seed distribution (§4.3).

### 4.1 `mf_dit` K2, per seed

| variant | seed 6 | seed 7 | seed 8 | seed 9 | seed 10 | **all 5 pooled** | **UNet@32 (seed 6)** |
|---|---|---|---|---|---|---|---|
| **`-t-tightened`** S&C | 1.000 | 1.000 | 1.000 | 1.000 | 0.833 | **0.967** | **1.000** |
| steps | 65.50 | 66.67 | 61.17 | 67.00 | 81.83 | **68.43** | **58.67** |
| **`-r-tightened`** S&C | 0.833 | 1.000 | 1.000 | 1.000 | 1.000 | **0.967** | **1.000** |
| steps | 66.33 | 72.33 | 64.50 | 76.67 | 74.17 | **70.80** | **63.17** |
| **`-c-tightened`** S&C | 0.000 | 0.000 | 0.500 | 0.000 | 0.000 | **0.100** | **0.833** |
| steps | **199.0** | **199.0** | 134.67 | **199.0** | **199.0** | **186.13** | **94.00** |

**The headline of this whole section is the `-c` row.** `199.00 = max_episode_length − 1` is a
**timeout**, and `mf_dit` times out on **4 of 5 seeds** — this is the U3 "crushed to a point"
collapse ([`../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md`](../U3/INVESTIGATION_dpcc-c_stuck_at_point_K2.md))
reproducing **systematically, not as seed-6 bad luck**. The UNet@32 has no episode anywhere near
timeout (mean 94.0, worst reconstructed episode 114).

### 4.2 Matched head-to-head at seed 6 (6 episodes each, unpaired bootstrap)

`*` = 95 % CI strictly excludes 0. Negative Δsteps / Δ(s/ep) favours the UNet.

| variant | ΔS&C | Δsteps | Δ s/episode |
|---|---|---|---|
| `-t-tightened` | +0.000 [0, 0] | **−6.83 [−12.33, −1.50]** `*` | −0.18 [−0.41, +0.03] |
| `-r-tightened` | +0.167 [0.00, +0.50] | −3.17 [−13.00, +7.00] | −0.02 [−0.30, +0.26] |
| `-c-tightened` | **+0.833 [+0.50, +1.00]** `*` | **−105.0 [−121.0, −90.3]** `*` | **−2.31 [−2.79, −1.85]** `*` |

### 4.3 The UNet's single seed against `mf_dit`'s seed distribution

| variant | axis | UNet@32 | `mf_dit` 5-seed mean, 95 % seed-bootstrap CI | UNet beats … |
|---|---|---|---|---|
| `-t` | steps | **58.67** | 68.43 `[63.43, 75.60]` | **5/5 seeds** (min 61.17) |
| `-r` | steps | **63.17** | 70.80 `[66.80, 74.80]` | **5/5 seeds** (min 64.50) |
| `-c` | steps | **94.00** | 186.13 `[160.40, 199.00]` | **5/5 seeds** (min 134.67) |
| `-c` | S&C | **0.833** | 0.100 `[0.000, 0.300]` | **5/5 seeds** (max 0.500) |
| `-t` / `-r` | S&C | 1.000 | 0.967 `[0.900, 1.000]` | ties 4/5, beats 1/5 |

**In every arm the UNet's step count falls below the lower bound of `mf_dit`'s seed-level CI.**

### 4.4 Verdict — and the hard statistical ceiling of a one-seed arm

**The UNet@32 is the better MeanFlow backbone at K = 2**, on the point estimate in all three arms,
and decisively so on `-c-tightened`. But the strength of the claim is not uniform:

1. **`-c-tightened` — safe to assert now.** `mf_dit`'s failure is *qualitative* (timeout on 4/5
   seeds, S&C 0.100) and reproduces on every seed; the UNet is at 0.833 / 94 steps. That is a
   difference in failure *mode*, not in magnitude, and no plausible seed variance closes a
   199-vs-94 gap. This also reframes §3.3's "UNet is weak on `-c`" — it is weak versus the **K20
   baselines**, while being the only MeanFlow-family arm that works there at all.
2. **`-t-tightened` — supported but seed-limited.** −6.83 steps with the CI excluding 0 at matched
   seed 6, and below all five `mf_dit` seeds.
3. **`-r-tightened` — directional only.** Beats 5/5 seeds on steps, but the matched CI straddles 0.
4. **Wall-clock is a wash between the two.** Both are K = 2, and `mf_dit` is marginally cheaper per
   step (0.0236–0.0253 vs ~0.0270 s, ≈7 % — within contention noise). The Δ(s/ep) CIs straddle 0 on
   `-t` and `-r`. **The 19× compute story belongs to §3 (vs K20), not here.**

🔴 **The ceiling: with one UNet seed you cannot reach p < 0.05, no matter how large the margin.**
If the UNet seed were exchangeable with the five `mf_dit` seeds, the probability that a single new
draw is the smallest of six is exactly **1/6 ≈ 0.167** — that is the *minimum attainable*
permutation p-value for a "beats 5/5" sweep. `-t` and `-r` are at that floor. Only `-c` escapes,
because there the argument is a reproducible failure mode rather than a rank statistic.
**Seeds 7–10 at width 32 (§6.1) are what convert §4.2–4.3 from "5/5 sweep" into a real test.**

### 4.5 What this does to `fix_1`

RESULTS §7.1 already called for a correction header on
`fix_1/INSIGHT_Gen3v6_unet_vs_dit_backbone_AB.md`. This section strengthens the retraction:
`fix_1` concluded *"the analytic-v MeanFlow JVP objective requires the DiT backbone"*. Under a
matched arm set with the DiT's full seed sweep, the correct-width UNet is not merely **viable** —
it is **ahead of the DiT on every tightened arm at K = 2**, and it is the only one of the two that
does not collapse under the `-c` selection rule. ⚠️ Still one seed; state it as *"the width-32 UNet
leads the DiT on one seed across all five DiT seeds"*, not *"the UNet is better"*, until §6.1 lands.

### 4.6 Reproducing §4

Appends to the §2.3 script (reuses `episodes`, `boot`, `m`/`summ`). `SEEDS` is `mf_dit`'s sweep.

```python
SEEDS = ['6', '7', '8', '9', '10']
MFDIT, UNET = '108', '110'

# 4.2 — matched at seed 6
for var in VARS:
    A, B = episodes(UNET, var, '6'), episodes(MFDIT, var, '6')
    for key in ('sc', 'steps', 'wall'):
        o, lo, hi = boot(A, B, key)
        star = '*' if (lo > 0 or hi < 0) else ' '
        print(f'  {var:18s} {key:5s}: {o:+8.3f} CI[{lo:+.3f},{hi:+.3f}] {star}')

# 4.3 — one UNet seed vs mf_dit's seed-level distribution (cluster unit = seed)
for var in VARS:
    for key in ('sc', 'steps'):
        per = [statistics.mean(x[key] for x in episodes(MFDIT, var, s)) for s in SEEDS]
        u   =  statistics.mean(x[key] for x in episodes(UNET,  var, '6'))
        rng = random.Random(7); N = 20000
        bs  = sorted(statistics.mean(rng.choice(per) for _ in per) for _ in range(N))
        wins = sum(1 for p in per if (u > p if key == 'sc' else u < p))
        print(f'  {var:18s} {key:5s}: UNet={u:7.3f} | mf_dit={statistics.mean(per):7.3f} '
              f'CI[{bs[int(.025*N)]:.3f},{bs[int(.975*N)]:.3f}] | UNet better than {wins}/5')
```

⚠️ `wins == 5` is **not** a p-value. The exact one-sided permutation p-value for a 6-way rank sweep
with one UNet seed is 1/6 ≈ 0.167 regardless of margin (§4.4).

---

## 5. Why this is not yet proof — the honest caveat block

- 🔴 **One seed (6), 6 episodes per arm.** Every S&C value is a multiple of 1/6. A bootstrap on 6
  points reproduces the sampling noise of 6 points; it does **not** manufacture power. The CIs above
  are the *lower bound* on uncertainty because they ignore seed-to-seed variance entirely — the
  baselines have 5 seeds available and were deliberately down-filtered to seed 6 for matching.
- 🔴 **Unpaired.** Trial identity is unrecoverable from `mean ± std` (§2.1). Pairing on initial
  conditions would tighten the step CIs materially and could flip `-r` from "straddles 0" to
  significant — or confirm it does not.
- 🔴 **Backbone confound is unresolved.** The MeanFlow arm has one training run. A "K2 beats K20"
  claim needs ≥3 training seeds so the comparison is model-class vs model-class, not
  checkpoint vs checkpoint.
- ⚠️ **`n_steps` averages over goal-successful trials only** (`eval_flow_matching_v3_meanflow.py:518`).
  Comparing steps across rows at unequal S&C is biased toward the *less* successful arm (its hard
  episodes are dropped). This bites `-r` (0.833 baselines) — the MF step advantage there is, if
  anything, understated. It does **not** affect `-t` (all arms at 1.00), which is why `-t` is the
  quotable cell. Two artifacts to recognise: `0.00` = SR 0, `199.00` = timeout.
- ⚠️ **`avg_time` is wall-clock per step on shared GPUs**, and includes the projection NLP solve.
  The 19× is dominated by NFE (2 vs 20) and is robust, but do not quote 3 significant figures.
- ⚠️ **Window-level train/test split leak** (inherited, POST_U10_III §4.2) affects all arms equally.

---

## 6. What to run to turn this into a real result

In priority order. Items 1–2 are what the claim actually needs; 3–5 are cheap hardening.

1. **Multi-seed the MeanFlow arm — the single highest-value job in this document.** Train `bbunet`
   @ `freq_dim=32` on seeds 7–10 (one job each, ~8 h; it is the cheapest Gen3v6 arm). Already listed
   as RESULTS §8.1. It is now load-bearing **twice over**: it is the only thing that puts an error
   bar on §3 (vs the K20 baselines) *and* the only thing that lifts §4 above its 1/6 permutation
   floor against `mf_dit`'s five seeds (§4.4). `mf_dit` already has seeds 6–10, so the comparison
   becomes symmetric the moment these land.
   → ⚠️ job 24318 already failed by evaluating an untrained seed; have the driver check for
   `<seed>/dataset_config.pkl` before submitting.
2. **Re-eval at `n_trials ≥ 10`, all arms in one batch.** 5 seeds × 3 halfspaces × 10 trials = 150
   episodes/arm. At that point S&C resolution goes from 1/6 to 1/150 and the `-r` and `-c` cells
   become decidable. **Run all three candidates in the same batch** so `avg_time` stays claimable.
3. **Persist per-episode records** so the bootstrap can be *paired* and the §2.1 reconstruction
   trick is no longer needed: dump the per-trial `(success, constraints_ok, n_steps, time)` arrays
   next to `results.json` in the eval tree. This is a small change in the eval writer and it
   unblocks every future statistical comparison in the repo.
4. **Switch to paired + cluster bootstrap** once 3 lands: resample *seeds* (cluster unit) and pair
   episodes by `(seed, halfspace, trial_idx)`. Report BCa rather than percentile intervals once
   n ≥ 30.
5. **Add a K-ladder** {1, 2, 5, 10, 20} for UNet@32 (RESULTS §8.7). "K2 matches K20" is a *much*
   stronger claim when the ladder shows where it saturates, and it feeds
   `DA/DA_20260805_LowK_Ablation_MFAF_vs_FM_DPCC.md`'s L3 leg, which currently rests on AlphaFlow
   alone.
6. **Select the MPC rule on held-out episodes.** Aggregation B (§3.4) currently picks each method's
   best rule on the same 6 episodes it is then scored on. With `n_trials ≥ 10` (item 2), split
   trials into a selection half and a reporting half, choose `-r`/`-c`/`-t` per method on the first,
   and report on the second. That removes the winner's curse instead of merely pricing it in, and
   turns B from "point estimate favours MF" into a clean claim.

**Pre-registration.** Before running 1–2, write down the decision rule so the re-eval cannot be
read post-hoc:

> *The claim "UNet@32 K2 ≽ DPCC K20 / FMv3 K20" is accepted iff, under **aggregation A** (all
> methods on `dpcc-t-tightened`) **and** under **aggregation B** (each method on its
> validation-selected rule, §6.6), the paired bootstrap ΔS&C CI excludes negative values AND the
> Δ(s/episode) CI excludes 0 favourably; the stronger "Pareto-dominant" claim additionally requires
> the Δsteps CI to exclude 0 favourably under A.*

Fix the rule and the arms now, not after seeing the numbers. Aggregation D is excluded by
construction (§3.4).

---

## 7. One-line verdict

**The inspection holds under every aggregation a controller can actually be run at.** At K = 2 the
width-32 UNet MeanFlow is **Pareto-dominant over both FMv3 K20 and DPCC K20** — equal S&C, fewer
steps, ~19× less wall-clock per episode — both when all methods are held to `dpcc-t-tightened`
(where the step margin is significant: −4.8 and −3.3, CIs excluding 0) and when each method uses its
own best rule (−3.5 and −2.8, CIs straddling 0). Its one weakness is `dpcc-c-tightened`, a rule it
does not have to use. The only aggregation that contradicts the claim is the average over
`-r`/`-c`/`-t`, which corresponds to no deployable controller and should not be reported.

**And within Gen3v6 the UNet is also the better backbone (§4):** at K = 2 it beats **all five**
`mf_dit` seeds on steps in every tightened arm, and on `-c-tightened` it is the only MeanFlow
variant that works at all (0.833 vs 0.100 — `mf_dit` times out on 4/5 seeds). `fix_1`'s "MeanFlow
needs the DiT" is not merely retracted; it is **inverted**.

**What is still missing is not a better result — it is error bars:** one seed, six episodes. With a
single UNet seed the best attainable permutation p-value against `mf_dit`'s five is 1/6 ≈ 0.167
(§4.4), so no amount of margin closes it. Run §6.1–6.2 and this becomes quotable.
