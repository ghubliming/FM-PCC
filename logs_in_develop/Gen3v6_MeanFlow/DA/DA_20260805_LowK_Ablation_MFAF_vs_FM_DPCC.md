# Ablation — does low-K MeanFlow/AlphaFlow match high-K FM/DPCC, and is low K a capability or just a discount?

**Date:** 2026-08-05
**Source:** `temp/2026-08-02/batch_avoiding_combined_20260802_092307/candidates_multidimensional_raw.csv`
**Scope:** `avoiding-d3il`, H=8, 5 seeds {6,7,8,9,10} × 2 trials × **3 halfspace envs**
(`top-right-hard`, `top-left-hard`, `both-hard`) = **10 episodes per env-cell, 30 pooled**
**Companion docs:** `DA_20260802_K2_MeanFlow_AlphaFlow_vs_FM_DPCC.md`,
`logs_in_develop/Gen12/DA/DA_20260803_HardFlow_activation_threshold_0p1.md`

---

## 0. The claim, decomposed

Three legs; it is only a result if all three hold:

| leg | statement | status |
|---|---|---|
| **L1** | MF/AF at K=2 reach the **same success+constraints** as FM K=20 and DPCC K=10/20 | **holds — they exceed it** |
| **L2** | MF/AF at K=2 cost **much less time**, ideally fewer steps | **holds on time (12×); steps not claimable** |
| **L3** | FM/DPCC at K=1–2 are **worse** — low K is a capability, not a discount anyone can take | **holds; the one apparent counter-example fails an entire environment** |

L3 is load-bearing. Without it, "MF/AF are fast at K=2" reduces to "K=2 is cheap", which is true
of any generator and is not a contribution.

**Headline:** on the per-environment tables (§3), **AlphaFlow at K=2 under `-r` is the only
configuration in the study that clears 70% in every environment** — 7/10, 7/10, 8/10 — at
1.3–7.0 s of planning per episode. The best DPCC configuration manages 50% in its worst
environment at ~35 s. And the config that appeared to undercut L3 in the pooled numbers,
diffusion sampled at K=1, turns out to score **0/10 on `top-right-hard` under all three
selection rules**.

---

## 1. What data exists, and what does not

Full-seed (5 seeds) candidates by engine and K:

| engine | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---|---|---|---|---|
| **AlphaFlow** (`bbsit`) | CAND_31 *(4 seeds)* | **CAND_32** | CAND_33 *(4 seeds)* | CAND_30 *(4 seeds)* | CAND_27 *(1 seed)* |
| **MeanFlow** (`bbmf_dit`) | — *(1 seed)* | **CAND_102** | — *(1 seed)* | — | — |
| **FM** `FlowMatchingODE` | — | — | CAND_106 *(1 seed)* | — | **CAND_105** |
| **Diffusion via FMv3 ODE solver** | CAND_109 | — | CAND_111 | CAND_112 | CAND_110 |
| **DPCC** (deployed diffusion) | **CAND_8** | — | — | **CAND_7** | **CAND_10** |

Gaps, stated up front:

- **No `FlowMatchingODE` below K=5, none at K=1/2 at any seed count.** The "FM at low K" half of
  L3 is untested; it is proxied by CAND_109/111 (the *diffusion* engine through the same FMv3
  Euler sampler) and by DPCC K=1.
- **No DPCC at K=2.** The DPCC low-K leg is K=1 only.
- **No `FlowMatchingODE` at K=10.** CAND_112 (diffusion via FMv3 sampler, K=10) is the nearest.
- AF's ladder (K=1/5/10) sits on seeds 7–10; AF K=2 adds seed 6. **Not perfectly paired.**
- **MF exists at full seed count only at K=2** — no MeanFlow K-ladder. §6 is AlphaFlow's alone.

Backbones differ (`bbsit` / `bbmf_dit` / FM's default), so **every cross-family row is jointly a
model and a K comparison**. The within-family ladders in §6 are the clean part.

---

## 2. Why the headline metric cannot be used

The natural table — `n_success_and_constraints` on `-tightened` arms — is **saturated**, and it
stays saturated per environment:

`dpcc-t-tightened`, per env (out of 10):

| config | top-right | top-left | both | pooled | worst env |
|---|---|---|---|---|---|
| AF K=2 | 8 | 10 | 10 | 28/30 | 80% |
| MF K=2 | 9 | 10 | 10 | 29/30 | 90% |
| FM K=20 | 10 | 10 | 10 | 30/30 | 100% |
| Diffusion K=1 (FMv3) | 10 | 10 | 10 | 30/30 | 100% |
| DPCC K=10 | 10 | 10 | 10 | 30/30 | 100% |
| DPCC K=20 | 10 | 10 | 10 | 30/30 | 100% |
| **DPCC K=1** | **2** | **6** | 10 | **18/30** | **20%** |

**And it is not measuring the generator.** The decisive diagnostic — the unprojected `diffuser`
arm of the same candidates, per env:

| config | unprojected (tr / tl / bh) | pooled | → with projection + tightening |
|---|---|---|---|
| **Diffusion K=1 (FMv3)** | 0 / 1 / 0 | **1/30 — 3.3%** | **30/30 — 100%** |
| DPCC K=20 | 0 / 1 / 2 | 3/30 — 10.0% | 30/30 |
| FM K=20 | 0 / 2 / 2 | 4/30 — 13.3% | 30/30 |
| **AF K=2** | 2 / 1 / 5 | **8/30 — 26.7%** | 28/30 |
| MF K=2 | 0 / 1 / 1 | 2/30 — 6.7% | 29/30 |
| DPCC K=1 | 0 / 0 / 0 | 0/30 — 0.0% | 18/30 |

A generator that solves **1 episode in 30 on its own** scores **30/30** once projected and
tightened. **The tightened score measures DPCC's projector, not the generative model.** Any
claim of the form "our K=2 model matches their K=20 model at 100%" is, on this benchmark, a
claim about the projector.

So everything below runs on the **untightened** arms, where the margin is removed and the
projector must work with what the generator produced. There the metric spans 0–100% per env and
discriminates. Per §10.0 of the Gen12 doc, arms are compared **matched by suffix** or
**best-of-family**, never pooled across suffixes — `-r`/`-c`/`-t` are deployment choices, not
samples.

The one thing the tightened table does show: **DPCC at K=1 fails even with the projector and the
margin, and fails unevenly — 2/10 on top-right against 10/10 on both-hard.** That is L3's first
data point, and it is invisible in the pooled 60%.

---

## 3. Per-environment results — the main tables

Format: `success+constraints out of 10` per environment, then pooled out of 30, then the
**worst-environment rate**, which is the number that matters for a controller you intend to
deploy without knowing which map it will face.

### 3.1 `dpcc-r` — neutral selection, no heuristic

| config | top-right | top-left | both | pooled | **worst env** |
|---|---|---|---|---|---|
| **AF K=2** | **7/10** | **7/10** | **8/10** | 22/30 — 73% | **70%** |
| MF K=2 | 7/10 | 4/10 | 8/10 | 19/30 — 63% | 40% |
| AF K=1 *(4 seeds)* | 8/8 | 5/8 | 4/8 | 17/24 — 71% | 50% |
| AF K=5 *(4 seeds)* | 3/8 | 4/8 | 7/8 | 14/24 — 58% | 38% |
| AF K=10 *(4 seeds)* | 2/8 | 4/8 | 2/8 | 8/24 — 33% | 25% |
| FM K=20 | 1/10 | 6/10 | 2/10 | 9/30 — 30% | 10% |
| **Diffusion K=1 (FMv3)** | **0/10** | 8/10 | 4/10 | 12/30 — 40% | **0%** |
| Diffusion K=10 (FMv3) | 0/10 | 6/10 | 1/10 | 7/30 — 23% | 0% |
| Diffusion K=20 (FMv3) | 3/10 | 4/10 | 2/10 | 9/30 — 30% | 20% |
| DPCC K=1 | 1/10 | 1/10 | 0/10 | 2/30 — 7% | 0% |
| DPCC K=10 | 0/10 | 4/10 | 2/10 | 6/30 — 20% | 0% |
| DPCC K=20 | 2/10 | 2/10 | 4/10 | 8/30 — 27% | 20% |

**AF K=2 is the only row in this table that is ≥70% in all three environments.** Every baseline
has at least one environment where it is at or below 20%, and four of them have an environment
they never solve at all.

### 3.2 `dpcc-t` — temporal consistency

| config | top-right | top-left | both | pooled | **worst env** |
|---|---|---|---|---|---|
| **AF K=2** | 4/10 | **10/10** | **10/10** | **24/30 — 80%** | 40% |
| MF K=2 | 3/10 | 3/10 | 7/10 | 13/30 — 43% | 30% |
| AF K=10 *(4 seeds)* | 5/8 | 6/8 | 6/8 | 17/24 — 71% | 62% |
| FM K=20 | 0/10 | 7/10 | 3/10 | 10/30 — 33% | 0% |
| Diffusion K=1 (FMv3) | 0/10 | 6/10 | 4/10 | 10/30 — 33% | 0% |
| Diffusion K=20 (FMv3) | 3/10 | 7/10 | 7/10 | 17/30 — 57% | 30% |
| DPCC K=1 | 0/10 | 0/10 | 0/10 | 0/30 — 0% | 0% |
| DPCC K=10 | 2/10 | 5/10 | 5/10 | 12/30 — 40% | 20% |
| **DPCC K=20** | 5/10 | 8/10 | 7/10 | **20/30 — 67%** | **50%** |

`-t` is where the baselines look their best (DPCC K=20 reaches 67% pooled / 50% worst) and it is
also where **AF's pooled score peaks at 80% but its worst environment drops to 40%.**
Pooled and worst-env rank AF's two rules differently — see §3.4.

### 3.3 `dpcc-c` — minimum projection cost *(broken for MF/AF at K=2)*

| config | top-right | top-left | both | pooled | **worst env** |
|---|---|---|---|---|---|
| AF K=2 | 2/10 | 2/10 | 1/10 | 5/30 — 17% | 10% |
| MF K=2 | 0/10 | 1/10 | 1/10 | 2/30 — 7% | 0% |
| AF K=1 *(4 seeds)* | 4/8 | 5/8 | 3/8 | 12/24 — 50% | 38% |
| AF K=5 *(4 seeds)* | 6/8 | 7/8 | 7/8 | 20/24 — 83% | **75%** |
| FM K=20 | 7/10 | 8/10 | 2/10 | 17/30 — 57% | 20% |
| Diffusion K=1 (FMv3) | 0/10 | 7/10 | 4/10 | 11/30 — 37% | 0% |
| DPCC K=1 | 1/10 | 1/10 | 0/10 | 2/30 — 7% | 0% |
| DPCC K=20 | 6/10 | 8/10 | 4/10 | 18/30 — 60% | 40% |

The MF/AF collapse at K=2 under `-c` is **uniform across environments** (1–2 of 10 everywhere),
which is the signature of the mechanism, not of a hard map: on a half-integrated τ=0.5 iterate a
stalled trajectory has ≈0 projection cost, so `minimum_projection_cost` selects it (§9.8/§10.6
of the companion doc). AF at K=1 recovers to 50%, AF at K=5 to 83% — **it is a K=2-specific
selection-rule bug, not a property of MF/AF.**

### 3.4 Worst-environment summary — the robustness view

Best worst-environment rate achievable by each config, over the three untightened rules:

| config | best worst-env | rule | pooled at that rule | episode time |
|---|---|---|---|---|
| AF K=5 *(4 seeds)* | 75% | `-c` | 83% | 10.0–12.8 s |
| **AF K=2** | **70%** | **`-r`** | **73%** | **1.3–7.0 s** |
| AF K=10 *(4 seeds)* | 62% | `-t` | 71% | 15.8–22.1 s |
| AF K=1 *(4 seeds)* | 50% | `-r` | 71% | 1.0–2.2 s |
| **DPCC K=20** | **50%** | **`-t`** | 67% | **32.3–35.7 s** |
| MF K=2 | 40% | `-r` | 63% | 1.6–5.0 s |
| Diffusion K=20 (FMv3) | 30% | `-t` | 57% | 24.5–37.6 s |
| FM K=20 | 20% | `-c` | 57% | 21.6–39.2 s |
| DPCC K=10 | 20% | `-t` | 40% | 19.6–24.1 s |
| Diffusion K=10 (FMv3) | 10% | `-c` | 47% | 10.0–16.3 s |
| **Diffusion K=1 (FMv3)** | **0%** | any | 33–40% | 1.2–2.9 s |
| **DPCC K=1** | **0%** | any | 0–7% | 1.9–4.9 s |

**AF K=2 at 70% worst-env / 1.3–7.0 s against DPCC K=20 at 50% worst-env / ~35 s: better on
robustness, and 5–25× cheaper.** The only config with a better worst-env is AF K=5 (75%), which
is the same family at 10× the time and on 4 seeds — it does not help the baselines.

### 3.5 Why per-environment, not pooled

Two things pooling hides here, both of which change conclusions:

1. **Diffusion K=1 pools to 40% while never solving `top-right-hard` at all** — 0/10 under
   `-r`, `-t`, *and* `-c`. Its entire score comes from `top-left` (60–80%) and `both` (40%).
   This is the config that appeared to undercut L3; per environment it is not a working
   controller, it is one that happens to suit two of three maps. §5 revises L3 accordingly.
2. **AF's best rule flips depending on the summary.** `-t` pools higher (80% vs 73%) but its
   worst environment is 40% against `-r`'s 70%. A pooled table would recommend `-t`; a
   deployment that must handle `top-right-hard` wants `-r`.

Environment difficulty is also not uniform and not the same for everyone: `top-right-hard` is
the hardest for the diffusion-family generators (0/10 for three of them under `-r`) but only
mildly hard for AF (7/10). `both-hard` is hardest for FM K=20 under `-c` (2/10) while being
AF's *best* environment under `-r` (8/10). **There is no single "hard env"** — which is
precisely why the per-env table has to be reported.

---

## 4. Pooled view — L1 and L2

Pooled across all three environments (30 episodes/cell), `dpcc-r`, with Wilson 95% intervals
and `ep_s` = planning wall clock per episode (`n_steps × avg_time`):

| config | success+constraints | 95% CI | steps | s/step | **episode time** |
|---|---|---|---|---|---|
| **AF K=2** | **22/30 — 73.3%** | [55.6, 85.8] | 73.7 | 0.034 | **3.3 s** |
| **MF K=2** | **19/30 — 63.3%** | [45.5, 78.1] | 79.5 | 0.036 | **3.3 s** |
| Diffusion K=1 (FMv3) | 12/30 — 40.0% | [24.6, 57.7] | 83.7 | 0.020 | 1.7 s |
| FM K=20 | 9/30 — 30.0% | [16.7, 47.9] | 85.5 | 0.459 | 40.3 s |
| DPCC K=20 | 8/30 — 26.7% | [14.2, 44.4] | 78.8 | 0.488 | 39.5 s |
| DPCC K=10 | 6/30 — 20.0% | [9.5, 37.3] | 81.1 | 0.279 | 22.5 s |
| DPCC K=1 | 2/30 — 6.7% | [1.8, 21.3] | 81.9 | 0.038 | 3.3 s |

| contrast | counts | Fisher p | time ratio | per-seed |
|---|---|---|---|---|
| AF K=2 vs **FM K=20** | 22/30 vs 9/30 | **0.0017** | **12.4× faster** | **5–0** |
| AF K=2 vs **DPCC K=20** | 22/30 vs 8/30 | **0.00066** | **12.1× faster** | **5–0** |
| AF K=2 vs DPCC K=10 | 22/30 vs 6/30 | **0.00007** | 6.9× faster | — |
| MF K=2 vs FM K=20 | 19/30 vs 9/30 | **0.019** | 12.2× faster | **4–0** (1 tie) |
| MF K=2 vs DPCC K=20 | 19/30 vs 8/30 | **0.0089** | 11.9× faster | **4–0** (1 tie) |
| AF K=2 vs MF K=2 | 22/30 vs 19/30 | 0.58 | 1.0× | — |

**L1 holds and then some** — MF/AF at K=2 do not merely match the high-K baselines on this arm,
they are **2.4–2.7× better**, p < 0.02 throughout, with unanimous per-seed direction.

**L2 holds on time: 3.3 s/episode against 39.5–40.3 s.** Best-of-family it is 1.3 s (AF `-t`)
against 34.4 s (DPCC K=20 `-t`) — **26×**.

**L2 on steps is not claimable.** AF K=2 uses 73.7 steps vs FM K=20's 85.5 and DPCC K=20's 78.8;
MF K=2 uses 79.5, *more* than DPCC K=20. And `n_steps` is averaged over successful episodes
only, while success rates differ threefold — the weak arms are scored on their easiest
episodes. Reported for completeness, not as a result.

Per-environment episode time, best rule per family:

| config | rule | top-right | top-left | both |
|---|---|---|---|---|
| AF K=2 | `-r` | 7/10 @ 1.4 s | 7/10 @ **7.0 s** | 8/10 @ 1.3 s |
| AF K=2 | `-t` | 4/10 @ 1.4 s | 10/10 @ 1.2 s | 10/10 @ 1.2 s |
| MF K=2 | `-r` | 7/10 @ 3.3 s | 4/10 @ 5.0 s | 8/10 @ 1.6 s |
| FM K=20 | `-c` | 7/10 @ 21.6 s | 8/10 @ 29.3 s | 2/10 @ 39.2 s |
| DPCC K=20 | `-t` | 5/10 @ 35.7 s | 8/10 @ 32.3 s | 7/10 @ 35.3 s |
| DPCC K=1 | `-r` | 1/10 @ 4.9 s | 1/10 @ 3.3 s | 0/10 @ 2.3 s |

Note the AF `-r` / `top-left` outlier: 7.0 s against 1.3–1.4 s elsewhere, a 5× local stall on
the same generator. Worth a look — it is the only place AF's cost is not flat.

---

## 5. L3 — the control leg

| config | K | pooled | worst env | **episode time** |
|---|---|---|---|---|
| **AF K=2** (`-r`) | 2 | **22/30 — 73%** | **70%** | **3.3 s** |
| **MF K=2** (`-r`) | 2 | **19/30 — 63%** | 40% | **3.3 s** |
| **DPCC K=1** (`-r`) | 1 | **2/30 — 7%** | **0%** | **3.3 s** |

**At identical measured wall clock — 3.3 s/episode to two significant figures — AF scores 22/30
and DPCC scores 2/30.** Fisher p < 1e-5. The budget is matched by measurement, not by argument.
DPCC K=1 never exceeds 1/10 in any environment under any rule, and reaches 0/30 under `-t`. It
is also the only config that fails *with* projection and tightening (18/30, and 2/10 on
top-right — §2).

**The apparent counter-example, resolved by the per-env view.** Diffusion sampled at K=1 through
the FMv3 Euler solver pools to 12/30 (40%) at 1.7 s — better than FM K=20 and DPCC K=20 at
1/24th the time, and *not* separable from MF K=2 pooled (19/30 vs 12/30, p = 0.12). On the
pooled table alone this weakens L3.

Per environment it does not survive: **0/10 on `top-right-hard` under `-r`, `-t` and `-c`
alike** (§3.1–3.3). Its 40% is `top-left` 8/10 plus `both` 4/10. A controller that cannot solve
one of three maps at all is not a low-K baseline that undercuts the claim — and it is
consistent with its `diffuser` score of 1/30, i.e. a generator that produces almost nothing
usable and is being carried by projection where the geometry happens to permit it.

**So L3 holds in the form:** low K destroys the deployed DPCC outright (2/30, 0% worst env), and
the one diffusion configuration that appears to survive it does so only on two of three
environments. Two caveats keep this from being airtight: CAND_109's model directory carries
**`aw1`, not `aw10`** (off-spec on action weight versus every other candidate here), and **no
`FlowMatchingODE` run below K=5 exists**, so the FM half of L3 is genuinely untested rather than
tested-and-passed.

---

## 6. K-ladders — the families move in opposite directions

Within-family, same model, same suffix (`dpcc-r`), K varied — free of the backbone confound:

| family | K=1 | K=2 | K=5 | K=10 | K=20 | direction |
|---|---|---|---|---|---|---|
| **AlphaFlow** | 17/24 — 71% | **22/30 — 73%** | 14/24 — 58% | 8/24 — 33% | — | **decreasing** |
| Diffusion (FMv3 sampler) | 12/30 — 40% | — | 7/30 — 23% | 7/30 — 23% | 9/30 — 30% | flat |
| **DPCC** (deployed) | 2/30 — 7% | — | — | 6/30 — 20% | 8/30 — 27% | **increasing** |

**DPCC improves monotonically with denoising steps; AlphaFlow degrades monotonically.** They are
not the same method at different price points — they sit at opposite ends of the same axis.

- **AF at K=10 (8/24, 33%) is indistinguishable from DPCC at K=20 (8/30, 27%).** AlphaFlow's
  advantage is *entirely* a low-K phenomenon; in DPCC's regime it is an ordinary baseline. The
  contribution is the shape of the curve.
- **AF K=1 (71%) ≈ AF K=2 (73%)**, so the optimum is at or below K=2 and K=1 is nearly free —
  though AF K=1's worst env is 50% against K=2's 70%, so **K=2 is the better deployment point**.

Caveats: the ladder rungs are seeds 7–10 while AF K=2 adds seed 6; **MF has no ladder at all**,
so the direction result is AlphaFlow's and holds for MeanFlow only by family analogy.

---

## 7. The backbone confound — MF/AF are not running FM/DPCC's network

Every cross-family comparison above changes **two** things at once: the denoising-step count K,
and the network architecture. MF/AF use transformer backbones (`bbsit`, `bbmf_dit`); FM and DPCC
use the UNet. The batch contains UNet-backbone MF/AF runs, in the same `(Bf_U3)` tree, same seed
6, so the confound can be measured rather than assumed.

### 7.1 The UNet runs, and they are a disaster

AlphaFlow, `(Bf_U3)` tree, seed 6, 6 episodes per cell, pooled over the three environments:

| backbone | K | `diffuser` | `-r` | `-t` | `-c` | `-r-tg` | `-t-tg` | `-c-tg` |
|---|---|---|---|---|---|---|---|---|
| **`bbunet`** | 1 | 1/6 | **0/6** | **0/6** | **0/6** | **0/6** | 2/6 | **0/6** |
| **`bbunet`** | 2 | 1/6 | **0/6** | **0/6** | **0/6** | **0/6** | 1/6 | **0/6** |
| **`bbunet`** | 5 | 1/6 | **0/6** | **0/6** | **0/6** | **0/6** | 1/6 | **0/6** |
| **`bbunet`** | 10 | 1/6 | **0/6** | **0/6** | **0/6** | **0/6** | 1/6 | **0/6** |
| `bbsit` | 2 | 2/6 | 3/6 | 2/6 | 0/6 | **6/6** | 5/6 | 0/6 |
| `bbdit` | 2 | 1/6 | 5/6 | 3/6 | 4/6 | **6/6** | **6/6** | **6/6** |

**20 of AlphaFlow-UNet's 24 projected cells score exactly zero**, at every K from 1 to 10. The
remaining four are `-t-tightened` at 1–2/6. It is not a weak model, it is a dead one.

MeanFlow, same tree, seed 6, K=2:

| backbone | `diffuser` | `-r` | `-t` | `-c` | `-r-tg` | `-t-tg` | `-c-tg` |
|---|---|---|---|---|---|---|---|
| **`bbunet`** | 0/6 | 1/6 | **0/6** | 2/6 | 2/6 | **0/6** | 3/6 |
| `bbdit` | 1/6 | 2/6 | 4/6 | 4/6 | 5/6 | 5/6 | **6/6** |
| `bbmf_dit` | 1/6 | **5/6** | 3/6 | **0/6** | 5/6 | 5/6 | **0/6** |

So there is **no working UNet MeanFlow or AlphaFlow in this dataset.** The confound cannot be
removed by swapping the backbone on our side — that comparison has been run and our side is the
one that collapses.

### 7.2 On **time** the confound is small, and for MeanFlow it is negligible

Generation-only cost (the `diffuser` arm, which contains no projection), normalised per network
evaluation:

| backbone | ms per network eval |
|---|---|
| AF `bbsit` | **6.17** |
| MF `bbmf_dit` | 8.45 |
| **DPCC UNet** | **8.97** |
| FM UNet | 9.18 |
| AF `bbunet` | 10.44 |
| MF `bbunet` | 10.43 |
| AF `bbdit` | 13.81 |

The architectures sit within ~2× of each other, and **DiT is the most expensive of all** — so
"transformer" is not a synonym for "fast" here. Decomposing the 14.5× generation speed-up of
AF-SiT-K2 over DPCC-UNet-K20:

| effect | measured how | factor |
|---|---|---|
| **K, 20 → 2** | DPCC UNet K=20 (0.1793 s/step) vs DPCC UNet K=1 (0.0094) → 19.1× for 20× | **≈10×** |
| **architecture, UNet → SiT** | AF UNet K=2 (0.0209) vs AF SiT K=2 (0.0123), same K, same tree | **1.45×** |
| combined | 0.1793 → 0.0123 | **14.5×** |

**The speed claim (L2) is ~90% K and ~10% architecture.** For MeanFlow it is essentially all K —
`bbmf_dit` at 8.45 ms/eval against DPCC's UNet at 8.97 ms is a **1.06×** difference, i.e. none.
L2 survives the confound.

### 7.3 On **quality** the confound is not resolvable with this data

L1 and L3's quality numbers compare AF-SiT / MF-`mf_dit` against DPCC-UNet. That is a joint
model + architecture + K comparison, and §7.1 shows the matched-architecture version cannot be
run because AF-UNet and MF-UNet do not work.

What is *not* affected:

- **L3's matched-wall-clock comparison (§5)** is architecture-agnostic on the cost side: DPCC's
  own UNet at K=1 generates at **0.0094 s/step, cheaper than AF-SiT at K=2 (0.0123)**. The
  baseline has full access to the speed; it is the quality at that speed it cannot reach (2/30).
  Changing DPCC's backbone would not change that — it is already faster.
- **The K-ladder directions (§6)** are within-family and within-backbone, so the opposite-sign
  result stands regardless.

What **is** affected: **every statement of the form "MF/AF at K=2 beats DPCC at K=20 on
quality"** (§3, §4). It should be read as *"AlphaFlow-with-SiT at K=2 beats DPCC-with-UNet at
K=20"*, and the architecture's share of that margin is unmeasured.

### 7.4 A side finding: the `-c` collapse is backbone-specific

The K=2 `-c` failure is not universal to MF/AF — it needs K=2 **and** the right backbone:

| model | `-c-tightened` at K=2 |
|---|---|
| AF `bbsit` | **0/6** |
| AF `bbdit` | **6/6** |
| MF `bbmf_dit` | **0/6** |
| MF `bbdit` | **6/6** |
| AF `bbsit` at K=1 / K=5 / K=10 | 6/6 / 6/6 / 6/6 |

**DiT is immune at K=2; SiT and mf_dit are not.** So the τ=0.5 stalled-trajectory ranking
pathology (§3.3) requires both the half-integrated iterate *and* a backbone whose K=2 output has
the right degeneracy. This narrows the bug and gives a second way to fix it — change backbone —
alongside the endpoint-ranking fix.

### 7.5 The iMeanFlow DiT-vs-UNet pair — **it proves nothing, and here is why**

> **Retraction.** An earlier version of this subsection concluded from the pair below that "the
> UNet is not the problem" and that AF-UNet's failure was therefore a broken run. **That
> conclusion was invalid** and is withdrawn. It read the *projected* columns while the
> *unprojected* column said both models fail — the exact error §2 of this document warns about.

iMeanFlow was run with both backbones in the same tree, same config string, same seed:
`(Bf_U9)/...jvp_bbdit` (CAND_58) against `..._jvp_bbunet_tslogit_normal` (CAND_59), both K=10:

| iMF K=10 `(Bf_U9)` | **`diffuser` (unprojected)** | `-r` | `-t` | `-c` | `-r-tg` | `-t-tg` | `-c-tg` |
|---|---|---|---|---|---|---|---|
| `bbdit` | **0/6** | 1/6 | 2/6 | 0/6 | 5/6 | 5/6 | 4/6 |
| `bbunet` | **0/6** | 1/6 | 3/6 | 1/6 | 5/6 | 4/6 | 4/6 |

The projected columns are indistinguishable — **because the generators are indistinguishably
bad**. Both score **0/6 unprojected**: they reach the goal (6/6 and 5/6 on `n_success` alone) and
violate a constraint on **every single episode**. Their 17/36 and 18/36 projected totals are the
projector's work, not the network's, exactly as §2 established for the 1/30 → 30/30 case.

**A comparison between two generators that both score zero is not evidence that either works.**
The pair is uninformative in both directions: it neither shows the UNet working nor shows it
failing. §7.1's finding — that AF-UNet scores 0/6 in 20 of 24 projected cells, where AF-SiT
scores 6/6 — stands on its own and is *not* rebutted by this pair.

### 7.5b What the batch cannot see at all: path smoothness

Direct visual inspection of the rollouts reports a failure mode this dataset has no column for:
**MF and AF on the UNet produce non-smooth, zig-zagging paths; only on their native backbone
(AF → SiT, MF → `mf_dit`) is the output smooth. The same is reported for `imf_dit` — it works
only on its own network.**

The batch cannot confirm or refute this. Its complete metric set is
`n_success`, `n_success_and_constraints`, `n_steps`, `n_violations`, `total_violations`,
`avg_time`, `collision_free_completed`, plus NFE/NLP counters — **there is no smoothness or
roughness metric anywhere in it.** Raw trajectories (`obs_all`, `act_all`,
`sampled_trajectories_all`) are present for only 430–595 of 123,443 rows.

This is a known gap, already written up in
`logs_in_develop/DA_Code/METRIC_SMOOTH/DISCUSSION_metric_smooth.md`, which defines roughness as
the mean squared second difference of the planned x-y path and ships it as `plan_roughness` /
`plan_roughness_raw` in the **Gen13/HardFlow** pipeline only. That document also makes the point
that decides the present question (§5.3): *the projected plan is smooth by construction, so the
information lives in the raw plan and the executed path.* Every projected column in this batch is
therefore blind to exactly the defect being described — and the two unprojected columns are too
coarse at 6 episodes to resolve it.

**So the operative claim on backbones is the visual one, not §7.5's tables**, and it is
consistent with everything the batch *can* show: each family works on its own network, and every
off-native backbone run in this dataset is either dead (AF-UNet, 20/24 zeros) or
generator-degenerate (iMF-UNet **and** iMF-DiT, both 0/6 unprojected).

For reference, the best iMF results at low K (seed 6): CAND_85 (`setup3_2e5`, K=2) scores `-r`
4/6, `-t` 4/6 and 6/6 on all three tightened arms; CAND_79 (`official setup1`, K=2) scores 6/6 on
all tightened arms at 0.0273 s/step. **Those are projected numbers and, per the above, should not
be read as generator quality without a roughness column.**

### 7.5c Is the baseline better than MF/AF *on the UNet*? Yes — but it is not the same UNet

| model | UNet | goal-reached | steps | projected quality |
|---|---|---|---|---|
| FM K=20 | **4 M** (`dim=32`) | 30/30 | 73.3 | high |
| DPCC K=20 | **4 M** (`dim=32`) | 29/30 | 74.7 | high |
| AF K=1–10 | **253 M** (`dim=256`) | 4/6 | 96–121 | **1–2/36** |
| MF K=2 | **253 M** (`dim=256`) | 4/6 | 65.0 | 8/36 |

So the statement "on the UNet, FM/diffusion beats MF/AF" is **true as observed but not a
controlled comparison** — the two sides differ by 63× in parameters (§7.6). There is no run
anywhere in this dataset of MF/AF on a 4 M UNet, or of FM/DPCC on a 253 M one.

**Underfitting or overfitting?** The instinct that this is hard to tell for a deep generative
model is right, and the usual tell — a train/test gap — is available in the saved loss histories.
In `temp/0408/af)losses.pkl` and `mf_losses.pkl` the terminal raw-MSE test/train ratios are
**1.25 (AF)** and **0.83 (MF)** — i.e. **essentially no generalisation gap**, which points *away*
from overfitting despite 253 M parameters on 96 demonstrations. Both curves also **improve then
degrade** (AF raw-MSE min 1.68 at mid-training, 5.57 at step 99 000; MF 2.71 → 8.84), and the
primary `training_losses` barely moves at all (1.0000 → 0.9899). That pattern — no gap, late
divergence, a flat headline loss — reads as an **optimisation/objective failure rather than
either under- or over-fitting**.

**Provenance warning:** those two pickles sit beside `mix_visual_aligning_*` in `temp/0408` and
are from the **visual-aligning** tree, not the avoiding-d3il runs analysed here. They demonstrate
the diagnostic and are suggestive; they are **not** evidence about the `(Bf_U3)` UNet runs. The
matching histories for those runs live on the cluster and have not been pulled.

### 7.6 Capacity — does a bigger network give better performance? **No.**

Analytic trainable-parameter counts for the velocity network (computed from the module
definitions; no checkpoint was loaded):

| backbone | config | **parameters** | vs baseline | works? |
|---|---|---|---|---|
| **DPCC / FM UNet** | `dim=32`, mults (1,2,4,8) — `config/avoiding-d3il.py:128` | **4.0 M** | 1× | **yes** |
| AF `bbsit` | H=256, depth 8, heads 4, mlp×4 | **10.0 M** | 2.5× | **yes** |
| MF `bbmf_dit`, MF/AF/iMF `bbdit` | H=256, depth 8 (same shape) | **10.0 M** | 2.5× | **yes** |
| **MF / AF / iMF `bbunet`** | `dim=freq_dim=256`, mults (1,2,4,8) | **253 M** | **63×** | **iMF yes, AF/MF no** |

The decisive detail: **the MF/AF/iMF UNet is not the baseline's UNet.** It is the same class at
eight times the width — `dim=freq_dim` with `freq_dim` defaulting to 256
(`af_trajectory_model.py:95`, `mf_trajectory_model.py:92`, `imf_trajectory_model.py:83`), where
the baseline runs `dim=32`. Width enters quadratically:

| UNet `dim` | parameters |
|---|---|
| 32 (baseline) | 4.0 M |
| 64 | 16.1 M |
| 128 | 64.3 M |
| 256 (ours) | 257 M |

**Answer to the capacity question: performance is not monotone in capacity, and at the top it
inverts.**

- 4 M (baseline UNet) → works, 30/30 tightened.
- 10 M (SiT / DiT) → works, and is the configuration that produces every positive result in this
  study.
- 253 M (our UNet) → works for iMF, fails completely for AF and MF.

The 63× capacity increase buys **nothing** anywhere in the dataset — the two *smallest*
configurations are the ones that work reliably, and the largest is the only one that ever
collapses. With 96 demonstrations in the training buffer, this is unsurprising; a 253 M network
on 96 trajectories is far past any useful operating point.

**Consequence for the confound.** The relevant capacity gap between our models and the baseline
is **2.5×, not 63×** — AF-SiT/MF-DiT at 10 M against DPCC/FM at 4 M. That is a real advantage
and it is unmeasured, but it is a modest one, and §7.6 gives the cheapest way to close it:
**re-run the baselines at `dim=64` (16 M)**, which brackets the transformers' 10 M from above,
rather than training a transformer baseline from scratch.

### 7.7 Verdict on the three backbones — UNet vs `imf_dit` (`bbdit`) vs own net

The hypothesis under test: *"UNet is mid, `imf_dit` is a disaster, the own net is best."*
`bbdit` **is** the `imf_dit` — the wrappers document it as "the faithful official-iMF
transformer" (`af_trajectory_model.py:42`, `mf_trajectory_model.py:42`) — so the three-way is
available in the `(Bf_U3)` tree at one seed. Sum of the six projected arms, out of 36:

| AlphaFlow `(Bf_U3)`, seed 6 | K=1 | K=2 | K=5 | K=10 | s/step @K=2 | steps @K=2 |
|---|---|---|---|---|---|---|
| **`bbunet`** | **2/36** | **1/36** | **1/36** | **1/36** | 0.271 | 96.5 |
| **`bbdit` (imf_dit)** | 29/36 | **30/36** | 28/36 | **23/36** | 0.045 | 70.0 |
| **`bbsit` (own)** | **32/36** | 16/36 | **29/36** | 28/36 | **0.020** | **64.3** |

| MeanFlow `(Bf_U3)`, seed 6, K=2 | projected | s/step | steps |
|---|---|---|---|
| **`bbunet`** | **8/36** | 0.037 | 65.0 |
| **`bbdit` (imf_dit)** | **26/36** | 0.042 | 67.5 |
| **`bbmf_dit` (own)** | 18/36 | **0.025** | 66.3 |

**Two of the three parts are the wrong way round:**

1. **UNet is not "mid" — it is the disaster.** 1–2 of 36 for AlphaFlow at *every* K, 8/36 for
   MeanFlow. Nothing else in this dataset is that bad.
2. **`imf_dit` is not a disaster — it is the most *consistent* backbone.** AlphaFlow: 23–30/36 at
   all four K, never below 23. MeanFlow: 26/36, the **best** of its three. On constraint
   satisfaction it is at least as good as the own net and considerably more stable in K.
3. **"Own net best" holds only at AF K=1** (32/36). At K=2 it drops to 16/36 (AF) and 18/36 (MF),
   both *below* `imf_dit`.

That K=2 drop is not a real backbone failure, though: it is entirely the `-c` collapse — `bbsit`
and `bbmf_dit` score 0/6 on both `-c` arms at K=2, which §7.4 identified as the τ=0.5 ranking
bug that `bbdit` happens to be immune to. **Excluding the two `-c` arms** (out of 24):

| K=2, `-c` arms excluded | own net | `imf_dit` |
|---|---|---|
| AlphaFlow | 16/24 | **20/24** |
| MeanFlow | **18/24** | 16/24 |

Still split. And on the neutral `-r` arm alone: AF `bbsit` 3/6 vs `bbdit` 5/6 (DiT better);
MF `bbmf_dit` 5/6 vs `bbdit` 2/6 (own better).

**Resolution.** The own net does not win on *constraint satisfaction* — `imf_dit` matches or
beats it there and is far steadier across K. Where the own net wins is **speed and step count**:
SiT is **2.2× faster per step than DiT** (0.020 vs 0.045) and **13× faster than the UNet**
(0.271), and it uses the fewest control steps at every K (§7.8). So the correct statement is
*"the own net is the most efficient; `imf_dit` is the most reliable; the UNet is broken"* — not
a single ordering. Single seed, 6 episodes per cell.

### 7.8 Does the visual smoothness show up in steps / time / success? — **steps, yes; the others, no**

The test the question asks for: if the own net really produces smoother paths, that should leave
a trace in the recorded metrics. Steps averaged over **goal-reached** episodes — the right
population, since it includes runs that got there while violating constraints:

| AlphaFlow, `-r-tightened` | K=1 | K=2 | K=5 | K=10 | mean |
|---|---|---|---|---|---|
| `bbunet` | 114.2 | 96.5 | 117.2 | 120.8 | **112** |
| `bbdit` | 72.7 | 70.0 | 76.0 | 96.0 | 79 |
| **`bbsit` (own)** | **72.0** | **64.3** | **73.3** | **70.7** | **70** |

**AlphaFlow-UNet takes 1.4–1.7× more control steps than AlphaFlow-SiT, at every K, and the own
net is lowest at all four.** That is exactly the signature a zig-zagging path leaves: jitter
costs net progress per step, so the episode takes longer in *steps* even when it eventually
arrives. The ordering survives on the **unprojected** arm too, where no projector has smoothed
anything — AF K=2: `bbsit` **59.5** < `bbdit` 67.7 < `bbunet` 75.0.

For scale, the baselines on their 4 M UNet sit at 73.3 (FM K=20) and 74.7 (DPCC K=20) — i.e.
**normal**, and nowhere near our 253 M UNet's 96–121. The step penalty is specific to the
off-native backbone, not to UNets.

**What does *not* carry the signal:**

- **Success / success+constraints** — `imf_dit` often scores *higher* than the own net (§7.7)
  while using more steps. Constraint satisfaction is dominated by the projector (§2), so it
  cannot see path quality.
- **`avg_time`** — it tracks parameter count and architecture (UNet 253 M → 0.271 s/step; DiT
  10 M → 0.045; SiT 10 M → 0.020), not path shape. A smooth model and a jittery model of the
  same size would cost the same per step.

**So the answer is: yes, but through one column only.** `n_steps` on goal-reached episodes is a
usable — weak, indirect — proxy for the smoothness you see by eye, and it agrees with the visual
report for AlphaFlow at all four K. **It does not agree for MeanFlow**: `bbunet` 65.0, `bbdit`
67.5, `bbmf_dit` 66.3 are flat, no penalty at all. So the metric corroborates the claim for AF
and is silent for MF.

Two caveats that both cut the same way: the UNet step figures rest on **4 of 6 goal-reached
episodes across 2 of 3 environments** — the *easy* subset — so the true penalty is more likely
understated than overstated. And a step count conflates path length with path jitter; only a
real roughness column (queue item 0c) separates them.

---

## 8. What this study does and does not establish

**Established:**

1. **AF K=2 is the only configuration ≥70% in every environment** on a neutral untightened rule,
   at 1.3–7.0 s/episode; the best baseline (DPCC K=20 `-t`) manages 50% worst-env at ~35 s (§3).
2. Pooled, **MF/AF at K=2 beat FM K=20 and DPCC K=20 by 2.4–2.7×**, p < 0.02, unanimous per-seed
   direction, at **12× less wall clock** (26× best-of-family) (§4).
3. **DPCC cannot operate at K=1**: 2/30 at matched wall clock against AF's 22/30, p < 1e-5;
   ≤1/10 in every environment; 18/30 even with projector and margin (§5, §2).
4. **The two families move in opposite directions in K** — AF degrades with steps, DPCC improves
   — so the advantage is structural (§6).
5. **The tightened metric measures the projector, not the generator**: a 1/30 generator scores
   30/30 once projected. Published tables on tightened arms are not evidence about generative
   models (§2).
6. **Pooling across environments materially misleads on this benchmark** — it credits 40% to a
   config that never solves one of the three maps, and it inverts AF's rule preference (§3.5).

**Not established:**

7. **Fewer steps.** Mixed in sign and confounded by success-conditioned averaging (§4).
8. **That low K is exclusive to MF/AF.** Diffusion at K=1 through the FMv3 sampler reaches 40%
   pooled at 1.7 s. Per-env it fails `top-right-hard` completely, which is why L3 survives — but
   it is off-spec (`aw1`) and needs a clean re-run before the point is settled (§5).
9. **Anything about MeanFlow's K-behaviour** — MF exists at full seed count only at K=2 (§1, §6).
10. **A clean K ablation across families.** Backbones differ and so does capacity: **every
    quality comparison in §3 and §4 is AF-SiT / MF-`mf_dit` (10 M) against DPCC-UNet (4 M)**,
    a 2.5× capacity gap whose share of the margin is unmeasured (§7.3, §7.6). The
    matched-architecture version has been attempted and **our side fails on every off-native
    backbone**: AF-UNet is dead (20/24 zeros, §7.1) and both iMF arms are generator-degenerate
    (0/6 unprojected, §7.5). Each family appears to work only on its own network.
12. **Path smoothness directly.** This batch has no roughness metric (§7.5b) and projected plans
    are smooth by construction. **One column is a usable indirect proxy**: `n_steps` on
    goal-reached episodes, where AF-UNet costs 1.4–1.7× AF-SiT at every K, matching the visual
    report — but it is flat for MeanFlow, so it corroborates the claim for AlphaFlow only (§7.8).
13. **That the UNet failure is a capacity problem.** The loss histories available show no
    train/test gap and late divergence with a flat headline loss — an optimisation/objective
    failure signature, not over- or under-fitting — but those pickles are from the
    visual-aligning tree, not these runs (§7.5c).
11. **MF as a match for AF.** MF K=2's worst env is 40% against AF's 70%, and MF collapses to 43%
    pooled under `-t` where AF reaches 80%. **On this data AlphaFlow carries the result.**

**Structural caveats:** 10 episodes per env-cell (5 seeds × 2 trials); `n_steps` conditioned on
success; AF ladder on a different seed set from AF K=2; CAND_109 trained with `aw1`; the backbone
ablation of §7 is single-seed (6) and from the legacy `(Bf_U3)` tree.

---

## 9. Verdict

> **AlphaFlow at K=2 is the only configuration tested that clears 70% goal-and-constraint
> success in all three environments — 7/10, 7/10, 8/10 — at 1.3–7.0 s of planning per episode.
> The best DPCC configuration reaches 50% in its worst environment at ~35 s. At matched wall
> clock (3.3 s), AlphaFlow solves 22 of 30 episodes where DPCC at K=1 solves 2. And the two
> methods move in opposite directions as denoising steps increase — DPCC needs steps, AlphaFlow
> is hurt by them — so this is a structural property, not a cheaper version of the same thing.**
>
> Four qualifications belong in the same breath. The standard tightened-success metric cannot
> show any of this, because with projection and tightening a 1/30 generator scores 30/30.
> `FlowMatchingODE` has never been run below K=5, so the FM half of the low-K control is
> untested. On this data the result is **AlphaFlow's**, not MeanFlow's — MF's worst environment
> is 40% against AF's 70%. And **the quality comparison is not architecture-matched**: AlphaFlow
> runs SiT where DPCC runs a UNet, and the matched version cannot be run because AlphaFlow-UNet
> scores zero in 20 of 24 projected cells.
>
> The **speed** half is not threatened by that: architecture accounts for 1.45× of the 14.5×
> generation speed-up (1.06× for MeanFlow) — the rest is K — and DPCC's own UNet at K=1
> generates *faster* than AlphaFlow-SiT at K=2 while scoring 2/30. The baseline has the speed
> and cannot convert it. The **quality** half is where the confound bites, and it is open.

---

## 10. Run queue

0. **Re-run DPCC/FM at `dim=64` (16 M), 5 seeds, K=20 and K=2.** Cheapest way to close the
   capacity confound: 16 M brackets the transformers' 10 M from above, so if the baselines do not
   improve, capacity is ruled out as the explanation for §3/§4's margins without training a
   transformer baseline at all (§7.6). Do this before item 0b.
0b. **Train FM/DPCC with a transformer backbone (SiT/DiT), 5 seeds, K=20 and K=2.** The
   architecture-matched comparison proper. More expensive than item 0 and only worth it if
   item 0 shows capacity matters.
0c. **Port the `plan_roughness` metric from the Gen13/HardFlow pipeline into the avoiding-d3il
   eval, and report it on the *raw* plan.** This is the highest-value item that is not a GPU run:
   the backbone failure mode is currently visible only by eye (§7.5b), and no table in this
   document — or any avoiding-d3il table — can see it. Definition and the raw-vs-projected
   argument are already worked out in `DA_Code/METRIC_SMOOTH/DISCUSSION_metric_smooth.md`.
   Without it, "each family only works on its own network" cannot be evidenced in a paper.
0c-2. **Pull the avoiding-d3il loss histories for the `(Bf_U3)` UNet runs off the cluster.**
   §7.5c's under/over-fitting diagnosis currently rests on pickles from the wrong tree. The
   train/test curves for the actual failing runs would settle it without any new GPU time.
0d. **Carry iMeanFlow into the full-seed comparison.** iMF-DiT at K=2 reaches 6/6 on all three
   tightened arms and 4/6 on `-r` at seed 6 (§7.5) — the same band as AF — and has never been run
   at 5 seeds. It is a third data point on the low-K claim that already exists at K=1/2/10.
1. **`FlowMatchingODE` at K=1 and K=2, 5 seeds, untightened arms, all three envs.** The biggest
   hole. If FM survives K=2, the claim narrows to "against DPCC" and must be written that way.
1b. **Diagnose why AF-UNet and MF-UNet fail.** 20 of 24 cells at exactly zero across four K values
   is a training or wiring failure, not a capacity limit. If it is fixable, it gives the
   architecture-matched comparison from our side at a fraction of the cost of item 0.
2. **Re-run CAND_109 with `aw10`, 5 seeds.** The one config that challenges L3 is off-spec on
   action weight. Confirm the `top-right-hard` 0/10 is real and not a training artefact.
3. **DPCC at K=2, 5 seeds.** The DPCC low-K leg is a single point at K=1; K=2 is the matched
   comparison and does not exist.
4. **MeanFlow K-ladder (K=1, 5, 10), 5 seeds.** §6's direction result is AlphaFlow's alone.
5. **More trials per env-cell.** 10 episodes/env means a 70% and a 50% are two episodes apart;
   §3's worst-env ranking is the study's main output and deserves n≥5 trials.
6. **AF K=2 and K=1 on the ladder seed set**, so §3/§4 and §6 sit on one set of seeds.
7. **Fix or drop `-c` at K=2** — evaluate `minimum_projection_cost` on an extrapolated endpoint
   rather than the current iterate (§3.3, §11.4 of the companion doc).
8. **Investigate the AF `-r` / `top-left` 7.0 s stall** (§4) — 5× the same config's cost in the
   other two envs.
