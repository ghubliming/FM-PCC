# DA — Mission 1: the α-Flow U-Net K-sweep on `pillars` (K = 1 / 2 / 5)

*Gen15 · campaign `Campaign_20260907_five_missions` · 2026-09-10.
Source batch: **`temp/0909/batch_uav_20260910_092309`**. Candidates **C47** (K=1) · **C49** (K=2) ·
**C50** (K=5), all `u7hg`, `EPlatest`, seed 6, n=10.
Matched on the **10 DPCC-family variants** present at every K.*

## 0. TL;DR

1. **Budget helps, but almost all of it arrives by K=2.** Mean S&C **0.060 → 0.280 → 0.320**.
   K=1→K=2 buys **+0.220**; K=2→K=5 buys **+0.040** for 2.5× the network time. §2
2. 🔴 **K=5 is where the arm starts producing physically unsafe plans.** At K=1 and K=2, **all 10
   variants are `phys_safe` = 1.000**. At K=5, **four go unsafe** — `dpcc-r-tightened` down to
   **0.200**. Budget widens the spread in both directions. §3
3. **The best cell costs 21× more at K=5 than at K=2 for +0.2 S&C** — `dpcc-t` at K=2 is 0.500 S&C
   at **75.1 ms**; `dpcc-t-tightened` at K=5 is 0.700 at **1564.5 ms**. §4
4. **The raw plan does not improve with budget — it peaks at K=2 and collapses at K=5**
   (S&C 0.000 → 0.200 → 0.000). §5

---

## 1. Provenance

| | |
|---|---|
| model | `AlphaFlowODE_9D_as1_ae0.2_bbunet` — **U-Net, 3,969,222 params (3.97 M)**, α floored at 0.2 |
| jobs | K=1 **25497** · K=2 **25499** · K=5 **25501 + 25589** (resume) |
| scene · geo | `pillars` · `pillars_hg_bounds+dynamics+geo_bounds+obstacles` (bounds, hs=0, obs=6) |
| eval tag | `Eaf_K{1,2,5}_mpc4_pid_stopgo_T0.5_EPlatest_u7hg` · seed 6 · n=10 |

**Comparison scope.** K=1 and K=2 carry **10** variants, K=5 carries **17**: the 7 HardFlow rows are
correctly **blocked as degenerate** at K≤2 (at A=0.5, `n_genuine = 0`, so the arm would be
`Π_S(Euler sample)` = DPCC in disguise). Every cross-K statement below therefore uses **only the 10
DPCC-family variants**. The K=5 HardFlow rows are analysed in
[`DA_20260910_T3_hardflow_vs_dpcc_pillars_K5.md`](DA_20260910_T3_hardflow_vs_dpcc_pillars_K5.md).

⚠️ Single seed, n=10. One rollout = 0.1 S&C; differences below ~0.2 are noise.

---

## 2. The budget axis — front-loaded gains

| | mean S&C | best cell | mean `goal_reached` | mean `phys_safe` | `fm_ms` |
|---|---|---|---|---|---|
| **K=1** | 0.060 | 0.200 | 0.060 | **1.000** | 9.3 |
| **K=2** | **0.280** | 0.500 | 0.280 | **1.000** | 18.1 |
| **K=5** | 0.320 | **0.700** | 0.320 | **0.850** | 44.6 |

Per-variant S&C:

| variant | K=1 | K=2 | K=5 |
|---|---|---|---|
| `diffuser` | 0.000 | **0.200** | 0.000 |
| `dpcc-r` | 0.000 | 0.200 | 0.000 |
| `dpcc-c` | 0.000 | 0.200 | **0.600** |
| `dpcc-t` | 0.200 | **0.500** | 0.600 |
| `dpcc-r-tightened` | 0.000 | 0.300 | 0.000 |
| `dpcc-c-tightened` | 0.000 | 0.300 | 0.000 |
| `dpcc-t-tightened` | 0.200 | 0.400 | **0.700** |
| `dpcc-r-geo_free` | 0.100 | 0.200 | 0.500 |
| `dpcc-c-geo_free` | 0.000 | 0.100 | 0.300 |
| `dpcc-t-geo_free` | 0.100 | 0.400 | 0.500 |

**`dpcc-t*` (temporal consistency) is the best selector at every K** — it holds the top cell in all
three columns. `-r` (random) never exceeds 0.500 and is 0.000 at K=5.

Four variants are **non-monotone**, peaking at K=2 and collapsing to 0.000 at K=5: `diffuser`,
`dpcc-r`, `dpcc-r-tightened`, `dpcc-c-tightened`. More sampling steps is not uniformly more skill.

---

## 3. 🔴 K=5 is where safety breaks

| K | cells with `phys_safe` < 1.0 |
|---|---|
| **1** | **none** (10/10 at 1.000) |
| **2** | **none** (10/10 at 1.000) |
| **5** | **4** — `dpcc-r-tightened` **0.200** · `dpcc-r` 0.600 · `dpcc-c-tightened` 0.800 · `dpcc-c` 0.900 |

At K=1 and K=2 the α-Flow arm never puts the aircraft in a physically unsafe state on this scene, on
any of 200 rollouts. At K=5, 4 of 10 variants do.

All four failures are `-r` or `-c`; **every `-t` and every `-geo_free` row stays at 1.000 across all
three K.** This matches the mission-3 finding that all eight unsafe DPCC cells in that study were
`-r`/`-c` rows — the random and cost-based selectors are the ones that break, and raising the budget
is what exposes them.

**Reading:** K=5 does not simply improve the arm, it *widens its spread* — the best cell improves
(0.500 → 0.700) while the worst becomes dangerous (1.000 → 0.200 safe).

---

## 4. What the best cell costs

| K | best variant | S&C | `steps_to_goal` | `avg_time_ms` |
|---|---|---|---|---|
| 1 | `dpcc-t` | 0.200 | 429.0 | **77.1** |
| **2** | `dpcc-t` | 0.500 | 421.6 | **75.1** |
| 5 | `dpcc-t-tightened` | **0.700** | **375.4** | **1564.5** |

K=5's best cell is genuinely better on both S&C (+0.200) and path length (−46 steps) — but it costs
**21× the wall-clock** of K=2's best cell. The projection is what explodes: `dpcc-t` projection runs
≈ 68 ms at K=1 and **1581 ms at K=5**, a 23× rise, because the projector works far harder on the
K=5 plan distribution.

Per the Pareto rule this is **a trade-off, not a dominance**: K=5 wins the primary axis, K=2 wins
cost by more than an order of magnitude and is the only one of the two that is uniformly safe.
**If a single operating point has to be chosen from this sweep, K=2 `dpcc-t` is the defensible one**
— 0.500 S&C, 75 ms, 10/10 variants safe.

*Timing raw; the 30.3 ms budget line in the job logs is a data-rate artefact, not a real-time target.*

---

## 5. The raw plan does not improve with budget

| K | `diffuser` S&C | `goal_reached` | `n_violations` | `track_err` |
|---|---|---|---|---|
| 1 | 0.000 | 0.000 | 162.4 | 0.328 |
| **2** | **0.200** | **0.200** | 156.0 | 0.356 |
| 5 | 0.000 | 0.000 | **197.7** | 0.339 |

Unprojected α-Flow peaks at K=2 and is back to zero at K=5, with its **highest** violation count
there. This is the pillars counterpart of the 2026-09-07 `s_curve` observation that raw-plan success
*falls* with K (0.60 → 0.10 → 0.20 at K = 1/2/5), and it says the same thing: **extra NFE is not
buying the α-Flow generator a better plan on this scene.** Whatever S&C the arm achieves at K=5 is
supplied by the projector, not the generator.

---

## 6. What this licenses

**Supported.** On `pillars` with honest geometry, the af U-Net's S&C rises 0.060 → 0.280 → 0.320
across K = 1/2/5, with ~85 % of the gain realised by K=2. `dpcc-t*` is the best selector at every K.
K=1 and K=2 are uniformly `phys_safe` = 1.000 (200 rollouts); K=5 breaks safety on 4 of 10 variants.
The best K=5 cell costs 21× the best K=2 cell for +0.200 S&C. Unprojected α-Flow does not improve
with budget.

**Not supported.** Any multi-seed claim (seed 6). Differences below ~0.2 S&C. Anything about
HardFlow at K≤2 (correctly blocked as degenerate). Any cross-engine ranking — that is mission 3,
which finds **`mf > fm > af`** on this same scene at K=5.

**Next.**
1. **K=3 is unmeasured and is where the interesting boundary sits** — it is the lowest K at which
   HardFlow becomes non-degenerate, and it lies between the safe/cheap K=2 and the unsafe/expensive
   K=5. One job would close it.
2. Investigate why `-r`/`-c` selectors lose physical safety specifically at K=5 while `-t` and
   `-geo_free` do not.
3. Read this together with mission 3: the arm's budget behaviour is not the reason it loses to
   MeanFlow — it loses at K=5, its own best budget.
