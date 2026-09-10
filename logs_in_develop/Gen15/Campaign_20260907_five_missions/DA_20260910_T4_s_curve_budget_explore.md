# DA — Mission 4: is the `s_curve` failure a **budget** failure?

*Gen15 · campaign `Campaign_20260907_five_missions` · 2026-09-10.
Source batch: **`temp/0909/batch_uav_20260910_092309`**. Candidates **C93** (fm K=2) · **C92** (fm K=20) ·
**C96** (mf K=2) · **C95** (mf K=10) — all `s_curve`, `u7hg`, `pid_stopgo`, seed 6, n=10.*

## 0. TL;DR — **no**, and raising K makes it worse

1. 🔴 **Across all 40 `u7hg` `s_curve` cells in this batch, the maximum S&C is 0.100.** Four cells
   reach it; every other cell is 0.000. The scene has no usable dynamic range on the primary axis
   at any budget, engine or projector tested. §2
2. 🔴 **More NFE strictly degrades the scene.** fm K=2 → K=20: `goal_reached` falls on **all five**
   shared variants (0.70→0.60, 0.70→0.20, 0.60→0.00, 0.70→0.00, 0.80→0.60) at **9.9×** the network
   cost. mf K=2 → K=10 falls or flatlines on all four, at 4.9×. §3
3. **At matched K=2, `fm` beats `mf` on this scene by a wide margin** — mean `goal_reached`
   **0.630 vs 0.100**, mean `phys_safe` **0.620 vs 0.070**. That is the **reverse** of mission 3's
   `pillars` K=5 result (`mf > fm`). Engine ranking on this arm is **scene-dependent**. §4
4. **High-K HardFlow reaches the goal by scraping the floor** — mf K=10 HardFlow rows post
   `goal_reached` up to 0.900 at `phys_min_z` **0.008–0.183**. This independently reproduces the
   mission-5 finding across the whole HardFlow family. §5

Together with mission 5 (`mjpc` fixes goal-reaching but not S&C), **`s_curve` is now excluded as a
ranking scene on two independent grounds: not a budget problem, not a controller problem.** §6

---

## 1. Provenance

| candidate | engine | K | variants | job |
|---|---|---|---|---|
| **C93** | fm | 2 | 10 | earlier sweep |
| **C92** | fm | **20** | 9 | 25500 + **25588** (resume) |
| **C96** | mf | 2 | 10 | earlier sweep |
| **C95** | mf | **10** | 8 | 25502 |

All `u7hg` honest geometry, `s_curve_hg_bounds+dynamics+geo_bounds+halfspace+obstacles`
(bounds, hs=4, obs=2), `controller=pid_stopgo`, seed 6, **n=10** on every cell.

The high-K jobs carry the U9 8-variant subset, so the budget axis is compared on the **shared**
variants only: 5 for fm (`diffuser`, `dpcc-c`, `dpcc-r`, `dpcc-t`, `dpcc-t-geo_free`), 4 for mf.
HardFlow rows exist only at high K (degenerate and correctly blocked at K=2), so §5 is within-K.

⚠️ Single seed, n=10 per cell.

---

## 2. 🔴 The ceiling: S&C = 0.100, everywhere

Every `u7hg` `s_curve` cell in the batch, ranked:

| candidate | engine | K | variant | S&C | goal | safe |
|---|---|---|---|---|---|---|
| C93 | fm | 2 | `dpcc-t` | **0.100** | 0.700 | 0.700 |
| C93 | fm | 2 | `dpcc-t-geo_free` | **0.100** | 0.800 | 0.800 |
| C95 | mf | 10 | `hardflow_sls-c` | **0.100** | 0.400 | 0.100 |
| C95 | mf | 10 | `hardflow_sls-t` | **0.100** | 0.900 | 0.200 |
| *all remaining 36 cells* | | | | **0.000** | | |

**Max S&C over 40 cells = 0.100** — a single successful, constraint-clean rollout out of ten, in the
four best cells the scene has ever produced. For comparison, `pillars` K=5 reaches 1.000 (mf
`dpcc-t-geo_free`) and `corridor` saturates at 1.000 on 30 of 40 cells.

Note the decoupling in the table: `hardflow_sls-t` reaches the goal **0.900** of the time and still
scores S&C 0.100, because `phys_safe` is 0.200. On this scene **goal-reaching and constraint-clean
flight are nearly disjoint events.**

---

## 3. 🔴 The budget axis runs the wrong way

### fm: K=2 → K=20 (5 shared variants)

| variant | S&C K2 → K20 | `goal_reached` K2 → K20 | `fm_ms` |
|---|---|---|---|
| `diffuser` | 0.000 → 0.000 | 0.700 → **0.600** | 17.4 → 169.5 |
| `dpcc-c` | 0.000 → 0.000 | 0.700 → **0.200** | 17.5 → 171.9 |
| `dpcc-r` | 0.000 → 0.000 | 0.600 → **0.000** | 17.5 → 171.7 |
| `dpcc-t` | **0.100 → 0.000** | 0.700 → **0.000** | 17.5 → 172.0 |
| `dpcc-t-geo_free` | **0.100 → 0.000** | 0.800 → **0.600** | 17.5 → 171.5 |

**Every variant gets worse.** Both cells that held S&C 0.100 at K=2 fall to 0.000 at K=20. The
network cost rises **9.9×**.

### mf: K=2 → K=10 (4 shared variants)

| variant | `goal_reached` K2 → K10 | `fm_ms` |
|---|---|---|
| `diffuser` | 0.100 → **0.000** | 18.1 → 88.4 |
| `dpcc-c` | 0.000 → 0.000 | 18.3 → 90.1 |
| `dpcc-r` | 0.000 → 0.000 | 18.3 → 90.2 |
| `dpcc-t` | 0.200 → **0.100** | 18.4 → 89.9 |

Same direction, 4.9× the cost.

**Mission 4's question is answered: the `s_curve` failure is not a budget failure.** Ten times the
sampling budget does not recover the scene — it costs an order of magnitude more and returns strictly
fewer goals. This is the same non-monotonicity seen on `pillars` (mission 1 §5) and on `s_curve` in
the 2026-09-07 af sweep, now measured on two more engines at much larger K.

---

## 4. ⛔ RETRACTED — "at matched K=2, `fm` beats `mf` on `s_curve`"

> **Withdrawn by [`CLOSURE_20260910_uav_engine_ladder_final.md`](CLOSURE_20260910_uav_engine_ladder_final.md) §3.1.**
> Both arms below ran `pid_stopgo`, and the funnel's Stage 1 shows the tracker — not the engine —
> decides the outcome on this scene: the *same* MeanFlow model at the *same* K=10 scores
> `goal_reached` **0.000 under `pid_stopgo` and 1.000 under `mjpc`** (T5, paired, p = 0.0035).
> The comparison is tracker-confounded and is **not** an engine result. The numbers are kept below
> as a record; the "reversal" conclusion is retracted.

### (retained for the record)

Ten matched variants, same scene, same K, same tag, same seed:

| | mean S&C | best S&C | mean `goal_reached` | mean `phys_safe` |
|---|---|---|---|---|
| **fm** (C93) | **0.020** | **0.100** | **0.630** | **0.620** |
| **mf** (C96) | 0.000 | 0.000 | 0.100 | 0.070 |

`fm` reaches the goal **6.3×** as often and stays physically safe **8.9×** as often. `mf` posts
`phys_safe` = 0.000 on **7 of 10** variants.

Set against mission 3, where on `pillars` K=5 the mean S&C ordering is **mf 0.635 > fm 0.359 > af
0.235**, this is a direct reversal. **There is no scene-independent ordering of these engines in the
current data** — `mf` dominates on `pillars`, `fm` dominates on `s_curve`. Any ladder claim that
quotes one scene is quoting a scene effect.

---

## 5. High-K HardFlow: arrives, but on the floor

| engine · K | variant | S&C | goal | safe | `n_viol` | **`phys_min_z`** | `proj_ms` |
|---|---|---|---|---|---|---|---|
| mf K=10 | `hardflow_sls` | 0.000 | **0.900** | 0.000 | 97.6 | **0.101** | 294.9 |
| mf K=10 | `hardflow_sls-r` | 0.000 | 0.700 | 0.000 | 166.9 | **0.008** | 1587.5 |
| mf K=10 | `hardflow_sls-c` | 0.100 | 0.400 | 0.100 | 150.8 | **0.146** | 1889.7 |
| mf K=10 | `hardflow_sls-t` | 0.100 | **0.900** | 0.200 | 79.8 | **0.183** | 1067.2 |
| fm K=20 | `hardflow_sls` | 0.000 | 0.400 | 0.300 | 52.1 | 0.629 | 886.6 |
| fm K=20 | `hardflow_sls-r` | 0.000 | 0.400 | 0.200 | 40.5 | 0.568 | 3163.1 |
| fm K=20 | `hardflow_sls-c` | 0.000 | **0.800** | **0.800** | **31.9** | **0.840** | 2787.0 |
| fm K=20 | `hardflow_sls-t` | 0.000 | 0.500 | 0.200 | 36.5 | 0.653 | 3437.6 |

**All four mf K=10 HardFlow rows fly at `min_z` ≤ 0.183** — ground level. Mission 5 found exactly
this for `hardflow_sls-r` under `pid_stopgo` and showed `mjpc` lifts the same plans to ≈ 1.1 m;
**the pathology is now confirmed across the entire HardFlow family at this budget**, not a
single-variant artefact.

`fm` K=20 `hardflow_sls-c` is the one high-K row that both arrives (0.800) and stays airborne
(`min_z` 0.840, `phys_safe` 0.800, fewest violations at 31.9) — yet still scores **S&C 0.000**. Its
projection costs 2787 ms.

---

## 6. Verdict on `s_curve`

Three independent attempts to recover this scene have now been measured:

| hypothesis | tested by | result |
|---|---|---|
| more sampling budget | **mission 4** (this DA) | ❌ strictly worse, 5–10× the cost |
| a stronger tracker | mission 5 | ⚠️ fixes goal-reaching, **S&C unchanged at 0.000** |
| a different engine | §4 + mission 3 | ⚠️ changes the ordering, ceiling still 0.100 |

**`s_curve` should be retired as a ranking scene** until its constraint set is revisited. It is not
measuring policy quality; it is measuring which failure mode a configuration lands in.

That leaves the campaign with `corridor` saturated at 1.000, pre-U7 `pillars` void, `s_curve` floored
at 0.100 — and **only post-U7 `pillars` K=5 carrying usable dynamic range**, which is why mission 3
is the campaign's load-bearing result.

---

## 7. What this licenses

**Supported.** Max S&C across 40 `u7hg` `s_curve` cells is 0.100. Raising K degrades every shared
variant on both engines (fm 5/5, mf 4/4 down-or-flat) at 4.9–9.9× the network cost. At matched K=2,
fm > mf on this scene (goal 0.630 vs 0.100, safe 0.620 vs 0.070). All four mf K=10 HardFlow rows fly
at `min_z` ≤ 0.183.

**Not supported.** Any multi-seed claim (seed 6). Differences below ~0.2. A scene-independent engine
ordering — §4 and mission 3 disagree by design. Anything about `af` on `s_curve` at these budgets
(no af arm in this batch at K=10/20).

**Next.**
1. Treat the **constraint set**, not the policy, as the `s_curve` defect. `phys_safe` = 0.000 on 7 of
   10 mf K=2 variants suggests the scene's geometry may be close to infeasible for this platform.
2. If `s_curve` is kept, report `phys_min_z` beside `goal_reached` always — §5 shows the two can
   disagree completely.
3. Do not spend further budget-sweep compute on this scene.
