# DA — Mission 3: HardFlow-SLSQP vs the DPCC projector, `pillars` K=5, three engines

*Gen15 · campaign `Campaign_20260907_five_missions` · 2026-09-10.
Source batch: **`temp/0909/batch_uav_20260910_092309`** (`DA_UAV_v1`, 1299 units, 0 failed).
Candidates **C50** (af) · **C74** (mf) · **C62** (fm) — all 17 variants, n=10, `u7hg`, seed 6.*

## 0. TL;DR — four answers, and one of them hurts

1. 🔴 **The claim ladder is reversed at the top. On `pillars` K=5 the ranking is `mf > fm > af`.**
   Mean S&C over 17 variants: **mf 0.635 · fm 0.359 · af 0.235**. α-Flow U-Net has **zero** cells at
   S&C ≥ 0.8; MeanFlow has 7. Corridor could not separate af from mf (`DA_20260908_T2`); `pillars`
   can, and **af comes last**. §2
2. **HardFlow does not beat the DPCC projector on S&C** — 8 wins, 9 losses, 1 tie across the 18
   matched selector pairs. But the split is entirely engine-dependent: HF wins **5/6 on fm**, loses
   **5/6 on af**. §3
3. ✅ **HardFlow's real win is cost: 8.2–21.3× cheaper projection** than full-geometry DPCC, 5.6–12.4×
   end-to-end. §4 — **but the advantage evaporates against `dpcc-*-geo_free`, which is cheaper than
   HardFlow in all 9 pairs.** §5
4. ✅ **HardFlow never produced an unsafe rollout: 0 of 210.** DPCC produced **43 of 270**. Every
   safety failure in the study is a DPCC row. §6

---

## 1. Provenance and gates

| | |
|---|---|
| scene · K · geo | `pillars` · **K=5** · `pillars_hg_bounds+dynamics+geo_bounds+obstacles` (bounds, hs=0, obs=6) |
| engines | **af** `AlphaFlowODE_9D_as1_ae0.2_bbunet` (U-Net, 3.97 M) · **mf** `MeanFlowODE_9D_dp0.5_bbunet` (U-Net) · **fm** `FlowMatchingODE_9D` |
| jobs | af **25501 + 25589** (resume) · mf **25503** · fm **25496** |
| seed · trials · variants | 6 · **n = 10** · **17/17** on every engine |
| tag | `u7hg` honest geometry (af additionally `EPlatest`) |

✅ **HardFlow genuineness gate PASSED.** `hf_degenerate = 0` and **`hf_n_genuine = 2`** on all 21
HardFlow cells. At K=5 with activation threshold A=0.5 the arm runs real HardFlow arithmetic, so
every HardFlow row below is citable — unlike the K=1/K=2 arms, which were correctly blocked.

⚠️ Single seed, n=10 per cell. Differences below ~0.2 in S&C are within sampling noise
(a 0.1 step = one rollout).

---

## 2. 🔴 The engine ranking — `af` is last

| engine | mean S&C (17 variants) | best cell | cells ≥ 0.8 |
|---|---|---|---|
| **mf** (MeanFlow U-Net) | **0.635** | **1.000** (`dpcc-t-geo_free`) | **7 / 17** |
| **fm** (naive Flow Matching) | 0.359 | 0.900 (`diffuser`) | 2 / 17 |
| **af** (α-Flow U-Net) | **0.235** | 0.700 (`dpcc-t-tightened`) | **0 / 17** |

Both orderings — by mean and by best cell — give **mf > fm > af**. The thesis ladder wants
`af > mf > fm`; the top two rungs are inverted and α-Flow does not even clear naive FM.

Taken with `DA_20260908_T2` (corridor: af ≡ mf, indistinguishable), the position is now:

| rung | corridor | **pillars K=5** |
|---|---|---|
| `af > mf` | ✗ not detectable | 🔴 **refuted** — af 0.235 vs mf 0.635 |
| `mf > fm` | ✓ supported | ✓ **supported** — 0.635 vs 0.359 |
| `fm > diffusion` | — untestable | — untestable (no diffusion arm) |

**`af > mf` is no longer merely unsupported; on the one scene with dynamic range it is contradicted.**

Note also what carries `af`: its best cell needs the projector (`diffuser` = 0.000, S&C only appears
under projection), whereas `fm`'s best cell **is** `diffuser` — the unprojected plan. §7.

---

## 3. HardFlow vs DPCC on S&C — no overall win

Matched selector, same engine, same K:

| selector | af DPCC → HF | mf DPCC → HF | fm DPCC → HF |
|---|---|---|---|
| `-r` | 0.000 → **0.200** ✅ | 0.200 → **0.900** ✅ | 0.000 → **0.400** ✅ |
| `-c` | **0.600** → 0.100 ❌ | **0.800** → 0.700 ❌ | 0.400 → **0.500** ✅ |
| `-t` | **0.600** → 0.100 ❌ | 0.400 → **0.700** ✅ | 0.400 → **0.500** ✅ |
| `-r-geo_free` | **0.500** → 0.000 ❌ | **0.900** → 0.600 ❌ | 0.200 → **0.400** ✅ |
| `-c-geo_free` | **0.300** → 0.200 ❌ | 0.400 → 0.400 = | 0.300 → **0.800** ✅ |
| `-t-geo_free` | **0.500** → 0.100 ❌ | **1.000** → 0.700 ❌ | **0.500** → 0.300 ❌ |
| **tally** | HF 1, DPCC 5 | HF 2, DPCC 3, tie 1 | **HF 5, DPCC 1** |

**Overall: HF 8 · DPCC 9 · tie 1.** No aggregate advantage.

The structure is the interesting part: **HardFlow rescues the weaker generator and degrades the
stronger one.** On naive FM it wins 5/6; on α-Flow it loses 5/6. A plausible reading is that
HardFlow's in-ODE guidance substitutes for plan quality the generator did not supply, and interferes
where the generator already produces a good plan — but this is one scene and one seed, and it is a
hypothesis, not a result.

---

## 4. ✅ The cost result — HardFlow is an order of magnitude cheaper than full DPCC

| engine | sel | DPCC `proj_ms` | HF `proj_ms` | ratio | DPCC `avg_ms` | HF `avg_ms` | ratio |
|---|---|---|---|---|---|---|---|
| af | `-r` | 2215.5 | **191.4** | **11.6×** | 2260.8 | **257.4** | 8.8× |
| af | `-c` | 1534.7 | **187.5** | 8.2× | 1580.0 | **253.4** | 6.2× |
| af | `-t` | 1580.9 | **192.7** | 8.2× | 1626.8 | **258.7** | 6.3× |
| mf | `-r` | 1870.7 | **87.8** | **21.3×** | 1915.9 | **154.0** | **12.4×** |
| mf | `-c` | 1726.4 | **124.9** | 13.8× | 1771.9 | **190.9** | 9.3× |
| mf | `-t` | 2092.0 | **128.5** | 16.3× | 2137.3 | **194.5** | 11.0× |
| fm | `-r` | 1077.0 | **64.2** | 16.8× | 1120.6 | **127.5** | 8.8× |
| fm | `-c` | 747.2 | **78.2** | 9.6× | 790.8 | **141.6** | 5.6× |
| fm | `-t` | 843.4 | **81.4** | 10.4× | 887.1 | **144.8** | 6.1× |

This is the axis the benchmark hierarchy asks HardFlow to win on, and it wins it decisively against
the full-geometry projector. HardFlow's generator is ~45 % dearer (`fm_ms` 63–66 vs 43–46 ms) because
the NLP solves live inside the ODE, but that is dwarfed by the projection saving.

*Timing is reported raw. The `budget = 30.3 ms` / 33 Hz line in the job logs is a data-rate artefact
plus cluster latency, not a real-time target; the `real_time_OVER×N` counters are not reproduced.*

---

## 5. ⚠️ …but `dpcc-*-geo_free` is cheaper than HardFlow, in all 9 pairs

| engine | sel | DPCC `proj_ms` | HF `proj_ms` | DPCC total | HF total | |
|---|---|---|---|---|---|---|
| af | `-r/-c/-t-geo_free` | 33.8 / 33.8 / 33.9 | 34.9 / 34.9 / 35.0 | **79.3 / 79.2 / 79.3** | 100.4 / 100.7 / 100.7 | HF **dearer** |
| mf | `-r/-c/-t-geo_free` | 33.7 / 33.6 / 33.5 | 34.7 / 34.8 / 34.8 | **78.8 / 78.7 / 78.7** | 100.6 / 100.7 / 100.7 | HF **dearer** |
| fm | `-r/-c/-t-geo_free` | 33.7 / 33.6 / 33.7 | 34.8 / 34.7 / 34.9 | **77.1 / 77.1 / 77.3** | 98.0 / 97.8 / 98.1 | HF **dearer** |

Once the geometry constraints are dropped, the two projectors cost **the same** (≈ 33.7 vs ≈ 34.8 ms),
and HardFlow's dearer generator makes it **~27 % more expensive end-to-end in every pair**.

**So the honest statement of the cost claim is narrower than it first appears:** HardFlow is far
cheaper than the *full-geometry* DPCC projector, but it is **not** the cheapest way to obtain a
projected plan on this scene. That title belongs to `dpcc-*-geo_free`, which also holds the single
best S&C cell in the study (mf `dpcc-t-geo_free`, S&C = 1.000, 0 violations, 78.7 ms).

---

## 6. ✅ The safety result — every unsafe rollout in the study is a DPCC row

| arm | cells | rollouts | cells with `phys_safe` < 1.0 | unsafe rollouts |
|---|---|---|---|---|
| **HardFlow** (7 × 3 engines) | 21 | 210 | **0** | **0** |
| **DPCC** (9 × 3 engines) | 27 | 270 | **8** | **43** |
| `diffuser` (unprojected) | 3 | 30 | 0 | 0 |

The eight failing DPCC cells, worst first: af `dpcc-r-tightened` **0.200**, mf `dpcc-r` **0.200**,
mf `dpcc-r-tightened` **0.200**, fm `dpcc-r` 0.300, fm `dpcc-r-tightened` 0.500, af `dpcc-r` 0.600,
af `dpcc-c-tightened` 0.800, af `dpcc-c` 0.900.

**Seven of the eight are `-r` (random candidate selection), and both `-r` and `-r-tightened` fail on
all three engines.** The random selector is not merely weaker — it is the only configuration in this
study that puts the aircraft in a physically unsafe state, and it is simultaneously the *most
expensive* projection row on every engine (`proj_ms` 1077–2215).

**This is HardFlow's strongest claim from mission 3:** at matched budget it holds S&C parity overall,
costs 8–21× less to project than full DPCC, and did not produce a single unsafe rollout in 210.

---

## 7. Projection is not universally worth it

| engine | `diffuser` (no projection) | best projected cell | verdict |
|---|---|---|---|
| **af** | **0.000** | 0.700 (`dpcc-t-tightened`) | projection is **essential** |
| **mf** | 0.900 | **1.000** (`dpcc-t-geo_free`) | marginal gain |
| **fm** | **0.900** | 0.800 (`hardflow_sls-c-geo_free`) | projection **hurts** |

For naive FM the unprojected plan is the best cell on the scene, at the lowest cost (43.1 ms, zero
violations). For α-Flow the unprojected plan never reaches the goal at all. Whether the projector
earns its cost is an engine-level question, not a global one.

---

## 8. What this licenses

**Supported.** On `pillars` K=5 with honest geometry: `mf > fm > af` on S&C by both mean and best
cell. HardFlow holds no aggregate S&C advantage over DPCC (8–9–1) but wins 5/6 on fm and loses 5/6
on af. HardFlow's projection is 8.2–21.3× cheaper than full-geometry DPCC and 5.6–12.4× end-to-end.
HardFlow produced 0 unsafe rollouts in 210; DPCC produced 43 in 270, concentrated in `-r`.
`dpcc-*-geo_free` is cheaper end-to-end than HardFlow in all 9 pairs.

**Not supported.** Any multi-seed claim (seed 6). Any S&C difference below ~0.2 (one rollout = 0.1).
Anything about the diffusion baseline (no `diffusion` arm exists on `pillars`). Any causal account of
*why* HardFlow helps fm and hurts af — §3's reading is a hypothesis.

**Next.**
1. 🔴 The `af > mf` rung now has one refutation and one non-detection. Before more af compute goes in,
   decide whether the ladder claim survives or the thesis position changes.
2. Re-run at **≥3 seeds** — every headline here rests on 10 rollouts at one seed.
3. Retire `dpcc-r` / `dpcc-r-tightened` from future sweeps, or run them only as a control: most
   expensive, only unsafe rows, and never the best cell on any engine.
4. A `diffusion` arm on `pillars` K=5 under `u7hg` would place the actual baseline.
