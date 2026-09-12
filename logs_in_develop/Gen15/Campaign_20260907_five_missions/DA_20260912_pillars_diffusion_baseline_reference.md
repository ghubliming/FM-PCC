# DA — the DPCC-diffusion baseline on `pillars`: a reference row

*Gen15 · campaign `Campaign_20260907_five_missions` · 2026-09-12.
Source batch: **`temp/1209/batch_uav_20260912_201035`**. Candidate **C59** — `diffusion`, `pillars`,
**K = 20**, `u7hg`, seed 6, **n = 10** on all 5 cells.
Jobs: **25611 → train 25635 → eval 25636** (4 variants) + **25681** (the 5th, after 25636 hit its
24 h wall). Closes [`CLOSURE_20260910…`](CLOSURE_20260910_uav_engine_ladder_final.md) §7.2.*

> 🔴 **This is a REFERENCE ROW, not a matched-budget claim.** Single seed; the budgets are
> deliberately unmatched (**K is a *training* parameter for DPCC diffusion** — the checkpoint is
> literally `…GaussianDiffusion_9D_K20` — while it is inference-only for the flow family, which runs
> at K=5); and only the DPCC projector family was run (5 variants, no HardFlow). Its purpose is to
> put the reader's scale on the page. Cite it as a reference, in a table captioned as such.

## 0. TL;DR

1. **The diffusion baseline never reaches the goal on `pillars`: `goal_reached` = 0.000 on all five
   variants, S&C 0.000.** §2
2. **It is not crashing — it is missing.** `phys_safe` = 1.000, zero divergences, normal altitude
   (`phys_min_z` 1.127); it simply stops **0.50 – 0.57 m** short of a goal threshold that sits
   around 0.3–0.4 m, and clips the pillars 101–156 times on the way. §3
3. **It is also the most expensive engine on the scene by a wide margin** — 182.9 ms/step
   unprojected (4.1× the flow family's ~44) and 3612–4425 ms/step projected. §4
4. **MeanFlow at K=5 solves the scene for 78.7 ms/step.** The baseline at K=20 does not solve it at
   any price. §5

## 1. Provenance and gates

| | |
|---|---|
| candidate | **C59** · `diffusion` · `pillars` · **K = 20** · `pid_stopgo` · `u7hg` |
| model | `H8_Dmodels.ddpm_diffusion.GaussianDiffusion_9D_K20` (trained by job 25635 — no pillars diffusion checkpoint existed before) |
| variants | **5**, DPCC family only: `diffuser`, `dpcc-c`, `dpcc-t`, `dpcc-c-tightened`, `dpcc-t-tightened` |
| seed · n | 6 · **10 on every cell** |

**Gates green:** `n_cb_tripped` = 0, `cb_sentinel` = 0, `timing_missing` = 0, `hf_degenerate` = 0,
`divergence_aborted` = **0.000 on all five**.

`-r` was excluded deliberately (T3: the only unsafe family, never the best cell) and HardFlow is
irrelevant to a DPCC-diffusion reference.

## 2. The row

| variant | S&C | `success` | `goal_reached` | `cfree` | `n_violations` | `phys_safe` | `avg_ms` |
|---|---|---|---|---|---|---|---|
| `diffuser` | 0.000 | 0.000 | **0.000** | 0.000 | 132.30 | 1.000 | **182.94** |
| `dpcc-c` | 0.000 | 0.000 | **0.000** | 0.000 | 142.80 | 1.000 | 3611.55 |
| `dpcc-t` | 0.000 | 0.000 | **0.000** | 0.000 | 156.10 | 1.000 | 3755.54 |
| `dpcc-c-tightened` | 0.000 | 0.000 | **0.000** | 0.100 | 139.10 | 1.000 | 4330.64 |
| `dpcc-t-tightened` | 0.000 | 0.000 | **0.000** | **0.300** | **101.00** | 1.000 | 4424.50 |

`steps_to_goal` and `track_err` are `nan` throughout — a consequence of never arriving, not missing
data.

The projector does work here: `dpcc-t-tightened` cuts violations 132.3 → **101.0** (−24 %) and gets
**3 of 10** rollouts collision-free. It just cannot convert that into a completed traverse.

## 3. The failure is a near-miss, not a crash

`goal_dist` against `goal_reached`, across engines in the same batch — this locates the threshold:

| arm | variant | `goal_dist` | `goal_reached` |
|---|---|---|---|
| fm K=5 | `diffuser` | 0.303 | 0.900 |
| mf K=5 | `diffuser` | 0.361 | 0.900 |
| mf K=5 | `dpcc-t-tightened` | 0.366 | 0.800 |
| fm K=5 | `dpcc-t` | 0.381 | 0.400 |
| mf K=5 | `dpcc-t` | 0.478 | 0.400 |
| fm K=5 | `dpcc-t-tightened` | 0.494 | **0.000** |
| **diffusion K=20** | `dpcc-t-tightened` | **0.499** | **0.000** |
| **diffusion K=20** | `dpcc-t` | **0.512** | **0.000** |
| **diffusion K=20** | `diffuser` | **0.567** | **0.000** |

Below ~0.37 m the goal is reached 80–90 % of the time; at ~0.49 m and beyond, never. **Every
diffusion cell sits at 0.499–0.567** — consistently on the far side of the line, and consistently
worse than the worst flow-family cell. The aircraft flies safely at normal altitude and stops short.

## 4. Cost

| engine | K | cheapest cell | `avg_ms` | best-S&C cell | `avg_ms` |
|---|---|---|---|---|---|
| fm | 5 | `diffuser` | **43.06** | `diffuser` (0.900) | 43.06 |
| mf | 5 | `diffuser` | 44.72 | **`dpcc-t-geo_free` (1.000)** | **78.74** |
| af | 5 | `diffuser` | 44.62 | `dpcc-t-tightened` (0.700) | 1564.50 |
| **diffusion** | **20** | `diffuser` | **182.94** | — *(none exceeds 0.000)* | 3612 – 4425 |

The **generator alone** costs 182.9 ms/step against ~44 for all three flow engines — **4.1×**, which
is the K=20-vs-K=5 gap made concrete. Adding the projector takes it to 3.6–4.4 s per control step.

*Raw times. The `budget = 30.3 ms` / 33 Hz line in the job logs is a data-rate artefact plus cluster
latency, not a real-time target, and is not reproduced as a verdict.*

## 5. What the reference row says

> **MeanFlow at K = 5 attains S&C 1.000 on `pillars` at 78.7 ms per control step. The DPCC-diffusion
> baseline at its own K = 20 operating point attains S&C 0.000 at 183–4425 ms per control step, and
> never reaches the goal on any of 50 rollouts.**

That is the sentence the row exists to support, and it is the strongest form the UAV leg can carry
without a second seed. It also supplies what CLOSURE §4 flagged as missing: **the bottom rung of the
ladder now has UAV evidence.** `mf ≫ diffusion` holds here by a margin that does not need statistics
— one engine solves the scene and the other does not finish it.

## 6. What this licenses

**Supported.** On `pillars` with honest geometry, `diffusion` K=20 scores S&C 0.000 and
`goal_reached` 0.000 on all five DPCC variants at n=10, with zero divergences and `phys_safe` 1.000;
it stops 0.50–0.57 m short against a threshold near 0.35 m; its generator costs 4.1× the flow
family's and its projected cells 3.6–4.4 s/step; `dpcc-t-tightened` reduces its violations 24 % and
clears 3/10 rollouts without ever completing one.

**Not supported.** Any matched-budget claim (K=20 vs K=5, by construction). Any multi-seed claim
(seed 6). Anything about HardFlow on this arm (not run). Any statement about `diffusion` on
`corridor` (no arm exists) — the `s_curve` diffusion arm is C97, analysed separately.

**Next.** None for this row — it is complete at 5/5 and n=10, and its purpose is served. The
outstanding UAV run remains **seeds on `pillars` K=5** (CLOSURE §7.1), which would also let this
baseline be restated with a confidence interval rather than as a reference.
