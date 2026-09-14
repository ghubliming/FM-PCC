# DA — the DPCC-diffusion baseline on `pillars`: a reference row (rev 2, full)

*Gen15 · campaign `Campaign_20260907_five_missions` · first written 2026-09-12, **fully redone 2026-09-13**.
Source: **`temp/1209/batch_uav_20260912_201035`**: `uav_aggregated_long.csv` (mask `all`), `per_rollout_detail.csv`,
`data_quality.csv`, `candidate_axes.csv`. Diffusion training facts come from `temp/1209/2026-09-10/15_08_10_uav_mix_train_25635.log`.*

> 🔴 **rev 2 replaces rev 1.** Rev 1 said the baseline "never reaches the goal", "stops short" and "does not finish".
> **That was wrong:** every diffusion rollout crosses the finish line. Rev 1 reported only the strict goal metric
> and misread `goal_dist` (a final distance to the goal point) as a distance short of the line. Rev 1's strict
> numbers were correct and are unchanged here. Rev 2 adds the crossed-line definition, the like-for-like flow
> candidates at every K, the DA-target comparison, route choice, training convergence and parameter counts.

> **Reference row, not a matched-budget claim.** One seed. K is a *training* parameter for DPCC diffusion
> (checkpoint `…GaussianDiffusion_9D_K20`) but inference-only for the flow family, so budgets are unmatched by
> construction. Diffusion runs no HardFlow (no velocity field). Cite it in a table captioned as a reference.

---

## 0. TL;DR

1. **The diffusion baseline completes every traverse:** it crosses the finish line on **50/50** rollouts. It **never
   passes within 0.30 m of the goal point** (strict success 0/50), and it violates the pillar clearance on **101–156
   steps per flight**. The clearance is scored with the 0.31 m drone-body margin; there are zero physical contacts in MuJoCo. §3
2. **S&C: strict 0.00 on all five variants; crossed-line ≤ 0.30** (`dpcc-t-tightened`). The flow family's top cells at K = 5
   reach **strict 0.70–1.00 and crossed 0.90–1.00**: mf 1.00 / 1.00, fm 0.90 / 1.00, af 0.70 / 0.90. mf's and fm's
   strict tops are generator-only cells (`geo_free`, `diffuser`); the top enforced-projection cells are 0.90 / 0.50 / 0.70 (§5.1). The gap is carried by **clearance violations and goal precision,
   not by finishing**: crossing the line is 1.00 for every engine. §2, §4
3. **It is not route collapse.** Diffusion flies (R,R,R) on all 10 unprojected flights, but fm K = 5 does too and
   still reaches the goal on 9/10. Diffusion ends 0.57 m from the goal point on average; fm ends 0.30 m. §6
4. **Training converged.** 100 k steps, test loss 0.528 → **0.00172**, with the same U-Net backbone as fm
   (**3,955,177 params**). The failure is not an undertrained checkpoint. §1
5. **Against the DA Target** (diffusion's top projection cell, `dpcc-t-tightened`: S&C 0.00 / 0.30, 634 steps,
   4 425 ms/step), **every engine at K = 5 has cells that are Pareto-dominant**: higher S&C, fewer steps and lower time. §5
6. **Architecture-matched:** all four engines are 1-D U-Nets of 3.96–3.97 M parameters. **But the diffusion arm
   uses `action_weight = 1`, not DPCC's 10** (U3 decision), so this is "DDPM under Gen15's UAV config", not a
   reproduction of the DPCC paper row. §8

---

## 1. Provenance and gates

| candidate | engine | K | backbone · params | tag | rows used |
|---|---|---|---|---|---|
| **C59** | **diffusion** (`GaussianDiffusion_9D_K20`) | **20** (train) | U-Net · **3,955,177** | `u7hg` | 5 variants |
| C78 | mf (`MeanFlowODE_9D_dp0.5_bbunet`) | 5 | U-Net · 3,969,222 | `u7hg` | 17 variants |
| C66 | fm (`FlowMatchingODE_9D`) | 5 | U-Net · 3,955,177 | `u7hg` | 17 variants |
| C53 | af (`AlphaFlowODE_9D_as1_ae0.2_bbunet`, EP latest) | 5 | U-Net · 3,969,222 | `u7hg` | 17 variants |
| C74 / C64 / C52 | mf / fm / af | 2 | as above | `u7hg` | 10 each |
| C50 | af | 1 | as above | `u7hg` | 10 |

**Common to all rows:**
- scene `pillars`, geometry `pillars_hg_bounds+dynamics+geo_bounds+obstacles` (honest geometry; **no halfspaces**, so the HardFlow `x_active` bug does not apply);
- `pid_stopgo`, mpc4, T = 0.5, seed 6, **n = 10 per cell**;
- default projector (geometry on the actual drone `p`), `FMPCC_SAFE_EPS_FRAC=1e-3`.

mf and fm have **no `u7hg` K = 1 candidate** in this batch (coverage gap).

**Gates, all 8 candidates:**
- 0 circuit-breaker trips, 0 missing timing, npz complete, no degenerate HardFlow cell;
- divergence aborts 0.00 everywhere except one fm K = 5 cell at 0.10.

**Diffusion training (job 25635):**
- `engine=diffusion backbone=unet params=3,955,177 (3.96 M)`
- `action_weight: 1`, `loss_type: l2`, `n_train_steps: 100000`
- test loss 0.528 (init) → 0.0602 (5 k) → **0.00172 (final)**; train loss 0.0006

---

## 2. Metric definitions (read before the tables)

| metric | definition in the eval |
|---|---|
| `goal_crossed_line` / `success_relaxed` | the drone was **ever on the goal side** of the finish line (and safe) |
| `goal_reached` / `success_strict` | the drone was **ever within 0.30 m** of the goal point (3.2, 0, z) (and safe); this is the paper metric used for pillars / s_curve |
| S&C strict / S&C crossed | the respective success **and** zero constraint violations |
| `cfree` | collision-free: zero violating steps |
| `viol` | violating steps per rollout, scored on the actual drone |
| `goal_dist` | **final** 3-D distance to the goal point, not a distance short of the line |
| `steps` | steps to the goal latch; **634 = full budget**, i.e. the strict goal was never latched |
| `avg_ms` | wall time per control step = `fm_ms` + `proj_ms` (raw cluster time, not a real-time verdict) |

---

## 3. The diffusion row (C59, K = 20)

| variant | strict succ | crossed succ | **S&C strict** | **S&C crossed** | cfree | viol (± sd) | goal_dist | steps | safe | div | avg_ms | fm_ms | proj_ms |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `diffuser` | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 132.3 ± 15.3 | 0.567 | 634 | 1.00 | 0.00 | 182.9 | 182.9 | 0.0 |
| `dpcc-c` | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 142.8 ± 18.5 | 0.511 | 634 | 1.00 | 0.00 | 3611.5 | 176.1 | 3435.5 |
| `dpcc-t` | 0.00 | **1.00** | 0.00 | 0.00 | 0.00 | 156.1 ± 24.2 | 0.512 | 634 | 1.00 | 0.00 | 3755.5 | 176.4 | 3579.2 |
| `dpcc-c-tightened` | 0.00 | **1.00** | 0.00 | 0.10 | 0.10 | 139.1 ± 50.1 | 0.501 | 634 | 1.00 | 0.00 | 4330.6 | 176.2 | 4154.4 |
| `dpcc-t-tightened` | 0.00 | **1.00** | 0.00 | **0.30** | **0.30** | 101.0 ± 85.0 | 0.499 | 634 | 1.00 | 0.00 | 4424.5 | 176.1 | 4248.4 |

**Reading:**
- The aircraft is always safe (no crash, normal altitude, `phys_contact_frac` 0.000 on every cell) and **always crosses the finish line**.
- The 101–156 violating steps are breaches of the scored clearance (pillar radius + 0.31 m drone body), not MuJoCo contacts.
- It **never passes within 0.30 m of the goal point**: every cell sits at 0.50–0.57 m final distance and uses the full 634-step budget.
- **The projector helps.** `dpcc-t-tightened` cuts violations 132 → 101 (−24 %) and makes 3/10 flights collision-free, which is the only non-zero S&C (crossed, 0.30).
- Its cost is high: 3.4–4.2 s of projection per step at K = 20, because T = 0.5 projects 10 of the 20 denoising steps.

---

## 4. The same five variants, every engine (K = 5 main, K ≤ 2 below)

All rows are U-Nets: mf/af 3.97 M, fm/diffusion 3.96 M params.

| variant | engine | K | S&C strict | S&C crossed | strict succ | crossed succ | cfree | viol | steps | avg_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| `diffuser` | **diffusion** | 20 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 132.3 | 634 | 182.9 |
| `diffuser` | mf | 5 | **0.90** | **0.90** | 0.90 | 1.00 | 0.90 | 18.2 | 445 | 44.7 |
| `diffuser` | fm | 5 | **0.90** | **1.00** | 0.90 | 1.00 | 1.00 | 0.0 | 469 | 43.1 |
| `diffuser` | af | 5 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 197.7 | 634 | 44.6 |
| `dpcc-c` | **diffusion** | 20 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 142.8 | 634 | 3611.5 |
| `dpcc-c` | mf | 5 | 0.80 | 0.80 | 0.80 | 1.00 | 0.80 | 46.9 | 431 | 1771.9 |
| `dpcc-c` | fm | 5 | 0.40 | 1.00 | 0.40 | 1.00 | 1.00 | 0.0 | 558 | 790.8 |
| `dpcc-c` | af | 5 | 0.60 | 0.90 | 0.60 | 0.90 | 0.90 | 22.9 | 489 | 1580.0 |
| `dpcc-t` | **diffusion** | 20 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 156.1 | 634 | 3755.5 |
| `dpcc-t` | mf | 5 | 0.40 | 0.40 | 0.40 | 1.00 | 0.40 | 140.7 | 524 | 2137.3 |
| `dpcc-t` | fm | 5 | 0.40 | 0.80 | 0.40 | 1.00 | 0.80 | 42.7 | 550 | 887.1 |
| `dpcc-t` | af | 5 | 0.60 | 0.80 | 0.60 | 1.00 | 0.80 | 41.3 | 480 | 1626.8 |
| `dpcc-c-tightened` | **diffusion** | 20 | 0.00 | 0.10 | 0.00 | 1.00 | 0.10 | 139.1 | 634 | 4330.6 |
| `dpcc-c-tightened` | mf | 5 | 0.80 | 0.90 | 0.80 | 1.00 | 0.90 | 20.1 | 432 | 1594.0 |
| `dpcc-c-tightened` | fm | 5 | 0.10 | 0.90 | 0.10 | 1.00 | 0.90 | 20.8 | 614 | 863.2 |
| `dpcc-c-tightened` | af | 5 | 0.00 | 0.00 | 0.00 | 0.80 | 0.00 | 225.8 | 634 | 2374.1 |
| `dpcc-t-tightened` | **diffusion** | 20 | 0.00 | 0.30 | 0.00 | 1.00 | 0.30 | 101.0 | 634 | 4424.5 |
| `dpcc-t-tightened` | mf | 5 | 0.80 | 0.80 | 0.80 | 1.00 | 0.80 | 53.2 | 417 | 1765.6 |
| `dpcc-t-tightened` | fm | 5 | 0.00 | 0.40 | 0.00 | 1.00 | 0.40 | 131.6 | 634 | 1175.2 |
| `dpcc-t-tightened` | af | 5 | 0.70 | 0.90 | 0.70 | 1.00 | 0.90 | 22.4 | 453 | 1564.5 |

**Win / tie / loss against diffusion on the five shared variants** (S&C; W = higher, T = equal, L = lower):

| engine | K | strict S&C | crossed S&C | exceptions |
|---|---|---|---|---|
| mf | 5 | **5 W** | **5 W** | none |
| fm | 5 | 4 W · 1 T | **5 W** | strict tie on `dpcc-t-tightened` (0.00) |
| af | 5 | 3 W · 2 T | 3 W · 1 T · **1 L** | ties on `diffuser` (0.00); `dpcc-c-tightened`: tie strict, **loses** crossed (0.00 vs 0.10) |
| mf | 2 | **5 W** | **5 W** | none |
| fm | 2 | **5 W** | 4 W · **1 L** | **loses** crossed `dpcc-t-tightened` (0.20 vs 0.30) |
| af | 2 | **5 W** | **5 W** | none |
| af | 1 | 2 W · 3 T | 2 W · 2 T · **1 L** | ties `diffuser`, `dpcc-c`; loses crossed `dpcc-c-tightened` (0.00 vs 0.10) |

- **Unprojected generator:** mf and fm fly the scene at 0.90 strict S&C at K = 5, while diffusion and af are at 0.00
  (af's raw plans violate 197.7 steps, more than diffusion's 132.3).

**K ≤ 2, same variants** (S&C strict / crossed; steps; avg_ms):

| variant | mf K=2 | fm K=2 | af K=2 | af K=1 |
|---|---|---|---|---|
| `diffuser` | 0.30 / 0.30; 570; 18.2 | 0.30 / 0.30; 579; 17.1 | 0.20 / 0.20; 595; 18.1 | 0.00 / 0.00; 634; 9.3 |
| `dpcc-c` | 0.40 / 0.40; 555; 80.6 | 0.20 / 0.20; 595; 95.3 | 0.20 / 0.20; 596; 93.8 | 0.00 / 0.00; 634; 106.2 |
| `dpcc-t` | 0.40 / 0.40; 551; 79.8 | 0.30 / 0.30; 572; 91.0 | 0.50 / 0.50; 528; 75.1 | 0.20 / 0.20; 593; 77.1 |
| `dpcc-c-tightened` | 0.40 / 0.40; 553; 80.2 | 0.50 / 0.50; 540; 78.9 | 0.30 / 0.30; 575; 86.0 | 0.00 / 0.00; 634; 120.6 |
| `dpcc-t-tightened` | 0.40 / 0.40; 550; 80.4 | 0.20 / 0.20; 594; 104.3 | 0.40 / 0.40; 549; 84.1 | 0.20 / 0.70; 592; 85.0 |

At K = 2 (2 NFE vs diffusion's 20), mf and af beat diffusion on every shared variant under both definitions, and fm
on all but crossed `dpcc-t-tightened`. Their projected cells cost **75–104 ms/step vs 3 611–4 425 ms** (35–59× less),
and the unprojected ones 17–18 vs 183 ms (~10× less).

---

## 5. Top cells and the DA Target

### 5.1 Top cell per candidate

Ties are broken by fewer steps, then lower `avg_ms`.

| engine | K | top **enforced** cell (strict S&C) | strict S&C | top enforced cell (crossed S&C) | crossed S&C | top cell incl. unprojected / `geo_free` |
|---|---|---|---|---|---|---|
| **diffusion** | 20 | *(all 0.00)* | **0.00** | `dpcc-t-tightened` | **0.30** | `diffuser` 0.00 / 0.00 |
| mf | 5 | `hardflow_sls-r` ✅ (442 st, 154 ms) | **0.90** | `dpcc-c-tightened` (432 st) | 0.90 | `dpcc-t-geo_free` **1.00 / 1.00** (411 st, 78.7 ms) |
| fm | 5 | `hardflow_sls-t` ✅ (536 st, 145 ms) | 0.50 | `dpcc-c` (558 st) | **1.00** | `diffuser` 0.90 / 1.00 (469 st, 43.1 ms) |
| af | 5 | `dpcc-t-tightened` (453 st, 1565 ms) | 0.70 | `dpcc-t-tightened` | 0.90 | same |
| mf | 2 | `dpcc-r` (491 st, 60 ms) | 0.70 | `dpcc-r` | 0.70 | same |
| fm | 2 | `dpcc-r` (538 st, 78 ms) | 0.50 | `dpcc-r` | 0.50 | same |
| af | 2 | `dpcc-t` (528 st, 75 ms) | 0.50 | `dpcc-t` | 0.50 | same |
| af | 1 | `dpcc-t-tightened` (592 st, 85 ms) | 0.20 | `dpcc-t-tightened` | 0.70 | same |

✅ = HardFlow cell with genuine NLP steps (K = 5, A = 0.5 → 2 genuine; not degenerate).
**Caution:** mf's only 1.00 strict cell is `dpcc-t-geo_free` (geometry **not** enforced) and fm's highest is its
unprojected `diffuser`. At K = 5 these generators fly the scene cleanly on their own, so those cells measure the
generator, not the projector.

### 5.2 The DA Target (rule: the baseline's top projection variant)

**Target = diffusion K = 20 `dpcc-t-tightened`:** S&C strict 0.00 / crossed 0.30 · cfree 0.30 · viol 101.0 · 634 steps · 4424.5 ms/step.

A cell beats the Target when it holds S&C and wins on at least one axis. **Pareto-dominant** means higher-or-equal
S&C **and** fewer steps **and** lower time.

| engine K=5 | cell | S&C strict / crossed | steps | avg_ms | vs Target |
|---|---|---|---|---|---|
| mf | `hardflow_sls-r` | 0.90 / 0.90 | 442 | 154.0 | **Pareto-dominant** |
| mf | `dpcc-t-tightened` (same variant) | 0.80 / 0.80 | 417 | 1765.6 | **Pareto-dominant** |
| fm | `hardflow_sls-t` | 0.50 / 0.90 | 536 | 144.8 | **Pareto-dominant** |
| fm | `dpcc-c` | 0.40 / 1.00 | 558 | 790.8 | **Pareto-dominant** |
| af | `dpcc-t-tightened` (same variant) | 0.70 / 0.90 | 453 | 1564.5 | **Pareto-dominant** |

On the variant-for-variant row (`dpcc-t-tightened`), mf and af dominate the Target outright. fm has 0.00 / 0.40
there with the same 634 steps, so it wins on crossed S&C, collisions (cfree 0.40 vs 0.30) and time (1175 vs
4425 ms) but not on strict S&C.

---

## 6. Why diffusion fails strict success: goal precision, not route choice

Columns: **match** = flown route equals the commanded route; **crossed** = rollouts that crossed the finish line;
**reached | crossed** = of those, how many also passed within 0.30 m of the goal point.

| cell | engine | K | route flown | match | crossed | **reached \| crossed** | mean goal_dist |
|---|---|---|---|---|---|---|---|
| `diffuser` | **diffusion** | 20 | (R,R,R) ×10 | 2/10 | 10/10 | **0/10** | 0.567 |
| `diffuser` | fm | 5 | **(R,R,R) ×10** | 2/10 | 10/10 | **9/10** | 0.303 |
| `diffuser` | mf | 5 | (L,L,L) ×9, (R,R,R) ×1 | 2/10 | 10/10 | 9/10 | 0.361 |
| `diffuser` | af | 5 | (R,R,R) ×10 | 2/10 | 10/10 | 0/10 | 0.941 |
| `dpcc-t-tightened` | **diffusion** | 20 | (R,R,R) ×10 | 2/10 | 10/10 | **0/10** | 0.499 |
| `dpcc-t-tightened` | mf | 5 | (L,L,L) ×8, (R,R,R) ×2 | 5/10 | 10/10 | 8/10 | 0.366 |
| `dpcc-t-tightened` | fm | 5 | (L,L,L) ×6, (R,R,R) ×4 | 4/10 | 10/10 | 0/10 | 0.494 |
| `dpcc-t-tightened` | af | 5 | (L,L,L) ×9, (R,R,R) ×1 | 2/10 | 10/10 | 7/10 | 0.351 |

**Findings:**
- **Every engine crosses the line on every flight.** `success_relaxed` does not discriminate on this scene;
  **strict success and collisions do.**
- **Single-route behaviour is not diffusion-specific.** The generators are unconditioned: commanded and flown routes
  match only when they happen to coincide. fm K = 5 flies (R,R,R) 10/10, exactly like diffusion, yet reaches the goal
  point 9/10.
- **Diffusion's miss is terminal precision.** It crosses the finish line but finishes 0.50–0.57 m from the goal
  point, never passing within 0.30 m. The CSV does not split lateral from altitude error.
- The same pattern shows in two flow cells (af K = 5 `diffuser`, fm K = 5 `dpcc-t-tightened`), so it is not unique to
  the DDPM, but diffusion is the only engine where it holds on every cell.

---

## 7. Cost

| engine | K | generator `fm_ms` (unprojected) | projected cells `avg_ms` | cheapest cell with strict S&C ≥ 0.90 |
|---|---|---|---|---|
| **diffusion** | 20 | **182.9** | 3611–4425 | none |
| mf | 5 | 44.7 | 1594–2336 (DPCC) · 107–195 (HardFlow) | `diffuser` 44.7 ms (unprojected) · `hardflow_sls-r` 154.0 ms (enforced) |
| fm | 5 | 43.1 | 791–1390 (DPCC) · 78–145 (HardFlow) | `diffuser` 43.1 ms (unprojected) |
| af | 5 | 44.6 | 1565–2656 (DPCC) · 122–259 (HardFlow) | none |
| mf / fm / af | 2 | 17–18 | 75–104 | none |

The DDPM generator costs **4.1×** the flow family at K = 5; with DPCC projection it costs **1.4–5.6×** the flow
family's DPCC cells (791–2 656 ms). HardFlow on the flow engines (78–259 ms) is **~14–57× cheaper** than
DPCC-projected diffusion.

---

## 8. What this licenses, and what it does not

**Supported (pillars, honest geometry, seed 6, n = 10):**
- The U-Net DDPM baseline at K = 20 completes every traverse but scores **strict S&C 0.00 on all five DPCC variants**
  and **crossed-line S&C ≤ 0.30**.
- It never passes within 0.30 m of the goal point, and ≥ 7/10 flights violate the pillar clearance on every variant
  (with zero MuJoCo contacts).
- At K = 5, **mf, fm and af all have Pareto-dominant cells over the baseline's top projection cell**. mf beats
  diffusion on every shared variant under both definitions; fm and af have ties and one loss each (§4 table).
- At K = 2, mf and af beat diffusion on every shared variant under both definitions, and fm on all but one (crossed
  `dpcc-t-tightened`), at ~2–3 % of diffusion's projected time.
- Architecture-matched: all U-Nets, 3.96–3.97 M params.

**Not supported:**
- **A reproduction of the DPCC paper baseline.** This arm uses `action_weight = 1` (DPCC uses 10). It is "DDPM under
  Gen15's UAV config". If the paper-faithful row is needed, train an `aw10` twin, and add an `aw` token to the
  experiment name first so the two cannot collide.
- A matched-budget claim (K = 20 training vs K = 5 inference).
- Any multi-seed claim or confidence interval (seed 6 only).
- HardFlow on the diffusion arm (not applicable).
- A K = 1 comparison for mf/fm (no `u7hg` K = 1 candidates).
- **Pooling with the corridor_v2 paper run.** That run uses a different projector configuration
  (`-bounds_free-pdes-tightened`, `FRAC=1.0`); state it or re-run pillars under those settings.
- "The projector solved pillars" for mf/fm at K = 5. Their generator alone already scores 0.90, so the projector's
  contribution on this scene is best shown on diffusion (132 → 101 violations) and on af (198 → 22).

## 9. How to cite it in the paper

- **Table:** one reference row per engine at its own operating point, with **both** S&C strict and S&C crossed
  columns, backbone + params, K and ms/step.
- **Caption:** single seed; unmatched budgets by construction; diffusion trained at `action_weight = 1` under the same
  UAV config as the flow arms.
- **Headline sentence:** *"Under identical U-Net backbones (3.96–3.97 M), the DDPM baseline at K = 20 completes every
  pillars traverse but never within the 0.30 m goal tolerance and never clear of the pillar margin on more than 3/10 flights
  (S&C 0.00 strict / ≤ 0.30 crossed-line), while MeanFlow at K = 5 reaches S&C 0.80–0.90 with projection enforced at
  154–1772 ms per step."*
- **Next** (optional, for a CI instead of a reference): seeds 7–8 on pillars K = 5 for mf/fm/af plus diffusion K = 20
  (CLOSURE §7.1). For a paper-faithful baseline: an `aw10` diffusion twin.
