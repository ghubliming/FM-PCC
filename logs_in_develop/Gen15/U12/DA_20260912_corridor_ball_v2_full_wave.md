# DA — `corridor_ball_v2` full wave: the ball is right, the projector works, and it still cannot go around

*Gen15 · U12 · 2026-09-12. Source batch: **`temp/1209/batch_uav_20260912_090334`**
(`DA_UAV_v1`, 1384 units, 0 failed). Jobs **25663–25668 → 25669–25674**, 3 engines × 2 tiers,
seed 6, **n = 10** on all 34 cells, geo `corridor_hgb2`.
Changelog: [`CHANGELOG_20260911_corridor_ball_v2_on_trajectory.md`](CHANGELOG_20260911_corridor_ball_v2_on_trajectory.md).*

## 0. TL;DR

1. ✅ **Gate 1 passes.** The unprojected plan goes from **0.00** violations on `corridor_hg` to
   **25.5 – 27.3** on `corridor_ball_v2`, all six arms. The small on-path ball binds. §2
2. 🔴 **Gate 2 fails.** `collision_free_completed` is **0.000 in 32 of 34 cells**; S&C is **0.000 in
   all 34**. The smaller, correctly-placed ball changed the numbers but not the outcome. §3
3. 🔴 **The decisive measurement: `phys_min_z` = 1.1264 – 1.1266 across all 33 cells** — a spread of
   **0.2 mm**, over three engines, two budgets, seven projector configurations and 340 rollouts.
   The escape requires z ≥ 1.56. **Nothing ever climbs.** §4
4. **The projector is not idle — it is confined.** It removes up to 5.7 violations, shortens the path
   by 53 steps and spends up to 1795 ms/step doing it. Every bit of that happens **along x**. §5

**The ball placement was never the problem, and neither was its size.** §6

---

## 1. Provenance and gates

| | |
|---|---|
| geo | `corridor_ball_v2` → `corridor_hgb2_…`, **obs=5** (4 wall caps + ball) |
| ball | `[0.0, 0.0, 1.13]`, r = **0.12** — on the measured flown path, keep-out 0.43 m |
| ceiling | `ub[2]` = **2.80** (synthetic; raised from 1.80) → escape slot z ∈ [1.56, 2.49] = **0.93 m** |
| tiers | **A** K=2, 4 variants · **B** K=5, 7 variants (HardFlow genuine, `n_genuine = 2`) |
| engines | af (U-Net 3.97 M) · mf (U-Net) · fm |
| seed · n | 6 · **10** on every cell |

Provenance verified in every child log: `[ U11 ] geo variants for 'corridor': ['corridor_ball_v2']`,
`E9 geo … (bounds=True, hs=2, obs=5)`, results under `corridor_hgb2_`.

---

## 2. ✅ Gate 1 — the ball binds, and it is *better* placed than U11's

`diffuser` (unprojected; identical plans under all three geometries — only scoring differs):

| arm | `_hg` | `_hgb` (U11, r=0.35 @ z=0.75) | **`_hgb2`** (U12, r=0.12 @ z=1.13) |
|---|---|---|---|
| af K=2 | 0.00 | 34.60 | **27.30** |
| mf K=2 | 0.00 | 34.70 | **27.10** |
| fm K=2 | 0.00 | 32.80 | **25.50** |
| af K=5 | — | 34.40 | **27.00** |
| mf K=5 | — | 34.60 | **26.70** |
| fm K=5 | — | 33.40 | **26.00** |

~22 % fewer violations than U11 — exactly what a smaller sphere predicts, since the violating
segment is the chord the straight path cuts through it. The ball is on the route and blocking.

---

## 3. 🔴 Gate 2 — nothing routes around it

**S&C = 0.000 in all 34 cells. `collision_free_completed` = 0.000 in 32 of 34.**

The two exceptions are both **fm K=2**, and neither is avoidance:

| cell | `cfree` | `n_success` | reading |
|---|---|---|---|
| fm K=2 `dpcc-t-tightened` | 0.200 | 0.800 | 2 clean rollouts, but success fell to 0.800 — and S&C is still 0.000, so **no rollout was both clean and successful** |
| fm K=2 `dpcc-t-geo_free` | 0.100 | 0.800 | 🔴 **geometry is OFF on this row** — it cannot be routing around the ball deliberately. Incidental. |

That a `-geo_free` row scores as well as the geometry-aware rows is itself the tell: the geometric
constraint is not what is producing the occasional clean rollout.

---

## 4. 🔴 The measurement that settles it — the altitude never moves

`phys_min_z`, across **all 33** `corridor_hgb2` cells:

| | |
|---|---|
| range | **1.1264 – 1.1266 m** |
| spread | **0.2 mm** |
| flown band (raw plan) | 0.956 – 1.236 m |
| **altitude the escape requires** | **z ≥ 1.56 m** |

Three engines, two budgets, seven projector configurations, 340 rollouts — and the minimum altitude
is constant to a fifth of a millimetre. The projector never lifts the plan, not by a centimetre, not
once. The 0.93 m slot U12 opened above the ball is never entered.

This is the same signature the 12-minute injection test showed (`min_z == final_z` on 10/10), now
measured across the whole wave.

---

## 5. The projector is working — in one axis

It is emphatically not inactive:

| arm | best reduction | variant | steps (diffuser → best) | `proj_ms` |
|---|---|---|---|---|
| af K=5 | **−5.70** | `dpcc-t` | 268.9 → 227.8 | **1549** |
| mf K=5 | **−5.10** | `dpcc-t` | 269.2 → 230.6 | 1490 |
| fm K=2 | −5.50 | `dpcc-t-tightened` | 273.3 → 275.1 | 205 |
| fm K=5 | −2.00 | `dpcc-t` | 271.7 → 246.0 | 539 |
| af K=2 | −1.50 | `dpcc-t` | 270.9 → 251.6 | 142 |
| mf K=2 | −1.10 | `dpcc-t` | 271.7 → 254.8 | 141 |

It converges, it changes the trajectory, it shortens the path by up to 53 steps, and it spends up to
**1.8 s per control step** doing so. Note also that U12's smaller ball made it *more* effective in
absolute terms at K=5 (af −3.80 → **−5.70**, mf −3.10 → **−5.10**) — the projector is responding to
the geometry, just not in the axis that would clear it.

Two secondary confirmations:

* **`-geo_free` is consistently *worse* than no projection at K=5** (af +1.40, mf +2.10): dropping the
  geometric family leaves the projector perturbing the plan without accounting for the ball. The
  negative control behaves correctly.
* **HardFlow is indistinguishable from the unprojected plan** — `hardflow_sls` sits within −1.6 to
  −1.4 of `diffuser` on all three engines while DPCC removes 2–5.7. Same as U11 §4(c).

---

## 6. 🔴 Verdict — the obstacle was never the variable

U11 blocked 100 % of the trained band with a 0.66 m keep-out and an 0.08 m escape slot; U12 blocks it
with a 0.43 m keep-out, on the measured flight path, with a **0.93 m** slot. **Same result.**

| | U11 `_hgb` | U12 `_hgb2` |
|---|---|---|
| escape slot | 0.08 m | **0.93 m** (11.6×) |
| ball keep-out | 0.66 m | 0.43 m |
| on measured flight path | no (z=0.75 vs flown 1.13) | **yes** |
| `diffuser` violations | 32.8 – 34.7 | 25.5 – 27.3 |
| **S&C** | **0 / 34** | **0 / 34** |
| **`collision_free` > 0** | 2 / 34 | 2 / 34 |
| **altitude ever changed** | no | **no** |

Eleven times the headroom, a correctly-placed obstacle, and the altitude still does not move by a
millimetre. **A geometry that cannot be escaped by any ball size, at any height, with any amount of
headroom, is not a geometry problem.**

### 6.1 The standing hypothesis

From the eval's own output on this scene:

```
Fix_16 DEGENERATE actions[1]/[2]: constant in the expert data — no training signal
action_bounds=auto → lb=[1.24e-04 -2.20e-05 -2.20e-05] ub=[4.3886e-02 2.2000e-05 2.2000e-05]
```

The corridor expert flies a straight line at constant y **and** z, so Δy and Δz have zero variance in
training; `action_bounds='auto'` derives the projector's per-step action cap from that range and
leaves ≈ **2.2 × 10⁻⁵ m/step** in both axes — **8.7 mm** over a 396-step episode, against the 0.43 m
climb the ball demands. The projector's whole authority is the **4.4 × 10⁻² m/step** it has in x,
which is exactly what §5 shows it spending.

Measured across scenes:

| scene | Δx | Δy | Δz |
|---|---|---|---|
| **corridor** | 4.4e-02 | **±2.2e-05** 🔴 | **±2.2e-05** 🔴 |
| pillars | 4.4e-02 | **±3.97e-02** ✅ | ±3.1e-05 🔴 |
| s_curve | 2.2e-02 | **±2.29e-02** ✅ | ±1.1e-05 🔴 |

**Δz is degenerate on every UAV scene** — every expert cruises at fixed altitude — so a *vertical*
detour is unreachable everywhere. `corridor` uniquely has no lateral authority either.

⚠️ **Not yet proven.** Job **25682** tests it directly: `dpcc-t-bounds_free` keeps dynamics and
geometry but drops the action-magnitude family (`eval_mix_uav.py:1259`).
`collision_free > 0` confirms the cap is the blocker; `= 0` refutes it and the search moves to the
projector call path.

---

## 7. What this licenses

**Supported.** `corridor_ball_v2` binds (`diffuser` 0.00 → 25.5–27.3 violations, all six arms) and is
~22 % lighter than U11's ball. No configuration achieves S&C > 0 in 34 cells, and `collision_free` is
0.000 in 32 of them. `phys_min_z` is constant to 0.2 mm across all 33 cells and 340 rollouts. The
DPCC projector removes up to 5.7 violations and 53 steps at up to 1.8 s/step, entirely along x.
`-geo_free` is worse than no projection at K=5. HardFlow is indistinguishable from no projection.

**Not supported.** Any engine ranking (S&C floored; violation counts within ~1.5 across af/mf/fm).
Any multi-seed claim (seed 6). The action-bound explanation itself — §6.1 is a hypothesis with one
job outstanding.

**Next.**
1. **Read job 25682 first.** It decides whether §6.1 stands, and therefore whether `corridor` can ever
   host a detour obstacle.
2. If confirmed: either set `action_bounds` explicitly for this scene, or **move the obstacle test to
   `pillars`**, whose expert weaves laterally and which carries ±0.040 m/step of Δy authority — 1800×
   corridor's.
3. Do **not** spend seeds or further geometry iterations on `corridor_ball_*` until 1 is answered.
