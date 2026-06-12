# Gen11 E4 U9 — Smooth Trajectories (Stop-and-Go Elimination, Option A)

**Date:** 2026-06-12
**Plan executed:** `U8_Stop_and_Go/STOP_AND_GO_ANALYSIS.md` — Option A (corner blends + global speed profile)
**Scope:** Coding only. Recollection + rejection-gate re-run are cluster-side (Section 5).

---

## 1. What changed

| # | File | Change |
|---|---|---|
| C1 | `uav_env_test/trajectories.py` | New primitive `blended_path(waypoints, radius, duration, yaw)` |
| C2 | `uav_expert_data_collect/trajectories.py` | `pillar_path` → blended chain (same 8-waypoint skeleton) |
| C3 | `uav_expert_data_collect/trajectories.py` | `s_curve_scene_path` → blended chain (same Z-route skeleton), **`T_HOVER` hovers dropped** |
| C4 | `uav_expert_data_collect/generator.py` | s_curve duration range reverted [18,24] → [16,22] s (the +2 s was the hover budget — U4 Fix A) |
| C5 | `uav_expert_data_collect/verify_blends.py` | New numerical verifier for clearance + smoothness (numpy-only, no mujoco) |

`empty_path` and `corridor_path` are untouched — single-segment paths were never stop-and-go
(U8 §3). `traverse_line` / `s_curve_path` base primitives unchanged (Epoch-3 consumers).

### C1 — `blended_path` primitive

- Straight segments between waypoints; every non-collinear interior corner cut by a circular
  fillet of radius ≤ `radius`, tangent to both adjacent segments. Fillet tangent offset
  `d = r·tan(β/2)` is clamped to half of either adjacent segment (β = turn angle); collinear
  breakpoints get no fillet; near-reversal corners (>~170°) raise `ValueError`.
- **One global cosine speed profile** over total arc length L: `s(t) = L·½(1−cos(πt/T))` —
  v > 0 for all interior t, v = 0 only at episode start/end. **Peak speed π·L/(2T) is
  identical** to the old length-proportional per-segment chain, so the speed regime the E4
  rejection gates were validated at is preserved.
- Returned acceleration includes the centripetal term `ṡ²/r` on fillets — correct
  feedforward for the cascaded PID.

### C2 — `pillar_path`

Waypoint skeleton unchanged (`x ∈ {−3.2, −2.5, −1.5, −0.5, +0.5, +1.5, +2.5, +3.2}`,
Fix_1/U3 channel y's). The 7× `traverse_line` chain (v=0 at all 6 interior waypoints) is
replaced by `blended_path(wps, BLEND_RADIUS=0.30, T)`. Straight portions near the pillars —
where the 8 cm minimum clearance lives — are untouched; fillets only cut corners ≥ 0.5 m in
x from every pillar. For uniform homotopies (LLL/RRR) the 4 mid-corners are collinear and
get no fillet automatically.

### C3 — `s_curve_scene_path`

Same U7 Z-route waypoints `(−3.2,y1) → (−0.5,y1) → (0,y1) → (0,y2) → (+0.5,y2) → (+3.2,y2)`.
The `(∓0.5, y)` breakpoints are collinear → no fillet (this also caps the Z-corner fillet
tangent offset at 0.25 m via the half-segment clamp, keeping fillets entirely inside the
x ∈ [−0.5, +0.5] gap). The two 90° Z-corners get fillets; **both 1.0 s hovers are removed** —
the hover existed to make a sharp corner trivially trackable; the blend removes the sharp
corner instead (U8 §3).

---

## 2. Verification (run locally, ALL PASS — 28/28 checks)

```bash
python3 uav_expert_data_collect/verify_blends.py
```

Checks per scene/homotopy/jitter/duration: (a) min clearance on the nominal path,
(b) interior speed never near zero (stop-and-go gone), (c) analytic v consistent with
finite-difference of p at dt=0.002 (no kinks at fillet joints — PID feedforward valid).

| Scene | Clearance gate | Measured | Interior v_min | v_peak | a_peak |
|---|---|---|---|---|---|
| pillars LLL/RRR | ≥ 0.43 m to pillar axes (0.12 r + 0.31 reach) | **0.510 m** | 0.12–0.19 m/s | 0.74–1.19 m/s | 1.1–2.9 m/s² |
| pillars LRL/RLR | ≥ 0.43 m | **0.510 m** | 0.16–0.25 m/s | 1.01–1.61 m/s | 3.4–8.6 m/s² |
| s_curve (jit ±0.04) | ≥ 0.31 m (rotor reach) to wall boxes | **0.410 m** | 0.09–0.12 m/s | 0.56–0.76 m/s | 1.2–2.3 m/s² |
| s_curve A/B corners | ≥ 0.31 m | **0.500 m** | — | — | — |

The s_curve numbers reproduce the U7 straight-leg clearances exactly (0.41 m worst-jitter
wall clearance, 0.50 m corner clearance) — the fillets do not reduce any pinch-point margin.

**Stats-validator note:** mean nominal speed stays inside the 0.15–0.80 m/s gate — pillars
L/T ≈ 0.64–1.0 m/s peak but mean ~0.65; s_curve mean = 8.0 m / [16,22] s ≈ 0.36–0.50 m/s
(removing the hovers + reverting the duration range cancel out).

---

## 3. Known risk (the one tuning lever)

Pillar mixed homotopies (LRL/RLR) at the shortest duration (T=10 s) demand **8.6 m/s²
(≈0.88 g) peak lateral accel** at the diagonal-exit fillets — the corner is now taken at
~1.35 m/s instead of v=0. The PID receives the centripetal feedforward, but this is the
most likely place for tracking error against the strict pillars contact gate (0.001).

If the pillars rejection rate exceeds 30% after recollection, tune in this order
(U8 §5 budgeted one iteration):
1. Raise `BLEND_RADIUS` 0.30 → 0.45 in `uav_expert_data_collect/trajectories.py`
   (centripetal accel scales 1/r; pillar corners have ≥ 0.5 m open space — re-run
   `verify_blends.py` to confirm the 0.43 m gate still holds).
2. Or raise the pillars duration floor `rng.uniform(10.0, 16.0)` → `(12.0, 16.0)` in
   `generator.py` (accel scales 1/T²).

---

## 4. What is deliberately NOT changed

- `traverse_line`, `s_curve_path` (uav_env_test) — Epoch-3 demo consumers keep old behaviour.
- `empty_path`, `corridor_path` — already single-segment, no internal stops.
- Rejection thresholds (`SCENE_MAX_CONTACT_FRACTION`) — gates stay as the empirical test.
- No mixing with the old dataset: per U8 §5, the existing 1,952 stop-and-go episodes are
  behaviourally incompatible with blended episodes — **full recollection required**.

---

## 5. What is NOT done here (cluster-side, Docker has no mujoco runtime)

1. Full E4 recollection with the new generator (all 4 scenes; only pillars + s_curve
   change behaviour, but recollect all to keep one consistent dataset version).
2. Re-run per-scene rejection gates (<30%) + `stats_validator.py`.
3. Regenerate E5 trajectory/physics GIFs — visual confirmation the stop-and-go is gone.
4. If pillars gate fires → one tuning iteration per Section 3, then recollect pillars.
