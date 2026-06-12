# U9 Fix_1 — BLEND_RADIUS 0.30 → 0.45 m

**Date:** 2026-06-12
**Trigger:** U9 pillars rejection 45.2% — exceeds <30% gate (see U9_EVAL_RESULTS.md)
**Root cause:** LRL/RLR mixed-homotopy fillet accel 8.6 m/s² at T=10s saturated cascaded PID

---

## Change

| File | Change |
|------|--------|
| `uav_expert_data_collect/trajectories.py` | `BLEND_RADIUS = 0.30 → 0.45` |

One-line edit. `pillar_path` and `s_curve_scene_path` both read `BLEND_RADIUS`; both are updated by this change. The `blended_path` primitive in `uav_env_test/trajectories.py` is unchanged.

---

## Why this fixes it

Peak centripetal acceleration at a fillet scales as `ṡ²/r = (π·L/2T)²/r`.

| r | LRL/RLR T=10s peak accel | PID budget (~0.7–0.8 g lateral) |
|---|--------------------------|----------------------------------|
| 0.30 m | **8.6 m/s²** (≈0.88 g) | ❌ over budget → motor saturation |
| 0.45 m | **5.7 m/s²** (≈0.58 g) | ✅ within budget |

The LLL/RRR homotopies have collinear interior corners — `blended_path` generates zero fillet for them — so they are unaffected by this change and stay at ~100% acceptance.

The s_curve Z-corners are also unaffected in practice: those corners already had generous clearance at r=0.30 (0.41–0.50m) and the fillet tangent offset at the s_curve (d = r·tan(45°) = r) stays well inside the gap.

---

## Verification — verify_blends.py (28/28 PASS)

```
pillars — clearance gate: ≥ 0.43 m to pillar axes:
  [LLL T=10] dist=0.510 m  v_min=0.185  a_peak=2.02 m/s²   PASS
  [LLL T=16] dist=0.510 m  v_min=0.116  a_peak=0.79 m/s²   PASS
  [LRL T=10] dist=0.510 m  v_min=0.249  a_peak=5.64 m/s²   PASS  ← was 8.6
  [LRL T=16] dist=0.510 m  v_min=0.156  a_peak=2.20 m/s²   PASS
  [RLR T=10] dist=0.510 m  v_min=0.249  a_peak=5.64 m/s²   PASS  ← was 8.6
  [RLR T=16] dist=0.510 m  v_min=0.156  a_peak=2.20 m/s²   PASS

s_curve — clearance gate: ≥ 0.31 m to wall boxes:
  [jit=±0.04, T=16/22]  wall dist ≥ 0.410 m  corner dist = 0.500 m   PASS (×6)
```

Key results vs r=0.30 baseline:

| Metric | r=0.30 | r=0.45 | Change |
|--------|--------|--------|--------|
| LRL/RLR T=10s a_peak | 8.6 m/s² | **5.64 m/s²** | −34% |
| LRL/RLR T=16s a_peak | 3.4 m/s² | **2.20 m/s²** | −35% |
| Clearance (all pillars) | 0.510 m | **0.510 m** | unchanged |
| s_curve wall clearance | 0.410 m | **0.410 m** | unchanged |
| s_curve corner clearance | 0.500 m | **0.500 m** | unchanged |

Clearance gate is unchanged because the fillet stays inside the open corridor between pillar pairs (≥ 0.5 m clear space). The larger radius cuts deeper into the corner but the straight portions near the pillars are untouched.

---

## What to do on the cluster

Recollect **pillars only** (empty/corridor/s_curve from the U9 run are clean — keep them):

```bash
sbatch Slurm_Codes/sbatch/uav_expert_data/collect_pillars.sh
# scene=pillars  n_trials=500  homotopy=all  seed=0
```

Acceptance target: rejection rate **< 30%** overall, with LRL/RLR individually > 30% acceptance (i.e., ≥ ~38 episodes each out of ~125 slots).

If the gate still fails after this run → apply the secondary lever (CHANGELOG §3 step 2): raise pillars duration floor `(10.0, 16.0) → (12.0, 16.0)` in `generator.py`. Combined effect with r=0.45 would reduce T=10s peak accel further to ~4.0 m/s².

---

## What is NOT changed

- `uav_env_test/trajectories.py` — `blended_path` primitive unchanged
- `uav_expert_data_collect/generator.py` — duration range unchanged (`(10.0, 16.0)`)
- s_curve, empty, corridor paths — no behavioural change
- Rejection thresholds (`SCENE_MAX_CONTACT_FRACTION`) — unchanged
