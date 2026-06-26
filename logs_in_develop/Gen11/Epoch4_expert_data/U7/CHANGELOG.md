# Gen11 E4 U7 — Changelog

**Date:** 2026-06-10  
**Branch:** update_into_FM  
**Implements:** Gen11E4U7_Plan_Fable.md (C1 + C2; C3 is run strategy, no code)

---

## C1 — Replace Seg B diagonal with 3-leg Z-route (`trajectories.py`)

### Why the diagonal was infeasible (root cause, definitively established in U7 plan)

The diagonal path (−0.5, y1) → (+0.5, y2) passes **0.291 m** from the gap-side wall
corners on the **nominal, zero-error path** — inside the 0.31 m rotor reach (0.019 m
penetration with no tracking error needed). Every prior fix attempt was treating
symptoms of a geometric infeasibility that cannot be resolved by speed, gain, or
time-budget changes. The full history:

- U4: attributed crash at f≈0.25 of Seg B to "motor saturation" — saturation was the
  downstream effect of the corner-contact impulse, not the cause.
- U5 Step 5: relocating diagonal to x=∓0.7 moved the penetration point into the
  walled section, making it worse and adding a *second* contact mechanism.
- U6: reverted to x=±0.5, confirmed crash resumed at same timing — geometry unchanged.
- F6 rejects #3 (min_z=0.741) and #7 (min_z=0.959): wall-dragging at healthy altitude
  with clip=30% and 24% — the smoking-gun signature of a persistent corner contact that
  sometimes causes a floor crash and sometimes just drags.

Pillars comparison (computed before plan): pillars inter-channel diagonals run at
1.12 m/s / 1.04 m/s² (5× more aggressive than Seg B) and pass at 90.4% — proving
trajectory aggressiveness is not the s_curve killer.

### Change (`uav_expert_data_collect/trajectories.py`, `s_curve_scene_path`)

The Seg B single diagonal is replaced by three piecewise `traverse_line` legs that
route through the gap centerline at x=0. This is the same `pillar_path` design
pattern (waypoint-stop, proportional-distance allocation) proven at 90.4% under
harsher dynamics.

**Before (6-phase, diagonal):**
```
Seg A → Hov 1 → Seg B (diagonal) → Hov 2 → Seg C
```

**After (7-phase, Z-route):**
```
Seg A → Hov 1 → Leg B1 → Leg B2 → Leg B3 → Hov 2 → Seg C
```

| Phase | From | To | Distance | Direction |
|-------|------|----|----------|-----------|
| Seg A | (−3.2, y1, z) | (−0.5, y1, z) | 2.7 m | pure-x |
| Hov 1 | (−0.5, y1, z) | — | 1.0 s | hover |
| **Leg B1** | **(−0.5, y1, z)** | **(0, y1, z)** | **0.5 m** | **pure-x** |
| **Leg B2** | **(0, y1, z)** | **(0, y2, z)** | **≈1.6 m** | **pure-y at x=0** |
| **Leg B3** | **(0, y2, z)** | **(+0.5, y2, z)** | **0.5 m** | **pure-x** |
| Hov 2 | (+0.5, y2, z) | — | 1.0 s | hover |
| Seg C | (+0.5, y2, z) | (+3.2, y2, z) | 2.7 m | pure-x |

**Verified clearances:**
- Leg B1, B3 (pure-x): ≥ 0.55 m from both gap-side corners.
- Leg B2 (pure-y at x=0): 0.50 m from both corners — well above 0.31 m rotor reach.
- At every pinch point, the path runs **parallel to the nearest wall**: tracking lag
  is along-path and cannot convert into lateral wall penetration (the old diagonal
  converted its 0.185 m documented lag directly into corner penetration).

**Time allocation:** proportional to Euclidean distance, no weighting:
```python
d_total = d_a + d_b1 + d_b2 + d_b3 + d_c   # ≈ 8.0 m when jitter=0
t_i = T_move * d_i / d_total
```
The U5 Seg-B 2× time weight is removed — it was compensating for a non-existent
speed problem; proportional allocation gives Leg B2 roughly 3.2 s for 1.6 m
(v_peak ≈ 0.39 m/s), well inside the pillars-proven envelope.

Docstring updated with the corner-geometry constraint and clearance numbers so the
next editor understands what must not be changed.

---

## C2 — Log accepted-episode clip stats (`collect.py`)

**Why:** F6 could only report clip% for *rejected* episodes. Without the accepted
baseline, the 17–60% reject-clip figures were uninterpretable — it was unknown
whether healthy s_curve episodes would also show 15–20% saturation (chronic-but-benign,
like pillars) or would be clean (saturation is the failure signal). Adding this closes
the diagnostic gap.

**Changes (`uav_expert_data_collect/collect.py`):**

- `accepted_clip_list = []` initialised before the trial loop.
- Each time a rollout is saved: `accepted_clip_list.append(rollout.get('motor_clip_frac', 0.0))`.
- At summary: `acc_clip_mean` and `acc_clip_max` computed and written to `run_summary.json`:

```json
"accepted_clip_mean": 0.0034,
"accepted_clip_max":  0.0412
```

Predicted values for U7 validation:
- empty / corridor: ≈ 0 (no dynamics stress).
- pillars: non-zero (known saturation on contact episodes that pass), probably 1–5% mean.
- s_curve (if C1 works): should show accepted-clip ≈ pillars range or lower — the
  corner contact was the confound; without it, saturation should be benign.

---

## Files changed

| File | Change |
|------|--------|
| `uav_expert_data_collect/trajectories.py` | C1: 3-leg Z-route replaces diagonal |
| `uav_expert_data_collect/collect.py` | C2: accepted-episode clip stats in summary |

---

## Validation plan

```bash
# s_curve smoke (expect near-0% — failure was deterministic geometry, now removed)
python uav_expert_data_collect/collect.py --scene s_curve --n-trials 20 --seed 100

# full 500
python uav_expert_data_collect/collect.py --scene s_curve --n-trials 500 --seed 0
```

If smoke > 30%: read histogram first.
- `contact` with healthy min_z → geometry, re-derive (don't guess).
- `floor` with clip > 50% → then and only then revisit gains/u_max.
