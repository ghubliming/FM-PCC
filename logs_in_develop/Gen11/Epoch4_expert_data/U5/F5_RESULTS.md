# Gen11 E4 U5 — F5 Run Results

**Date:** 2026-06-10  
**Jobs:** 21408 (empty), 21409 (corridor), 21410 (s_curve), 21411 (pillars)  
**Node:** i6-gpu-1

---

## Summary table

| Scene | Saved | Trials | Rejection | Histogram | Status |
|-------|-------|--------|-----------|-----------|--------|
| empty | 500 | 500 | 0.0% | none | ✅ PASS |
| corridor | 500 | 500 | 0.0% | none | ✅ PASS |
| s_curve | 0 | 21 | 100% (ABORT) | floor=13, contact=8 | ❌ SAME AS F4 |
| pillars | 33 | 83 | 60.2% (ABORT) | contact=33, floor=17 | ❌ REGRESSION from F4 (20.8%) |

---

## empty / corridor — ✅ Clean

Both scenes collected 500/500 with 0 rejects and healthy stats. No action needed.

---

## s_curve — ❌ Still 100% rejection, different root cause

### Key new evidence from instrumentation (Step 1)

All 21 rejects show **`clip=0.0%`** — no motor saturation occurred on any step. This eliminates the U4 P&S root cause (attitude loop saturation → thrust corruption → altitude collapse). The fix in Steps 3+4 is working: the motor-saturation path is completely gone.

Yet the drone still crashes to `min_z ≈ 0.083–0.187` on every trial. Since the failure mode changed while the outcome is the same, there is a **new, different crash mechanism** introduced in U5.

### Root cause identified: Step 5 introduced a wall clip during Seg B

**Geometric proof:**

Step 5 moved the Seg B diagonal from `(-0.5, y1) → (+0.5, y2)` to `(-0.7, y1) → (+0.7, y2)`. The first 0.2 m of the new diagonal (x ∈ [-0.7, -0.5]) is inside the walled section of corridor 1, where `seg1_wall_pos` occupies `x ∈ [-3.0, -0.5]`, `y ∈ [-0.35, -0.25]`.

As the drone traverses this section, its y-coordinate moves toward +0.8 while the wall is fixed at y ∈ [-0.35, -0.25]. The rotor first enters the wall's contact zone at:

```
f = 0.0875 (8.75% along Seg B)
x = −0.577 m   (wall present ✓)
y = −0.660 m
clearance to wall edge (y=−0.35) = 0.310 m ≈ rotor_reach = 0.31 m  → CONTACT
```

By x = −0.5 (wall exit), clearance has already dropped to **0.221 m < 0.31 m** — the rotor is inside the wall for the last 0.12 m of the walled section.

In contrast, the U4 original diagonal started at x = −0.5 (the wall exit), so the **entire diagonal was in the gap (no walls)**. Clearance at start: 0.45 m, safely above rotor_reach.

**Evidence match:** 8 contact rejects (direct wall hit), 13 floor rejects (wall hit disrupts attitude → altitude collapse). The floor rejects without contact are cases where the disturbance knocked the drone off the cosine profile just enough to fall, without triggering the obstacle-contact geom.

### What worked

- **Step 1 (instrumentation)**: functioning perfectly — reject histogram and per-reject `min_z` / `clip%` gave the diagnostic above in one run.
- **Step 2 (z range [0.90,1.30])**: no negative effect observed; empty/corridor still 0%.
- **Steps 3+4 (Seg B 2× time + thrust-priority)**: clip=0.0% confirms motor saturation is eliminated. These fixes are sound — s_curve would succeed if the geometry were correct.
- **Step 5 (hover relocation)**: the hover positions (x=∓0.7) are geometrically safe for hovering. The bug is that the **diagonal start/end was also moved** to x=∓0.7, which put the first and last 0.2 m of the diagonal inside the walled section.

---

## pillars — ❌ Regression (20.8% → 60.2%)

### Homotopy breakdown (83 trials, 4 homotopies cycling)

| Homotopy | Approx. trials | Saved | Success rate |
|----------|---------------|-------|-------------|
| (L,L,L) | ~21 | 16 | ~76% |
| (R,R,R) | ~20 | 14 | ~70% |
| (L,R,L) | ~21 | 1 | ~5% |
| (R,L,R) | ~21 | 2 | ~10% |

L,L,L and R,R,R are performing comparably to U4 F4. The regression is entirely from **(L,R,L) and (R,L,R) cross-channel homotopies at ~90–95% rejection rate**.

### Analysis

- Some pillar floor rejects show `clip > 0%` (e.g. clip=1.3%, 3.0%) — these are contact-disruption → saturation → altitude collapse, the old mechanism, still active for pillars.
- Many floor rejects show `clip=0.0%` — not saturation-driven; likely altitude collapse from large attitude errors after pillar contacts or aggressive lateral maneuvers.
- The cross-channel homotopies require diagonal transitions that sweep ≈ 2.2 m laterally. These stress the controller most.
- In U4 F4, the overall 20.8% rejection was with a **homotopy filter (Fix B)** active for R_R_R recovery. It is unclear whether L_R_L and R_L_R were also collected at high rejection rates in U4 F4 or were simply not the ABORT trigger due to the 500-trial ceiling. With U5's 500-trial run hitting 60% ABORT at trial 83, the cross-channel homotopies are clearly the cause.

---

## Required fixes for U6

### Fix A (critical — s_curve): revert Seg B endpoints to x=±0.5

The diagonal MUST start and end at the gap boundary (x = −0.5 and x = +0.5) where the walls end. This is the U4 original design and is geometrically correct. The hover positions at x=∓0.7 are fine (safe lateral clearance to corridor walls), but the **diagonal endpoints must not be moved inside the walled section**.

Design for U6 s_curve (7 phases):

```
Seg A:    (−3.2, y1, z) → (−0.7, y1, z)   pure-x, d=2.5 m
Hover 1:  hover at (−0.7, y1, z)            1.0 s
Trans A→B: (−0.7, y1, z) → (−0.5, y1, z)  pure-x, d=0.2 m  ← bridge to gap entry
Seg B:    (−0.5, y1, z) → (+0.5, y2, z)   diagonal, d≈1.89 m (original), 2× time weight
Trans B→C: (+0.5, y2, z) → (+0.7, y2, z)  pure-x, d=0.2 m  ← bridge from gap exit
Hover 2:  hover at (+0.7, y2, z)            1.0 s
Seg C:    (+0.7, y2, z) → (+3.2, y2, z)   pure-x, d=2.5 m
```

Alternative (simpler): revert hover AND diagonal back to x=±0.5 entirely. At x=±0.5 the hover has 0.14 m lateral clearance to corridor y-walls, which was safe in U4 F4 even without Steps 3+4.

### Fix B (pillars): restrict homotopy pool or increase trial budget

Cross-channel homotopies (L,R,L) and (R,L,R) need either:
- Separate high-budget collection runs (e.g. 300 trials each, expect ~30 saves at 90% rejection)
- OR: run pillars with `--homotopy` restricted to `(L,L,L)` and `(R,R,R)` for the main 500-trial run (as U4 Fix B did), and collect cross-channel separately

Do NOT run all 4 homotopies in one 500-trial run — the cross-channel failures pull the overall rate above the ABORT threshold.

---

## Pre-training gate status

| Scene | Gate (< 30%) | Action |
|-------|-------------|--------|
| empty | ✅ 0% | Done |
| corridor | ✅ 0% | Done |
| s_curve | ❌ 100% | Fix A → rerun |
| pillars | ❌ 60% | Fix B → rerun (restricted homotopy pool) |
