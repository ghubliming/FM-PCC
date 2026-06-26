# Gen11E4U6 — Plan

**Date:** 2026-06-10 · **Scope:** s_curve + pillars collection, fixing the two U5 regressions
**Inputs:** `U5/{CODING_PLAN.md, CHANGELOG.md, F5_RESULTS.md}`, F4 homotopy breakdown (`U4/F4_RESULTS.md`)
**Honest accounting:** U5 Steps 1–3 were correct. Step 5 broke s_curve geometry; Step 4's allocation broke pillars cross-channel homotopies; Step 1's clip detector under-reports. All three are fixed here.

---

## What U5 actually did (evidence)

| Scene | F4 | U5 | Verdict |
|-------|----|----|---------|
| empty / corridor | 0% | 0% | unchanged ✅ |
| s_curve | 100% (saturation mid-diagonal) | 100% (**new** cause: wall clip) | failure mode replaced, not fixed |
| pillars L_L_L / R_R_R | ~70–76% success | ~70–76% success | unchanged |
| pillars **L_R_L / R_L_R** | **~67% / ~59% success** | **~5% / ~10% success** | ❌ regression from Step 4 |

Two proven mechanisms:

1. **s_curve wall clip (Step 5 bug):** moving the Seg B diagonal endpoints from x=±0.5 to x=∓0.7 put its first/last 0.2 m inside the walled section. Rotor enters `seg1_wall_pos` contact zone at x=−0.577 (clearance 0.31 m = rotor reach); at the wall exit clearance is 0.221 m. Histogram matches: contact=8 direct hits, floor=13 knock-down crashes, all with no saturation involved.
2. **pillars attitude starvation (Step 4 bug):** the thrust-priority binary search halves torque scale until motors fit — it over-cuts attitude authority by up to 2× vs the exact feasible scale (needed 0.49 → gets 0.25). Cross-channel homotopies have the most aggressive diagonals and need that torque; F4's plain clip kept most of the motor differential, U5's scaling destroyed it → lateral tracking error → pillar contact (contact=33 dominant).
3. **clip telemetry blind spot (Step 1 flaw):** `n_clip` fires only when the *final* u touches u_max/u_min. The torque-scaled output lands strictly inside bounds, so U5's `clip=0.0%` lines do NOT prove saturation never engaged. F5_RESULTS' "saturation eliminated" claim is unproven — fix the detector before re-concluding.

---

## Changes (in order)

### C1 — s_curve: revert Step 5 geometry, keep everything else (`trajectories.py`)

Full revert of hover/diagonal positions to the U4 layout — hover and diagonal endpoints back at x=±0.5:

```
Seg A: (−3.2,y1,z)→(−0.5,y1,z) · Hov1 at (−0.5,y1,z) · Seg B: (−0.5,y1,z)→(+0.5,y2,z)
Hov2 at (+0.5,y2,z) · Seg C: (+0.5,y2,z)→(+3.2,y2,z) · d_a=d_c=2.7, d_b≈1.89
```

**Keep** the Step 3 2× time weight on Seg B (`d_total = d_a + 2.0*d_b + d_c`).
Rationale for full revert over bridge segments: F4 ran hovers at ±0.5 with zero contact rejects — the "wall end-face risk" never materialized; the diagonal must start at the gap boundary regardless. Simplest change that is provably safe.

### C2 — controller: exact-scale allocation with torque floor (`flight_controller.py`)

Replace the binary search in `CascadedPID.compute`:

```python
# exact largest scale that fits all motors (no over-cutting)
hi = (self.u_max - thrust_cmd); lo = (thrust_cmd - self.u_min)
caps = [hi / tc if tc > 0 else lo / -tc for tc in torque_comp if abs(tc) > 1e-9]
scale = min(1.0, min(caps)) if caps else 1.0
scale = max(scale, 0.5)            # torque floor: never give up >50% attitude authority
u = np.clip(thrust_cmd + scale * torque_comp, self.u_min, self.u_max)
```

- Exact scale removes the up-to-2× over-reduction.
- The 0.5 floor means: below it, prefer slight thrust corruption (the final clip) over attitude loss — F4 proved partial-torque clipping is survivable; U5 proved zero-torque is not.
- Record `self.last_raw_saturated = bool(raw u out of bounds)` and `self.last_torque_scale = scale` each call (for C3).
- Optional, behind a check: raise `u_max` 2.0→2.6×u_hover to shrink saturation frequency — **first verify** the MuJoCo actuator `ctrlrange` doesn't already cap below that, else it's a silent no-op.

### C3 — fix the clip telemetry (`generator.py`)

Count saturation from the controller's own flag, not from boundary-touching output:

```python
u = pid.compute(...)
n_clip += int(pid.last_raw_saturated)     # raw demand exceeded limits this step
```

Optionally also log mean `last_torque_scale` over the episode in the reject dict. This makes `clip=…%` mean "saturation events" again.

### C4 — pillars run strategy

After C2+C3, rerun pillars with all 4 homotopies (the F4 baseline says cross-channel should return to ~60–67%). Decision rule on the smoke result:

- overall < 30% → full 500, done.
- cross-channel still lagging (> 50% rejection) → two-track: main 500-trial run with `--homotopy` limited to `(L,L,L)`/`(R,R,R)`, plus dedicated 300-trial runs per cross-channel label; **and** add 2× time weight to the two inter-channel diagonal segments in `pillar_path` (same physics as s_curve Step 3 — halve peak lateral velocity where the transition happens).

### C5 — keep, don't touch

z range [0.90, 1.30] (Step 2), reject-histogram instrumentation (Step 1 minus the detector fix), Seg B 2× weight (Step 3). empty/corridor configs unchanged.

---

## Validation gates

| Run | Expectation | Gate |
|-----|-------------|------|
| s_curve 20-trial smoke (C1+C2+C3) | wall clip gone; saturation handled with torque floor | < 30% rejection |
| pillars 20-trial smoke (C2+C3) | cross-channel back to ≥ F4 levels | < 30% overall, no homotopy > 50% |
| then | full 500 each → E5 gate | — |

If the s_curve smoke still rejects > 30%, read the histogram first: `floor` with `clip>0` → revisit C2 floor value; `contact` → geometry, re-derive (don't guess).

**Definition of done:** all four scenes < 30% rejection on full runs, reject histograms in every `run_summary.json`, and `clip%` reflecting raw saturation.
