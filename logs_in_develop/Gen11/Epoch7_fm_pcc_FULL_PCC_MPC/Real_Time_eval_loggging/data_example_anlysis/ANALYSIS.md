# Log Analysis — `rollout_corridor_C_10001.log`

**Episode:** `corridor_C_10001`  
**Variant:** `dpcc-t` (temporal consistency, dynamics constraint)  
**Scene:** corridor, homotopy C (most complex route)  
**Node:** i6-gpu-1  
**Date analysed:** 2026-06-25

---

## Quick verdict

| Question | Answer |
|----------|--------|
| Real-time safe? | **NO** — 100% of steps over budget |
| Task success? | **FAIL** — drone crashes at T=1.576s, frozen on floor for remaining 6s |
| Root cause? | FM z-drift → floor impact at step 52 → PID can't recover |
| Projector working? | **Yes** on normal steps; proj_cost spikes reveal the crash moment precisely |

---

## 1. Timing — far over the 33 Hz budget

The cluster (i6-gpu-1) cannot run this policy at real-time speed:

```
Budget:    30.3 ms   (1000 / 33 Hz)

fm_ms:     mean=85.4   max=89.1   p95=86.1   → 2.8× budget alone
proj_ms:   mean=59.8   max=62.6   p95=61.0
total_ms:  mean=145.2  max=149.4  p95=146.5  → 4.8× budget
over_budget: 243/243 (100%)
```

**FM inference is the bottleneck**, not the projector:

- `fm_ms = 85.4 ms` = 59% of total
- `proj_ms = 59.8 ms` = 41% of total (SLSQP with 24 equality constraints, B=4 batch)

Even removing the projector entirely (`diffuser` variant), the 85 ms FM inference alone is
**2.8× over the 30 ms budget**. The system cannot close the UAV control loop in real time
on this hardware.

**Timing is rock-solid flat** — essentially zero variance:

```
total_ms: min=143.5  max=149.4  range=5.9 ms
fm_ms:    min=84.7   max=89.1   range=4.4 ms
proj_ms:  min=58.6   max=62.6   range=4.0 ms
```

No spikes, no GIL pauses, no outliers. The model is completely deterministic in compute
cost per step — good for profiling; bad because it means there is no easy win (no pathological
slow step to fix, just a structural throughput problem).

### Implication

The cluster GPU gives `fm_ms ≈ 85 ms`. To fit the 30 ms budget the model would need to
run **~2.8× faster** — either model compression (distillation, quantisation), fewer ODE
steps, or deployment on faster hardware. The projector's 59 ms is a separate optimisation
target (e.g. warm-starting SLSQP, reducing batch size).

---

## 2. Failure timeline — floor crash at T=1.576s

The rollout has a sharp, readable structure:

```
Phase 1  T=0.000–0.394s  (steps  0–12)  Normal flight, low proj_cost (0.8–7.6)
Phase 2  T=0.394–0.909s  (steps 13–29)  proj_cost spikes (100→13976), p_des_z collapses
Phase 3  T=0.909–1.576s  (steps 30–52)  21 contacts over 0.87s, track_err explodes
Phase 4  T=1.576–7.364s  (steps 52–242) Drone frozen at (-2.316, -0.477, 0.087), episode bleeds out
```

### Phase 1 — Normal (steps 0–12)

p_des_z starts at 1.186 m and is already falling toward 0.86 m within 12 steps. The FM
is predicting a descending trajectory from the start. proj_cost is low (0.8–7.6), meaning
the SLSQP projector finds a consistent dynamics solution cheaply.

```
step 0:  p_des_z=1.186  p_z=1.186  proj_cost=1.19
step 5:  p_des_z=1.180  p_z=1.186  proj_cost=1.22
step 9:  p_des_z=1.237  p_z=1.186  proj_cost=6.24   ← first sign of tension
```

The drone's actual z stays at 1.186 m (PID holding altitude) but p_des is oscillating in
z, drifting down. track_err is still < 0.1 m.

### Phase 2 — proj_cost explosion (steps 13–29)

proj_cost becomes a **crash leading indicator**:

```
step 13:  proj_cost=147
step 14:  proj_cost=457
step 15:  proj_cost=830
step 16:  proj_cost=1223       ← track_err first exceeds 0.3 m
step 24:  proj_cost=2424
step 30:  proj_cost=13014      ← first contact
```

A high proj_cost means the SLSQP projector had to move the FM's raw trajectory a long
way to satisfy the Euler consistency constraint `act[t] = Δp_des[t]`. The FM was proposing
a trajectory so internally inconsistent (or so far from current state) that enforcing
dynamics required a large correction. This is the projector fighting the FM rather than
nudging it.

The spike sequence **predicts the crash 0.5 s before first contact** — proj_cost is a
leading diagnostic, not just a post-hoc indicator. In a real deployment, a proj_cost alert
at step 13 could trigger a recovery behaviour before the crash at step 30.

### Phase 3 — contact storm (T=0.909–1.576s, 21 contacts)

All 21 contacts happen in a 0.87 s window. First contact at T=0.909s with proj_cost=13014.
After contact begins, track_err explodes:

```
T=0.909s  contact  track_err=0.304 m
T=1.152s  contact  track_err=1.552 m    ← FM horizon z shows +0.997 m commanded
T=1.455s  contact  track_err=1.747 m    ← peak
T=1.576s  z=0.194 m → floor hit, drone frozen
```

The FM horizon at the contact steps shows large z-axis commands (e.g. `fm_horizon0_z = 1.000`,
`0.458`, `-0.357`) — the FM is oscillating wildly in z, consistent with multi-modal mode
blending between homotopy classes in corridor. The dynamics constraint keeps the executed
actions internally consistent, but cannot prevent the FM from predicting a downward
trajectory into the ground.

### Phase 4 — frozen (steps 52–242, 5.8 s)

After floor impact the drone stops completely:

```
STATE p=(-2.316, -0.477, 0.087)  v≈(0, 0, 0)   [constant for 190 steps]
```

Meanwhile the FM keeps running, p_des keeps accumulating (drifting to x=−0.286 by end):

```
p_des x progress:  −2.800 → −0.286   (+2.514 m)
drone x progress:  −2.800 → −2.316   (+0.484 m, all before the crash)
final track_err:   2.072 m
goal_dist:         5.254 m            (corridor is 5.6 m long; drone barely moved)
```

The policy has no awareness the drone is dead — it continues to generate actions that the
frozen PID integrates into growing p_des drift. This is the same open-loop integration
problem as the diffuser z-axis divergence, just playing out in x after a crash.

proj_cost returns to normal (~0.8–1.2) during Phase 4 because the drone velocity is ~0
and the FM predicts small actions consistent with a stationary drone. Low proj_cost ≠
healthy flight; it means the FM and drone agree that "nothing is moving."

---

## 3. Observation: proj_cost as a real-time health signal

The proj_cost spike sequence is the most actionable insight from this log:

| proj_cost range | meaning | action |
|---|---|---|
| 0.5–2.0 | FM proposal is near-dynamics-consistent — normal | none |
| 2–50 | FM slightly strained — mild disagreement between FM and dynamics | monitor |
| 50–500 | FM proposal far from dynamics consistency — approaching instability | warning |
| >500 | SLSQP projecting large correction — FM has lost coherence with actual state | trigger recovery |

A threshold like `proj_cost > 200` would have fired at step 14 — 0.5 s before first
contact. **Without the projector (diffuser variant) this signal would not exist.** The
projector both constrains the trajectory AND reveals via its cost how much it had to fight
the FM.

---

## 4. What this episode tells us about corridor homotopy C

Corridor homotopy C is the most challenging route (sharp corners or the longest path). The
FM trained on all homotopies is mode-blending — the z-commands oscillate between "climb"
and "descend" from step 0. Unlike the pillars scene where dpcc succeeded, here:

- The corridor has floor/ceiling geometry that punishes z errors immediately
- The FM's z-axis oscillation (visible in the horizon: `+0.997, -0.357, +0.458` at crash
  steps) cannot be corrected by the dynamics constraint alone — dynamics enforces `act[t] =
  Δp_des[t]` but does not prevent p_des_z from drifting toward the floor
- Spatial constraints (floor `z > z_min`, ceiling `z < z_max`) would address this directly —
  exactly the halfspace/bounds constraints that are wired but empty this epoch (U3)

---

## 5. Conclusions and what to do next

### Confirmed findings

1. **Cluster is not real-time for this model.** 145 ms total vs 30.3 ms budget = 4.8× over.
   FM inference (85 ms) is the larger bottleneck; projector (60 ms) is secondary. No amount
   of projector tuning closes this gap — the FM itself needs to be faster or the hardware
   better.

2. **Dynamics constraint alone is not enough for corridor.** In pillars it brought
   goal_dist from 6.5 m to 1.0 m. Here the drone crashes at 0.5 m progress. The FM's
   z-axis mode blending is fatal in a floor-ceiling constrained scene. Adding spatial
   constraints (`z > 0.3 m` lower bound, `z < 2.5 m` upper bound) is the next required step.

3. **proj_cost is a leading crash indicator.** Spikes 0.5 s before first contact. Could be
   used as a real-time safety trigger — suspend or re-plan when proj_cost exceeds a
   threshold. This is only possible because we have the projector; diffuser has no such signal.

4. **After crash, the FM does not know the drone is dead.** p_des drifts 2.5 m while the
   drone is frozen. A termination / re-initialisation trigger is needed: if track_err >
   threshold for N consecutive steps → halt or reset.

### Recommended next steps (priority order)

| Priority | Action | Expected fix |
|----------|--------|------|
| 1 | Add `z_min` / `z_max` bounds to `obstacle_constraints` in `uav_eval.yaml` | Prevent FM z-drift from reaching floor/ceiling |
| 2 | Add corridor halfspace constraints for walls | Restrict lateral FM mode-blending |
| 3 | Add `track_err > 0.5 m` termination in rollout | Stop dead-drone policy from running 6 s |
| 4 | Profile FM on faster hardware or reduce ODE steps | Close the real-time gap |
| 5 | Use `proj_cost > 200` as a real-time anomaly flag | Early warning for recovery |

---

## Appendix — raw numbers

| Metric | Value |
|--------|-------|
| Steps | 243 |
| Duration | 7.36 s |
| fm_ms mean/max/p95 | 85.4 / 89.1 / 86.1 ms |
| proj_ms mean/max/p95 | 59.8 / 62.6 / 61.0 ms |
| total_ms mean/max/p95 | 145.2 / 149.4 / 146.5 ms |
| Over budget | 243/243 (100%) |
| proj_cost mean/median/max | 414.6 / 0.92 / 13977 |
| proj_cost spikes (>100) | 37/243 steps |
| First proj_cost spike | step 13 (T=0.394s) |
| First contact | step 30 (T=0.909s) |
| Floor hit (z<0.2) | step 52 (T=1.576s) |
| Frozen from step | 52 onward |
| x-progress (p_des) | 2.514 m of 5.6 m total |
| x-progress (drone) | 0.484 m |
| Final track_err | 2.072 m |
| goal_dist | 5.254 m |
| Contacts | 21 (all in 0.87 s window) |
| contact_frac | 0.063 |
