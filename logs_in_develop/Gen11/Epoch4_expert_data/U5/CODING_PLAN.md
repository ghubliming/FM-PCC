# Gen11 E4 U5 — Coding Plan

**Date:** 2026-06-10  
**Source:** Gen11E4U4 P&S MD (strictly followed)  
**Goal:** Fix s_curve 100% rejection so collection proceeds to 500 saved episodes.

---

## Step 1 — Instrument reject reasons (do first, ~20 lines)

**File:** `uav_expert_data_collect/generator.py` → `run_trial`

Currently both reject paths return `None` indistinguishably. Change:

```
# Before
if contact_frac > contact_limit:
    return None
if min_z < Z_FLOOR_MARGIN:
    return None

# After — return a reject dict so collect.py can log it
if contact_frac > contact_limit:
    return {"rejected": True, "reason": "contact", "contact_frac": contact_frac, "min_z": min_z}
if min_z < Z_FLOOR_MARGIN:
    return {"rejected": True, "reason": "floor", "min_z": min_z, "contact_frac": contact_frac}
```

**File:** `uav_expert_data_collect/collect.py` (or equivalent runner)

- Check return type; accumulate reject reasons into a counter.
- At end of batch print: `REJECT HISTOGRAM: floor=N  contact=M  total=K`
- Also print per-reject: `min_z=X.XX  at_step=NNN  motor_clip_frac=Y` (if instrumented).

**Also in `run_trial`:** record `min_z`, `max_contact_frac`, and optionally per-step motor clip fraction (fraction of steps where any motor was clipped) to the reject dict.

**Validation:** run 20 trials s_curve; confirm histogram printed. This tells us which check fires before any trajectory tuning.

---

## Step 2 — Altitude headroom (1 line)

**File:** `uav_expert_data_collect/generator.py` → `_build_traj_and_init` (or equivalent init)

Find the s_curve z-sampling line:

```python
# Before
z = np.random.uniform(0.70, 1.10)

# After
z = np.random.uniform(0.90, 1.30)
```

Rationale: observed z-dip is 0.30–0.45 m; raising floor by +0.20 m converts most floor-rejects to accepts. Walls are 1.5 m tall — [0.90, 1.30] stays inside corridor.

**Validation gate:** 20-trial smoke. Expect rejection well under 60% after Steps 1+2.

---

## Step 3 — Slow the diagonal (1 line, apply if rejection still > 30%)

**File:** `uav_expert_data_collect/trajectories.py` → s_curve phase time allocation

Find where Seg B (diagonal) gets its time budget. Double its relative weight so peak lateral accel/velocity is roughly halved (overshoot scales quadratically with peak velocity).

```python
# Example — exact variable names TBD after reading the file
# Before: equal-weight allocation across segments
seg_times = allocate_by_distance([d_a, d_b, d_c], T_move)

# After: give Seg B 2x weight
seg_times = allocate_by_distance([d_a, 2.0 * d_b, d_c], T_move)
```

---

## Step 4 — Thrust-priority allocation in CascadedPID (principled fix)

**File:** `uav_env_test/flight_controller.py` → `CascadedPID.compute`

Current clipping corrupts collective thrust when attitude torques saturate:

```python
u = np.clip(u, self.u_min, self.u_max)   # line ~61 — clips AFTER M_inv @ wrench
```

Replace with thrust-priority allocation:

```python
# After M_inv @ wrench → u (4-vector, per-motor)
u_desired = u.copy()
over = u_desired > self.u_max
under = u_desired < self.u_min
if over.any() or under.any():
    # Scale torque components down, preserve collective thrust
    thrust_cmd = u_desired.mean()                        # collective (mean of 4 motors)
    torque_comp = u_desired - thrust_cmd                 # per-motor torque offset
    scale = 1.0
    for _ in range(10):                                  # binary search on scale
        u_try = thrust_cmd + scale * torque_comp
        if u_try.max() <= self.u_max and u_try.min() >= self.u_min:
            break
        scale *= 0.5
    u = np.clip(thrust_cmd + scale * torque_comp, self.u_min, self.u_max)
else:
    u = u_desired
```

Alternative (simpler): raise `u_max = 2.6 * u_hover` instead of 2.0 to give attitude correction more headroom before saturation.

This fix also resolves the pillars end-of-episode speed spikes (Fix C closes for free).

---

## Step 5 — Remove / relocate hover pauses

**File:** `uav_expert_data_collect/trajectories.py` → s_curve hover definitions

The F4 hover pauses at x=±0.5 (wall end-face plane, only 0.14 m lateral margin) did nothing for the rejection rate and sit in a risky position. Two options:

- **Remove entirely** if Steps 1–4 bring rejection < 30%.
- **Relocate** hover to x=∓0.7 (inside the corridor) if settling is still wanted.

---

## Execution order

| Step | Change | Lines | Gate |
|------|--------|-------|------|
| 1 | Instrument reject reasons | ~20 | Print histogram on 20-trial run |
| 2 | z range [0.90, 1.30] | 1 | rejection < 60% |
| 3 | Slow Seg B (2× weight) | 1 | rejection < 30% |
| 4 | Thrust-priority CascadedPID | ~15 | rejection < 10% |
| 5 | Remove/relocate hover pauses | 2–5 | Cosmetic, after gate passed |

**Pre-training gate:** rejection < 30% on 20-trial smoke → run full 500-trial collection → proceed to E5 GIF generation.
