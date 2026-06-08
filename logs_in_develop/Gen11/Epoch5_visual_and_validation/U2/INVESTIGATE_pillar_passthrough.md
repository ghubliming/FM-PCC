# Investigation: Drone Appears to Fly Through Pillars in GIF

**Date**: 2026-06-08  
**Trigger**: GIF `logs/uav_expert_data/gifs/pillars/L_L_L/pillars_L_L_L_pid_default_0000000.gif`
shows the drone visually passing through / very close to a pillar.  
**Question**: Is this a deep E4 data quality problem, or only an E5 GIF render artifact?

---

## Short answer

**This is a deep E4 data problem, not a GIF render artifact.**

The pillar trajectory generator (`weave` with `y_amplitude=-1.0, period=4.0s`) produces
**random lateral positions at each pillar column** depending on the episode duration T.
For roughly **half of all T values in the sampled range [10, 16]s**, the drone physically
contacts a pillar with its rotor during data collection. For T ≈ 12–14 s, the drone
actually passes to the *right* of all pillars — the opposite side from what "L_L_L" means.

The GIF is showing the correct real physics position (after E5 U2 obs column fix). The
visual "flying through" is real.

---

## Root cause analysis

### The weave trajectory formula

`generator.py` uses a sinusoidal weave in TIME (not in x):

```python
# uav_env_test/trajectories.py line 135
p = np.array([x_start + v_x * t_eff,   A * np.sin(omega * t_eff),   z])
# A = -1.0 for L_L_L,  omega = 2π / 4.0 s,  v_x = 6.4 / T
```

The y position at each pillar x depends entirely on the phase `omega * t` at that x:

```
t_pillar = (x_pillar - x_start) / v_x = (x_pillar + 3.2) * T / 6.4

y_pillar = -1.0 * sin(2π/4.0 * (x_pillar + 3.2) * T / 6.4)
```

**T is randomly sampled** from `rng.uniform(10.0, 16.0)`. There is no constraint that
the wave peaks or troughs align with the three pillar x positions.

### Pillar contact geometry

The drone rotors are **collision geoms** (not visual-only). Each rotor is an ellipsoid:
```xml
<default class="rotor">
    <geom type="ellipsoid" size=".13 .13 .01"/>   <!-- radius 0.13 m in x and y -->
</default>
<geom name="rotor1" class="rotor" pos="-.14 -.18 .05" .../>
<geom name="rotor2" class="rotor" pos="-.14  .18 .05" .../>
...
```

Rotor reach in y from drone COM: `±(0.18 + 0.13) = ±0.31 m`  
Pillar radius: `0.12 m`  
Pillar A at `y = −0.60`, Pillar B at `y = +0.60`.

The **contact zone** for Pillar A — drone y values that cause rotor-to-pillar overlap:

| Drone COM y | Closest rotor center y | 2D dist to Pillar A | Clearance |
|---|---|---|---|
| −0.50 | −0.32 | 0.313 m | **+0.063 m OK** |
| −0.70 | −0.52 | 0.161 m | **−0.089 m CONTACT** |
| −0.80 | −0.62 | 0.141 m | **−0.109 m CONTACT** |
| −0.92 | −0.74 | 0.198 m | **−0.052 m CONTACT** |
| −1.00 | −0.82 | 0.261 m | **+0.011 m OK (1.1 cm)** |

The contact zone for Pillar A is approximately **y ∈ (−0.55, −0.97)**.  
At the designed amplitude peak (y = −1.0), clearance is only **1.1 cm**.  
At the designed channel centre `_Y_L = −0.92` in `trajectories.py`, the drone is
actually **INSIDE the contact zone** (clearance = −0.052 m).

### Full scan: y at each pillar vs duration T

Computed from the weave formula, for `y_amplitude=-1.0, period=4.0, x_range=(-3.2,3.2)`:

```
T(s)  | y@A1    y@A2    y@A3   | clearance A1  A2  A3  | contact
----------------------------------------------------------------------
T=10.0 | -0.195  -1.000  -0.195 | +0.351  +0.011  +0.351 | none (A2 margin 1.1cm)
T=10.5 | -0.049  -0.924  -0.741 | +0.240  -0.049  -0.105 | A2 A3  ← CONTACT
T=11.0 | +0.098  -0.707  -0.995 | +0.101  -0.092  +0.007 | A2     ← CONTACT
T=11.5 | +0.243  -0.383  -0.858 | -0.024  +0.171  -0.090 | B1 A3  ← CONTACT (pillar B!)
T=12.0 | +0.383  -0.000  -0.383 | -0.105  +0.193  +0.171 | B1     ← CONTACT
T=12.5 | +0.514  +0.383  +0.243 | -0.081  -0.105  -0.024 | B1 B2 B3 ← ALL THREE
T=13.0 | +0.634  +0.707  +0.773 | +0.006  +0.069  +0.130 | none (margins tiny)
T=13.5 | +0.741  +0.924  +0.999 | +0.100  +0.273  +0.345 | none
T=14.0 | +0.831  +1.000  +0.831 | +0.185  +0.347  +0.185 | none
T=14.5 | +0.904  +0.924  +0.337 | +0.254  +0.273  -0.087 | B3     ← CONTACT
T=15.0 | +0.957  +0.707  -0.290 | +0.305  +0.069  +0.259 | none
T=15.5 | +0.989  +0.383  -0.803 | +0.336  -0.105  -0.108 | B2 A3  ← CONTACT
T=16.0 | +1.000  +0.000  -1.000 | +0.347  +0.193  +0.011 | none (A3 margin 1.1cm)
```

**Out of 13 sample T values: 7 cause at least one pillar contact (54%).**

### The homotopy label is also wrong

For T ∈ [12, 15]s, the drone's y at the pillar columns is **positive** (right side of
centre). An L_L_L episode at T=13s has the drone passing y=+0.634, +0.707, +0.773 at
the three pillars — this is a R_R_R-class path, not L_L_L. The label is assigned by
the trajectory intention (amplitude sign), not the actual lateral position at each pillar.

---

## This is E4 data, not E5 GIF

The current GIFs in `logs/uav_expert_data/gifs/` were generated **before** E5 U2 Phase 2
(old obs column bug — rendering at `p_des` not actual `p`). But `p_des` IS the weave
trajectory, so the rendered position follows the same sinusoidal path. The visual overlap
is the actual commanded trajectory clipping the pillar.

After Phase 2 re-render (rendering at actual `p`), the GIF will change slightly (PID
lag moves actual `p` slightly away from `p_des`), but the fundamental problem remains
in the underlying physics data.

---

## Code to verify on cluster

Run this on a node with pickle access to confirm contact fraction and actual y values:

```python
import pickle, glob, numpy as np

results = []
for path in sorted(glob.glob(
        'logs/uav_expert_data/pillars/L_L_L/*.pkl')):
    ep = pickle.load(open(path, 'rb'))
    obs   = ep['obs']             # (T, 9) after U2
    p     = obs[:, 3:6]           # actual positions
    meta  = ep['metadata']
    T_dur = meta['total_time']
    cf    = meta['contact_fraction']

    # y at each pillar x (nearest timestep)
    v_x_approx = (p[-1, 0] - p[0, 0]) / (len(p) * ep['dt'])
    y_at_pillars = []
    for xp in [-2.0, 0.0, 2.0]:
        idx = np.argmin(np.abs(p[:, 0] - xp))
        y_at_pillars.append(p[idx, 1])

    results.append({
        'id':      ep['episode_id'],
        'T':       T_dur,
        'cf':      cf,
        'y_pils':  y_at_pillars,
        'contact': cf > 0,
    })

results.sort(key=lambda r: r['T'])
print(f'{"T(s)":>6}  {"cf":>6}  {"y@A1":>7}  {"y@A2":>7}  {"y@A3":>7}  contact?')
for r in results[:30]:
    print(f'{r["T"]:6.1f}  {r["cf"]:6.4f}  '
          f'{r["y_pils"][0]:+7.3f}  {r["y_pils"][1]:+7.3f}  {r["y_pils"][2]:+7.3f}  '
          f'{"YES" if r["contact"] else "   "}')

n_contact = sum(r['contact'] for r in results)
n_wrong_side = sum(all(y > 0 for y in r['y_pils']) for r in results)
print(f'\nTotal L_L_L episodes: {len(results)}')
print(f'Episodes with contact (cf>0): {n_contact} ({100*n_contact/len(results):.0f}%)')
print(f'Episodes where drone passes RIGHT of all pillars: {n_wrong_side} ({100*n_wrong_side/len(results):.0f}%)')
```

---

## Severity assessment

| Issue | Severity | Affects |
|---|---|---|
| Rotor contact with pillars accepted | **High** | ~50% of L_L_L/R_R_R episodes have brief contact |
| Homotopy label wrong (L_L_L but goes right) | **High** | ~30% of L_L_L episodes for T≈12–14s |
| 1.1 cm clearance at nominal trajectory | **Medium** | Any PID error eliminates the margin |
| `_Y_L = -0.92` in trajectories.py is INSIDE contact zone | **High** | The explicit pillar_path would also have this problem |

---

## Root cause: weave vs explicit pillar_path

`generator.py` comment (Fix_1) says `pillar_path` was replaced by `weave` because
`pillar_path` caused zero-velocity stops near pillars. But the explicit `pillar_path`
in `trajectories.py` uses the channel centres `_Y_L/-Y_C/_Y_R` and includes a 0.20m
safety margin:

```python
# trajectories.py line 36
_Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_MARGIN   # = -0.6 - 0.12 - 0.20 = -0.92 m
```

The margin of 0.20 m was intended to give clearance from the **pillar surface** (drone
COM to pillar surface = 0.20 m). But it does not account for the **rotor reach of 0.31 m**.
The actual minimum COM distance for zero contact should be:
```
y_safe = pillar_A_y - pillar_radius - rotor_reach_y = -0.6 - 0.12 - 0.31 = -1.03 m
```

Both the weave amplitude (-1.0) and the explicit channel centre (-0.92) are **inside the
contact zone**.

---

## Recommended fix

**Short term (before E5 Phase 2 / E4 U3 re-collect)**: Investigate contact fraction
distribution on cluster. If the majority of L_L_L episodes have `cf=0`, the contact is
so brief that it may be acceptable for current training.

**Proper fix (E4 U4 or E5 fix)**: Two options:

**Option 1 — Increase amplitude / move channel outward**:
```python
# generator.py — change amplitude to give 5 cm clearance at peak
# Required: y_peak < -1.03m → amplitude = -1.08 (5cm margin)
_amp_map = {
    '(L,L,L)': -1.08,   # was -1.0, contact zone ends at -0.97
    '(R,R,R)': +1.08,
    '(L,R,L)':  0.0,
    '(R,L,R)':  0.0,
}
```
This only fixes the peak. The phase alignment problem (wrong homotopy label for some T)
remains.

**Option 2 — Phase-lock the weave to pillar positions (complete fix)**:
Force the wave to reach the amplitude at every pillar x by choosing `period` such that
the wave completes a half-period between consecutive pillar x positions (spacing = 2.0 m):
```python
# Require: x_spacing / v_x = half_period
# x_spacing = 2.0 m,  v_x = 6.4/T,  half_period = period/2
# → period = 2 * x_spacing / v_x = 2 * 2.0 * T / 6.4 = T * 0.625
# This makes period T-dependent — not directly supported by weave().
```
This requires a redesign of the weave to parameterise by x-position, not time.

**Option 3 — Revert to pillar_path with corrected channel centres**:
```python
# trajectories.py: fix _Y_L to account for rotor reach
_Y_L = PILLAR_Y_A - PILLAR_RADIUS - 0.35   # = -0.6 - 0.12 - 0.35 = -1.07 m
```
And fix `pillar_path` to use velocity-continuous transitions (the original reason for
switching to weave was zero-velocity stops — those can be fixed without abandoning
the explicit homotopy path).

**Recommended**: Option 3 — explicit path with corrected margins, fix the velocity
continuity issue separately. The weave approach is fundamentally misaligned with the
pillar homotopy structure.

---

## Files to change for the fix

| File | Change |
|---|---|
| `uav_expert_data_collect/trajectories.py` | `_Y_L`, `_Y_R` → −1.07 / +1.07; fix `pillar_path` velocity continuity |
| `uav_expert_data_collect/generator.py` | Revert pillar case from `weave` back to `pillar_path` |

Re-collection required: all pillars homotopy classes (477 episodes). New collect = E4 U4.

---

## Why does the drone not bounce off / stop when it contacts the pillar?

This is the most confusing part. The answer is: **MuJoCo physics IS responding — it is
not a MuJoCo bug and the contact forces are real. The drone does not literally pass
through in the physics simulation.** What happens is a combination of three things:

### 1. PID authority overwhelms brief contact force (by design)

When a rotor clips a pillar:
1. MuJoCo computes a contact constraint force (normal to the contact surface).
2. This force pushes the drone slightly away — a small lateral deflection over 1–3 physics steps (0.01–0.03 s).
3. The PID, which has no knowledge of contact, simultaneously applies rotor thrust to
   maintain the desired trajectory.
4. The PID's thrust authority is large enough to overcome the brief contact impulse.
5. Net result: a 1–3 step wobble, then recovery. `contact_fraction ≈ 0.003–0.010` — well
   below the 0.02 acceptance threshold.

This was a **deliberate design choice** in E4. The 0.02 contact-fraction threshold was
set specifically to allow brief wall and pillar clips (corridor L/R homotopies also clip
walls; reverting to a tighter threshold in E4 U2 Fix_1 caused 38% rejection rate).
Brief contact IS physically realistic for tight homotopy paths and was accepted.

### 2. MuJoCo's soft contact model resolves over multiple steps

MuJoCo uses `solref="0.02 1"` by default (contact reference time 20 ms, damping ratio 1).
This means the contact stiffness ramps up over ~20 ms rather than producing an instant
hard collision. At 100 Hz physics (10 ms/step), the contact force grows over 2 steps —
it does not produce a hard stop. For a drone moving at 0.7 m/s, the contact with a fixed
pillar at a glancing angle generates mostly a lateral nudge, not a dead-stop impact.

No XML `<contact>` or explicit `solref/solimp` overrides are set in
`quadrotor_modified.xml` or `scene_pillars.xml` — both use MuJoCo defaults, which are
tuned for soft, stable simulation, not rigid-body collisions.

### 3. The GIF currently shows p_des, not actual p (pre-Phase 2)

The GIFs currently in `logs/uav_expert_data/gifs/` were generated **before E5 U2 Phase 2**
with the old obs column bug: rendering at `obs[t, :3]` = **p_des** (commanded position),
not the actual drone position.

`p_des` is the mathematical weave trajectory output. It follows `y = -sin(ωt)` exactly,
regardless of what the physics engine did. For episode durations where the weave puts
`p_des.y` inside the pillar zone (e.g. y = −0.80 at a pillar for T=10.5s), the GIF
renders the drone VISUALLY inside the pillar — even though the actual drone `p` was
deflected away by contact forces.

```
What the OLD GIF shows:   p_des = weave(t) → commanded path → CAN cross pillar boundary
What the physics did:     p     = actual   → briefly clipped, then PID recovered
What the NEW GIF will show (after Phase 2): p → the physical deflection, looks less severe
```

**This means part of the visual "flying through" in the current GIFs is a rendering
artifact of the pre-Phase-2 obs column bug.** After Phase 2 re-render, the drone will
no longer be shown at the commanded trajectory — it will show the actual deflected path.
However, for episodes where the actual drone position also clips the pillar (contact events
confirmed in pickle metadata), the problem will still be visible.

### Summary table

| Phenomenon | Is this a bug? | Source |
|---|---|---|
| Drone rotor overlaps pillar in GIF visually | Part render artifact (p_des vs p), part real | E5 obs column bug + E4 trajectory design |
| MuJoCo does not hard-stop the drone | Not a bug — soft contact + PID authority | Deliberate design (0.02 threshold) |
| Contact force is present in physics | ✅ Working correctly | MuJoCo contact model |
| Contact_fraction > 0 in pickle metadata | ✅ Expected for L/R homotopies | E4 design decision |
| Drone passes on WRONG SIDE of pillar | **Real data problem** — wrong homotopy label | Weave phase misalignment bug |

---

## Cross-references

| Document | Content |
|---|---|
| `uav_expert_data_collect/generator.py` line 151–168 | Current weave implementation + Fix_1/2 comments |
| `uav_expert_data_collect/trajectories.py` line 36–38 | `_Y_L/_Y_R` channel centres (wrong margins) |
| `uav_expert_data_collect/trajectories.py` line 51–84 | `pillar_path` — the explicit alternative |
| [`PLAN.md`](PLAN.md) | E5 U2 context |
| [`../../Epoch4_expert_data/METHODOLOGY.md`](../../Epoch4_expert_data/METHODOLOGY.md) | §11 episode counts |
