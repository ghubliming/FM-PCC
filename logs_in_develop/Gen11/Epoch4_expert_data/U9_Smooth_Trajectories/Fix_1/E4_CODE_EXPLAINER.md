# Gen11 E4 — Expert Data Collection: Full Code Explainer

**Date:** 2026-06-12
**Scope:** Everything in `uav_expert_data_collect/` and `uav_env_test/` that runs the E4
collection pipeline, including Fix_1 geometry fix.
**Purpose:** Study reference — understand every layer from scene geometry through physics
simulation to the pickle schema written to disk.

---

## Table of Contents

1. [System overview — what the pipeline does](#1-system-overview)
2. [File map — what each file is responsible for](#2-file-map)
3. [Scene geometry — the four environments](#3-scene-geometry)
4. [Homotopy classes — what they mean and why they matter](#4-homotopy-classes)
5. [Trajectory layer — the math of motion planning](#5-trajectory-layer)
   - 5.1 `traverse_line` — single-leg cosine profile
   - 5.2 `blended_path` — multi-waypoint smooth path (U9 core)
   - 5.3 `pillar_path` — homotopy routing through 3 pillar pairs
   - 5.4 `s_curve_scene_path` — two-corridor Z-route
   - 5.5 `corridor_path` and `empty_path`
6. [Cascaded PID controller — the "expert"](#6-cascaded-pid-controller)
   - 6.1 Position outer loop
   - 6.2 Attitude inner loop (SO(3))
   - 6.3 Motor allocation
   - 6.4 Thrust-priority saturation handling
7. [Physics simulation — MuJoCo integration](#7-physics-simulation)
8. [Trial generation and rejection criteria](#8-trial-generation-and-rejection)
9. [Dataset writing — from rollout to pickle](#9-dataset-writing)
10. [Collection orchestrator — `collect.py`](#10-collection-orchestrator)
11. [Verification tools — `verify_blends.py` and `stats_validator.py`](#11-verification-tools)
12. [End-to-end data flow](#12-end-to-end-data-flow)
13. [Key design decisions and their rationale](#13-key-design-decisions)

---

## 1. System Overview

The goal of E4 is to produce a **supervised training dataset for a Flow Matching
trajectory policy (FM-PCC)**. The policy learns to imitate an expert. Here the "expert"
is a cascaded PID controller that follows a hand-designed reference trajectory. We collect
many episodes across four environments (scenes), randomise start positions, durations, and
gain settings, and save the resulting state-action pairs.

The high-level loop for each episode:

```
1. Sample scene + homotopy + random parameters (altitude, duration, jitter)
2. Build a reference trajectory function  traj(t) → (p_des, v_des, a_des, yaw_des)
3. Step MuJoCo physics:
   for each physics step k:
       query traj(k·dt) → commands
       run PID → motor thrusts
       step simulator
       record (p, v, p_des, q)
4. Quality-check the rollout (contact fraction, floor crash)
5. If accepted: downsample to 33 Hz, add per-episode noise, write pickle
```

What comes out: `logs/uav_expert_data/<scene>/<homotopy_safe>/<episode_id>.pkl`

Each pickle is a self-contained episode dict. The FM model trains on `(obs, actions)` from
this dict. `obs` is `[p_des(3) | p(3) | v(3)]` = 9D; `actions` is `Δp_des` = 3D.

---

## 2. File Map

```
uav_env_test/
  trajectories.py       — primitive trajectory factories (traverse_line, blended_path,
                          circle, weave, s_curve_path, hover_at, step_to)
  flight_controller.py  — CascadedPID (the "expert" controller)

uav_expert_data_collect/
  trajectories.py       — scene-specific wrappers (pillar_path, corridor_path,
                          s_curve_scene_path, empty_path) + geometry constants
                          BLEND_RADIUS, PILLAR_CHANNELS, CORRIDOR_CHANNELS
  generator.py          — run_trial(): builds trajectory, runs MuJoCo loop,
                          returns raw rollout or reject dict
                          SCENE_XMLS, SCENE_OBSTACLES, HOMOTOPY_CLASSES, GAIN_VARIANTS
  dataset_writer.py     — rollout_to_episode(): downsamples, adds noise, packs schema
                          save_episode(): writes pickle
  collect.py            — main driver: loops n_trials, calls generator + writer,
                          tracks stats, writes run_summary.json
  stats_validator.py    — post-collection sanity check: speed, action norms, counts
  verify_blends.py      — numpy-only geometric check of blended_path clearances
```

---

## 3. Scene Geometry

Four MuJoCo XML scenes live in
`d3il/environments/d3il/models/mj/robot/quadrotor/scenes/`.

### empty

No obstacles. The drone flies a random point-to-point with minimum separation 1.0 m.
Start/end both drawn from `[-1.8, 1.8]^2` in xy, `[0.70, 1.10]` in z.
Duration: `max(4.0, dist / 0.4)` — at least 4 s or long enough for 0.4 m/s average.

### corridor

Two axis-aligned box walls:
- `wall_y_neg` at `y = -0.5`, half-extents `[2.0, 0.05, 0.75]`
- `wall_y_pos` at `y = +0.5`, half-extents `[2.0, 0.05, 0.75]`

Clear channel: `y ∈ (-0.45, +0.45)`, width 0.90 m.

The drone enters from `x = -2.8` and exits at `x = +2.8` along one of three lateral
channels:
- `L: y = -0.12` (left-biased, stays 0.33 m from left wall centre)
- `C: y = 0.0` (centred, ±3 cm jitter)
- `R: y = +0.12` (right-biased)

Channel values were tightened from `±0.18` (U1/U2) to `±0.12` (U3) because at `±0.18`
the rotor ellipsoid (reach 0.31 m from COM) contacted the wall on every L/R episode.
At `±0.12`: clearance = `0.45 - 0.12 - 0.31 = 0.02 m` (2 cm, just inside contact-free).

### s_curve

Two offset corridors connected by a gap:
- Segment 1: `x ∈ [-3, -0.5]`, corridor centred at `y = -0.8`
- Segment 2: `x ∈ [+0.5, +3]`, corridor centred at `y = +0.8`
- Gap: `x ∈ [-0.5, +0.5]`, open air

Critical geometry: the gap-side wall corners are at `A = (-0.5, -0.25)` and
`B = (+0.5, +0.25)`. A straight diagonal from the Seg 1 channel to the Seg 2 channel
passes only 0.291 m from these corners — inside the 0.31 m rotor reach — making a direct
diagonal geometrically impossible regardless of speed. This forced the Z-route design
(§5.4).

### pillars

Six vertical cylinders, radius 0.12 m, at:
- Column A (y = -0.6): x ∈ {-2, 0, +2}
- Column B (y = +0.6): x ∈ {-2, 0, +2}

The drone enters from `x = -3.2` and exits at `x = +3.2`, choosing one of two lateral
channels at each of the three pillar pairs.

**Channel positions (Fix_1 U3 geometry):**
```
_Y_L = -0.6 - 0.12 - 0.31 - 0.08 = -1.11   (left of col A)
_Y_C =  0.0                                   (centre between cols)
_Y_R = +0.6 + 0.12 + 0.31 + 0.08 = +1.11   (right of col B)
```

The margin formula: pillar_y ± (radius + rotor_reach + safety).
- `radius = 0.12 m` — physical pillar radius
- `rotor_reach = 0.31 m` — maximum COM→rotor edge distance in any direction
- `safety = 0.08 m` — 8 cm buffer above the contact-free threshold

At `_Y_L = -1.11`, the nearest pillar axis (column A at y = -0.6) is 0.51 m away —
0.08 m beyond the rotor-reach boundary. That 0.08 m is the minimum safety margin.

---

## 4. Homotopy Classes

A homotopy class labels *which topological path* the drone takes through the environment.
Two trajectories have the same homotopy if one can be continuously deformed into the other
without passing through an obstacle.

| Scene | Homotopy classes | Meaning |
|-------|-----------------|---------|
| `empty` | `['N/A']` | Only one topological class (open space) |
| `corridor` | `['L', 'C', 'R']` | Left / centre / right of corridor |
| `s_curve` | `['default']` | Only one route through the S |
| `pillars` | `['(L,L,L)', '(L,R,L)', '(R,L,R)', '(R,R,R)']` | Per pillar pair: pass left or right |

For the pillars scene, the label `(L,R,L)` means: pass left of pair 1, right of pair 2,
left of pair 3. There are 2³ = 8 theoretical combinations but only 4 are collected: the
four symmetric ones (all-same and the two alternating). The other 4 (`LLR`, `LRL`
excluding the symmetric ones, etc.) are skipped for dataset balance.

**Why homotopy balance matters for FM training:** if one homotopy dominates the dataset,
the policy learns to prefer it and will fail on underrepresented routes. The collection
loop cycles homotopies with `i % len(homotopy_pool)` so all classes get equal trial
counts.

---

## 5. Trajectory Layer

### 5.1 `traverse_line` — single-leg cosine profile

**File:** `uav_env_test/trajectories.py:65`

The simplest non-trivial trajectory. Given start `p_s` and end `p_e` and duration `T`:

**Position:** `p(t) = p_s + s(t) · Δp`
where `Δp = p_e - p_s` and the scalar progress variable:

```
s(t) = ½ (1 − cos(π t/T))
```

This is a **cosine blend** (also called a haversine or smoothstep). It starts at 0, rises
smoothly to 1, and ends at 1.

**Velocity:** `ṡ(t) · Δp` where
```
ṡ(t) = (π / 2T) sin(πt/T)
```

Peak speed at `t = T/2`: `v_peak = π‖Δp‖ / (2T)`

**Acceleration:** `s̈(t) · Δp` where
```
s̈(t) = (π/2T)² cos(πt/T)
```

Acceleration at `t=0` and `t=T` is `(π/2T)² ‖Δp‖`, not zero — the function is smooth but
not zero-acceleration at the endpoints. However it is zero-**jerk** at the endpoints
(the derivative of `cos` is `−sin` which is zero at 0 and `T`).

**Key properties:**
- `v(0) = v(T) = 0` — drone is stationary at start and end
- `v_peak = π‖Δp‖/(2T)` — set `T` to control peak speed
- Analytic `a(t)` is passed as feedforward to the PID → better tracking

### 5.2 `blended_path` — multi-waypoint smooth path (U9 core)

**File:** `uav_env_test/trajectories.py:114`

The central innovation of U9. Replaces the old `s_curve_path` / `traverse_line` chains
(which forced `v=0` at every interior waypoint — stop-and-go) with a **single continuous
smooth path** through all waypoints.

**Geometry construction:**

Given waypoints `[w₀, w₁, …, wₙ]` and a blend radius `r_max`:

For each interior corner `wᵢ` (not first or last):
1. Compute the turn angle `β = arccos(û_{i-1} · û_i)` where `û` are unit vectors of
   adjacent segments.
2. If `β < 1e-6` (collinear): **no fillet** — the corner is simply passed through.
   This is the key reason LLL/RRR pillars work at 100%: all three pillar pairs are on the
   same y-channel, so every interior corner is collinear.
3. Otherwise: compute tangent offset `d = r_max · tan(β/2)`, clamped so it does not
   consume more than half of either adjacent segment:
   ```
   d = min(r_max · tan(β/2),  0.5 · L_{i-1},  0.5 · Lᵢ)
   ```
   Then the actual fillet radius is `r = d / tan(β/2)` (≤ r_max).
4. The fillet is a circular arc tangent to both segments. Entry tangent point:
   `p1 = wᵢ − d · û_{i-1}`. Exit tangent point: `p2 = wᵢ + d · û_i`.
   Arc center: `center = p1 + r · n̂` where `n̂` is the unit normal pointing toward the
   turn. Arc length: `r · β`.

**Path elements:** The complete path is assembled as an ordered list of elements —
alternating straight segments and circular arcs. Each element has a length and an
evaluation function `eval(s_local) → (p, tangent, curvature)`.

For a **straight segment** from `a` in direction `û`:
```
p = a + s · û,   tangent = û,   curvature = 0
```

For a **circular arc** with center `c`, radius `r`, entry radial `e_r0`, entry tangent `û_0`:
```
φ = s / r
radial(φ) = cos(φ) · e_r0 + sin(φ) · û_0
p = c + r · radial
tangent = −sin(φ) · e_r0 + cos(φ) · û_0
curvature = −radial / r      (points toward center, magnitude 1/r)
```

**Global speed profile:**

Total arc length `L = Σ element_lengths`.

One cosine profile over the entire path:
```
s(t) = L · ½ (1 − cos(πt/T))
ṡ(t) = L · (π/2T) sin(πt/T)       ← arc-length speed (scalar)
s̈(t) = L · (π/2T)² cos(πt/T)      ← arc-length acceleration (scalar)
```

**Velocity and acceleration in world frame:**
```
v(t) = ṡ(t) · tangent(s(t))
a(t) = s̈(t) · tangent(s(t))  +  ṡ(t)² · curvature(s(t))
```

The second term `ṡ² · curvature` is the **centripetal acceleration**. On a circular
fillet with radius `r`, it has magnitude `ṡ²/r` and points toward the arc center. This is
what caused the Fix_1 bug:

At `r = 0.30 m`, T = 10 s (shortest pillar episode), the LRL/RLR path length is
approximately L ≈ 8.7 m. Peak arc-length speed:
```
ṡ_peak = L · π / (2T) ≈ 8.7 · π / 20 ≈ 1.37 m/s
```
Peak centripetal acceleration at the fillet:
```
a_centripetal = ṡ_peak² / r ≈ 1.87 / 0.30 ≈ 6.2 m/s²
```

In practice with the actual path length this works out to approximately **8.6 m/s²** —
about 0.88g lateral. This saturated the PID's lateral channel. Fix_1 raised `r` to
`0.45 m`, reducing it to ~5.7 m/s² (0.58g), inside the PID budget.

**Why this is a better primitive than chained `traverse_line`:**
- No interior `v=0` stops — the drone maintains positive speed throughout
- All derivative information (tangent, curvature) is analytic → clean feedforward
- One temporal parameter T controls the whole trajectory → easy to reason about
- The speed profile for a given path length and duration is identical to an equivalent
  single-segment `traverse_line` → the validated speed regime (E4 rejection gates) is
  preserved exactly

### 5.3 `pillar_path` — homotopy routing through 3 pillar pairs

**File:** `uav_expert_data_collect/trajectories.py:66`

**Waypoint layout (8 points, 7 segments):**

```
x:  [-3.2,  -2.5,  -1.5,  -0.5,  +0.5,  +1.5,  +2.5,  +3.2]
y:  [  0,   y₀,    y₀,    y₁,    y₁,    y₂,    y₂,     0  ]
```

Where `y₀, y₁, y₂` are the channel y-values for homotopy `[h₀, h₁, h₂]`:
- `'L'` → `_Y_L = -1.11`
- `'R'` → `_Y_R = +1.11`

The drone starts and ends at `y=0` (centre), ramps out to the first channel, transitions
between channels in the open space between pillar pairs, and ramps back to centre at exit.

**Why transitions happen BETWEEN pillars (not AT pillar x-positions):**

The quadrotor body has rotors at approximately `±0.14 m` (x) and `±0.18 m` (y) from the
COM. If the drone were still transitioning its y-channel at `x = -2.0` (pillar 1), the
front rotor (at COM_x + 0.14) would approach the pillar while still off-channel, and the
rear rotor (at COM_x − 0.14) would drag through the pillar after passing. The
x-coordinates for y-transitions are placed at `-1.5` and `+0.5` (between pairs), leaving
≥ 0.5 m in x from the nearest pillar before and after each corner.

**Clearance on the straight segments (8 cm minimum):**

At pillar x-positions `{-2, 0, +2}`, the path is straight and horizontal (y = constant).
Channel `L` is at y = -1.11; nearest pillar axis (col A) is at y = -0.6. Distance:
`1.11 - 0.6 = 0.51 m`. Rotor reach: 0.31 m. Clearance: `0.51 - 0.31 = 0.20 m` from
rotor edge. But the actual minimum during the diagonal transition is 0.08 m (8 cm) at the
closest approach to the pillar when the path is diagonally cutting between channels in
open space. `verify_blends.py` confirms all transitions stay ≥ 0.43 m from pillar axes
(0.12 + 0.31 = 0.43 m = minimum safe distance COM → pillar surface).

**For LLL/RRR:** `y₀ = y₁ = y₂`, so all interior waypoints are on the same y-channel.
The `blended_path` interior corner detection finds `β ≈ 0` at every waypoint → no fillet,
path is a single straight line. This is why LLL/RRR always achieves ~100% acceptance.

**For LRL/RLR:** interior corners at `x = -1.5` and `x = +1.5` involve real y-changes
(`|Δy| ≈ 2.22 m`) → large `β` → fillets are generated. These are the corners that caused
the Fix_1 saturation.

**Duration range:** `[10.0, 16.0]` seconds (drawn uniformly).

### 5.4 `s_curve_scene_path` — Z-route through two offset corridors

**File:** `uav_expert_data_collect/trajectories.py:123`

**Waypoint skeleton (6 points, 5 segments):**

```
(-3.2, y1, z) → (-0.5, y1, z) → (0.0, y1, z) → (0.0, y2, z) → (+0.5, y2, z) → (+3.2, y2, z)
```

Where `y1 = -0.8 + jitter`, `y2 = +0.8 + jitter`, `jitter ∈ [-0.04, +0.04]`.

**The Z-route explained:**

The naive straight diagonal from `(-0.5, y1)` to `(+0.5, y2)` fails geometrically
(0.291 m clearance < 0.31 m rotor reach). The Z-route adds a waypoint at `(0, y1)` and
`(0, y2)`, creating:
- **Leg B1** (pure x): `(-0.5, y1)` → `(0, y1)` — move right in corridor 1 y-lane
- **Leg B2** (pure y at x=0): `(0, y1)` → `(0, y2)` — move laterally in the gap
- **Leg B3** (pure x): `(0, y2)` → `(+0.5, y2)` — enter corridor 2

At `x=0`, the nearest gap-side corner is at `(-0.5, -0.25)` (corner A) or `(+0.5, +0.25)`
(corner B). For `y1 = -0.8`, leg B1 stays at `y = -0.8`, clearance from corner A (at
`y = -0.25`) is `|−0.8 − (−0.25)| = 0.55 m` > 0.31 m → safe.
Leg B2 is at `x=0`, corner A is at `x = -0.5` → `|0 − (−0.5)| = 0.5 m` in x → safe.

**With `blended_path`:**
- The corners at `(-0.5, y1)` and `(+0.5, y2)` — where the path turns 90° from x to y
  (or y to x) — are collinear with their outer neighbours (the long corridor legs are
  purely x-directional), so `β ≈ 0` → **no fillet**.
- The two 90° corners in the gap at `(0, y1)` and `(0, y2)` get fillets of radius
  `BLEND_RADIUS = 0.45 m`. Clearance from the gap-side corners at these fillets is ≥ 0.55 m.

**Duration:** `[16.0, 22.0]` seconds.

### 5.5 `corridor_path` and `empty_path`

**`corridor_path`:** calls `traverse_line` directly. A single straight segment from
`(-2.8, y, z)` to `(+2.8, y, z)`. No fillets needed — one segment.

**`empty_path`:** also calls `traverse_line`. Random 3D point-to-point.

---

## 6. Cascaded PID Controller

**File:** `uav_env_test/flight_controller.py`

This is the "expert" whose behaviour the FM policy learns to imitate. It implements the
Lee/Mellinger cascaded architecture standard in quadrotor control.

**Architecture overview:**

```
p_des, v_des, a_des, yaw_des
          ↓
   [1] Position PD + feedforward
          ↓   F_world (desired force vector in world frame)
   [2] Force direction + yaw → R_des (desired rotation)
          ↓
   [3] SO(3) attitude PD (Lee 2010)
          ↓   [T, τ_x, τ_y, τ_z] (total thrust + torques)
   [4] Motor allocation  M · u = wrench
          ↓
   u ∈ R^4  (motor thrusts, Newtons)
```

### 6.1 Position Outer Loop

**Inputs:** actual `p`, `v`; desired `p_des`, `v_des`, `a_des`

Position PD with feedforward:
```
e_p = p − p_des          (position error, world frame)
e_v = v − v_des          (velocity error, world frame)
a_cmd = −Kp_pos · e_p − Kd_pos · e_v + a_des
```

**Gains (default):**
- `Kp_pos = [4.0, 4.0, 8.0]` — stiffer in z (altitude) than xy
- `Kd_pos = [3.0, 3.0, 4.0]`

High-gain variant: `Kp_pos × 1.2`, `Kd_pos × 1.0`.
Low-gain variant: `Kp_pos × 0.8`, `Kd_pos × 0.9`.

**Required thrust vector:**
```
F_world = m · (a_cmd + [0, 0, g])
```

This is Newton's second law: the drone must produce `F_world` to achieve `a_cmd`, where
`[0, 0, g]` is the gravitational compensation.

### 6.2 Attitude Inner Loop

**Step 1 — Desired orientation from force direction:**

The thrust direction must align with `F_world`. Desired body-z axis:
```
b3_des = F_world / ‖F_world‖
```

Desired body-y axis (from yaw constraint):
```
x_c = [cos(yaw_des), sin(yaw_des), 0]
b2_des = (b3_des × x_c) / ‖b3_des × x_c‖
b1_des = b2_des × b3_des
```

`R_des = [b1_des | b2_des | b3_des]` (columns)

**Step 2 — SO(3) attitude error (Lee 2010):**

The skew-symmetric error matrix:
```
E = ½ (R_des^T R − R^T R_des)
```

Extract the three independent components as the error vector:
```
e_R = [E₂₁, E₀₂, E₁₀]   (vee map on the skew-symmetric part)
```

This error lives in the Lie algebra so(3) and is zero when `R = R_des`.

**Step 3 — Torque command:**

```
gyro = ω_body × (I · ω_body)     (gyroscopic term, I = inertia diagonal)
τ = −Kp_att · e_R − Kp_omega · ω_body + gyro
```

**Gains:**
- `Kp_att = [70.0, 70.0, 4.0]` — strong in roll/pitch, weak in yaw
- `Kp_omega = [2.5, 2.5, 1.0]`

Note `ω_des_body = 0` (acceptable for slow trajectories — the error is zero when the
body is not rotating, which is fine at the slow speeds we command).

**Total thrust scalar:**

Project the world-frame force onto the actual body-z axis `b3 = R[:, 2]`:
```
T = F_world · b3
```

If `T < thrust_floor = 0.1 · m · g`, clamp to avoid free-fall.

### 6.3 Motor Allocation

The four-motor allocation matrix `M ∈ R^{4×4}` (columns = motors, rows = wrench components):

```
M[:, i] = [1, r_y_i, −r_x_i, κ_i]
```

Where `(r_x_i, r_y_i)` is motor `i`'s site position in body frame and `κ_i` is the yaw
torque coefficient from the MuJoCo actuator gear (gear[5]).

Rows of wrench `[T, τ_x, τ_y, τ_z]`:
- Row 0 (T): all motors contribute equally (coefficient 1)
- Row 1 (τ_x / roll): `+r_y` — motors further in +y produce more roll
- Row 2 (τ_y / pitch): `−r_x` — motors further in −x produce more pitch
- Row 3 (τ_z / yaw): `κ_i` — alternating sign (co-rotating pairs cancel, counter-rotating pairs sum)

Solve:
```
u = M_inv · [T, τ_x, τ_y, τ_z]
```

### 6.4 Thrust-Priority Saturation Handling

If any motor demand exceeds `[u_min, u_max]`, we must scale back — but which to cut?
Naive clipping (`np.clip`) distorts the torque ratios unpredictably. The U6 fix uses
**thrust-priority with analytic scaling:**

```
thrust_cmd = mean(u)        (the per-motor average = T/4)
torque_comp = u − thrust_cmd   (the per-motor torque contribution)
```

For each motor `i` with `torque_comp_i ≠ 0`, compute the maximum scale that keeps it
in bounds:
```
if torque_comp_i > 0:   cap_i = (u_max − thrust_cmd) / torque_comp_i
if torque_comp_i < 0:   cap_i = (thrust_cmd − u_min) / (−torque_comp_i)
```

Final scale: `scale = max(min(all caps), 0.5)`.

The `0.5` floor ensures attitude authority is never reduced by more than half — cutting
it further would let the drone tumble.

```python
u = np.clip(thrust_cmd + scale * torque_comp, u_min, u_max)
```

**Saturation telemetry:** `pid.last_raw_saturated` is `True` if the raw `u` exceeded
bounds. `generator.py` counts these steps: `n_clip += int(pid.last_raw_saturated)`.
The `motor_clip_frac = n_clip / n_step` goes into the episode metadata and the run
summary, and was the key diagnostic that confirmed PID saturation for the Fix_1 bug
(LRL/RLR rejected episodes showed 29–78% clip rates).

---

## 7. Physics Simulation

**File:** `generator.py:run_trial()`, lines 206–282

The MuJoCo model is loaded fresh for each trial:
```python
model = mujoco.MjModel.from_xml_path(SCENE_XMLS[scene])
data  = mujoco.MjData(model)
```

**Initial state:**
```python
data.qpos[:3]  = init_pos        # x, y, z from _build_traj_and_init
data.qpos[3:7] = [1, 0, 0, 0]   # unit quaternion → level hover orientation
data.qvel[:]   = 0.0             # zero velocity
mujoco.mj_forward(model, data)   # propagate initial state (computes contacts etc.)
```

**Physics timestep:** `dt = float(model.opt.timestep)` — typically 0.01 s (100 Hz).

**Per-step loop:**
```python
for k in range(n_step):          # n_step = round(duration / dt)
    t = k * dt

    # 1. Query reference trajectory
    p_des, v_des, a_des, yaw_des = traj_fn(t)

    # 2. Read current state from simulator
    p  = data.qpos[:3].copy()    # position (world frame)
    v  = data.qvel[:3].copy()    # linear velocity (world frame)
    q  = data.qpos[3:7].copy()   # quaternion [w, x, y, z]
    om = data.qvel[3:6].copy()   # angular velocity (body frame)

    # 3. Run PID → motor thrusts
    u = pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)

    # 4. Count raw saturation events
    n_clip += int(pid.last_raw_saturated)

    # 5. Apply control and step simulator
    data.ctrl[:4] = u
    mujoco.mj_step(model, data)

    # 6. Check obstacle contacts (not floor)
    hit = any(_is_obstacle_contact(model, data.contact[ci])
              for ci in range(data.ncon))
    n_hit += int(hit)

    # 7. Record step data
    steps.append({'p': p, 'v': v, 'p_des': p_des, 'q': q})
```

**Contact detection:**
```python
def _is_obstacle_contact(model, contact):
    n1 = model.geom(contact.geom1).name
    n2 = model.geom(contact.geom2).name
    return n1 != 'floor' and n2 != 'floor'
```

Floor contacts are excluded because the drone body sometimes grazes the floor near
the start of an episode (during the vertical settling) and these are not obstacle
collisions. The floor rejection is handled separately by `Z_FLOOR_MARGIN`.

---

## 8. Trial Generation and Rejection Criteria

**File:** `generator.py:_build_traj_and_init()` (parameters) and `run_trial()` (criteria)

### Randomised parameters

Each trial is seeded with `seed = base_seed + i`. The RNG controls:

| Parameter | Scene | Range |
|-----------|-------|-------|
| altitude z | all | `uniform(0.90, 1.30)` |
| start/end (x,y) | empty | `uniform(−1.8, +1.8)^2` |
| duration | empty | `max(4.0, dist/0.4)` |
| y_jitter | corridor C | `uniform(−0.03, +0.03)` |
| duration | corridor | `uniform(6.0, 10.0)` |
| y_jitter | s_curve | `uniform(−0.04, +0.04)` |
| duration | s_curve | `uniform(16.0, 22.0)` |
| duration | pillars | `uniform(10.0, 16.0)` |
| gain_variant | all | cycles `[pid_default, pid_high_gain]` |

### Rejection criteria

**Contact fraction:** `contact_frac = n_hit / n_step`. If this exceeds the scene threshold:

```python
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,   # up to 2% steps can have obstacle contact
    'corridor': 0.02,   # same — brief wall grazes allowed
    's_curve':  0.08,   # higher because wall end-faces at x=±0.5 are unavoidably grazed
    'pillars':  0.001,  # near-zero — any pillar contact is a trajectory design bug
}
```

**Floor crash:** `min_z = min(p[2] for all steps)`. If `min_z < 0.50 m`, reject with
reason `'floor'`. This catches episodes where the drone drops (motor saturation, unstable
control) without hitting an obstacle — those would slip through the contact check.

**Abort on mass rejection:** if rejection rate exceeds `reject_limit = 0.60` after at
least 20 trials, the run aborts early and prints the rejection histogram by reason.

**Reject dict format (U5 Step 1):**
```python
{'rejected': True, 'reason': 'contact' | 'floor' | ...,
 'contact_frac': float, 'min_z': float, 'motor_clip_frac': float}
```

Having structured reject dicts (instead of bare `None`) lets `collect.py` build a
histogram: `contact=156  floor=70` — exactly what diagnosed the Fix_1 pillar issue.

---

## 9. Dataset Writing

**File:** `dataset_writer.py`

### Downsampling

Physics runs at 100 Hz (dt = 0.01 s). Training targets 33 Hz:
```python
stride = max(1, round(1.0 / (dt_physics * 33)))  # = 3 at 100 Hz
steps_33Hz = steps[::stride]
```

So every 3rd physics step is kept → `T ≈ duration × 33` frames per episode.

### Observation construction (9D)

```python
obs[t] = [p_des(3) | p(3) | v(3)]
```

Three concatenated 3-vectors:
- `p_des`: commanded position (what the trajectory said at this moment)
- `p`: actual drone position (what MuJoCo reported)
- `v`: actual drone velocity

The FM model uses this to learn the mapping "given where I was commanded to go,
where I actually am, and how fast I'm moving → what should the next command be?"

### Quaternion field (D-prep, added E4 U3+)

```python
q[t] = [w, x, y, z]   float32   # actual body quaternion from data.qpos[3:7]
```

Not used in training (the policy is position-only, not attitude-aware). Stored for
**visual rendering**: when generating GIFs and camera images, the quaternion lets the
MuJoCo renderer show the drone's actual tilt, not a forced-level avatar.

### Noise injection

To prevent the policy from overfitting to a single exact trajectory (and to thicken the
data manifold), a small constant offset is added to the **targets** (the p_des sequence):

```python
offset = rng.normal(0.0, noise_sigma=0.02, size=(1, 3))   # one constant per episode
targets = targets + offset
```

**Why constant per-episode, not per-step:** per-step noise makes `actions = diff(targets)`
noise-dominated — the signal is `σ/step ≈ 0.012 m/step` but per-step noise std is
`√2 · 0.02 ≈ 0.028 m/step`. A constant offset shifts the whole trajectory rigidly, so
`diff(targets + offset) = diff(targets)` — the actions are unaffected.

### Actions

```python
actions[t] = targets[t+1] − targets[t]   # forward difference
```

This is `Δp_des` in world frame. At 33 Hz with average speed 0.4 m/s:
```
‖Δp_des‖ ≈ 0.4 / 33 ≈ 0.012 m/step
```

The FM model learns to predict `Δp_des` given `obs`. At inference, the output is
accumulated: `p_des_new = p_des_old + action_predicted`.

### Episode ID

```python
ep_id = f'{scene}_{homotopy_safe}_{gain_variant}_{seed:07d}'
```

The 7-digit zero-padded seed is the critical field. `generate_physics_gifs.py` recovers
it with `int(ep_id.split('_')[-1])` to reconstruct the exact same random sequence and
re-run the physics for visual replay.

### Pickle schema (full)

```python
{
    'episode_id': str,            # e.g. "pillars_L_R_L_pid_default_0000042"
    'scene':      str,            # 'empty' | 'corridor' | 's_curve' | 'pillars'
    'homotopy':   str,            # '(L,R,L)' | 'L' | 'default' | 'N/A'
    'controller': str,            # 'pid_default' | 'pid_high_gain'
    'dt':         float,          # dataset timestep ≈ 0.0303 s (33 Hz)
    'obs':        (T, 9) float32, # [p_des | p | v]
    'actions':    (T-1,3) float32,# Δp_des (forward diff of targets)
    'targets':    (T, 3) float32, # absolute p_des (with noise offset)
    'q':          (T, 4) float32, # actual quaternion [w,x,y,z]
    'obstacles':  list[dict],     # geometry metadata (type, center, radius/extents)
    'metadata': {
        'start_pos':        list[float],
        'total_time':       float,
        'dt_physics':       float,
        'contact_fraction': float,
        'controller_gains': str,
        'noise_sigma':      float,
    }
}
```

---

## 10. Collection Orchestrator

**File:** `collect.py:main()`

```
parse args
    → scene, n_trials, seed, gain_variant, noise_sigma, reject_limit

for i in range(n_trials):
    homotopy = homotopy_pool[i % len(pool)]   # round-robin balance
    trial_seed = base_seed + i

    rollout = run_trial(scene, homotopy, gain_variant, seed=trial_seed)

    if rejected:
        rejected += 1
        reject_counter[reason] += 1
        if rejection_rate > reject_limit after 20 trials: ABORT

    ep_id = build_episode_id(scene, homotopy, gain_variant, trial_seed)
    episode = rollout_to_episode(rollout, ep_id, noise_sigma, rng=trial_seed+99991)
    save_episode(episode, out_dir/homotopy_safe/)

write run_summary.json
```

**Round-robin homotopy:** `homotopy_pool[i % len(pool)]` guarantees that the first `k`
trials contain exactly `floor(k / num_classes)` of each homotopy, for any prefix. This
keeps counts balanced as the run progresses.

**Two separate RNGs per episode:**
- `rng = np.random.default_rng(trial_seed)` inside `run_trial()` — controls trajectory
  geometry (altitude, duration, jitter)
- `rng = np.random.default_rng(trial_seed + 99991)` inside `collect.py` — controls the
  noise offset added to targets. The large offset avoids correlation between the two
  streams.

**Progress logging:** prints every 50 saved episodes with rate and ETA.

**`run_summary.json` contents:**
```json
{
    "scene": "pillars",
    "saved": 475,
    "rejected": 25,
    "rejection_rate": 0.0500,
    "reject_histogram": {"contact": 7, "floor": 18},
    "accepted_clip_mean": 0.132,
    "accepted_clip_max": 0.443,
    "elapsed_s": 1234.5,
    "sec_per_episode": 2.6
}
```

---

## 11. Verification Tools

### `verify_blends.py`

Pure numpy (no MuJoCo). Samples the nominal trajectory densely (`dt = 0.002 s`) and
checks three properties for every homotopy × duration combination:

1. **No stop-and-go:** interior speed `v_min` (excluding first/last 5% of time) > 0.05 m/s

2. **Finite-difference consistency:** `‖FD_velocity − analytic_velocity‖_max < 0.05 m/s`.
   This verifies there are no kinks or jumps at element boundaries. If the path were
   discontinuous, the PID feedforward `a_des` would be wrong at those points.

3. **Geometric clearance:** minimum COM-to-pillar-axis distance ≥ 0.43 m (for pillars)
   or minimum COM-to-wall-surface distance ≥ 0.31 m (for s_curve).

Run before every E4 collection campaign whenever `trajectories.py` is changed. All 28
checks (4 homotopies × 2 durations × pillars, plus 3 jitters × 2 durations × s_curve)
pass at `BLEND_RADIUS = 0.45`.

### `stats_validator.py`

Post-collection check. Loads all pickles in a directory and computes:
- Speed distribution (mean, median, p5, p95, max) from `obs[:, 6:9]` (velocity columns)
- Episode length distribution
- Action norm `‖Δp_des‖` distribution (mean, p95)
- Homotopy counts

Checks:
- Speed mean ∈ [0.15, 0.80] m/s (expected: 0.30–0.50)
- Action mean ∈ [0.002, 0.05] m/step (expected: 0.009–0.015 at 0.3–0.5 m/s, 33 Hz)

Writes `dataset_stats.json` next to the pickles.

---

## 12. End-to-End Data Flow

```
Slurm sbatch (collect.sh)
    → collect.py --scene pillars --n-trials 500 --seed 0
        → for each trial i:
            generator.run_trial(
                scene='pillars',
                homotopy='(L,L,L)' | '(L,R,L)' | '(R,L,R)' | '(R,R,R)',
                gain_variant='pid_default' | 'pid_high_gain',
                seed=i
            )
                → rng = default_rng(i)
                → z = rng.uniform(0.90, 1.30)
                → dur = rng.uniform(10.0, 16.0)
                → traj_fn = pillar_path(['L','R','L'], altitude=z, duration=dur)
                    → blended_path(8 waypoints, BLEND_RADIUS=0.45, T=dur, yaw=0)
                        → builds 14 path elements (7 straights + 6 fillets + 1 straight)
                        → total arc length L
                        → returns closure: traj(t) → (p, v, a, yaw)
                → model = MjModel.from_xml(scene_pillars.xml)
                → data = MjData(model)
                → pid = CascadedPID(model) with scaled gains
                → for k in range(int(round(dur/dt))):
                    → p_des, v_des, a_des = traj(k*dt)
                    → u = pid.compute(state, reference)
                    → mj_step(model, data)
                    → record step
                → if contact_frac > 0.001 or min_z < 0.50: return reject dict
                → return rollout dict (steps, metadata)

            dataset_writer.rollout_to_episode(rollout, ep_id='pillars_L_R_L_pid_default_0000042')
                → downsample steps[::3] (100 Hz → 33 Hz)
                → obs = stack [p_des | p | v]  shape (T, 9)
                → q   = stack quaternions       shape (T, 4)
                → targets = p_des column        shape (T, 3)
                → targets += rng.normal(0, 0.02, (1, 3))   (constant noise offset)
                → actions = diff(targets)        shape (T-1, 3)
                → return episode dict

            save_episode(episode, 'logs/uav_expert_data/pillars/L_R_L/')
                → pickle.dump to logs/uav_expert_data/pillars/L_R_L/pillars_L_R_L_pid_default_0000042.pkl

        → write run_summary.json

Slurm sbatch (stats_validator.sh)
    → stats_validator.py --data-dir logs/uav_expert_data/pillars
        → load all .pkl files
        → compute speed, action norm, homotopy counts
        → print pass/fail vs targets
        → write dataset_stats.json
```

---

## 13. Key Design Decisions

### Why PID + reference trajectory instead of RL?

A PID controller following a known reference trajectory is fully **deterministic and
verifiable**. The trajectory math can be checked analytically (`verify_blends.py`). The
rejection criterion is a hard geometric gate, not a reward signal. The resulting dataset
has clean structure — smooth velocities, balanced homotopies, bounded action norms —
which is easier for FM to learn from than noisy RL rollouts.

### Why cosine speed profile for `traverse_line`?

The cosine blend `s(t) = ½(1 − cos(πt/T))` has three desirable properties:
1. `v(0) = v(T) = 0` — zero-velocity boundary conditions, matching the hovering state
2. Monotonically increasing then decreasing — no oscillation in speed
3. Analytic derivatives `v, a` — enables feedforward in the PID without finite differences

### Why `blended_path` instead of chained `traverse_line`?

Chained `traverse_line` enforces `v=0` at every waypoint (the "stop-and-go" bug in U8).
At 33 Hz, a `v=0` interior waypoint creates a step in `Δp_des = diff(targets)`. The
model must learn both "accelerate through corridor" and "brake at waypoint" behaviors, but
those waypoints are artefacts of the route parameterisation — they are not physically
meaningful to the avoidance task. `blended_path` removes them: the only `v=0` points are
the genuine episode start and end.

### Why the large constant noise offset instead of per-step noise?

Training data augmentation via trajectory offset serves two purposes:
1. Thicken the manifold — small shifts in the commanded trajectory teach the model that
   nearby positions are also valid
2. Prevent overfitting to the exact trajectory geometry

Per-step noise adds σ√2 randomness to every action, dwarfing the signal (σ = 0.02 m,
signal ≈ 0.012 m/step). A constant offset has zero effect on actions and only shifts
the absolute position — exactly the right augmentation for position-conditioned FM.

### Why separate `q` field (not in obs)?

The policy (`obs → action`) operates on position and velocity only — `[p_des | p | v]`.
Attitude (quaternion `q`) is not a policy input because the position controller abstracts
it away: the policy outputs `Δp_des` (position increment), and the real drone's low-level
controller handles attitude tracking.

The `q` field exists only for **visual rendering** during human inspection (GIFs, camera
images). Storing it in the pickle avoids having to re-simulate to get the attitude for
rendering.

### Why scene-specific contact thresholds?

- `pillars 0.001` — pillar clearance (0.20 m from rotor edge) is large enough that any
  contact is a definitive trajectory bug. There is no "brief graze" excuse.
- `s_curve 0.08` — the wall end-faces at `x = ±0.5` are open edges that the drone must
  fly very close to. Even on a perfect trajectory, the rotor ellipsoid briefly touches the
  edge face on the way into the gap. This is a known geometric constraint, not a failure.
- `corridor 0.02` — L/R homotopies at `y = ±0.12` have only 2 cm clearance from the
  wall surface. Brief contact on PID jitter is expected and was the reason the original
  `0.01` threshold caused 38.6% rejection in U2/Fix_1.
