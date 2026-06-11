# Gen11 E4 U8 — Stop-and-Go: Why It Exists, Whether It Can Be Eliminated

**Date:** 2026-06-11
**Status:** Analysis only — no code changes
**Question:** The expert trajectories visibly stop at every waypoint (decelerate → pause → re-accelerate). Is this an E4 problem? Can it be replaced with smooth, continuous-momentum UAV flight?

---

## 1. Verdict: yes, it is E4 scope — user hypothesis confirmed

The stop-and-go is **not** a controller artifact, **not** a physics/simulation issue, and **not**
introduced by E5 rendering. E5 GIFs (both trajectory and physics) faithfully reproduce what E4
recorded. The behaviour is baked into the **reference trajectory generator** that E4 feeds to the
PID controller. Two independent sources, both in E4-scope code:

### Source 1 — `traverse_line` cosine velocity profile (every scene)

`uav_env_test/trajectories.py:65-89` (base factory, re-exported and composed by
`uav_expert_data_collect/trajectories.py`):

```python
s      = 0.5 * (1 − cos(π·t/T))          # position blend
s_dot  = 0.5 · π/T · sin(π·t/T)          # velocity: sin profile
```

By construction **v(0) = v(T) = 0 and a(0) = a(T) = 0 for every segment**. Every multi-segment
path is a chain of these primitives, so the commanded velocity drops to exactly zero at every
waypoint joint:

| Scene | Segments | Full stops per episode (excluding natural start/end) |
|---|---|---|
| empty | 1 | 0 — no internal stops |
| corridor | 1 | 0 — no internal stops |
| pillars (`pillar_path`) | 7 | **6** (at x = −2.5, −1.5, −0.5, +0.5, +1.5, +2.5) |
| s_curve (`s_curve_scene_path`) | 5 legs + 2 hovers | **4 stops + 2 explicit pauses** |

The docstring of `s_curve_path` states this openly: *"zero velocity at every waypoint — fine for
low-speed env demos."* It was a deliberate primitive choice, not a bug.

### Source 2 — Explicit 1.0 s hover pauses (s_curve only)

`uav_expert_data_collect/trajectories.py:159` (`T_HOVER = 1.0`, phases Hov 1 / Hov 2). Added in
U7 to let the PID stabilise attitude before the 90° lateral gap crossing. These are full
stationary holds — the most visible part of the stop-and-go.

---

## 2. Why it was designed this way (it is not an accident)

The per-segment cosine primitive buys three properties that the entire E4 safety story is built on:

1. **Analytically verifiable clearance.** The U3/U7 collision analyses (pillar margin = rotor
   reach 0.31 m + 8 cm; Z-route corner clearances 0.50–0.55 m) are computed on **straight-line
   nominal paths**. Straight segments parallel to walls at every pinch point mean PID tracking
   lag is *along-path* and "cannot reduce wall clearance" (`trajectories.py:140-141`). A curved
   path breaks this argument — lag acquires a lateral component toward obstacles.

2. **Zero-velocity corners are trivially trackable.** The Z-route has 90° direction changes. A
   sharp corner taken at speed is the hardest thing a cascaded PID can be asked to track; taken
   at v=0 it is trivial. The hover pauses exist for exactly this reason.

3. **Rejection-rate pressure.** E4's history is a war against episode rejection (U2/Fix_1: 71.4%
   rejection ABORT; U7 finally achieving 0% on s_curve). Every smoothness concession was traded
   away for rejection-free collection. The expert is conservative *by requirement*.

So: the trajectories are ugly because the generator optimises for **provable safety and
collection yield**, and aesthetics was never an objective. The result is a robotics-textbook
point-to-point primitive chain, not the continuous-momentum arcs a human pilot or optimal
planner would fly.

### Why it matters downstream

The dataset records (p, v, a) at every step. The FM-PCC policy will learn stop-and-go as the
*correct* expert mode — it is consistent, low-entropy behaviour, exactly the kind imitation
learning reproduces faithfully. It will not average out. If the deployed policy must fly
smoothly, the data must be smooth.

---

## 3. Can it be eliminated? Yes — geometric feasibility check per scene

The key question is whether smooth (non-zero-velocity) waypoint transitions can keep the
clearances that U3/U7 fought for. Checked against the actual scene geometry:

### empty / corridor — already smooth
Single segment each; the only v=0 points are the natural episode start/end. Nothing to do.

### pillars — feasible with corner blends
The 6 internal waypoints sit at x ∈ {−2.5, −1.5, −0.5, +0.5, +1.5, +2.5} — all chosen (Fix_1)
to be **between** pillar pairs with ≥ 0.5 m buffer. A circular corner blend of radius r deviates
from the sharp corner by at most r·(√2−1) ≈ 0.41·r. With r = 0.3 m the path moves ≤ 0.13 m off
the corner point, in open space away from pillars. The straight portions near the pillars (where
the 8 cm minimum clearance lives) are untouched. **Feasible; clearance re-verification needed
only at the 6 blend regions.**

### s_curve — feasible, surprisingly
This is the scene where the diagonal was *geometrically infeasible* (passes 0.291 m from wall
corners < 0.31 m rotor reach), which is why intuition says "smoothing is impossible here." But
the Z-route corners are at (0, ∓0.8) — and the critical wall corners are at A = (−0.5, −0.25)
and B = (+0.5, +0.25):

```
distance(Z-corner (0,−0.8) → wall corner A) = √(0.5² + 0.55²) ≈ 0.74 m
```

A blend radius of 0.3 m at the Z corners keeps the path ≥ 0.74 − 0.13 ≈ **0.61 m** from the wall
corners — still double the rotor reach. The infeasible region is the *diagonal across the gap*,
not the neighbourhood of the Z corners. **A rounded-Z at constant speed is geometrically fine.**

The hover pauses can then go too: a blended arc with continuous velocity is *easier* for the PID
to track than the sharp corner the pauses were protecting — the pause existed to make a hard
corner trivial; the blend removes the hard corner instead.

---

## 4. How to eliminate it — options ranked

### Option A — Corner blends + global speed profile (recommended)
Keep the exact U7 waypoint skeletons (they encode all the safety analysis). Replace the
per-segment cosine chain with:
- circular (or clothoid) blends of radius ~0.3 m at every interior waypoint,
- one **global** cosine/trapezoid speed profile over total arc length, so v > 0 throughout and
  v = 0 only at episode start/end,
- drop `T_HOVER` in `s_curve_scene_path`.

Cost: one new primitive (`blended_path(waypoints, radius, duration)`) in
`uav_env_test/trajectories.py` + swapping the chain constructors in E4 `trajectories.py`.
The analytical clearance argument survives almost intact: straight portions unchanged, blends
verified per Section 3. Re-run the rejection-rate gate (<30%) as the empirical check.

### Option B — Minimum-snap polynomial trajectories (the "elegant" answer)
The textbook quadrotor solution (Mellinger & Kumar): piecewise polynomials through waypoints
minimising snap, with continuity of p, v, a, jerk at joints. This is what produces the flowing,
cinematic UAV trajectories the user is imagining. Cost: a QP solve per episode, **and** corridor
clearance becomes a numerical-verification problem (the optimiser will cut corners unless
constrained — needs corridor constraints or post-hoc clearance sampling). Significantly more
machinery for a result Option A approximates at 10% of the effort.

### Option C — Keep stop-and-go (do nothing)
Defensible if downstream only needs obstacle avoidance, not smooth flight (per E5 CLOSURE.md).
But the user has now judged it "highly counter-intuitive and ugly" — so the requirement has
effectively changed, and C is off the table for any demo-facing use.

### Non-options
- Tuning PID gains, segment durations, or `T_HOVER` alone — cannot help; v=0 at joints is a
  property of the trajectory *primitive*, not its parameters.
- Smoothing at E5/rendering time — would falsify the data; the policy still learns from the
  recorded states.

---

## 5. Consequences of fixing it (be honest about the bill)

1. **Full E4 recollection.** All 1,952 episodes encode stop-and-go; a new generator means a new
   dataset and re-running the <30% rejection gates per scene.
2. **Rejection-rate risk.** Continuous-speed corners increase tracking error exactly where U7
   bought 0% rejection with pauses. Section 3 argues the margins hold (0.61 m ≫ 0.31 m), but the
   empirical gate is the real test — budget for one tuning iteration on blend radius/speed.
3. **Old + new datasets are behaviourally incompatible.** Don't mix them in one training run —
   the policy would learn a bimodal stop-vs-flow distribution at waypoints.

---

## 6. Recommendation

If smooth flight is now a requirement: **Option A as a new E4 U9 / Epoch 6 work item** —
`blended_path` primitive, drop hovers, keep U7 waypoint skeletons, re-verify the 8 blend regions
analytically (same style as the U7 Z-route proof), then recollect and re-gate. The s_curve
geometry does **not** block it; the diagonal infeasibility and corner-blend feasibility are
different questions with different answers.

This matches the open question already flagged in `Epoch5_visual_and_validation/CLOSURE.md`
("smooth trajectory redesign — deferred to Epoch 6+ if stop-and-go proves limiting"). The user's
judgement that it looks bad is that condition firing.
