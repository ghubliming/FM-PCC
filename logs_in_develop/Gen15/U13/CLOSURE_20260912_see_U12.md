# U13 closure → see U12

The `corridor_ball` investigation spans **U11 (r=0.35) · U12 (r=0.12) · U13 (r=0.05, r=0.01)** and is
closed as one line of work in a single document:

📄 [`../U12/CLOSURE_20260912_corridor_obstacle_investigation.md`](../U12/CLOSURE_20260912_corridor_obstacle_investigation.md)

**Headline:** on `corridor` the projector does not change the path — max \|Δy\| **0.0000 m**,
max \|Δz\| **0.0010 m** across 10 paired rollouts, against a 0.32 m detour requirement. The apparent
violation reduction (19.60 → 18.20) is a step-count artefact: the per-step rate is **0.0721 → 0.0716
(0.99×)**, unchanged.

The projector is **not** broken — on `pillars` K=5 (af) the same code cuts the violation *rate*
**0.3118 → 0.0584 (0.19×)** and `collision_free` 0.000 → 0.900.

**Ruled out across U11–U13:** ball radius (0.35 → 0.01, a 35× range), the ceiling (1.80 → 2.80),
obstacle placement (moved onto the measured flown path), and the action-magnitude cap
(`dpcc-t-bounds_free`). None moved the path.

**No further ball geometry should be built on `corridor`.** Successor: U14 — the obstacle moves to
the corridor's open approach region, where the wall halfspaces are `x_active`-inactive and y is
unbounded.
