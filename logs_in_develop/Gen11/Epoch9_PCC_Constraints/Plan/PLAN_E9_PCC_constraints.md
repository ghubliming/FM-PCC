# Epoch 9 PLAN — Bring back PCC: real constraint geometry (halfspace 2D, obstacle 3D)

**Date:** 2026-07-04
**Status:** PLAN only — concepts + implementation map, no code.
**Fills:** the empty placeholders E7 deliberately left (`PLAN.md §5` of
`../../Epoch7_fm_pcc_FULL_PCC_MPC/`): `workspace_bounds` / `halfspace_constraints` /
`obstacle_constraints` in `config/uav_projection.yaml`.
**Template:** the Gen7 visual-aligning bone — `fm_visual_aligning_test/eval_fm_visual_aligning.py`
(`setup_dpcc_projector`, `formulate_halfspace_constraints`, per-geometry config blocks,
exec-time violation metrics) — the same DPCC lineage the whole repo follows.

---

## §0 — TL;DR

E7 restored the full PCC/MPC bone (projector + candidate fan + selection + constraint
metrics) but ran **dynamics-only**; the spatial constraint slots exist and are gated off.
E9 designs the **real geometry** for those slots, per scene, following DPCC principles
exactly as the Gen7 visual-aligning eval does — **one full-stack constraint set per scene,
activation selected in the yaml**:

| Scene | `constraint_types` (set in yaml) | Geometry it carries |
|---|---|---|
| `empty` | **`[]` — NO constraints applied** (explicit baseline, marked) | none |
| `corridor` | `['dynamics','bounds','halfspace','obstacles']` | box bounds + 2 wall halfspaces + ball(s) |
| `pillars` | `['dynamics','bounds','halfspace','obstacles']` | box bounds + halfspace + 6 cylinder/ball obstacles |
| `s_curve` | `['dynamics','bounds','halfspace','obstacles']` | box bounds + per-segment switched halfspaces + ball(s) |

**`empty` is deliberately unconstrained** — it is the raw-FM result denominator and the
plumbing/regression baseline, mirroring the visual-aligning `no_constraint` entry
(`constraint_types: []`). We do **not** apply any projector geometry on `empty`; it is
marked as such in the config and its `results.json`.

Every **other** scene carries its **own complete geometry** — a box (bounds and/or
box-shaped exclusion), a ball/sphere (`sphere_outside`), and halfspace planes — and
**which of those are active is chosen in the yaml** via each scene entry's
`constraint_types` list plus the `active_geo_variants` selector, exactly the
dpcc / visual-aligning-FMPCC mechanism (§4).

**No new solver machinery.** The engine (`flow_matcher_v3_uav/sampling/projection.py`:
`SafetyConstraints` for lb/ub/ineq, `ObstacleConstraints` for quadratic sphere terms,
`DynamicConstraints` for the Euler rows) already supports every constraint form needed —
it is byte-identical to the FMv3ODE/DPCC engine. E9 is **geometry design + config wiring +
one helper robustness fix + exec-time violation metrics**.

**Obstacle primitive reality (box vs ball):** the engine's `ObstacleConstraints` provides
the quadratic **ball** primitive only — `sphere_outside` on `['x','y']` (a vertical
**cylinder**, the exact cross-section of a full-height pillar) or on `['x','y','z']` (a
true **3D sphere**). There is **no `box_outside` primitive**. So a "3D box":
- as **containment** (arena / altitude floor+ceiling) → `bounds` (lb/ub rows) — supported;
- as an **exclusion** (a wall you stay outside of) → its DPCC-native form is **halfspace
  planes** (each box face is a halfspace), because a convex box *exclusion* is non-convex
  from outside and is not a single primitive. Walls are therefore encoded as halfspaces
  (+ optional bounding ball for a cheap volumetric guard). If a literal box-exclusion
  primitive is ever wanted, it is a small engine addition (`box_outside` = per-face
  halfspace union with active-face selection) — **out of scope for E9**, noted for the record.

**The motivating failure:** E6-U2 closed with the pure-FM policy at **0 % success on
`pillars`**. Obstacle projection + candidate selection is exactly the DPCC mechanism that
should rescue it — E9 is the direct test.

---

## §1 — DPCC principles we commit to (and where each lives)

These are the load-bearing principles from the DPCC paper / `ralfroemer99/dpcc`, already
embodied in the avoiding-task and Gen7 visual-aligning code. E9 must not deviate:

1. **Constraints are linear/quadratic functions of the trajectory tensor** — rows over the
   flattened `(horizon × transition_dim)` plan, normalized-space handled internally by
   `ProjectionNormalizer`. We only supply *physical* coordinates; never hand-normalize.
2. **Projection during sampling, not only after** — `diffusion_timestep_threshold = 0.5`
   (project in the last 50 % of FM steps); the `post_processing` variant (threshold 0)
   remains the ablation. Unchanged from E7.
3. **Constraint tightening** for `-tightened` variants: spatial constraints shrunk by
   `enlarge_constraints` so the *executed* (tracked) path, which lags the plan, still
   clears the true geometry. In E7 the margin was computed but inert (no spatial
   constraints); E9 makes it real — and on the UAV it must additionally absorb the
   **p ↔ p_des tracking gap** (§4.3), which the arm never had.
4. **Candidate fan + selection**: `dpcc-t` (temporal consistency) / `dpcc-c` (minimum
   projection cost) / `dpcc-r` (random) over `batch_size` candidates. Already live since E7.
5. **`model_free` becomes meaningful**: spatial constraints *without* dynamics rows. In E7
   it was a documented no-op; E9 is the first epoch where the DPCC Table-1 variant grid is
   fully populated for the UAV.
6. **Violation accounting at execution time**, not just plan time: per-step signed margins
   of the *flown* trajectory against the same geometry (Gen7's
   `exec_halfspace_viol_count` / `exec_obstacle_viol_count` / `exec_max_*` pattern,
   `eval_fm_visual_aligning.py` ~L485–590). This is what makes
   `n_violations` / `collision_free` / `success_and_constraints` non-trivial.

---

## §2 — Where constraints bind on the UAV tensor (DPCC-faithful: geo on `p` only)

### §2.0 What the ORIGINAL DPCC actually constrains (verified, settles the question)

Read directly from the canonical avoiding-task lineage
(`config/projection_eval.yaml` + `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`).
Avoiding transition `[action(2) | p_des(2) | p(2)]`, index map
`{vx:0, vy:1, x_des:2, y_des:3, x:4, y:5}`:

| Constraint family | Binds to | Dims |
|---|---|---|
| **halfspace** | **actual position `p` ONLY** (`act_obs_indices['x','y']`) | 4,5 |
| **obstacles** | **actual position `p` ONLY** (`dimensions:['x','y']`) | 4,5 |
| **bounds** (in avoiding) | the **action** (velocity `vx,vy`) — *not* position | 0,1 |
| **dynamics** | **both** `p_des` **and** `p`, each coupled to the action via `deriv` | (2,3)&(4,5)←(0,1) |

**So the answer to "did DPCC apply the GEO constraint to p_des and p and the action?" is
NO.** The **geometric** constraints (halfspace, obstacles) bind to **actual position `p`
only**. Only the **dynamics** rows touch `p_des` (and they touch `p` too) — and they touch
the action, but as the Euler *link*, not as a geometric constraint on the action. Bounds in
avoiding happen to sit on the action because there they encode a *velocity limit*, which is
a different use of the `bounds` family, not a spatial box on position.

**Therefore the Gen7 visual-aligning code is CORRECT, not wrong** — its `_DIM` maps
halfspace/obstacle to `p` (dims 6,7,8) only, exactly matching DPCC. E9 does the **same**.
(This reverts a wrong call in an earlier draft of this plan that proposed binding spatial
rows to both `p` and `p_des`; that would have deviated from DPCC. Dropped.)

**Why `p`-only is sufficient (not a corner-cutting hole):** `p_des` and `p` are already
rigidly coupled by the DC_FIX dynamics rows (both are the running integral of the same
action from the same start). Constraining `p` therefore shapes `p_des` *through* the
dynamics coupling — the projector cannot push `p` outside an obstacle while leaving `p_des`
free, because the `deriv` rows bind them together. This is exactly why DPCC never needed a
second geometric copy on `p_des`, and neither do we. Any residual `p`↔`p_des` tracking slack
on the drone is absorbed by the **tightening margin** (§4.3), which is the DPCC-native knob
for it — not by adding non-canonical constraint rows.

### §2.1 UAV tensor layout

- **E7 12D** `[act(0:3) | p_des(3:6) | p(6:9) | v(9:12)]` (PID path)
- **E8 9D** `[act(0:3) | p_des(3:6) | p(6:9)]` (`pos_only`, MJPC path)

Spatial constraint dims `x,y,z → 6,7,8` (actual position `p`) — **identical indices in
both layouts**, so one geometry config serves both tensors and both controllers with zero
branching. The plan freezes this as an invariant (assert `traj_dim ∈ {9, 12}` and the `p`
slice at 6:9 when building spatial constraints).

**Velocity STATE (12D dims 9,10,11, the measured `v`) is IGNORED — no constraint projection
on it, ever.** It is neither a geometric-constraint target nor a dynamics-link target; the
projector leaves those columns untouched (padded ±inf in bounds, absent from every
halfspace/obstacle/deriv row). MJPC/PID stay pure trackers (E8 §4.3); all safety lives
upstream in the projector. *(Do not confuse this with the **action** dims 0,1,2 = Δp_des,
which DO get a magnitude bound — see §2.2. "velocity state `v`" ≠ "action Δp_des".)*

### §2.2 `bounds` is TWO orthogonal roles — the UAV needs BOTH

DPCC's `bounds` family was used for two *different* jobs across tasks, and they are **not**
substitutes (see the problem writeup
`../../../Gen7_FMPCC_Viusal_Aligning/Patch_Constraints/PROBLEM_bounds_velocity_vs_geo.md`):

| Role | Binds to | Purpose | avoiding | visual-aligning |
|---|---|---|---|---|
| **action-magnitude bound** | **action dims 0,1,2** (Δp_des) | keep the commanded step inside the dataset's normalized/trained range | ✅ (as `['vx','vy']`) | ❌ **lost** |
| **workspace box** | **actual `p` dims 6,7,8** | physical Cartesian safety envelope | ❌ | ✅ (as `workspace_bounds`) |

Visual-aligning **retired** the action bound when it added the workspace box ("Replaced"),
silently dropping the guard that kept sampled actions in range. **E9 (UAV) must carry BOTH**,
because they solve different failures and the UAV is exactly the plant where the missing one
bites (an unbounded commanded Δp_des destabilizes the second-order drone — cf. the E6
multi-mode explosion; a perfect-tracking arm rarely exposed it):

1. **Workspace box** — lb/ub on actual `p` (6,7,8) + altitude floor/ceiling. The wanted new
   behavior; per-scene arena envelope (§3).
2. **Action-magnitude bound** — lb/ub on the **action** dims (0,1,2), sized from the
   dataset's normalized Δp_des range / measured per-step step-size. Restores avoiding's
   dataset-normalization guard. Scene-independent (a property of the action space, not the
   arena), so one shared bound reused across every constrained scene.

Both are the same engine primitive (`formulate_bounds_constraints` → lb/ub rows via
`SafetyConstraints`), just on different dims — no solver change, only two entries instead of
one. Neither touches the velocity *state* (9,10,11), which stays ignored (§2.1).

**Sizing note (must measure, don't guess):** the action bound comes from the dataset's
normalized action range (the same range the normalizer already fits) and/or the measured
Δp_des step-size distribution; the workspace box comes from the scene XML arena extent. Both
are derived at impl time, not hard-coded here.

---

## §3 — Per-scene geometry design

Scene XMLs: `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_*.xml`.
All numbers below are read from the XMLs; the implementation must re-derive them from the
XML at design time (single source of truth), not trust this document.

**Structure:** one full-stack geo entry per scene (except `empty`), each carrying its
**own complete** bounds + halfspace + obstacle geometry. Activation is per-entry
`constraint_types` in the yaml (§4) — the same entry can be run at any subset of its
constraints by editing that one list, without touching geometry, exactly like the
visual-aligning `combined_*` slots.

### §3.1 Common: inflation radius

Every constraint surface is offset by `r_drone + margin_base`:
- `r_drone`: the drone's bounding radius in xy — **measure from the quadrotor body geoms
  in the XML** (rotor-tip to center), do not guess. Record the measured value in the
  changelog.
- `margin_base`: small static clearance (order 0.05 m) applied to *all* variants.
- `enlarge_constraints` (the `-tightened` extra, config: 0.025) is applied **on top**,
  only for `-tightened` variants, exactly as the existing `setup_dpcc_projector` already
  does (`radius + enlarge`, halfspace shift by `enlarge`).

So: plain variants plan against geometry ⊕ (r_drone + margin_base); tightened variants
against geometry ⊕ (r_drone + margin_base + enlarge). Execution-time violation metrics
check against geometry ⊕ r_drone only (physical collision truth, not the planning margin).

### §3.2 `empty` — NO constraints (explicit baseline, not applied)

`constraint_types: []`. The projector is a **no-op** on `empty` — raw FM output, no
geometric enforcement, exactly the visual-aligning `no_constraint` entry. This is the
result **denominator** and the plumbing/regression baseline. It is **marked** in the
config comment and stamped into `results.json` (`constraint_types: []`) so no reader
mistakes it for a failed-projection run. We do **not** design bounds/halfspace/obstacles
for `empty`.

The shared **altitude box** (z-floor > 0 off the ground plane, z-ceiling) is defined once
and reused by the three constrained scenes below — but it is **not** applied to `empty`.

### §3.3 `corridor` — full stack: box bounds + wall halfspaces + optional ball

Walls: boxes at y = ±0.5, half-thickness 0.05, spanning x ∈ [−2, 2]. Inner faces at
y = ∓0.45. Its own complete geometry:

- **bounds (box):** arena x∈[−2,2] envelope + altitude floor/ceiling — the containment box.
- **halfspace (the two walls, as box faces):** two constraints on (x, y):
  - feasible side `y ≥ −0.45 + inflation` (line along x, side `above`)
  - feasible side `y ≤ +0.45 − inflation` (line along x, side `below`)
  Config format is the existing `formulate_halfspace_constraints` triple
  `[point1, point2, side]` — same as the avoiding-task triangles. (These *are* the box
  walls in DPCC-native form; see §0 on why a box exclusion is halfspaces, not a `box_outside`.)
- **obstacle (ball, optional):** a `sphere_outside` bounding ball may be placed at the
  corridor mouth/exit as a volumetric guard, or left empty for the pure-corridor case —
  presence is toggled purely by keeping/removing the `obstacles` key from this scene's
  `constraint_types` in the yaml.

**⚠ Required robustness fix (the one real code-level fix in E9):**
`flow_matcher_v3_uav/utils/constraints_helpers.py::formulate_halfspace_constraints`
computes the boundary normal as `[-1, 1/m]` from the slope `m`. Corridor walls run along
x ⇒ `m = 0` ⇒ `1/m` divides by zero (and vertical walls ⇒ `m = inf` breaks the slope
itself). The fix is the standard perpendicular construction the Gen7 *plotting* code
already uses (`_hs_xy_draw`: `dir = p2 − p1`, `n = (−dy, dx)` sign-picked by `side`), and
an intercept form `n·x ≥ n·p_shifted` instead of slope-intercept. This must reduce to
byte-identical `(C_row, d)` for the sloped avoiding-task inputs (add a numeric
equivalence check against the old function on those inputs before switching) — the helper
is shared with the arm evals.

### §3.4 `pillars` — full stack: box bounds + halfspace + 6 cylinder/ball obstacles

6 pillars: rows at y = ±0.6, x ∈ {−2, 0, 2}, radius 0.12, spanning z ∈ [0, 2] — they
cover the whole flight altitude band. Its own complete geometry:

- **bounds (box):** arena envelope + altitude floor/ceiling.
- **obstacles (the pillars, as balls/cylinders):** the **exact** representation of a
  full-height vertical cylinder is `sphere_outside` on dims `[x, y]` only — the quadratic
  constraint reads `(x−cx)² + (y−cy)² ≥ r²` at every knot, altitude-independent:
  - 6 entries `{type: sphere_outside, dimensions: [x, y], center: [cx, cy], radius: 0.12 + inflation}`.
  This is *not* an approximation — for pillars taller than the flight band the 2D disc is
  the true cross-section. A **true 3D ball** (`sphere_outside` on `['x','y','z']`) is the
  documented pattern for any future floating/finite-height obstacle (same engine call,
  three dims), and is exactly the "3D ball" primitive available if a scene ever needs it.
- **halfspace:** an outer envelope halfspace (e.g. keep the drone on the intended side of
  the pillar field, or an entry-lane wall) so `pillars` also exercises the halfspace path
  in combination — toggled from the yaml `constraint_types`.

Engine precedent: exactly the avoiding-task obstacle pattern
(`visual_aligning_eval.yaml` obstacle entries; `ObstacleConstraints.build_matrices`
quadratic P/q/v form; SLSQP handles the non-convex "outside" constraint from a warm
FM-sample start).

### §3.5 `s_curve` — full stack; the honest hard case: non-convex, needs constraint switching

Its own complete geometry, same three families: **bounds (box)** arena + altitude
floor/ceiling; **obstacles (ball)** optional bounding balls at the inside corners of the
crossover as a volumetric backstop; and **halfspace** walls — but the halfspace walls need
per-segment **activation** (below), and *that activation is set in the yaml* (§4), which is
exactly the "when to activate is set in yaml" mechanism.

Walls: segment 1 at y ∈ {−1.3, −0.3} spanning x ∈ [−3, −0.5]; segment 2 at
y ∈ {0.3, 1.3} spanning x ∈ [0.5, 3]. The feasible set (lower channel → crossover →
upper channel) is **non-convex**, and DPCC halfspaces are *global* over the horizon: a
naive union of all 4 wall halfspaces has an **empty intersection** (y ≥ −0.25ish AND
y ≤ … contradictory across segments) — it would strangle every plan. This is a genuine
structural limit of the linear-constraint formulation, not a tuning issue.

**Chosen approach — per-replan active-set switching ("dynamic constraints"):**
at each MPC replan, select the active halfspaces from the drone's **current x** (which
channel/segment it is in), and constrain the plan with only that segment's two walls
(+ a lookahead rule near the crossover):

- x in segment-1 range → activate segment-1 walls only.
- x in the crossover gap `[−0.5, 0.5]` → activate no y-walls (or only the outermost
  envelope y ∈ [−1.3+i, 1.3−i]) — the gap is genuinely open.
- x in segment-2 range → segment-2 walls only.

Receding horizon makes this principled: each 8-step plan (short relative to segment
length at the dataset's step size) is locally inside one convex cell, and the replan
cadence re-selects before the cell changes. This is the same trick every corridor-MPC
uses (convex corridor decomposition), and the repo has prior art: the
`FIX_DYNAMIC_CONSTRAINTS` work (Gen7 + E8) already rebuilds constraint matrices
per-replan, so *rebuilding the constraint list each replan step is an already-paid cost*,
not new machinery. Implementation shape: the per-scene config stores wall segments
**with their x-validity interval**; a small `select_active_constraints(p_now, scene_geo)`
step in the eval loop filters the list before `setup_dpcc_projector` (or before a
lighter-weight `build_matrices` refresh) each replan.

**Rejected alternatives** (documented so they're not re-litigated): covering wall
segments with rows of `sphere_outside` discs (conservative, ~10+ quadratic constraints
per wall, chokes SLSQP and still leaks between discs); one global convex envelope
(empty/over-tight, see above); leaving s_curve unconstrained (gives up the hard case).

**Horizon-crossing caveat:** if a plan's horizon straddles the crossover, segment-1 walls
applied to the tail of a plan that should already be turning is *conservative* (delays
the turn), never unsafe. Accepted for v1; a per-knot active-set (different `C` rows for
different horizon knots — the engine's matrices are per-knot already, see
`build_matrices` row structure) is the documented v2 refinement if conservatism visibly
hurts s_curve success.

---

## §4 — Config & wiring (all additive; E7/E8 defaults untouched)

### §4.1 Per-scene geometry blocks — the visual-aligning / dpcc yaml pattern verbatim

Current `config/uav_projection.yaml` has *single global* placeholder keys — wrong shape
for per-scene geometry. Adopt the **exact** visual-aligning activation pattern
(`visual_aligning_eval.yaml`: `geo_constraint_variants` list + `active_geo_variants`
selector):

- **`geo_constraint_variants:`** — a list of named entries, **one per scene**. Each entry
  carries its **own complete** geometry: `constraint_types`, `workspace_bounds`,
  `halfspace_constraints`, `obstacle_constraints`. The entry name is the scene name.
  - `empty` → `constraint_types: []`, no geometry (the `no_constraint` analog, §3.2).
  - `corridor` / `pillars` / `s_curve` → full stack, each with its own box/ball/halfspace.
- **`active_geo_variants:`** — the list that selects which scene entries actually run,
  identical to the visual-aligning selector. **This is "when to activate, set in yaml":**
  which scenes run, and — via each entry's `constraint_types` — which of its bounds /
  halfspace / obstacle families are enforced, are both edited here, no code change. The
  `-tightened` sibling is auto-generated per active entry exactly as the visual-aligning
  geo loop already does.
- **Per-halfspace activation interval (s_curve):** each `halfspace_constraints` item may
  carry an optional `x_active: [lo, hi]` field (the drone-x range over which that wall is
  live — §3.5). Scenes without it build once, as today. This keeps the s_curve
  segment-switching declarative **in the yaml**, matching the user's "when to activate is
  set in yaml" requirement rather than hard-coding segment logic.

The eval resolves the active entry from the scene name at run start; the resolved geometry
is stamped into `results.json`/npz (Gen7 does this — `_gc` in the artifact writer) so
plots and `npz_analysis` draw exactly what was enforced. The legacy global keys stay as a
fallback (empty ⇒ scene entry wins), so nothing existing breaks and old configs still parse.

**`bounds` carries BOTH sub-roles (§2.2):** the `bounds` family in each constrained scene
supplies (a) the per-scene **workspace box** on `p` (`workspace_bounds`) **and** (b) a
**shared action-magnitude bound** on the action dims (`action_bounds`, scene-independent,
merged in from a top-level key so it is identical everywhere). Both are lb/ub rows; keeping
them as two named keys stops the "one replaces the other" mistake that dropped the action
guard in visual-aligning.

**Sketch (shape only — real numbers derived from the XMLs / dataset range at impl time):**
```yaml
active_geo_variants: [empty, corridor, pillars, s_curve]   # ← select what runs, in yaml

# Shared action-magnitude bound (dims 0,1,2 = Δp_des) — restores avoiding's dataset guard.
# Scene-independent; merged into every constrained scene's 'bounds' family. Size from the
# dataset's normalized action range / measured Δp_des step-size (§2.2). NOT on velocity state.
action_bounds: {lb: [...], ub: [...]}      # 3 action dims

geo_constraint_variants:
  - name: empty                     # baseline — NO constraints applied (marked)
    constraint_types: []            # no workspace box AND no action bound — raw FM

  - name: corridor                  # own full stack: box + action-bound + walls(+ball)
    constraint_types: ['dynamics', 'bounds', 'halfspace', 'obstacles']
    workspace_bounds: {lb: [...], ub: [...]}   # box on p (6,7,8)
    # action_bounds inherited from the shared top-level key (on dims 0,1,2)
    halfspace_constraints:
      - [[...],[...], 'above']
      - [[...],[...], 'below']
    obstacle_constraints: []        # or a bounding ball; toggled by keeping 'obstacles' above

  - name: pillars                   # own full stack: box + action-bound + 6 balls + envelope halfspace
    constraint_types: ['dynamics', 'bounds', 'halfspace', 'obstacles']
    workspace_bounds: {lb: [...], ub: [...]}
    halfspace_constraints: [ ... ]
    obstacle_constraints:
      - {type: sphere_outside, dimensions: ['x','y'], center: [...], radius: ...}   # ×6

  - name: s_curve                   # own full stack: box + action-bound + switched walls + corner balls
    constraint_types: ['dynamics', 'bounds', 'halfspace', 'obstacles']
    workspace_bounds: {lb: [...], ub: [...]}
    halfspace_constraints:
      - {line: [[...],[...]], side: 'above', x_active: [-3.0, -0.5]}   # segment-1 wall
      - {line: [[...],[...]], side: 'below', x_active: [ 0.5,  3.0]}   # segment-2 wall
    obstacle_constraints: [ ... ]   # optional corner balls
```

### §4.2 Eval changes (`FM_v3_uav_test/eval_fm_uav.py`)

1. `load_pcc_config`: resolve `scene_constraints[scene]` → the flat keys
   `setup_dpcc_projector` already reads. Zero change to the projector-builder contract.
2. `setup_dpcc_projector`: un-gate stays automatic (blocks fire once `constraint_types`
   lists them). Spatial rows stay bound to **`p` (dims 6,7,8) only**, DPCC-faithful and
   byte-verbatim with the existing Gen7 `_DIM` mapping — **no new channel rows** (§2.0).
   The `bounds` block builds **both** sub-roles (§2.2): the `workspace_bounds` box on `p`
   **and** the shared `action_bounds` lb/ub on the action dims (0,1,2) — two lb/ub row-sets,
   same primitive, restoring the action guard visual-aligning dropped. The only other
   addition is the §3.1 inflation arithmetic (r_drone + margin_base folded into the stored
   geometry at config-resolve time) so the projector body stays verbatim-Gen7.
3. §3.5 active-set selection hook in the replan loop (s_curve only; scenes without
   `x_active` intervals build once, exactly as today — no per-replan rebuild cost where
   it isn't needed).
4. Exec-time violation metrics (§1.6): port the Gen7 margin-check block; wire its counts
   into the already-existing `n_violations` / `total_violations` / `collision_free`
   fields (E7 put the fields in; they're trivially clean today).
5. Constraint drawing on the plan/overview plots: port Gen7's geometry drawing
   (halfspace boundary + infeasible shading, obstacle discs, bounds box — 2D xy panel and
   the 3D panel's wall plane/sphere, `eval_fm_visual_aligning.py` UF-15/16 blocks).
   Non-negotiable for debuggability: every plot must show what the projector believed.

### §4.3 What the tightening margin must cover on the UAV (measure, don't guess)

The arm's `enlarge_constraints=0.025` assumed near-perfect tracking. The UAV's executed
`p` lags `p_des` by a controller-dependent amount. Before locking margins: pull the
**tracking-error distribution** (`track_err` is already logged per step since E7/E8) from
existing free-space runs, per controller (PID vs MJPC), and set
`margin_base ≥ p95(track_err)`. If PID and MJPC need visibly different margins, the
margin becomes a per-controller config key — flag at implementation time, not now.

---

## §5 — What is explicitly NOT in scope

- **No solver changes** (SLSQP stays; gradient variant stays as-is).
- **No MJPC cost changes** — MJPC/PID remain pure trackers; safety is upstream (E8 §4.3).
- **No retraining** — sampling/eval-side only; every existing checkpoint (12D and 9D)
  is evaluated as-is.
- **No visual constraint extraction** — geometry comes from the scene config (ground
  truth), not perception. That is the Gen7-visual direction and a separate epoch.
- **No dataset/collection changes.**

---

## §6 — Validation sequence (cluster; ordered by de-risking)

1. **Baseline — `empty`, no constraints (`constraint_types: []`):** raw FM, projector is a
   no-op. This is the result denominator; it must reproduce the E6/E7 free-space numbers
   exactly (it *is* the unconstrained path). Confirms the no-op / marking is honored and
   gives the denominator every other scene is measured against.
2. **The flagship — `pillars` (its full stack, obstacles the key family):** the E6-U2
   0 %-success scene. Compare `diffuser` (unprojected) vs `dpcc-r/c/t` at the same
   checkpoint/seeds. Hypothesis: projection + selection lifts success above 0 and drives
   exec obstacle violations to ~0. Also run `model_free` (spatial-only, dynamics off via
   its `constraint_types`) — first real data for that variant on the UAV.
3. **`corridor` (its full stack, halfspace the key family):** validates the fixed helper
   on m = 0 walls end-to-end. Check exec halfspace violations ≈ 0 and no wall contact.
   Because bounds + halfspace are both active, also confirms combined constraints compose.
4. **`s_curve` + switched halfspaces:** the stress test. Watch for (a) infeasible-QP
   warnings near the crossover (active-set logic bug), (b) over-conservative turning
   (horizon-crossing caveat §3.5 — triggers the per-knot v2).
5. **Action-bound check (§2.2):** confirm the restored `action_bounds` on dims 0,1,2 keeps
   commanded Δp_des inside the dataset range without stalling flight — compare a run with vs
   without it (esp. on multi-mode scenes, where the E6 explosion came from unbounded commands).
   The bound should shave the pathological large steps while leaving nominal flight untouched.
6. **Tightened sweep:** `-tightened` variants after §4.3 margins are set from measured
   tracking error; confirm tightened plans trade success-rate for lower violation counts
   in the DPCC-paper direction.
6. **Budget check:** SLSQP with 6 quadratic obstacle constraints × batch 4 × horizon 8 —
   log `planner_ms` via the existing real-time recorder; `diffusion_timestep_threshold`
   is the pressure valve if it blows the 33 Hz budget.

Success criteria for the epoch: (2) shows a strictly positive pillars success rate for at
least one dpcc variant with `collision_free_rate` ≈ 1; (1) shows zero regression; every
per-scene plot renders its geometry.

---

## §7 — Risks

- **SLSQP infeasibility at the start state** (drone already inside the inflated margin at
  replan time): the engine's `skip_initial_state=True` exists for exactly this; verify it
  is honored on the quadratic obstacle path, not only the linear rows.
- **Non-convex `sphere_outside` local minima:** SLSQP projects to the *nearest* feasible
  point — a plan threaded straight at a pillar can be pushed to either side, and the
  candidate fan + selection is DPCC's own answer (that's why `batch_size>1` matters here
  more than anywhere). Don't "fix" this inside the solver.
- **Helper regression on the arm:** the §3.3 normal-construction fix touches a shared
  helper — the numeric-equivalence check on avoiding-task inputs is mandatory before
  merge.
- **Margin mis-set → misleading science:** too small ⇒ violations blamed on DPCC; too
  large ⇒ success-rate collapse blamed on DPCC. Hence §4.3's measure-first rule.
- **s_curve switching cadence vs plan horizon:** if the horizon at dataset step-size is
  longer than the crossover gap, conservatism may dominate — the per-knot active-set v2
  is the pre-planned escape hatch, not a redesign.

---

## §8 — References (the "linked codes")

- **Primary template:** `fm_visual_aligning_test/eval_fm_visual_aligning.py` —
  `setup_dpcc_projector` (constraint blocks), UF-15/16 geometry drawing, exec violation
  metrics (~L485–590), per-geo config resolution.
- **Engine (reuse as-is):** `flow_matcher_v3_uav/sampling/projection.py`
  (`SafetyConstraints`, `DynamicConstraints`, `ObstacleConstraints`, `Projector`),
  `flow_matcher_v3_uav/sampling/policies.py` (selection).
- **Helper to fix:** `flow_matcher_v3_uav/utils/constraints_helpers.py::formulate_halfspace_constraints`.
- **Slots to fill:** `config/uav_projection.yaml`; gated blocks in
  `FM_v3_uav_test/eval_fm_uav.py::setup_dpcc_projector`.
- **Scene truth:** `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_{empty,corridor,pillars,s_curve}.xml`.
- **Config-shape precedent:** `config/visual_aligning_eval.yaml` (named geo entries),
  `config/projection_eval.yaml` (avoiding-task halfspace/obstacle format).
- **Prior epochs:** E7 `PLAN.md`/`CHANGELOG.md` (the bone + placeholders),
  E8 `PLAN_MJPC_Thrust_Control.md` (9D/MJPC path, tracker-stays-pure principle),
  E6-U3 homotopy finding + E6-U2 pillars 0 % (the motivation),
  Gen7/E8 `FIX_DYNAMIC_CONSTRAINTS` (per-replan matrix rebuild precedent for §3.5).
