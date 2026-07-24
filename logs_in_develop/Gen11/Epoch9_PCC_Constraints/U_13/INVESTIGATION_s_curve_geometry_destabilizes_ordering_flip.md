# U_13 — Investigation: the UAV **s_curve** projection-variant ordering FLIPS (why removing geometry, not adding it, is what almost reaches goal)

**Date:** 2026-07-10. Status: **INVESTIGATION ONLY — no code changed. No bug found in the
ablation logic.** Sibling to this folder's
`INVESTIGATION_geo_free_model_free_worse_than_diffuser.md` (the **corridor** case). This one is
about **s_curve**, and the ordering is *not the same* — on this scene the roles of geometry and
of the action bound invert. Read the corridor doc first; this doc leans on its §2 mechanism
(dynamics = the action↔state coupling) and only re-derives what changes.

**Trigger (user's words), from a visual/GIF quality pass on the s_curve rollouts (traj
smoothness + does-it-reach-goal):**

> `bounds_free` (direct crash) `<` `diffuser` (crash in the first few steps) `≈`
> `geo_free-model_free` `≈` `model_free` `<` `post_processing` (crash *after* passing the first
> corner) `≈` `dpcc-c/t/r`
>
> `<<` (almost works, reaches goal) `geo_free-bounds_free` `≈` `geo_free`
>
> "and the tightened variants are no better."

**Verdict up front:** This is the **corridor result turned inside-out, and it is expected, not a
bug.** On corridor the feasible set is *benign* (a wide, convex box + two parallel infinite
walls), so the full geometry stack genuinely helped and `dpcc-*`/`post_processing` beat
`diffuser`. On s_curve the feasible set is **non-convex, narrow (~24 cm bands), switched
per-segment, and the horizon is 2× longer (750 vs 360 steps)**. Projecting onto *that* geometry
is a destabilizing perturbation, so **every variant that keeps geometry ON eventually crashes**,
and the best results come from **dropping geometry entirely and keeping only the dynamics
constraint**, which re-anchors and smooths the long-horizon plan enough to almost reach goal.
The one principle from the corridor doc still holds — **dynamics is the load-bearing family** —
but here a *second* fact dominates: **geometry only helps when the feasible set is easy; on a
non-convex tight scene it hurts.** A corollary shows up too: the **action bound becomes
load-bearing for stability** (removing it, `bounds_free`, is now the *worst* variant, not a
no-op).

---

## 1. Why s_curve is a different animal (verified against `config/uav_projection.yaml:196-224`)

| property | corridor | **s_curve** |
|---|---|---|
| feasible set | convex-ish: box + 2 **infinite** parallel walls | **NON-convex**: 2 offset segments (seg1 band ≈ y∈[-0.92,-0.68], seg2 band ≈ y∈[0.68,0.92]) joined by a **crossover** |
| walls | always live | **per-segment**, switched on the drone's current x via `x_active` (`eval_fm_uav.py:732-740`) |
| band width (w/ margin 0.33) | wide | **~24 cm**, plus a ~24 cm **corner gate** at the crossover |
| horizon | 360 steps (dur 6–10 s) | **750 steps (dur 16–22 s)** — `config/uav.py:44-51` |
| maneuver | fly straight down a lane | **cross from the y≈−0.8 lane to the y≈+0.8 lane through a narrow corner** |

The non-convexity is real geometry, not a config artifact: `halfspace_constraints` carry
`x_active` ranges (`config/uav_projection.yaml:200-203`) and the projector **includes a wall
only while `current_x` is inside that wall's x-segment** (`eval_fm_uav.py:736-738`). At the
crossover (x∈[-0.5,0.5]) the seg1 walls switch off and the seg2 walls switch on, with two
inside-corner cap balls (`:222-224`). This switching, plus the ~24 cm gate, is the hard part of
the scene — and it's exactly where the geometry-ON variants die.

---

## 2. The full ordering, decoded (variant → active families)

Toggle semantics (`config/uav_projection.yaml:28-31`, gates `eval_fm_uav.py:674,688,726,732,742`):
`geo_free` → geometry OFF (geo_bounds+halfspace+obstacles as a group), `bounds_free` → action
bound OFF, `model_free` → dynamics OFF; default = full stack.

| variant | dynamics | action bound | geometry | schedule | s_curve outcome |
|---|:--:|:--:|:--:|---|---|
| `bounds_free` | ✅ | ❌ | ✅ full | guided, last 50% | **direct crash — WORST** |
| `diffuser` | — | — | — | none | crash in first few steps |
| `geo_free-model_free` (bounds alone) | ❌ | ✅ | ❌ | guided, last 50% | ≈ diffuser |
| `model_free` (dyn off, full geometry) | ❌ | ✅ | ✅ full | guided, last 50% | ≈ diffuser |
| `post_processing` | ✅ | ✅ | ✅ full | **once, final step** | crash **after** first corner |
| `dpcc-c/t/r` | ✅ | ✅ | ✅ full | guided last 50% + select | ≈ post_processing |
| `geo_free` (dyn+bounds, no geometry) | ✅ | ✅ | ❌ | guided, last 50% | **almost works → goal** |
| `geo_free-bounds_free` (dynamics alone) | ✅ | ❌ | ❌ | guided, last 50% | **almost works → goal — BEST** |

Four regimes, top (worst) to bottom (best):

### Regime W — `bounds_free`: dynamics + tight geometry, **no cap on the correction** → direct crash
`bounds_free` keeps the full s_curve geometry AND dynamics, and removes *only* the action bound
(`eval_fm_uav.py:688`). On corridor removing that bound was a near-no-op (corridor doc §4: it
reduces to an idempotent clamp). **Here it is the worst variant** — and that is the tell. With
dynamics ON, DC_FIX couples the executed action to the state channels (`:726-730`); with the
~24 cm non-convex geometry ON, the projector must apply a **large** state correction to satisfy
the narrow, switching halfspaces; DC_FIX propagates that large correction straight into the
**executed action** `act` (dims 0,1,2, executed at `:956,971`). The action bound is normally the
thing that keeps that command inside the dataset's Δp_des range (`action_bounds:'auto'`,
`:712-724`). Strip it and the coupled correction is uncapped → the very first executed command
is out of range → **direct crash.** So on s_curve the action bound is not garnish; it is the
stability cap on a dynamics-coupled geometry correction that is *genuinely large* because the
geometry is tight.

### Regime D — `diffuser` ≈ `geo_free-model_free` ≈ `model_free`: the raw policy already fails early
The FM policy itself does not reliably fly s_curve: raw `diffuser` (`projector=None`) **crashes in
the first few steps.** s_curve is the longest, hardest scene, and the unconstrained ODE sample
can't hold the narrow lane. The two dynamics-OFF ablations land in the same bucket:
- `geo_free-model_free` (**bounds alone**) — clamp without coupling ≈ idempotent (corridor doc §4)
  → tracks `diffuser`.
- `model_free` (**full geometry, dynamics OFF**) — this is the corridor doc's "decoupled
  geometry yanks the state channels but not the action" corruption. On corridor it made things
  *worse* than diffuser. On s_curve it's ≈ diffuser only because **diffuser already crashes
  early** — there's no long stable segment left for the decoupled yanking to corrupt. Both die
  in the first few steps, so they're indistinguishable at the top of the range.

### Regime C — `post_processing` ≈ `dpcc-c/t/r`: coherent full stack survives seg1, dies at the crossover
These keep the **full stack including dynamics**, so (corridor doc §2) their projection is
coherent, and the action bound caps the correction — enough to hold the drone through segment 1
and **past the first corner**, which is strictly better than the crash-early group. But they
**crash after the first corner**, i.e. at/after the **crossover**, which is the one place s_curve
stops being corridor-like: the halfspaces *switch* (`x_active`, `:736-738`), the feasible band is
non-convex, and the ~24 cm corner gate must be threaded. The full geometry that *helped* on
corridor now has to enforce a non-convex switching constraint that fights the FM plan through the
turn → the projection destabilizes the executed action right where the maneuver is hardest.
`post_processing` (project once on the final sample, threshold 0.0, `:748-749`) and `dpcc-*`
(guided last 50% + candidate selection, `_selection_for` `:768-774`) behave the same because the
failure is **geometric**, not about scheduling or candidate choice — both are trying to satisfy
the same infeasible-in-practice corner.

### Regime G — `geo_free-bounds_free` ≈ `geo_free`: drop geometry, keep dynamics → almost to goal
The **best** variants **turn geometry OFF and keep dynamics ON.** With geometry gone there is
nothing to yank the plan at the narrow lanes or the crossover; with dynamics ON the projector
does a **consistent, current-state-anchored re-projection** of the FM plan every replan
(`skip_initial_state` anchors `b[0]=s_0`, corridor doc §2). Over a **750-step** horizon this
re-anchoring is not a no-op the way it was on 360-step corridor: it continually pulls the plan
back to the measured state and enforces `p[t+1]=p[t]+act[t]` self-consistency (`:726-730`),
**smoothing out the drift that made raw `diffuser` crash early.** That is why dynamics-only
**beats** diffuser here (it stabilizes the long rollout) whereas on short corridor it was ≈
diffuser. `geo_free` (dynamics + action bound, still no geometry) ≈ `geo_free-bounds_free`
(dynamics alone) because, with geometry off, the corrections are small and the action bound is
back to being an idempotent near-no-op (corridor doc §4) — the extra bound changes nothing.
Neither *fully* reaches goal every trial ("almost works"), because they carry **no** geometry
and the FM lane-tracking isn't perfect — but they stay safe and get to the endpoint far more
than anything with geometry ON.

---

## 3. The corridor → s_curve flip, in one table

| regime | corridor (benign feasible set) | **s_curve (non-convex, tight, long)** |
|---|---|---|
| full stack `dpcc-*`/`post_processing` | **best** (geometry genuinely helps) | **crashes at the crossover** (geometry can't be satisfied through the turn) |
| dynamics-only `geo_free-bounds_free` | ≈ diffuser, marginally below (re-projection ~no-op) | **best** (long-horizon re-anchoring stabilizes; no geometry to fight) |
| `bounds_free` (no action cap) | ~no-op (idempotent clamp) | **WORST — direct crash** (uncapped dynamics-coupled correction) |
| raw `diffuser` | mid | crashes early (hardest scene) |

The mechanism is one consistent story, not two contradictory ones:
1. **Dynamics is always load-bearing.** Every survivable variant keeps it; every dynamics-OFF
   variant (`model_free`, `geo_free-model_free`) tracks the crash-early raw baseline.
2. **Geometry is conditional.** It helps iff the feasible set is easy to satisfy (corridor:
   wide/convex/static). On a non-convex, ~24 cm, per-segment-switched set (s_curve) projecting
   onto it *destabilizes*, so geometry-ON = crash and geometry-OFF+dynamics = best.
3. **The action bound is load-bearing for stability exactly when geometry is tight**, because
   then the dynamics-coupled correction is large; `bounds_free` removing the cap is the worst
   variant here and a no-op on corridor.

---

## 4. "The tightened variants are no better" — consistent with the above

`-tightened` adds `enlarge_constraints: 0.025` (`config/uav_projection.yaml:73`) on top of the
always-on inflation — it **shrinks the feasible geometry further.** On a scene where geometry is
already the thing causing the crashes, making the ~24 cm bands and the corner gate *even
narrower* cannot help and predictably doesn't. Note the yaml already omits `-tightened` siblings
for the `geo_free` composites on purpose (`:36-38`: "tightening only affects geometry, which
geo_free removes → redundant/no-op") — so the tightened cells that *do* run are precisely the
geometry-ON ones, i.e. the ones already failing; tightening them changes a crash into a crash.

---

## 5. Direct answer to the visual observation ("are the constraints destroying the s_curve runs?")

**On s_curve, yes — the *geometry* constraints are, and here that's not because dynamics is
missing (as on corridor) but because the geometry itself is infeasible-in-practice for this
policy.** The scene is non-convex, the lanes and the crossover gate are ~24 cm, and the horizon
is 750 steps. Any variant that keeps geometry ON either can't cap its own correction
(`bounds_free` → direct crash) or gets through seg1 and dies at the switching corner
(`dpcc-*`/`post_processing`). The variants that **remove geometry and keep dynamics**
(`geo_free`, `geo_free-bounds_free`) let the FM plan flow along the lanes while dynamics
re-anchoring smooths the long rollout — and those are the only ones that almost reach goal. This
is the ablation working correctly: it says the current s_curve **geometry encoding is the
bottleneck**, not the dynamics coupling. The next lever is on the geometry side (see §6), not the
projector logic.

---

## 6. Diagnostic ladder / next steps to confirm on the cluster

Run all on **s_curve**, same seed, and read `results.json` + npz per variant. Each item tests
one claim above:

1. `diffuser` vs `geo_free-bounds_free` — expect **dynamics-only > diffuser** here (opposite of
   corridor). Confirms Regime G: long-horizon re-anchoring stabilizes.
2. `bounds_free` — log `‖act_projected − act_raw‖` at step 0 and `timing.proj_cost`. Expect a
   **large** step-0 executed-action move (the smoking gun for Regime W: uncapped
   dynamics-coupled geometry correction).
3. `dpcc-c` / `post_processing` — log the **x-position at crash** and the step index. Expect the
   crash clustered at the **crossover** (x≈±0.5, where `x_active` switches) — confirms Regime C
   is a *geometric* failure at the corner, not a scheduling one.
4. Split `success` into `goal_reached` vs `safe` (`eval_fm_uav.py:1068-1077`) and check
   `n_fm_steps` vs the 750 budget: the geometry-ON crashes should show `safe`↓ well before step
   750; the `geo_free` variants should show `safe`↑ and `goal_dist` small ("almost").
5. **Geometry-side fix probes** (the real lever, since §5 says geometry is the bottleneck): try a
   *wider* corner gate / larger corner-ball allowance, or a per-segment margin smaller than 0.33
   at the crossover only, and re-run `dpcc-c`. If widening the gate turns the Regime-C crash into
   a goal-reach, that confirms the failure is the ~24 cm switching corner, not the projector.

**Expected outcome if the analysis is right:** #1 dynamics-only > diffuser; #2 `bounds_free`
step-0 action move ≫ `dpcc-c`; #3 crashes concentrated at x≈±0.5; #5 widening the corner
recovers the geometry-ON variants.

---

## 7. Files inspected (no edits)
- `config/uav_projection.yaml` — s_curve geometry (`:196-224`: box, per-segment `x_active`
  halfspaces at inner faces, corner cap balls), `projection_variants` + toggle semantics
  (`:28-38, 39-65`), `enlarge_constraints` (`:73`), inflation.
- `config/uav.py:44-51` — `MAX_PATH_LENGTH_PER_SCENE` (corridor 360 vs **s_curve 750**), the
  horizon fact behind Regime G.
- `FM_v3_uav_test/eval_fm_uav.py` — variant gates (`:674` geo box, `:688` action bound, `:726`
  dynamics DC_FIX, `:732-740` **per-segment halfspace switching via `x_active`+`current_x`**,
  `:742` obstacles), schedule/threshold (`:747-749`), `_selection_for` (`:768-774`),
  executed-action path (`:956,971`), scene-aware success/goal metrics (`:1062-1090`),
  `_normalize_halfspace` (`:613-621`).
- Sibling: `INVESTIGATION_geo_free_model_free_worse_than_diffuser.md` — the corridor case whose
  §2 mechanism (dynamics = action↔state coupling) and §4 (action clamp ≈ idempotent when in
  range) this doc reuses and, for the tight-geometry regime, extends.
