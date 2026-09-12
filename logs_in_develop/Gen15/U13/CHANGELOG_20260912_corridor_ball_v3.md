# Gen15 U13 — `corridor_ball_v3`: the same ball, smaller

**Config only.** One new geo variant in `config/uav_projection.yaml`. No code touched, no other
scene touched.

## 0. The change

**v3 differs from v2 in one number: the ball radius, 0.12 → 0.05.** Centre, walls, wall caps,
workspace box, `constraint_types` and `planning_inflation` are byte-identical to
`corridor_ball_v2`.

| variant | tag | ball | keep-out |
|---|---|---|---|
| `corridor_ball` (U11) | `_hgb` | r = 0.35 @ z = 0.75 | 0.66 |
| `corridor_ball_v2` (U12) | `_hgb2` | r = 0.12 @ z = 1.13 | 0.43 |
| **`corridor_ball_v3`** | **`_hgb3`** | **r = 0.05 @ z = 1.13** | **0.36** |

## 1. Why smaller still blocks

The drone's own inflation (`r_drone` = 0.31) does nearly all the blocking, so shrinking the ball
costs almost nothing in coverage and buys escape room:

| | value |
|---|---|
| keep-out | 0.05 + 0.31 = **0.36 m** (ball itself 0.10 m across) |
| blocked at y = 0 | z ∈ **[0.77, 1.49]** |
| blocked at y = ±0.12 (L/R channels) | z ∈ [0.79, 1.47] |
| measured flown band | z ∈ **[0.956, 1.236]** → fully blocked on all three channels, ≥0.19 m margin |
| escape **up** | z ∈ **[1.49, 2.49] = 1.00 m** wide (v2: 0.93 m) |
| escape **down** | z ∈ [0.61, 0.77] = 0.16 m |
| climb needed | **0.36 m** (v2: 0.43 m) |
| lateral escape | **none** — walls give \|y\| ≤ 0.14, the ball needs \|y\| ≥ 0.36 |

The flown band is measured, not assumed: over 100 `diffuser` rollouts
(`batch_uav_20260911_202307_TEMP`) `phys_min_z` equals `phys_final_z` on every one — the drone holds
a constant altitude, z ∈ [0.956, 1.236], mean 1.127.

🔴 The sphere is **virtual** (no geom in `scene_corridor.xml`); it is scored as a constraint
violation only. No retraining, physics untouched.

## 2. ⚠️ The open risk, stated before the result

v2 already had a 0.93 m escape slot and a correctly-centred ball, and still scored **S&C 0/34** with
`phys_min_z` constant to **0.2 mm** across 340 rollouts — the projector never moved the plan
vertically at all. The measured vertical action authority on `corridor` is **2.2e-05 m/step**
(`action_bounds='auto'`, because the corridor expert flies a straight level line so Δy and Δz have
zero variance in training) = **8.7 mm** over a 396-step episode, against a 0.36 m climb.

**If that cap is what binds, a smaller ball does not change it.** Job **25682**
(`dpcc-t-bounds_free` on v2, action-magnitude family off, geometry on) tests the cap directly and is
queued. Read it alongside this one: together they separate "the ball was too big" from "the
projector cannot move vertically at all".

## 3. Verification

* YAML `safe_load` round-trips; confirmed `constraint_types`, `halfspace_constraints`,
  `workspace_bounds`, `planning_inflation` and the four wall caps are all `== corridor_ball_v2`.
* `_hgb3` keeps results in a fourth sibling geo folder, so `_hg` / `_hgb` / `_hgb2` / `_hgb3` can
  never pool in a DA.
* Not in `active_geo_variants`; selected per job with `UAV_MIX_GEO_VARIANTS` (U11).
* 🔴 Not run on the cluster at write time.

## 4. Run — injection test, job 25689 (2026-09-12)

`Slurm_Codes/temp_bash/eval_20260912_u13_injection.sh` →
mf · corridor · K=2 · `corridor_ball_v3` · seed 6 · n=10 · `diffuser, dpcc-t`. ~12 min.

### Checks

1. `[ U11 ] geo variants for 'corridor': ['corridor_ball_v3']`
2. `E9 geo … variant 'corridor_ball_v3' … (bounds=True, hs=2, obs=5)`
3. results path containing `corridor_hgb3_`
4. **GATE 1** — `diffuser` `n_violations` > 0 (the ball still binds)
5. **GATE 2** — `dpcc-t` `collision_free_completed` > 0 ← **decides v3**

**Awaiting results.**

---

## 5. `corridor_ball_v3_2` — the ball made genuinely tiny

v3 used **r = 0.05**, which is the *same radius as the four wall-end caps* — not small. v3_2 cuts it
to **r = 0.01**: a 2 cm ball, 1/5 of a cap, 1/31 of the drone. **Only the radius changes**; centre,
walls, caps, workspace box, `constraint_types` and `planning_inflation` are byte-identical to v3.

| variant | tag | r | keep-out | escape slot | climb |
|---|---|---|---|---|---|
| `corridor_ball` | `_hgb` | 0.35 | 0.66 | 0.08 m | — |
| `corridor_ball_v2` | `_hgb2` | 0.12 | 0.43 | 0.93 m | 0.43 m |
| `corridor_ball_v3` | `_hgb3` | 0.05 | 0.36 | 1.00 m | 0.36 m |
| **`corridor_ball_v3_2`** | **`_hgb32`** | **0.01** | **0.32** | **1.04 m** | **0.32 m** |

Blocked at y=0: z ∈ [0.81, 1.45]; at y=±0.12: [0.83, 1.43]. The flown band [0.956, 1.236] is still
fully covered on all three channels — the drone's own 0.31 m inflation does the blocking.

### 5.1 🔴 What the v3 and probe results already established

| geo | ball | `diffuser` nviol | `dpcc-t` nviol | `cfree` | S&C |
|---|---|---|---|---|---|
| `_hg` | none | 0.00 | 0.00 | 1.000 | 1.000 |
| `_hgb` | 0.35 | 34.70 | 36.20 | **0.000** | 0.000 |
| `_hgb2` | 0.12 | 27.10 | 25.80 | **0.000** | 0.000 |
| `_hgb3` | 0.05 | 22.00 | 20.80 | **0.000** | 0.000 |

And the action-bound hypothesis is **refuted** — job 25683, `_hgb2`:

```
dpcc-t              nviol 25.80   cfree 0.000
dpcc-t-bounds_free  nviol 27.70   cfree 0.000    <- action-magnitude cap REMOVED
diffuser            nviol 27.10
```

With the cap gone the projector did **worse than no projection**. `phys_min_z` is **1.1265–1.1266**
in every corridor row measured — four geometries, five projector configs, with and without the cap,
**including `corridor_hg` where there is no ball at all**. The projector has never changed the plan's
altitude under any condition tested.

**So neither ball size nor the action bound is the binding variable.** v3_2 should be read as closing
out the radius axis, not as a likely fix.

### 5.2 Run

`Slurm_Codes/temp_bash/eval_20260912_u13_v32_injection.sh` → mf · corridor · K=2 ·
`corridor_ball_v3_2` · seed 6 · n=10 · `diffuser, dpcc-t`. ~12 min. Results under `corridor_hgb32_`.

**Gate 1** `diffuser` `n_violations` > 0 · **Gate 2** `dpcc-t` `collision_free_completed` > 0.
