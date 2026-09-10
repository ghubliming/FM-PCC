# Gen15 U11 — `corridor_ball`, and a per-job geo-variant override

## 0. TL;DR

`corridor_hg` is **constraint-trivial**: every arm, projected and unprojected, reports
`n_violations = 0.00` and `collision_free_completed = 1.000`. S&C degenerates to plain `success`
and the projector spends 155–202 ms/step removing violations that never happened — the scene cannot
test constrained control at all. **U11 adds `corridor_ball`**, a forced-detour variant, plus
**`UAV_MIX_GEO_VARIANTS`** so it can be selected per job instead of by editing a shared yaml key.

## 1. The evidence that motivated it

From `Campaign_20260907_five_missions/CLOSURE_20260910_uav_engine_ladder_final.md` §1.1, over the
four `u7hg` corridor candidates (af K1, af K2, mf K2, fm K2), `batch_uav_20260910_092309`:

| arm | S&C | `n_violations` | `total_violations` | `collision_free_completed` |
|---|---|---|---|---|
| af K=1 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 |
| af K=2 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 |
| mf K=2 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 |
| fm K=2 `diffuser` | 1.000 | **0.00** | **0.00** | 1.000 |

The projected rows (`dpcc-t`, `dpcc-t-tightened`) report the same zeros. The metric is **not**
broken — the identical unprojected arm reports **197.70** violations with `cfree = 0.000` on
`pillars` and **24.30** on `s_curve`. `corridor` is genuinely trivial.

*Per the user's manual check (2026-09-10): `pillars` and `s_curve` geometry are fine as they are.
Only `corridor` needed this.*

## 2. 🔴 Why the detour is vertical — it is forced, not chosen

The obvious design is a cylinder in the middle that the drone flies around. **That is
geometrically impossible here, for any radius including zero:**

| quantity | value | source |
|---|---|---|
| wall inner faces | y = ±0.45 | `scene_corridor.xml`: `pos="0 ±0.5 0.75" size="2.0 0.05 0.75"` |
| planner drone radius | 0.31 | `planning_inflation.r_drone` (U7) |
| free lateral band for the planner | **\|y\| ≤ 0.14** | 0.45 − 0.31 |
| a centre sphere demands | **\|y\| ≥ r + 0.31** | `sphere_outside` |

`0.31 > 0.14` ⇒ **no lateral bypass exists**. An `['x','y']` cylinder at the centre would make the
scene *infeasible*, not hard. So the ball is **3-D** (`['x','y','z']`) and the route is over the top.
`_DIM` already maps `z → 8` (`eval_mix_uav.py:1195`), so no code change was needed for this.

### The numbers

| | |
|---|---|
| ball | centre `[0.0, 0.0, 0.75]`, radius **0.35** (0.70 m across, in a 0.90 m corridor, vs a 0.62 m drone) |
| blocked band at (x,y)=(0,0) | z ∈ **[0.09, 1.41]** (0.75 ∓ (0.35 + 0.31)) |
| expert cruise altitude | z ~ **U(0.90, 1.30)** — `uav_expert_data_collect/generator.py:124` |
| ⇒ | the expert band lies **entirely inside** the blocked band, on all three channels (`CORRIDOR_CHANNELS` L/C/R = −0.12 / 0.0 / +0.12) |
| escape | z ∈ **[1.41, 1.80]** — 0.39 m of headroom |
| under the ball | 0.75 − 0.66 = 0.09 < 0.30 workspace floor ⇒ **infeasible** |

One feasible route, and the policy has never flown it.

## 3. 🔴 The sphere is VIRTUAL

`scene_corridor.xml` contains only a floor and two wall boxes. **No geom is added.** The drone flies
through the ball in MuJoCo; it is scored as a constraint violation and nothing else.

That is deliberate: this variant asks whether the **projector** can route around a constraint the
policy has never seen — with **no retraining** and **no change to physics**, so the existing
corridor checkpoints stay valid. Do not describe it as an obstacle the drone collides with.

## 4. The changes

| file | change |
|---|---|
| `config/uav_projection.yaml` | new `geo_constraint_variants` entry **`corridor_ball`** — `corridor_hg`'s walls and all four wall-end caps unchanged, plus the virtual centre ball; `geo_tag_suffix: '_hgb'`. **Not** added to `active_geo_variants`. |
| `mix_uav_test/eval_mix_uav.py` | `UAV_MIX_GEO_VARIANTS` override inside `_resolve_active_geo_matches()`; validates against the yaml and against the scene, `exit 2` otherwise; echoes `[ U11 ]`. |
| `Slurm_Codes/sbatch/uav_mix/eval_mix_uav.sh` | exports `UAV_MIX_GEO_VARIANTS` + echo |
| `Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh` | exports `UAV_MIX_GEO_VARIANTS` + echo |

### Why an env override and not an `active_geo_variants` edit

`active_geo_variants` is a **shared** yaml key read at **run** time, while a job's environment is
captured at **submit** time. Editing it would re-point every queued corridor job — the U6 failure
mode, and the same reason U9 (`UAV_MIX_VARIANTS`) and U10 (`UAV_MIX_CONTROLLER`) exist. The override
can only narrow or re-point to entries the yaml already defines; it can never invent one.

`geo_tag_suffix: '_hgb'` makes the results path distinct (`corridor_hgb_bounds+dynamics+…`), so a
`corridor_ball` run can never pool with `corridor_hg` in a DA.

⚠️ `eval_scene()` runs **every** active entry for the scene in one job (Fix_6), so
`UAV_MIX_GEO_VARIANTS='corridor_hg,corridor_ball'` is a within-job A/B and roughly doubles runtime.
`load_pcc_config()` still requires exactly one match and raises if given two.

## 5. Verification

* `python3 -m py_compile mix_uav_test/eval_mix_uav.py` — OK
* `bash -n` on both sbatch scripts — OK
* YAML structurally validated against the `corridor_hg` sibling: identical key names and indent
  levels, one extra `obstacle_constraints` item, no tabs. *(No `yaml` module is installed in this
  container, so a `safe_load` round-trip was not possible here — do it once on the cluster.)*
* 🔴 **Not run on the cluster.** First-run checks below.

### First-run checks

1. `[ U11 ] geo variants = corridor_ball` in the wrapper log
2. `[ U11 ] geo variants for 'corridor': ['corridor_ball']` in the eval log
3. `[ eval ] E9 geo 'corridor' ← variant 'corridor_ball': … (bounds=True, hs=2, obs=5)` — **obs=5**,
   four caps plus the ball
4. a results path containing `corridor_hgb_`
5. 🔴 **the arm is only meaningful if `diffuser` now violates.** If the unprojected row still
   reports `n_violations = 0.00`, the ball is not binding and the geometry needs revisiting before
   any comparison is read.

## 6. What this variant is for

`corridor_hg` could not distinguish a projector that works from one that does nothing — both scored
zero violations. `corridor_ball` can: the unprojected plan is trained at z ∈ [0.90, 1.30] and should
violate on essentially every rollout, while a projector that does real work must lift the plan into
[1.41, 1.80]. That is the projected-vs-unprojected contrast the corridor scene has never supported.
