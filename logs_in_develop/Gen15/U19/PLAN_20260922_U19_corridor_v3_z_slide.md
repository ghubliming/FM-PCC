# PLAN — Gen15 U19 · `corridor_v3`: the corridor gets a constraint on **altitude**

**2026-09-22 · update plan for a coding agent · NO CODE WRITTEN, NOTHING RUN.**

> **STATUS 2026-09-22 (later the same day): BUILT, NOT RUN — and the author re-defined v3.** The hump of this plan is
> **not** the corridor v3; it is kept as the ablation **`corridor_v3_ablation_hump`**. The v3 the author imagined is
> **`corridor_v3_tilt`**: the v2 slide itself leaned over (−60° about the launch altitude) so one x–y–z plane pushes the
> drone sideways AND down. Both are coded, both pass gate G0 offline ([`figs/`](figs/README.md)), and the driver's
> `smoke` mode launches BOTH pilots. Changelog:
> [`CHANGELOG_20260922_U19_corridor_v3_coding1.md`](CHANGELOG_20260922_U19_corridor_v3_coding1.md).
> Next: `bash Slurm_Codes/temp_bash/eval_20260922_u19_corridor_v3.sh smoke` on the cluster.
Author's request (v3.65): *"I didn't see it slide or lean on the z axis, feels just x, y. Maybe test a corridor_v3 with
some angle on z, run the eval."* Predecessors: U16 (`corridor_v2_slide`, the corridor of the thesis), U11–U14 (the failed
corridor balls and gate; read [`../U12/CLOSURE_20260912_corridor_obstacle_investigation.md`](../U12/CLOSURE_20260912_corridor_obstacle_investigation.md) once — it says what NOT to repeat).

> **The idea in one line.** Keep the wide corridor exactly as it is — scene XML, model, checkpoint, routes, goals,
> projector stack `-bounds_free-pdes-tightened`, metrics — and swap the **test-time constraint**: instead of the
> 14° slide in the x–y plane, a **hump in the x–z plane** that the plan must climb over. Same mechanism as the slide
> (a sloped, `x_active`-windowed halfspace, virtual, scored only), rotated from the horizontal into the vertical plane.

---

## 0 · Why, in numbers (measured, v3.65)

| fact | value | source |
| :-- | :-- | :-- |
| corridor constraints acting on z | **none** — walls, slide and wall caps are all `[x, y]`; z is bounded by the workspace box only | `config/uav_projection.yaml` `corridor_v2_slide` |
| box on z, raw → after 0.31 m inflation | [0.30, 1.80] → **[0.61, 1.49]** | same entry, `planning_inflation` |
| flown z, all 96 flights of 4 paired cells | **0.86–1.37 m** (launch 1.11 m) — the box never binds | staged rollouts `temp/18-09-2026/logs/UAV_MIX/uav-corridor`, `obs_all[:, 5]` |
| projected − unprojected z, same launch | up to **0.14–0.22 m** (flow models), **0.27 m** (diffusion); grows toward the exit | same; v3.65 changelog §1 |
| with the action cap ON (U11–U13) | max Δz **0.001 m** — the cap from a zero-variance `dz` in the demonstrations froze z | U12 closure §0 |

So: the corridor tests a lateral move only; **the projector CAN move z now** (the `-bounds_free` stack released the
cap), which is the fact U11–U13 did not have. That is what makes a z-constraint worth one more attempt.

## 1 · The geometry — `corridor_v3_hump`

Scene XML **unchanged**: `scene_corridor_v2.xml` (walls y = ±1.0, 1.5 m tall, no ceiling geom). The hump is
**virtual** like the v2 slide (no geom; drone flies through it in MuJoCo; scored as a constraint violation). This keeps
the raw table comparable with v2 (unprojected flights *succeed* but *violate*), and needs no scene work.

**Hump:** the plan's altitude must stay above a roof that rises from the floor at x = −1.5 to height **H at x = 0**
and falls back to the floor at x = +1.5:

```yaml
  - name: corridor_v3_hump
    scene: corridor
    scene_xml: scene_corridor_v2.xml
    geo_tag_suffix: '_cv3h'          # new folder family, never pooled with _cv2s
    constraint_types: ['dynamics', 'geo_bounds', 'halfspace', 'obstacles', 'bounds']
    planning_inflation: {r_drone: 0.31, margin_base: 0.0}
    workspace_bounds:
      lb: [-4.0, -.inf, 0.30]
      ub: [ 4.0,  .inf, 2.80]        # ceiling raised 1.80 -> 2.80 (synthetic: no ceiling geom; same edit as U12/U13)
    halfspace_constraints:           # walls: identical to corridor_v2_slide (plane xy, default)
      - {line: [[-2.0, -0.95], [2.0, -0.95]], side: 'above', x_active: [-2.0, 2.0]}
      - {line: [[-2.0,  0.95], [2.0,  0.95]], side: 'below', x_active: [-2.0, 2.0]}
      # THE HUMP: two sloped halfspaces in the x-z plane, switched by x like the s_curve walls
      - {plane: xz, line: [[-1.5, 0.0], [0.0, 1.10]], side: 'above', x_active: [-1.5, 0.0]}
      - {plane: xz, line: [[ 0.0, 1.10], [1.5, 0.0]], side: 'above', x_active: [ 0.0, 1.5]}
    obstacle_constraints:            # the four wall-end caps, identical to corridor_v2_slide
      - {type: sphere_outside, dimensions: ['x', 'y'], center: [-2.0, -1.0], radius: 0.05}
      - {type: sphere_outside, dimensions: ['x', 'y'], center: [ 2.0, -1.0], radius: 0.05}
      - {type: sphere_outside, dimensions: ['x', 'y'], center: [-2.0,  1.0], radius: 0.05}
      - {type: sphere_outside, dimensions: ['x', 'y'], center: [ 2.0,  1.0], radius: 0.05}
```

**The numbers with H = 1.10** (r_drone 0.31, tightened +0.025; slope 1.10/1.5 = 0.733, so the inflated roof sits
0.31·√(1+0.733²) = **0.385 m** above the raw roof, measured vertically):

| | |
| :-- | :-- |
| required z at the peak (x = 0) | ≥ 1.10 + 0.385 (+0.03 tightened) ≈ **1.49–1.52 m** |
| flown band without projection | 0.86–1.24 m → **every flight violates at the peak**, by 0.25–0.65 m |
| climb needed from the 1.11 m launch | **≈ 0.40 m** — same order as the slide's 0.24–0.47 m lateral demand |
| escape slot above | z ∈ [1.52, 2.49] = **0.97 m** (needs the 2.80 ceiling; at 1.80 the slot would be −0.03 m — infeasible) |
| lateral escape | **none** by construction (the hump spans the full corridor width) — the detour is vertical or nothing |
| divergence guard | corridor envelope z ub 1.30 + 2.0 m slack = 3.30 m — a 0.4 m climb cannot trip it |
| the route end (x = 2.8) | 1.3 m after the hump ends; the finish line is a plane through the route end, so altitude does not affect *success*; the strict goal (≤ 0.30 m, 3-D) may drop if the drone is still high — **report both, as Ch 5/6 already do** |

**No walls kept from the v2 slide.** The slide is dropped on purpose so that the scene isolates z. A combined
scene (slide + hump, a 3-D detour) is a follow-up, not this update.

**Second candidate, milder: H = 0.90** → required z ≈ 1.29–1.32 m, climb ≈ 0.2 m; still violated by every flight
(band top 1.24). Run only if H = 1.10 fails gate G2 *and* the failure is a climb that stops short (G3 passes).

## 2 · Code changes — three sites, all small, all opt-in via the `plane` key

The halfspace machinery is written for a line in the x–y plane. The cleanest change is to let an entry name its
plane and map `y → z` at the three places that touch the coordinates. **Entries without `plane` are byte-identical
in behaviour.**

| # | file · site | change |
| :-- | :-- | :-- |
| 1 | `mix_uav_test/eval_mix_uav.py` `_normalize_halfspace` (≈ l.1201) | return a third value `plane = hs.get('plane', 'xy')` for the dict form (`'xy'` for the list form). Update its ~6 call sites to unpack three values. |
| 2 | `mix_uav_test/eval_mix_uav.py` `setup_dpcc_projector`, halfspace loop (≈ l.1344–1352) | `_hs = {'x': _DIM['x'], 'y': _DIM['z'] if plane == 'xz' else _DIM['y']}` per entry. `utils.formulate_halfspace_constraints` (`mix_uav/utils/constraints_helpers.py:4`) only reads the indices named `'x'` and `'y'`, so it needs **no change**: the sloped branch, the side convention (`'above'` = larger second coordinate feasible) and the tightening direction carry over. Under `-pdes` the `_DIM` table already points at p_des (l.1271), so the binding is inherited. **HardFlow needs nothing**: it builds its NLP from the same constraint list (`return_constraint_list=True`) and its per-replan `x_active` gate reads x only. |
| 3 | `mix_uav_test/eval_mix_uav.py` `_exec_constraint_violations` (≈ l.821–834) | in the halfspace loop use `q = p[2] if plane == 'xz' else p[1]` in place of `p[1]` for the signed distance. Same inflation `r_drone`, same "clear when feasible ≥ r_drone" rule. |
| 4 | drawing only — `eval_mix_uav.py` overview (≈ l.1034, 1072, 1140, 1182, 1470) and `eval_artifacts.py` (`halfspace_body_boundaries` l.289, the extent loop l.145, `write_mpc_foresight`) | **guard**: skip entries with `plane != 'xy'` in every x–y drawing so nothing crashes; optionally add one x–z panel (z of the flown path over x with the hump's raw and inflated roof) to `plot_overview` — this is the picture the thesis would show. |
| 5 | `config/uav_projection.yaml` | append `corridor_v3_hump` as above under a `[Gen15 U19]` comment block; **not** in `active_geo_variants`; select per job with `UAV_MIX_GEO_VARIANTS='corridor_v3_hump'` (U11 mechanism, l.463). |
| 6 | `Slurm_Codes/temp_bash/eval_2026MMDD_u19_corridor_v3_hump.sh` | copy of `eval_20260913_u16_corridor_v2_paper.sh` with geo `corridor_v3_hump`, eval tag **`u19cv3`**, `plan` / `smoke` / `submit` modes kept. `git add -f` (gitignored folder). |
| 7 | `Data_Analysis/DA_UAV_v1/discovery.py` | check that a geo_tag `corridor_cv3h_…` parses (l.117 reads the leading scene token, so it should); add nothing unless it does not. |

Also: `x_active` switching per replan already exists for s_curve (projector rebuilt each step when the scene has
windowed halfspaces, ≈ l.1672); it reads the current x (p_des under `-pdes`) — correct for the hump as it is.

**Do not** touch `Fix_16` / `action_bounds='auto'`: `-bounds_free` is part of the stack and is what lets z move.

## 3 · Gates — in this order, stop at the first failure

| gate | test | pass |
| :-- | :-- | :-- |
| **G0** offline, no GPU | a `temp/geo_demo`-style plot of the hump: raw roof, inflated roof, the flown band 0.86–1.24 m of v2, the 2.49 m slot | inflated roof at the peak above 1.24 m; slot ≥ 0.6 m |
| **G1** pilot, unprojected | `diffuser`, mf K = 3, 3 trials (L/C/R) | violating steps > 0 on **3/3** (the hump binds) |
| **G2** pilot, projected | `dpcc-t-bounds_free-pdes-tightened` and `hardflow_new-t-bounds_free-pdes-tightened`, same trials | collision-free ≥ 2/3 **and** success ≥ 2/3 on at least one arm |
| **G3** the U12 lesson | paired executed z, projected − unprojected, at x ∈ [−0.5, 0.5] | **> 0.15 m** on every projected flight — the plan *climbs*; a violation count that drops because the flight is shorter is not a pass |
| **G4** the wave | U16 paper layout on `corridor_v3_hump`: mf / fm / af at K ∈ {1, 3, 5}, diffusion K = 20; `diffuser` + `dpcc-{r,c,t}-bounds_free-pdes-tightened` (+ HardFlow at K ≥ 3); 12 flights (4 per route); seed 6 | read with the corridor's own DA (`DA_UAV_v1`), both success criteria printed |

Pilot cost ≈ 20 min GPU (U16 smoke was one short job). Wave ≈ 1 GPU-day (U16 was 24 h walls per job, ~10 jobs).

## 4 · What the thesis would do with it

- A pass gives §6.3 a corridor with a **vertical** test-time constraint beside the lateral one: same model, same
  projector, and a side-view figure (the `fig_uav_corridor_altitude` builder of v3.65 already draws z over x from
  `uav_paths.json`; the extract only needs a fifth panel group). It would also fill the slot the flawed UAV-pillars
  material leaves open (v3.65 §3) with a scene whose constraint is *new to the data*, which is the kind that ranks
  projectors (U18 plan §1).
- A G2/G3 failure is still a result: "the projector moves z as a by-product but does not climb on demand" — then the
  v3.65 `\guard` in §6.3.2 becomes a sentence with data behind it. Either way nothing in the current draft rests on U19.

## 5 · Out of scope

A physical ramp geom in a `scene_corridor_v3.xml` (crash instead of violation); the slide + hump combination;
retraining on a demonstration set with altitude changes (the current expert flies level, `dz` variance zero — that is
a different study). Never `pillars_xl` / `pillars_xxl`.

---
Claude (Fable 5.1, Claude Code) · 2026-09-22 · plan only; every line number is from the working tree of this date and
must be re-checked before editing. Nothing executed — the container has no MuJoCo.
