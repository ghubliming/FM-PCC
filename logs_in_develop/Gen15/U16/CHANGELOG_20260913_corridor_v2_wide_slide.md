# U16 — `corridor_v2`: wide corridor (new MuJoCo scene) + slide halfspace, no retraining

**Date:** 2026-09-13 · **Gen:** 15 · **Follows:** U15 rev2 (`../U15/CHANGELOG_20260913_u15r2_slide_and_plots.md`)
**Preview:** `temp/geo_demo/U16_corridor_v2/preview.png`

## 1. Why

The corridor is too narrow for a visible avoidance test. Walls are 0.90 m apart and the quadrotor
spans 0.62 m rotor tip to rotor tip (r_drone 0.31), leaving 0.28 m of centre-line room. Any
halfspace that visibly crosses a route leaves less than a drone underneath, so the corridor seals
(U15, `temp/geo_demo/U15_block_truth.png`). U16 widens the corridor instead of shrinking the gate.

## 2. What changed

| file | change |
| :-- | :-- |
| `d3il/.../quadrotor/scenes/scene_corridor_v2.xml` | **new**, a copy of `scene_corridor.xml` with both wall boxes moved outward in parallel, y = ±0.5 → **±1.0**. Parsed diff vs the original: only the model name and the two `pos` y values. Length (4 m), height (1.5 m), thickness, drone include, floor and lighting are identical. ASCII |
| `mix_uav_test/eval_mix_uav.py` | `_apply_geo_entry`: optional per-entry `scene_xml` (reset to `None` on every call, so pre-U16 entries are byte-identical in behaviour) |
| | `eval_scene`: if the entry has `scene_xml`, load that XML from the same `scenes/` folder for this entry only, and pass it to `_run_variant`. Otherwise use the scene's own model, loaded once as before. Missing file → `FileNotFoundError`. Logs a `[ U16 ]` line with the loaded walls |
| | new `_scene_obstacles_from_model`: static world-body box geoms, read back from the simulated model into `config['scene_obstacles']` (plain floats) |
| `mix_uav_test/eval_artifacts.py` | `plot_overview` and `write_mpc_foresight` draw `geo_config['scene_obstacles']` when present, instead of the hand-written `generator.SCENE_OBSTACLES['corridor']` (which still says ±0.5). Drawing only |
| `config/uav_projection.yaml` | + `corridor_v2_slide` (suffix `_cv2s`). Additions only; `corridor_hg`, `corridor_gate`, `corridor_gate_n1` and all shared keys unchanged |
| `Slurm_Codes/temp_bash/eval_20260913_u16_corridor_v2_slide.sh` | injection submitter (gitignored folder, so `git add -f`) |

**Not changed:** the original corridor XML and the corridor dataset, checkpoint, routes, starts,
goals and episode budget (`scene: corridor` still selects them all). Also unchanged: the scorer,
projector, HardFlow, controller and normaliser.

### Why reuse the corridor model works

The UAV policy is state-only: observations are commanded + actual position, 6-D, with no image.
It never sees the walls, so moving them changes nothing the model conditions on, only the physics
it flies through. Contact detection is `generator._is_obstacle_contact` (any non-floor contact),
which works with any scene. The divergence envelope (corridor y ±0.45 + 2.0 m slack) cannot trip
inside ±1.0 walls.

## 3. Geometry

| | corridor | corridor_v2 |
| :-- | --: | --: |
| wall centres | y = ±0.50 | y = ±1.00 |
| inner faces / clear width | ±0.45 / 0.90 m | ±0.95 / 1.90 m |
| drone-centre room | ±0.14 | **±0.64** |

**Slide:** `{line: [[-2.0, 0.95], [2.0, -0.05]], side: 'below', x_active: [-2.0, 2.0]}`. It runs from the
top wall at the corridor entrance, descends 14.0°, and visibly crosses route C at x = 1.80.

| scorer basis (r_drone 0.31) | value |
| :-- | :-- |
| drone-centre limit | entrance +0.630, exit **−0.370** |
| room under it at the exit | 0.27 m (raw line to bottom wall: 0.90 m vs the 0.62 m drone) |
| route L (y −0.12) | blocked x ∈ [1.01, 2.00], must drop 0.25 m |
| route C (y 0.00) | blocked x ∈ [0.53, 2.00], must drop 0.37 m |
| route R (y +0.12) | blocked x ∈ [0.05, 2.00], must drop 0.49 m |
| sliding needs | 0.0110 m/step = 50% of the `FMPCC_SAFE_EPS_FRAC=1.0` cap |
| after the exit | goal (2.8, route y) needs only y ≥ −0.30 (goal radius 0.30) |

## 4. ⚠️ Risks, stated before the run

- **Out of distribution.** The corridor model never saw |y| > ~0.13; the slide forces it to −0.37.
  Observations are not clipped on `normalize()`, so the model receives an unfamiliar state.
  Its plans there are untested.
- **Projector position box.** The SLSQP box is z ∈ [−5, 5], which for a p_y half-width of ~0.13
  (estimated, not measured) is about ±0.65 m. −0.37 is inside it; a deeper slide might not be.
- **FRAC = 1.0 side effects from U15 rev1:** the diffuser drifted 6–7 cm, DPCC overshot and missed the
  goal, and HardFlow flipped in both trials. They are unaddressed and may repeat.
- **Not testable locally** (no `mujoco` in Docker): the XML load, `_scene_obstacles_from_model` and
  the GIF overlay were only compile-checked. The script's child-log check (`[ U16 ] ... walls:
  y=±1.00`) is the confirmation.

## 5. Injection

Same as U15: `diffuser`, `dpcc-t-bounds_free`, `hardflow_new`; K = 3; 2 trials (L, C);
`FMPCC_SAFE_EPS_FRAC=1.0`; `record=gif` at 320 px. Pass criteria are in the script header.

## 6. Run record

| job | submitted | status |
| --: | :-- | :-- |
| (id not recorded) | 2026-09-13 | DONE. Results `temp/1309/corridor_cv2s_bounds+dynamics+geo_bounds+halfspace+obstacles/`, child log not downloaded |

### 6.1 First read (2 trials × 3 arms, one seed, K = 3: injection, not a DA)

**Scene check (indirect, no child log).** The projected drones flew at y ≈ −0.63 inside x ≤ 2 with
contact_frac ≤ 0.001. In the original XML that position is through the wall, so `scene_corridor_v2.xml`
was the scene simulated. FRAC = 1.0 reached the job: paths moved 0.5–0.7 m.

| arm | success | S&C | collision-free | slide viol. steps | slide depth (max) | y at x = 2.0 (limit −0.37) | goal dist |
| :-- | :-: | :-: | :-: | :-- | :-- | :-- | :-- |
| diffuser | 2/2 | 0/2 | 0/2 | L 39, C 58 | 29 / 42 cm | −0.07 / +0.07 (drifts 6–7 cm, as U15) | 0.30 / 0.29 |
| dpcc-t-bounds_free | **0/2** | 0/2 | 0/2 | L 30, C 30 (+3 bottom-wall) | **8 / 8 cm** | −0.62 / −0.63 | 0.73 / 0.77 |
| hardflow_new | **0/2** | 0/2 | 0/2 | L 30, C 32 (+1 wall, +1 cap) | **8 / 9 cm** | −0.64 / −0.64 | 0.75 / 0.81 |

No divergence aborts. **HardFlow did not flip this time.** No circuit-breaker trips.

**Route C timeline** (y at x = −1, 0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8):
- slide limit: +0.38, +0.13, +0.005, −0.12, −0.25, −0.37, n/a, n/a
- dpcc: −0.01, +0.01, +0.01, **−0.05**, **−0.53**, −0.63, −0.60, −0.64

**Reading:**
- **Works:** both projectors now change the flight and cut slide penetration depth by ~75–80%
  (29–42 cm → 8–9 cm), on L and C, with HardFlow stable.
- **Fails 1, late:** the slide engages at x ≈ 0.45, but the drone only starts descending at x ≈ 1.0,
  giving 30 shallow violation steps.
- **Fails 2, overshoot + no return:** it then drops to the floor limit (−0.63 vs the −0.37 needed) and
  stays there after the slide ends. The goal needs y ≥ −0.30 at x = 2.8, so success is 0/2 where the
  diffuser gets 2/2. Consistent with the model being out of distribution at |y| > 0.13.

**Verdict:** mechanism success, metric failure. Not ready for a paper run.
