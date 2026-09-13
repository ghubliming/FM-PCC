# U14 — `corridor_gate`: a slanted halfspace across the corridor

**Date:** 2026-09-12 · **Gen:** 15 · **Scope:** `config/uav_projection.yaml` only — **no code changed**

## 1. What changed

One new geo variant appended to `geo_constraint_variants`. Nothing else in the file was edited;
`corridor_hg`, `pillars_hg`, `s_curve_hg` and `corridor_ball*` are byte-identical to before.

```yaml
- name: corridor_gate
  scene: corridor
  geo_tag_suffix: '_hgg'
  halfspace_constraints:
    - {line: [[-2.0, -0.45], [2.0, -0.45]], side: 'above', x_active: [-2.0, 2.0]}   # wall, unchanged
    - {line: [[-2.0,  0.45], [2.0,  0.45]], side: 'below', x_active: [-2.0, 2.0]}   # wall, unchanged
    - {line: [[-2.0, -0.25], [2.0,  0.55]], side: 'below', x_active: [ 0.40, 2.0]}  # ← THE GATE
```

`corridor_gate` is NOT added to `active_geo_variants` — it is selected per job with
`UAV_MIX_GEO_VARIANTS`, the U11/U12/U13 pattern. Default behaviour of every other pipeline
is unchanged.

## 2. Why a halfspace, after four failed spheres

U11–U13 placed a sphere on the corridor route at r = 0.35 / 0.12 / 0.05 / 0.01. All four:
`collision_free = 0.000`, and the executed path **bit-identical** to the unprojected arm
(max |Δy| = 0.0000 m, max |Δz| = 0.0010 m, viol/step 0.0721 vs 0.0716 = 0.99×). Radius,
altitude, ceiling (1.80 → 2.80), the action-magnitude cap (`dpcc-t-bounds_free`) and the
projection threshold (T = 1) were each ruled out individually.

The halfspace family is the one that is **known to bite on this stack**: on `pillars` it takes
`af` from 197.70 → 22.90 violations (rate 0.3118 → 0.0584 = 0.19×) and `collision_free`
0.000 → 0.900. If the family works there and not here, the difference is not the family.

## 3. Geometry — verified, not assumed

`corridor_path(x_start=-2.8, x_end=2.8)` (`uav_expert_data_collect/trajectories.py:112`) →
**flight is left to right.** The gate rises left→right and starts at x = +0.40, so the drone
meets the **tightest point first** — a head-on wall — and the wedge opens behind it.

The x-mirror is not cosmetic. The descending version eases the drone in and squeezes it last;
the ascending version blocks it on arrival. The latter is the demo.

```
raw line          y = 0.15 + 0.20x        slope 0.20, +11.31°, sec = 1.0198
effective (plan)  y ≤ 0.20x − 0.1661      margin 0.31
effective (score) y ≤ 0.20x − 0.1865      margin 0.33 = r_drone + margin_base
```

### 🔴 Why `x_active` starts at 0.40 and not 0.13

0.13 is where the **planner's** slot opens. The **scorer** uses margin 0.33 and its slot does
not open until x = 0.3327. An `x_active` of 0.13 would score violations on x ∈ [0.13, 0.33]
that **no path can avoid**, so `collision_free` could never reach 1 even for a perfect run —
the metric would be broken before the experiment started. 0.40 clears the scorer by 67 mm.

### Effect on the three trained channels (scorer basis)

| channel | y | scored-blocked x-range |
| :-- | --: | :-- |
| **L** | −0.12 | **none — CLEAN for all x** |
| C | 0.00 | [0.40, 0.93] |
| R | +0.12 | [0.40, 1.53] |

Deliberately the easiest solvable form: the detour target is not a novel path, it is the **L
channel the generator already has demonstrations for** (`CORRIDOR_CHANNELS`). The projector
only has to move the plan 0.120 m in y, onto a route the model has seen.

## 4. 🔴 Prediction, on record before the run

`action_bounds:'auto'` derives the per-step action cap from this dataset's own observed range.
The corridor expert flies dead straight, so Δy is degenerate — from the eval's own line,
`temp/0809/2026-09-07/00_05_59_uav_mix_eval_25495.log` (`uav-corridor`):

```
action_bounds=auto → lb=[ 1.24e-04 -2.20e-05 -2.20e-05] ub=[4.3886e-02 2.2000e-05 2.2000e-05]
```

| scene | Δx | Δy cap | max followable angle | source |
| :-- | --: | --: | --: | :-- |
| **corridor** | 4.39e-02 | **2.20e-05** | **0.029°** | `…eval_25495.log` |
| pillars | 4.41e-02 | 3.97e-02 | 42.0° | `…eval_25499.log` |
| s_curve | 2.22e-02 | 2.29e-02 | 45.9° | `…eval_25502.log` |

Each scene derives its own cap from its own data (`--scene` sets the dataset string to
`uav-<scene>`, `config/uav_mix.py:15`).

- riding this boundary needs Δy/step = 0.20 × 4.3886e-02 = **8.78e-03 → 399× the cap**
- reaching L at all needs 0.120 m; 271 steps × 2.2e-05 = **6.0 mm available → 20× short**

**So the honest expectation is that this fails like U11–U13.** If it does, that IS the result:
the blocker is the **dataset's action range**, not the constraint family, not the shape, not
the angle — and the per-scene table above is the evidence. A pass would refute that and is the
more interesting outcome. Either way the run is decisive, which is why it is worth the slot.

## 5. Gates

| # | check | meaning |
| :-- | :-- | :-- |
| 1 | `diffuser` `n_violations > 0` | the gate actually bites |
| 2 | `dpcc-t` `collision_free_completed > 0` | the projector actually solves it |
| 3 | `dpcc-t` executed y differs from `diffuser` by **> 1 mm** | **the real question** — U12 failed this one |

Gate 3 is the one that matters. U11–U13 all passed gate 1 and failed gates 2 and 3 together;
what was never established is whether the projector *moves the path at all* on this scene.

## 6. Demo plots

`temp/geo_demo/U14_corridor_hs_repo/` (not version-controlled):

- `constraint_overview.png` — the repo's own `plot_geo_constraints`, 3-panel, with the gate in it
- `lr_mirror.png` — as-drawn vs x-mirrored; **the mirrored panel is what was approved**
- `mirror_and_depth.png` — why "push it further down" seals the corridor

⚠️ The repo plotter draws only the **raw** line, never the inflated boundary. The drone is
0.31 m wide, so what actually constrains it sits 0.31 m below the orange line. That gap is
what made the U11/U12 ball look like it was floating above the route.

## 7. Not done

- `active_geo_variants` not modified (env override instead)
- the `cz_mid` plot bug (`eval_mix_uav.py:1157`, `eval_artifacts.py:488,507`) still unfixed —
  obstacles are drawn at workspace mid-height, not their true z. Fix reviewed in
  `temp/geo_demo/fixed/`, not applied: no code changes authorised.

## 8. Run record

| job | mode | engine | K | seed | variants | submitted | log dir |
| --: | :-- | :-- | :-- | --: | --: | :-- | :-- |
| **25706** | `full` | mf | 2, 5, 10 | 6 | 7 | 2026-09-13 (cluster) | `Slurm_Codes/logs/2026-09-13/` |

Variants: `diffuser`, `dpcc-r`, `dpcc-c`, `dpcc-t`, `dpcc-t-tightened`, `dpcc-t-geo_free`,
`dpcc-t-bounds_free`. → 21 cells (7 variants × 3 K).

⚠️ The cluster rolled to **2026-09-13** at submit time; local date was 09-12. Logs and the
batch folder are under the **09-13** directory.

The injection step was skipped — `full` was submitted directly. Gate 1/2 can still be read off
the K=2 `diffuser` / `dpcc-t` cells, which is exactly what the injection would have produced.

**Status (2026-09-13): DA DONE — see §9 and the DA.** 25707/25708/25709 all ended "Job completed
successfully", 7/7 variants each, no traceback / cancel / time-wall. Raw result folders downloaded
to `temp/1309/` (21 `results.json`, 210 rollout logs; K identified by `fm_ms` ≈ 18 / 45 / 90 ms,
since the K2 folder carries no suffix). **No `batch_*` folder yet** — the CSVs come from a separate
`run_da_batch_uav.sh` job. No DA until the `batch_*` CSVs are downloaded
(`da-requires-csv-never-from-logs`); gate 3 additionally needs the rollout logs, so pull the
whole batch folder, not just the CSVs.

## 9. 🔴 Corrections after the run (2026-09-13)

Three statements above were wrong. The config entry itself is unaffected: the gate is valid and did
exactly what it was designed to do (gate 1 passed).

1. **Scorer margin is 0.31, not 0.33.** `_exec_constraint_violations` uses
   `config['inflation']['r_drone']` only (`eval_mix_uav.py:783`), not `r_drone + margin_base`.
   So the scorer slot opens at x = 0.1307, the same as the planner. The §3 argument for starting
   `x_active` at 0.40 instead of 0.13 rested on the wrong margin. 0.40 is harmless, just more
   conservative than needed. Corrected blocked ranges: **C x ∈ [0.40, 0.83], R x ∈ [0.40, 1.43]**
   (not 0.93 / 1.53). L is still clean. The DA used the correct 0.31 and its recount matches the eval
   205/210 exact, 210/210 within one step.
2. **Gate 2 was mis-specified.** `collision_free_completed > 0` is satisfied by the L channel by
   construction, so every cell read 0.40 without the projector doing anything. The DA counts C/R only.
3. **The §4 mechanism is refuted.** The failure was predicted correctly, but the stated cause
   (the `action_bounds='auto'` constraint) is ruled out by this wave's own `dpcc-t-bounds_free`
   control, whose executed path is identical to `dpcc-t`. See DA §5.1. The degenerate Δy channel
   still correlates with the null result. The route it acts through is open (DA §5.3).

**Result:** [`DA_20260913_corridor_gate_full_wave.md`](DA_20260913_corridor_gate_full_wave.md). All
three original gates evaluated: 1 PASS, 2 FAIL (0/90 C/R), 3 FAIL (max 1.0 mm, `-tightened` 3.0 mm).
