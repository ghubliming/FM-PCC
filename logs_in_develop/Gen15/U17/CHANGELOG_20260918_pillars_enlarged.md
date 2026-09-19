# Gen15 · U17 — UAV-pillars enlarged at test time (`pillars_xl`, `pillars_xxl`)

**2026-09-18.** Config + driver only. **No Python was touched, no model retrained, no MJCF edited.**

- Diagnosis: [`PENDING_20260918_pillars_geometry_redesign.md`](../../Writing/Working_Space/data_status/PENDING_20260918_pillars_geometry_redesign.md)
- Runbook: [`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](../../Writing/Working_Space/data_status/SLURM_RUNBOOK_20260919_pillars_enlarged.md)
- Control condition it is contrasted against: [`Gen15/U7`](../U7) (`pillars_hg`), kept intact.

---

## 1. Why

`pillars_hg` enforces the constraint the demonstration generator was **built to satisfy**:

```
uav_expert_data_collect/trajectories.py:48
    _Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_ROTOR_REACH - PILLAR_SAFETY   # = -1.11
```

The projector demands `|y| >= 0.6 + 0.12 + 0.31 = 1.03`; the demonstrated routes fly at
`|y| = 1.11`, **0.08 m inside the feasible set before any projection runs**. 4 of 4 routes satisfy
the set, tightening included, and the *unprojected* plan already scores S&C 1.00 (fm) / 0.90 (mf)
at K=5. Every projected cell therefore measured **how little of an already-feasible plan a method
disturbs**, not whether a method can make an infeasible plan feasible — which is why per-step
projection with random selection *destroyed* a clean plan (1.00 → 0.00) and endpoint projection
"won" by leaving it alone. That is why the four models could not be ordered on this scene, and it
is not a property of the models.

D3IL-avoiding does not have this problem because its halfspaces are introduced **at test time**
(0, 1 and 2 of 96 demonstrations satisfy them). UAV-corridor gets the same from the slide. U17
gives UAV-pillars that construction.

## 2. What changed

**One file: `config/uav_projection.yaml`** — two new `geo_constraint_variants` entries appended.
Nothing else in the repo changed.

| | `pillars_hg` (untouched) | **`pillars_xl`** | `pillars_xxl` |
| :-- | :-- | :-- | :-- |
| `geo_tag_suffix` | `_hg` | `_xl` | `_xxl` |
| `obstacle_constraints[*].radius` ×6 | 0.12 | **0.35** | 0.55 |
| `scene`, `constraint_types`, `planning_inflation`, `workspace_bounds` | — | identical to `_hg` | identical to `_hg` |
| in `active_geo_variants` | yes | **no** | **no** |
| submitted in the 19-09 wave | control | **yes** | **no** |

Deliberately **not** changed:

1. **`enlarge_constraints` stays `0.025`.** The tightening knob is a reported variable of the
   thesis (every table has a *tightened* column); doing the enlargement there would make the two
   indistinguishable. The enlargement lives in the entry's own `radius`. The `-tightened` variants
   of the new entries add the usual 0.025 on top, meaning what they mean everywhere else.
2. **`scene_pillars.xml` stays at `size="0.12 1.0"` ×6.** Enlarging the MJCF would invalidate the
   demonstrations and force a retrain. The enlarged pillars are **virtual**, exactly like the
   U11–U13 corridor balls: the drone flies through the extra 0.23 m in MuJoCo and it is scored as a
   constraint violation only.
3. **`active_geo_variants` is left as written.** A scene's two entries must never be active in the
   same job (Fix_6 multi-match: both run, and pillars at K=5 already walls at 24 h). Every job
   selects its geometry with `UAV_MIX_GEO_VARIANTS=pillars_xl` (U11), which keeps the shared file
   honest for corridor and s_curve.

## 3. The geometry

Planner margin `r_drone 0.31 + margin_base 0.0`; scorer margin `r_drone 0.31` (it ignores
`margin_base`) — **identical, so plan and score agree.** Pillars at `x ∈ {-2, 0, 2}`, `y = ±0.6`.

| | `_hg` R=0.12 | **`_xl` R=0.35** | `_xxl` R=0.55 |
| :-- | --: | --: | --: |
| keep-out radius `R + 0.31` | 0.43 | **0.66** | 0.86 |
| outer channel opens at `\|y\| >=` | 1.03 | **1.26** | 1.46 |
| demonstrated route `\|y\| = 1.11` | feasible by 0.08 | **violates by 0.15** | violates by 0.35 |
| straight centre lane (`y = 0`) | open (`\|y\| <= 0.17`) | **blocked** | blocked |
| … blocked for `\|x - x_col\| <` | — | 0.275 | 0.616 |
| free band to the box edge | [1.03, 2.19] | **[1.26, 2.19]** | [1.46, 2.19] |
| … width | 1.16 m | **0.93 m** | 0.73 m |

Both rungs stay **solvable**: 0.93 m (xl) / 0.73 m (xxl) of free corridor against a measured
`track_err` of ~0.30 m. `-tightened` takes another 0.025 off each side (xl: [1.285, 2.165], 0.88 m).
The 0.15 m the demonstrated route now sits inside the obstacle is comparable to the corridor
slide's 0.24–0.47 m — the scene this one is being made to resemble.

## 4. The two things that would have made the wave worthless — both checked

**① The violation scorer must use the enlarged radius.** ✅ **Verified in the code, not assumed.**
`_exec_constraint_violations` (`mix_uav_test/eval_mix_uav.py:792`) reads
`config['obstacle_constraints']` and scores each step at `radius + r_drone`, where `config` is the
**resolved geo entry** (`_apply_geo_entry`, :501). So the enlarged radius is the yardstick as well
as the tube, with no code change. Had it scored against the physical 0.12 pillars, every row would
have read collision-free and the experiment would say nothing. The driver's pre-flight greps for
that line, and group V re-checks it empirically on one cell before the wave goes out.

**② Physical contact and the enlarged constraint are now different things.** ✅ Intended, and
already separated: `phys_safe` is MuJoCo contact truth (`eval_artifacts.py:211`) and is
radius-independent, so it stays a genuinely separate column. A flight can be `phys_safe` and still
violate — which is exactly what D3IL-avoiding does.

**Expected, and not a bug:** `_warn_expert_route_infeasibility` (:904) will print
`WARNING pillars homotopy=…: expert route violates the PLANNING constraint set …` at job start.
That is now the **point** of the entry. The check is print-only and never blocks a run (:913).

## 5. The driver

`Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh` — plan-by-default, same shape as
`pipeline_20260918_pending_all.sh`.

⚠️ **`Slurm_Codes/temp_bash/` is gitignored** (`.gitignore:49`) — the file does **not** arrive with
a `git pull`. Copy it to the cluster by hand.

```bash
bash Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh                    # PLAN, group V
bash Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh submit             # the verify cell
WAVE="A B C D" bash Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh submit   # the wave
```

**10 driver jobs, 13 children.** Group V runs first, alone.

| grp | engines | K | variants | children | tag |
| :-- | :-- | :-- | --: | --: | :-- |
| **V** | mf | 5 | `diffuser` only | 1 | `u7xlchk` |
| A | fm, mf, af | 1, 2 | 7 per-step | 6 | `u7xl` |
| B | fm, mf, af | 5 | 5 per-step (half 1) | 3 | `u7xl` |
| C | fm, mf, af | 5 | 6 endpoint + `dpcc-t` (half 2) | 3 | `u7xl` |
| D | diffusion | 20 (training-fixed) | 7 per-step | 1 | `u7xl` |
| E *(opt-in)* | fm, mf, af | 2 | endpoint, `A=1.0` | 3 | `u7xla1` |
| X *(opt-in, do not run)* | fm, mf, af | 5 | `pillars_xxl` | 3 | `u7xxl` |

Four design decisions worth not re-deriving:

- **The eleven configurations** are `diffuser`, `dpcc-{r,c,t}` plain and `-tightened`, and
  `hardflow_new{,-r,-c,-t}`. `*-geo_free` rows are not part of the thesis and are not run.
- **K=5 is split into two disjoint jobs (B and C), not one.** The `pillars_hg` K=5 jobs
  (25318/25321) hit the 24 h cap with 5 of 17 variants done — `proj_ms` for `dpcc-c` is 63 ms at
  K=2 against 1751 ms at K=5 — and the enlarged constraint makes the solver work *harder*, because
  the plan is now genuinely infeasible. The halves are disjoint so two concurrent jobs can never
  race on the same variant folder, and C carries `dpcc-t` as the non-HardFlow companion the eval
  requires (it refuses a HardFlow-only subset).
- **Groups A/B carry no endpoint rows.** HardFlow is degenerate at K=1 (the only step is terminal)
  and at K=2 under the shipped A=0.5 (step 0 floors out); the eval drops those variants and writes
  `HF_DEGENERATE_SKIPPED.txt`. Endpoint at K=2 needs A=1.0 — that is opt-in group E, under its own
  tag, because A=1.0 rows are **not** interchangeable with group C's A=0.5 rows.
- **Group D has no K list.** K is a *training* property for DPCC diffusion (the checkpoint is
  literally K20), and endpoint projection is unavailable to that arm by construction
  (`engine_registry` sets `supports_hardflow=False` for ddpm — it needs a velocity field). D also
  closes the `dpcc-r` / `dpcc-r-tightened` gap the baseline had on the old geometry.

Tags `u7xl` / `u7xlchk` / `u7xla1` / `u7xxl` cannot collide with `u7hg`, and `geo_tag_suffix: '_xl'`
puts the results in their own folder — the new rows can never pool with the control condition.

## 6. Also to be done on the cluster (not this driver)

`scancel 25890 25891 25892 25893 25894 25895` — groups C/D/E of the 18-Sep wave are all
`pillars_hg` work filling a table the draft no longer prints. **Only those still pending or
running**; finished ones are kept as part of the control condition. Groups A/B/F/G/H of that wave
do not touch pillars and are unaffected.

## 7. Status

- [x] `pillars_xl` + `pillars_xxl` in `config/uav_projection.yaml`; YAML re-parsed, `pillars_hg`,
      `active_geo_variants` and `enlarge_constraints: 0.025` confirmed byte-unchanged
- [x] scorer path verified to follow the entry's radius; `phys_safe` confirmed independent
- [x] driver written, `bash -n` clean, PLAN mode prints 10 driver jobs with the pre-flight passing
- [x] **group V submitted and READ — PASSED both gates.** Driver 25923 -> child 25940,
      completed 2026-09-19 09:24:48-09:29:22 UTC, git rev `b8f1beb2`. mf / K5 / `diffuser`,
      10 flights, tag `u7xlchk`:

      | metric | `pillars_hg` | `pillars_xl` |
      | :-- | --: | --: |
      | S&C (`strict_and_constraints_rate`) | 0.90 | **0.00** |
      | `n_violations_mean` | 0 | **68.7** |
      | `collision_free_rate` | 1.00 | **0.00** |
      | plain `strict_rate` | 0.90 | 0.90 (unchanged) |
      | `physical.safe_rate` | 1.00 | 1.00 |

      `diffuser` is unprojected, so these are the SAME flights as the `_hg` rows re-scored —
      plain success is identical, which pins the whole S&C collapse on the constraint. The four
      expert-route probes flipped from passing to 52-54/200 violating samples. 9 of 10 rollouts
      show 51-53 violating steps at ~0.066 m mean penetration (predicted max 0.15 m); the 10th is
      the single failed flight (223 violations, goal missed) and alone lifts the mean to 68.7.
- [~] **groups A–C submitted (attempt 2): drivers 25970–25978**, 9 drivers -> 12 children,
      2026-09-19, on 43 GiB free. Running; results pending.
      - Attempt 1 (drivers 25942–25950 -> children 25952–25963) **died of a full filesystem**:
        25952 crashed `OSError: [Errno 28]` and the other eleven children never started, because
        Slurm could not create their log files on a disk with 2.9 GiB free. Post-mortem and the
        disk arithmetic: runbook §3c.
      - **Group D excluded from attempt 2 on purpose.** 25951 (diffusion, K20) from attempt 1 was
        still RUNNING on that exact output path; a second job there would have corrupted it with
        no error from either side. It also cannot finish as designed — `dpcc-r` measures 6809 s
        per trial, so one variant is ~19 h and seven are ~130 h against a 24 h cap. It needs one
        job per variant (runbook §3d).
- [ ] group D re-run, restructured as one job per variant
- [ ] all 12 children landed; child IDs into the runbook's run map
- [ ] `DA_in_Paper/analysis/pillars_grid.py` with `GEO_PREFIX = 'pillars_xl'`
- [ ] `v3/withheld/20260918_uav_pillars_section.tex` restored, every number recomputed
