# PENDING — UAV-pillars tests the wrong thing, and has to be re-evaluated with the obstacles enlarged

**2026-09-18 · opened by the v3 author's question on `fig:expert-uav`.** Supersedes the UAV-pillars half
of [`PENDING_20260918_verified_data_audit.md`](PENDING_20260918_verified_data_audit.md) §C.
Runbook: [`SLURM_RUNBOOK_20260919_pillars_enlarged.md`](SLURM_RUNBOOK_20260919_pillars_enlarged.md).

> # ⛔ OUTCOME 2026-09-22: the redesign was run in full and ABANDONED. `pillars_xl` returned S&C = 0.00 in all 94 cells — every projector routes into the forbidden centre corridor between the pillar rows (which the 0.66 m keep-out closes by only 6 cm). **The thesis falls back to `pillars_hg` with the §1 caveat stated in the prose.** Closure: [`Gen15/U17/CLOSURE_20260922_U17_abandoned.md`](../../../Gen15/U17/CLOSURE_20260922_U17_abandoned.md). The diagnosis in §1–§2 below remains correct; the fix in §3 did not work.
>
> ✅ **IMPLEMENTED 2026-09-18 as Gen15 U17** — config + driver only; no Python touched, no model
> retrained, no MJCF edited. `pillars_xl` (R 0.35) and `pillars_xxl` (R 0.55) are in
> `config/uav_projection.yaml`; `pillars_hg` is kept untouched as the control condition. Driver:
> `Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh` (gitignored — copy to the cluster by hand).
> Changelog: [`logs_in_develop/Gen15/U17/CHANGELOG_20260918_pillars_enlarged.md`](../../../Gen15/U17/CHANGELOG_20260918_pillars_enlarged.md).
> ✅ **2026-09-19 — the §2 verify cell has RUN and the redesign works.** Tag `u7xlchk`, geometry
> `pillars_xl`, analytic average-velocity matching at $\nfe=5$, ten flights, variant `diffuser`
> (unprojected): **S&C 0.00 with 51–53 violating control steps per flight**, against **0.90** for the
> same unprojected cell on `pillars_hg`. The enlarged radius reaches both the projector and the
> violation scorer, so warnings 2 and 3 below are cleared on this cell. The demonstrated reference
> paths now violate the set by $0.15$ m on all four routes, recomputed by
> `DA_in_Paper/plotting/extract/expert_paths.py`. **The full wave of §6 is still to be submitted.**
> The diagnosis below stands unchanged.

---

## 1. The finding

**The constraint set UAV-pillars is evaluated against is the one the demonstration generator was built to
satisfy.** The demonstrated channels are not a free choice — they are derived from the constraint:

```
uav_expert_data_collect/trajectories.py:48
    _Y_L = PILLAR_Y_A - PILLAR_RADIUS - PILLAR_ROTOR_REACH - PILLAR_SAFETY   # = -1.11
```

and the projection then enforces `|y| >= 0.6 + 0.12 + 0.31 = 1.03` at a pillar. The routes fly at
`|y| = 1.11`, i.e. **8 cm inside the feasible set before any projection runs**. The consequences are
visible in every number the scene produced:

| evidence | value | source |
| :-- | :-- | :-- |
| demonstrated routes satisfying the constraint set, tightening included | **4 of 4** | `fig:expert-uav`, `data/expert_paths.json` |
| minimum clearance of a demonstrated route to a pillar surface | $0.39$\,m, against the $0.31$\,m vehicle radius | same |
| unprojected plan, instantaneous-velocity matching, $\nfe=5$ | **S&C $1.00$**, 0 violating steps | `tab:uav-pillars`, variant `diffuser` |
| unprojected plan, analytic average-velocity matching, $\nfe=5$ | **S&C $0.90$** | same |

So the plan is feasible before the projection touches it, and what the projected configurations measure
is **how little of an already-feasible plan each method disturbs** — not whether a method can make an
infeasible plan feasible. Several results of the scene are artefacts of that: per-step projection with
random selection *destroys* a clean plan (FM $1.00 \to 0.00$), and endpoint projection "wins" mostly by
leaving the plan alone.

**This is why the models could not be ordered on this scene**, and it is not a property of the models.

## 2. Why the other two scenes do not have this problem

D3IL-avoiding is built the opposite way: the halfspaces are introduced **at test time** and 0, 1 and 2 of
the 96 demonstrations satisfy the three geometries. UAV-corridor gets the same thing from the slide, which
did not exist when the demonstrations were flown — all three lanes cross it, by $0.24$, $0.36$ and
$0.47$\,m (`fig:expert-uav`). UAV-pillars is the only scene whose constraint the data already respects.

## 3. The fix: enlarge the obstacle at test time

**Keep the physical scene, the demonstrations and the checkpoints. Change only the radius the projection
enforces.** No retraining. This reproduces exactly the construction D3IL-avoiding uses.

Geometry at a pillar, with constraint radius $R$ and vehicle radius $0.31$:

| quantity | formula | at $R=0.12$ (today) |
| :-- | :-- | :-- |
| outer channel opens at | $\lvert y\rvert \ge 0.6 + R + 0.31$ | $1.03$ |
| demonstrated route flies at | $\lvert y\rvert = 1.11$ | feasible by $0.08$ |
| centre channel open while | $R \le 0.29$ | open, $\lvert y\rvert \le 0.17$ |
| outermost usable $\lvert y\rvert$ (workspace $\pm2.5$) | $2.5 - 0.31$ | $2.19$ |

**The demonstrated route becomes infeasible for $R > 0.20$.** Proposed ladder, in the order to run it:

| entry | $R$ | demonstrated route violates by | centre channel | detour needed to | note |
| :-- | --: | --: | :-- | --: | :-- |
| `pillars_hg` | 0.12 | — (feasible) | open | — | **control, already run.** Keep as the contrast |
| **`pillars_xl`** | **0.35** | $0.15$\,m | closed | $\lvert y\rvert \ge 1.26$ | **primary.** Comparable to the corridor slide's $0.24$–$0.47$\,m |
| `pillars_xxl` | 0.55 | $0.35$\,m | closed | $\lvert y\rvert \ge 1.46$ | optional second rung, only if `xl` turns out too easy |

Both remain solvable: the free corridor between the required detour and the workspace edge is
$0.93$\,m wide at $R=0.35$ and $0.73$\,m at $R=0.55$.

### Three things the implementer must get right

1. **Enlarge the radius in the geometry entry, not `enlarge_constraints`.** The tightening knob
   ($0.025$) is a reported variable of the thesis — every table has a *tightened* column — and
   overloading it would make the two indistinguishable. Add new `geo_constraint_variants` entries with
   their own `geo_tag_suffix`, exactly as `pillars_hg` does; copy that block and change `radius` on the
   six `sphere_outside` entries plus the suffix and name. **The tightened variants of the new geometry
   still use $0.025$ on top**, as everywhere else.
2. **The violation scorer must use the enlarged radius too.** If the executed-violation check keeps
   scoring against the physical $0.12$ pillars, every row will read collision-free and the experiment
   says nothing. Verify on one cell before submitting the wave.
3. **Physical contact and the enlarged constraint are now different things.** The pillars in the MJCF
   stay at $0.12$, so a flight can satisfy physics and violate the constraint. That is intended and is
   what D3IL-avoiding does; the thesis reports constraint satisfaction, and `phys_safe` stays a separate
   column. Do **not** enlarge the MJCF pillars — that would invalidate the demonstrations and force
   retraining.

## 4. What this costs and what it buys

Evaluation only, one seed, ten flights per cell, the same eleven projection configurations the scene
already uses. It gives UAV-pillars the thing it is in the thesis for: a scene where the generative model
must produce a plan that is *not* the demonstrated one, and where the projection has to do real work —
so the four models can be ordered on something other than how little they disturb a plan that was
already correct.

## 5. Meanwhile, in the draft

`sec:res:uav:pillars` was **deleted** at v3.41 (withheld at v3.40). The section is replaced by a short statement of the
finding above and a `\hole` pointing here. The withdrawn text — three tables and the flown-path figure —
is kept verbatim at `v3/withheld/20260918_uav_pillars_section.tex` and is restored, with every number
recomputed, once the new evaluations land. Downstream: the UAV conclusion, both cross-environment summary
tables and the selected-combination table read *withheld* for this scene, and the projection comparison
for it is withdrawn with it. The draft now rests on UAV-corridor for the quadrotor claim.

---

## 6. Separately: figures waiting on a DOWNLOAD, not on a run

These are ready on the cluster and need fetching. None of them needs a job.

| figure | what to fetch | from |
| :-- | :-- | :-- |
| `fig:raw-plans` (Fig 6.3), FM $\nfe=1$ | `…/6/both-hard/…/diffuser.png` | `logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.FlowMatchingODE_a1.5_b1.0_aw10/H8_K1_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials` (seeds 6–10) |
| `fig:raw-plans`, FM $\nfe=2$ | same | same model folder, `H8_K2_Meuler_T0.5_…_msg20trials` |
| `fig:raw-plans`, CI-MeanFM $\nfe=1$ | same | `…flow_matcher_v3_alphaflow.models.AlphaFlowODE_msgafon02_s6`, K1 (seed 6) |
| `fig:raw-plans`, CI-MeanFM $\nfe=2$ | same | same, K2 (seed 6) |
| `sec:res:aligning:projection` — the ten alignment contexts | the executed end-effector paths | staged before a variant-naming bug in the stager was fixed (`_train_set` suffix); needs one more run of the **stager**, not of an evaluation |

**`fig:raw-plans`, diffusion $\nfe=2$: cannot be fetched and cannot be run as specified.** Every
`GaussianDiffusion` folder in the batch is `K20`, because a diffusion model's step count is fixed when its
noise schedule is discretised at training time. That cell stays empty unless a $K{=}2$ diffusion model is
trained. Details and the crop box: `DA_in_Paper/plotting/REQUEST_20260916_cluster_fm_plan_panels.md`.

Ledger row **R5** carries this.

## 7. Jobs of the 18-Sep wave that this makes pointless

From [`SLURM_RUNBOOK_20260918_pending_runs.md`](SLURM_RUNBOOK_20260918_pending_runs.md). **Cancel these if
they are still queued or running**; if they have already finished, leave the results, they cost nothing to
keep as part of the control condition.

| group | jobs | what it was for | why it is moot |
| :-- | :-- | :-- | :-- |
| C | **25890** (fm), **25891** (mf) | UAV-pillars, FM and MeanFM at $\nfe=1$, ten per-step configurations, `pillars_hg` | fills a hole in a table that is now withheld. The control condition needs no $\nfe=1$ row |
| D | **25892** (fm), **25893** (mf), **25894** (af) | UAV-pillars endpoint projection at $\nfe=2$, $A=1.0$, `pillars_hg` | same table; endpoint at $\nfe=2$ has to be re-run on the new geometry regardless |
| E | **25895** | UAV-pillars diffusion baseline, `dpcc-r` and `dpcc-r-tightened`, `pillars_hg` | completes the baseline's row on the old geometry; the baseline must be run in full on the new one anyway |

```bash
scancel 25890 25891 25892 25893 25894 25895     # only those still pending/running
```

Groups A, B, F, G, H of that runbook are **not** affected — none of them touches pillars.
