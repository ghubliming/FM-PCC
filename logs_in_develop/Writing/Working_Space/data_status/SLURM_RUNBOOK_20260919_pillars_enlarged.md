# SLURM RUNBOOK — UAV-pillars with the obstacles enlarged at test time

**Opened 2026-09-18 for the 19-09 wave.** Diagnosis and the geometry ladder:
[`PENDING_20260918_pillars_geometry_redesign.md`](PENDING_20260918_pillars_geometry_redesign.md).

> ✅ **IMPLEMENTED 2026-09-18 as Gen15 U17** — the config work of §1 is done and the driver of §3 is
> written. §2 is **not** done: it is a cluster run, and nothing else may be submitted until it reads
> clean. Changelog: [`logs_in_develop/Gen15/U17/CHANGELOG_20260918_pillars_enlarged.md`](../../../Gen15/U17/CHANGELOG_20260918_pillars_enlarged.md).
>
> | | |
> | :-- | :-- |
> | config | `config/uav_projection.yaml` — `pillars_xl` (R 0.35) and `pillars_xxl` (R 0.55) appended; `pillars_hg`, `active_geo_variants` and `enlarge_constraints: 0.025` byte-unchanged |
> | driver | `Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh` (plan-by-default) |
> | ⚠️ | `Slurm_Codes/temp_bash/` is **gitignored** — the driver does not arrive with a `git pull`. Copy it to the cluster by hand. |
>
> **Evaluation only.** No training job, no new checkpoints. The models are the ones already trained;
> only the constraint the projection enforces changes.

---

## 1. The config change to make first

One new entry in `config/uav_projection.yaml`, in `geo_constraint_variants`. **Copy the `pillars_hg`
block verbatim and change four things:**

| field | `pillars_hg` (leave it alone) | new entry |
| :-- | :-- | :-- |
| `name` | `pillars_hg` | `pillars_xl` |
| `geo_tag_suffix` | `'_hg'` | `'_xl'` |
| `obstacle_constraints[*].radius` | `0.12` (six entries) | **`0.35`** (all six) |
| everything else — `scene`, `constraint_types`, `planning_inflation` (`r_drone: 0.31`, `margin_base: 0.0`), `workspace_bounds` `lb [-4.0,-2.5,0.30]` / `ub [4.0,2.5,1.80]` | — | identical |

`enlarge_constraints` for pillars stays at **0.025**: the tightened variants of the new geometry mean the
same thing they mean everywhere else in the thesis. Do not use that knob to do the enlarging.

**Do not touch `scene_pillars.xml`.** The physical pillars stay at radius `0.12`. The enlargement is a
test-time constraint, exactly as D3IL-avoiding's halfspaces are, and the demonstrations must stay valid.

**`active_geo_variants`** currently reads
`['empty_no_constraint', 'corridor_hg', 'pillars_hg', 's_curve_hg']`. Per the warning in that file, a
scene's two entries must never be active in the same job — both would run and the job takes twice as
long, and pillars at $K{=}5$ already hits the wall clock. Either swap `pillars_hg` → `pillars_xl` there,
or leave the list alone and select per job with `UAV_MIX_GEO_VARIANTS=pillars_xl` (preferred — it keeps
the file honest for the other scenes).

An optional second rung, `pillars_xxl` with `radius: 0.55` and suffix `'_xxl'`, is specified in the
PENDING note. **Do not submit it in this wave.** Run `xl` first and look at the unprojected row.

> ✅ **Done, as specified.** Both entries are appended to `config/uav_projection.yaml`; the six
> `sphere_outside` radii are the only field that differs from `pillars_hg`, `active_geo_variants` is
> untouched, and every job selects its geometry with `UAV_MIX_GEO_VARIANTS` (the preferred option
> above). `pillars_xxl` is defined but is behind an `XXL_OK=1` gate in the driver.
>
> Point 2 of §1's three-things list — **the violation scorer must use the enlarged radius** — turns
> out to need **no code change, and this was verified rather than assumed**:
> `_exec_constraint_violations` (`mix_uav_test/eval_mix_uav.py:792`) reads
> `config['obstacle_constraints']` from the resolved geo entry and scores at `radius + r_drone`, so
> the enlarged radius is the yardstick as well as the tube. Point 3 likewise already holds:
> `phys_safe` is MuJoCo contact truth (`eval_artifacts.py:211`) and is radius-independent.
> §2 still runs — the code being right is not the same as the number coming back right.

## 2. Verify on ONE cell before submitting the wave

Two failure modes would make all 20 jobs worthless, and both are cheap to rule out. Run a single
evaluation — `mf`, `pillars`, seed 6, `K=5`, `UAV_MIX_GEO_VARIANTS=pillars_xl`, `UAV_MIX_VARIANTS=diffuser`
— and check:

1. **The unprojected plan must now violate.** Variant `diffuser` at `pillars_hg` reads S&C $0.90$ for
   `mf`. At `pillars_xl` it must drop well below that. If it is still $0.90$, the enlarged radius is not
   reaching the projector.
2. **The violation scorer must use the enlarged radius.** `n_violations` on that same row must be
   greater than zero. If the executed-violation check still scores against the physical $0.12$ pillars,
   every row in the wave will read collision-free and the experiment says nothing.

Expected geometry, for checking by hand: at a pillar the plan must keep $|y| \ge 0.6 + 0.35 + 0.31 =
1.26$; the demonstrated channels are at $|y| = 1.11$, so a plan that copies the demonstration is
**0.15 m inside the obstacle**; the centre channel is closed ($0.6 - 0.35 - 0.31 < 0$); free space runs
out to $|y| = 2.19$.

> This is **group V** of the driver, and it is the default wave, so the bare command runs exactly
> this one cell and nothing else:
>
> ```bash
> bash Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh           # PLAN — prints, submits nothing
> bash Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh submit    # the verify cell, alone
> ```
>
> It carries its own tag `u7xlchk`, so a throwaway check can never pool with the wave it is checking.
>
> **One warning is expected and must not be "fixed".** `_warn_expert_route_infeasibility`
> (`eval_mix_uav.py:904`) probes the expert routes against the planning set and will print
> `WARNING pillars homotopy=…: expert route violates the PLANNING constraint set …`. That is now the
> point of the geometry. The check is print-only and never blocks a run (:913).

## 3. The job matrix

Scene `pillars`, seeds `"6"`, `n_trials` = the yaml default (10 flights per cell), geometry
`pillars_xl`, the eleven projection configurations the scene already uses (`diffuser`; `dpcc-{r,c,t}`
plain and tightened; `hardflow_sls` and its three rules). `geo_free` variants are not part of the thesis
and are not requested.

**10 driver jobs.** `eval_k_sweep.sh` fans out to one child job per K, so the child count is larger.

| # | engine | K list | checkpoint / arm | children |
| --: | :-- | :-- | :-- | --: |
| 1 | `mf` | `"1 2 5"` | `MeanFlowODE_9D_dp0.5_bbunet` | 3 |
| 2 | `fm` | `"1 2 5"` | `FlowMatchingODE_9D` (U-Net; the folder carries no `bb` suffix) | 3 |
| 3 | `af` | `"1 2 5"` | `AlphaFlowODE_9D_as1_ae0.2_bbunet`, `UAV_MIX_AF_ALPHA_END=0.2`, `UAV_MIX_BONE_AF=unet`, epoch `EPlatest` | 3 |
| 4 | `diffusion` | `"20"` | `GaussianDiffusion_9D_K20` | 1 |

$K=20$ is the baseline's training budget and its only one. $K=1,2,5$ is the ladder the other scenes use,
and it closes the two holes the old geometry never had — `mf` and `fm` at $K=1$.

```bash
# PLAN first, every time.
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf        pillars "6" "1 2 5"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh fm        pillars "6" "1 2 5"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh af        pillars "6" "1 2 5"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh diffusion pillars "6" "20"
```

with, exported for every one of them:

```
UAV_MIX_GEO_VARIANTS=pillars_xl
FMPCC_UAV_EVAL_TAG=u7xl            # the tag the DA selects on; must not collide with u7hg
```

and for the `af` jobs additionally `UAV_MIX_BONE_AF=unet`, `UAV_MIX_AF_ALPHA_END=0.2`.

**Wall clock.** The `pillars_hg` jobs at $K=5$ already reached the 24 h limit with the per-step variants,
and the enlarged constraint will make the solver work harder, not less. Give the $K=5$ children a longer
`--time` or split their variant list with `UAV_MIX_VARIANTS` (per-step in one job, `hardflow_sls*` in
another) rather than discovering the wall at hour 24.

---

### 3a. As implemented — `eval_20260919_u17_pillars_xl.sh`

```bash
WAVE="A B C D" bash Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh           # PLAN
WAVE="A B C D" bash Slurm_Codes/temp_bash/eval_20260919_u17_pillars_xl.sh submit    # the wave
```

**10 driver jobs → 13 children**, as specified above, with the $K{=}5$ variant list split per the
wall-clock warning. Everything below is set by the driver, not by the operator.

| grp | engines | K | variants | children | tag |
| :-- | :-- | :-- | --: | --: | :-- |
| **V** | mf | 5 | `diffuser` only — **run first, alone** | 1 | `u7xlchk` |
| A | fm, mf, af | 1, 2 | 7 per-step | 6 | `u7xl` |
| B | fm, mf, af | 5 | 5 per-step (half 1) | 3 | `u7xl` |
| C | fm, mf, af | 5 | 6 endpoint + `dpcc-t` (half 2) | 3 | `u7xl` |
| D | diffusion | 20 (training-fixed) | 7 per-step | 1 | `u7xl` |
| E *(opt-in, `WAVE=E`)* | fm, mf, af | 2 | endpoint at `A=1.0` | 3 | `u7xla1` |
| X *(opt-in, needs `XXL_OK=1`)* | fm, mf, af | 5 | `pillars_xxl` | 3 | `u7xxl` |

The eleven configurations are `diffuser`, `dpcc-{r,c,t}` plain and `-tightened`, and
`hardflow_new{,-r,-c,-t}` (`hardflow_sls` is `hardflow_new` in this tree). `*-geo_free` rows are not
part of the thesis and are not run.

Three things the driver does that the §3 sketch does not, each with a reason:

- **$K{=}5$ is two jobs, B and C, with *disjoint* variant lists.** Disjoint matters: two concurrent
  jobs sharing a variant name would race on the same results folder. C keeps the `dpcc-t` pair as its
  non-HardFlow companion, which the eval requires — it refuses a HardFlow-only subset.
- **Groups A and B carry no endpoint rows.** HardFlow is degenerate at $K{=}1$ (the only step is the
  terminal one) and at $K{=}2$ under the shipped $A{=}0.5$; the eval drops those variants and writes
  `HF_DEGENERATE_SKIPPED.txt`. Endpoint at $K{=}2$ needs $A{=}1.0$ — that is **group E**, under its
  own tag, because $A{=}1.0$ rows are not interchangeable with group C's $A{=}0.5$ rows. Group E is
  the 18-Sep group D re-run on the new geometry (§5).
- **Group D has no K list**, and no endpoint half: $K$ is a training property for DPCC diffusion, and
  `engine_registry` sets `supports_hardflow=False` for ddpm (endpoint projection needs a velocity
  field). D does close the `dpcc-r` / `dpcc-r-tightened` gap the baseline had on the old geometry.

The driver's pre-flight refuses to submit unless `enlarge_constraints` is still `0.025`, the six MJCF
pillars are still `size="0.12 …"`, `pillars_xl` is **not** in `active_geo_variants`, and
`_exec_constraint_violations` still reads `config['obstacle_constraints']`.

### 3b. Run map — job IDs

Filled in from the submission output. `eval_k_sweep.sh` prints one child ID per K.

| grp | job | driver ID | child IDs | submitted | status |
| :-- | :-- | :-- | :-- | :-- | :-- |
| V | mf · K5 · diffuser | | | | ⬜ not yet submitted |
| A | fm · K1,2 | | | | ⬜ |
| A | mf · K1,2 | | | | ⬜ |
| A | af · K1,2 | | | | ⬜ |
| B | fm · K5 half 1 | | | | ⬜ |
| B | mf · K5 half 1 | | | | ⬜ |
| B | af · K5 half 1 | | | | ⬜ |
| C | fm · K5 half 2 | | | | ⬜ |
| C | mf · K5 half 2 | | | | ⬜ |
| C | af · K5 half 2 | | | | ⬜ |
| D | diffusion · K20 | | *(direct eval job, no children)* | | ⬜ |

## 4. What "done" looks like

A batch in which, for geometry `pillars_xl`, tag `u7xl`, seed 6:

- every one of the 10 model–budget cells exists with all eleven variants;
- the **unprojected** (`diffuser`) rows show real failure — that is the whole point of the wave, and if
  they do not, stop and raise the radius rather than reporting the result;
- `n_violations` is non-zero wherever S&C is below 1;
- the diffusion baseline has `dpcc-r` and `dpcc-r-tightened` this time — they were missing on the old
  geometry and left a gap in the baseline's row.

Then `DA_in_Paper/analysis/pillars_grid.py` runs against the new batch with `GEO_PREFIX = 'pillars_xl'`,
and `v3/withheld/20260918_uav_pillars_section.tex` is restored with every number recomputed.

## 5. Jobs to cancel from the 18-Sep wave

Groups C, D and E of [`SLURM_RUNBOOK_20260918_pending_runs.md`](SLURM_RUNBOOK_20260918_pending_runs.md)
are all `pillars_hg` work that fills a table this draft no longer prints:

```bash
scancel 25890 25891 25892 25893 25894 25895     # only those still pending or running
```

If they have already finished, keep the results — they belong to the control condition and cost nothing.
Groups A, B, F, G and H of that wave do not touch pillars and are unaffected.
