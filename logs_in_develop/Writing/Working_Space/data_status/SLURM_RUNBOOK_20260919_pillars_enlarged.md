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

Driver file on the cluster: `Slurm_Codes/temp_bash/Gen15_U17.sh` (renamed from
`eval_20260919_u17_pillars_xl.sh` on copy). Each driver's child ID is printed in its own log,
`Slurm_Codes/logs/<date>/<time>_eval_k_sweep_<driver>.log`.

| grp | job | driver ID | child IDs | submitted | status |
| :-- | :-- | :-- | :-- | :-- | :-- |
| V | mf · K5 · diffuser | **25923** | **25940** | 2026-09-19 | 🟢 COMPLETED 09:24:48–09:29:22 UTC · geometry confirmed live; flown-path numbers pending (see below) |

**Group V, what job 25940 established.** Git rev `b8f1beb2`, `EVAL_TAG=u7xlchk`,
`UAV_MIX_GEO_VARIANTS=pillars_xl`, `UAV_MIX_VARIANTS=diffuser`, `n_trials=10` from the yaml.

- ✅ **The entry is live and complete.** `E9 geo 'pillars' ← variant 'pillars_xl':
  constraint_types=[dynamics, geo_bounds, obstacles, bounds] (bounds=True, hs=0, obs=6)` — all six
  enlarged pillars loaded; results land in their own folder
  `pillars_xl_bounds+dynamics+geo_bounds+obstacles/`, which cannot pool with `_hg`.
- ✅ **The enlarged radius reaches the scoring code path — proven, not inferred.** All four
  homotopies now trip the expert-route probe:

  | homotopy | violating samples | total penetration |
  | :-- | --: | --: |
  | (L,L,L) | 52/200 | 4.73 m·samples |
  | (L,R,L) | 54/200 | 4.20 |
  | (R,L,R) | 54/200 | 4.20 |
  | (R,R,R) | 52/200 | 4.73 |

  Under `pillars_hg` these same routes **passed** (feasible by 0.08 m). The probe
  (`_warn_expert_route_infeasibility`) computes this by calling `_exec_constraint_violations` — the
  violation scorer itself — so the enlarged radius is demonstrably reaching the scorer, not only the
  projector. Mean penetration over violating samples is 4.73/52 = 0.091 m against a predicted
  maximum of 0.15 m, which is what a route that cuts the corner at three pillar columns should give.
  This is the §2 warning that was predicted, and it is the intended behaviour.
- ✅ **The virtual enlargement did not touch physics.** `safe=1.000` — every flight is contact-free
  in MuJoCo, exactly as designed: the MJCF pillars are still r=0.12 and `phys_safe` is independent
  of the constraint radius.
- ✅ **The policy is unharmed.** `success=0.900`, `success_relaxed=1.000`, `track_err=0.446`,
  `steps_to_goal=424/634`. The plan still reaches the goal; it is now merely *illegal*. That is the
  condition the scene was rebuilt to create.
- ✅ **§2 PASSED on both gates.** From `…/diffuser/results.json`:

  | metric | `pillars_hg` | **`pillars_xl`** | gate |
  | :-- | --: | --: | :-- |
  | `success.strict_and_constraints_rate` (S&C) | 0.90 | **0.00** | ① "well below 0.90" ✅ |
  | `constraint.n_violations_mean` | 0 | **68.7** | ② "> 0" ✅ |
  | `constraint.collision_free_rate` | 1.00 | **0.00** | 0/10 rollouts legal |
  | `constraint.total_violations_mean` | 0 | 9.80 m·steps | |
  | `success.strict_rate` (plain success) | 0.90 | 0.90 | **unchanged — see below** |
  | `physical.safe_rate` | 1.00 | 1.00 | physics untouched ✅ |

  **The control is exact.** `diffuser` is unprojected, so with the same seed and checkpoint these
  are the *same flights* as the `pillars_hg` rows — only the yardstick changed. `strict_rate` is
  0.90 in both, confirming it. So the entire S&C collapse 0.90 → 0.00 is attributable to the
  constraint and to nothing else, and `pillars_hg`'s S&C of 0.90 was plain success with
  `collision_free = 1.0` on every rollout. That is the artefact the U17 redesign set out to remove.

  **Per-rollout structure.** 9 of 10 rollouts: 51–53 violating steps out of ~424, penetration
  3.3–3.6 m·steps (≈ 0.066 m mean per violating step, against the 0.15 m predicted maximum — the
  route clips each of the three pillar columns for ~17 steps). The 10th (`rollout_8`) is the one
  failed flight: 223 violations, 66.7 m·steps, goal not reached, and the only rollout whose flown
  homotopy is (R,R,R). It alone lifts `n_violations_mean` to 68.7 from a median of ~52.

  **Incidental, and already known:** `homotopy_flown` is (L,L,L) on 9 of 10 rollouts whatever
  homotopy was commanded — the policy is unconditioned and does not track the commanded route
  (Fix_12). Not a U17 effect; it is why `homotopy_flown` exists.
**ATTEMPT 2 — the live wave.** Resubmitted 2026-09-19 after the disk was cleared to 43 GiB free
(attempt 1 died of `Errno 28`; its IDs and post-mortem are in §3c). Group D was **deliberately
excluded** — 25951 from attempt 1 was still RUNNING on that exact output path, and a second job
there would have corrupted it silently.

| grp | job | driver ID | child IDs | submitted | status |
| :-- | :-- | :-- | :-- | :-- | :-- |
| A | fm · K1,2 | 25970 | **25988** (K1), **25990** (K2) | 2026-09-19 | 🟢 both COMPLETE, 7/7 variants — K1 ended 20-09 04:08, K2 20-09 13:01 UTC |
| A | mf · K1,2 | 25971 | **25984** (K1), **25985** (K2) | 2026-09-19 | 🟢 both COMPLETE, 7/7 — 19-09 20:55 / 22:20 UTC |
| A | af · K1,2 | 25972 | **25986** (K1), **25987** (K2) | 2026-09-19 | 🟢 both COMPLETE, 7/7 — 19-09 23:54 / 20-09 01:15 UTC |
| B | fm · K5 half 1 | 25973 | **25989** | 2026-09-19 | 🟢 COMPLETE, 5/5 — 20-09 22:31 UTC |
| B | mf · K5 half 1 | 25974 | **25991** | 2026-09-19 | 🟢 COMPLETE, 5/5 — 21-09 07:44 UTC |
| B | af · K5 half 1 | 25975 | **25992** | 2026-09-19 | 🟢 COMPLETE, 5/5 — 21-09 17:26 UTC |
| C | fm · K5 half 2 | 25976 | **25994** | 2026-09-19 | 🟠 **RUNNING** (as of 21-09 ~21:50): variant 1/6 `dpcc-t`, trial 7/10 at 15,471 s. ~2,210 s/trial → ~6 h per DPCC variant; the four HardFlow variants are minutes each. **Expected to finish inside 24 h; not a resubmit** |
| C | mf · K5 half 2 | 25977 | **25993** | 2026-09-19 | 🟢 COMPLETE, 6/6 — 21-09 20:33 UTC |
| C | af · K5 half 2 | 25978 | **25995** | 2026-09-19 | 🟠 **RUNNING** — started 21-09 20:33:45 UTC (queued behind 25993), `dpcc-t` trial 2/10. Same ~13 h profile as 25993 → expected done ~22-09 09:30 UTC. **Not a resubmit** |
| D | diffusion · K20 | 25951 *(attempt 1)* | *(direct eval job)* | 2026-09-19 | 🔴 **CANCELLED AT TIME LIMIT** 20-09 11:18:55 UTC. Delivered `diffuser` (10/10) and `dpcc-r` (10/10, 67,033 s = 18.6 h for that one variant); `dpcc-r-tightened` stopped at trial 2/10 → **partial cell on disk, must be deleted before any diffusion re-run**. Five variants never started |

9 drivers → 12 children. Child IDs are printed in each driver's own log,
`Slurm_Codes/logs/2026-09-19/13_*_eval_k_sweep_<driver>.log`.

Before resubmitting, the one piece of real state from attempt 1 was removed —
`rm -rf logs/UAV_MIX/uav-pillars/plans/mix_uav_af/*/Eaf_K1_*_u7xl` (the three complete `dpcc-c`
trials 25952 wrote before it crashed). Everything else from attempt 1 produced no output at all.

**Why a plain resubmit is safe here.** `eval_mix_uav.py` has **no cell-level resume**: it runs
every variant it is given and writes `results.json` with `'w'` (:2392), and episode ids are
deterministic (`10_000 + i`, :2257), so a re-run rewrites byte-identical filenames. "Resume"
therefore always means "re-run and overwrite" — safe against one's own dead output, and unsafe
only against a job still writing to the same path. Hence the exclusion of D.

**ATTEMPT 1 — dead, retained for the record.** Drivers 25942–25950 → children 25952–25963.
All nine drivers worked correctly (10-minute thin submitters, clean logs); the failure was
entirely in the children. 25952 (af K1) crashed mid-run with `Errno 28`; 25953–25963 never
started and left no log file at all.

---

### 3c. 🔴 POST-MORTEM — the 19-09 wave died of a full filesystem

**Root cause, from the one child that left a log.** `13_18_34_uav_mix_eval_25952.log`
(af, K=1), at variant 4/7 `dpcc-c`, trial 4/10, 11:59:51 UTC:

```
OSError: [Errno 28] No space left on device:
  'logs/UAV_MIX/.../Eaf_K1_..._u7xl/6/pillars_xl_.../dpcc-c/rollout_pillars_(R,R,R)_10003.log'
```

**The other eleven children left no log file at all.** A job that starts always creates its
output file, so this is consistent with one thing: on a full filesystem Slurm cannot create the
log and the job fails at startup. All twelve children were launched at 11:18:35–11:18:36 UTC
against a filesystem with **2.9 GiB free**, and the disk was gone inside 40 minutes.

**The warning was in this runbook and was not acted on** (the 2.9 GiB note above). The estimate
there — ~0.8 GiB for the wave — was **too low**, and the reason is now measurable: it was scaled
from the group V `diffuser` cell (9.7 MiB), which runs with **no projector**. A projected variant
writes far more per rollout. Treat 9.7 MiB/variant as a floor, not an average, when sizing the
re-run.

**Nothing in the wave is salvageable, and the partial tree is actively dangerous.** 25952 wrote
three complete `dpcc-c` trials and one truncated rollout log before dying, so the `u7xl` tree now
contains half-written cells that a DA pass would happily average. **Delete the whole `u7xl` tree
for the af arm before resubmitting**, and check the other arms for stubs:

```bash
du -sh logs/UAV_MIX/uav-pillars/plans/mix_uav_*/*/E*_u7xl          # what exists
rm -rf  logs/UAV_MIX/uav-pillars/plans/mix_uav_af/*/Eaf_K1_*_u7xl  # the crashed cell
```

`u7xlchk` (group V) is **untouched and remains valid** — it completed hours earlier.

### 3d. ✅ Group D (25951) — CLOSED BY DECISION, 2026-09-21: the diffusion baseline is reported UNPROJECTED only

**Decision (author, 2026-09-21).** The diffusion K=20 arm on UAV-pillars is **not re-run**. The thesis
reports the baseline on this scene from its **raw network output only** — variant `diffuser`, no
projection of any kind — and states that per-step projection at K=20 is not evaluable within the
cluster's 24 h job limit. The delivered `dpcc-r` cell stays on disk but is **not printed**: a single
projected rule with no `c`/`t`/tightened siblings would be an incomplete row, not a comparison.
The 2/10 partial `dpcc-r-tightened` cell is to be deleted:

```bash
rm -rf logs/UAV_MIX/uav-pillars/plans/mix_uav_diffusion/*/Ediffusion_K20_*_u7xl/6/pillars_xl_*/dpcc-r-tightened
```

**The numbers that make it infeasible** — all measured on 25951, none estimated:

| quantity | value | source |
| :-- | --: | :-- |
| `proj_ms` per control step, `dpcc-r`, K=20 | **10,387 ms** | `TIMING` line |
| same, flow models at K=5 (for scale) | 2,272–3,355 ms | 25989/25991/25992 |
| same, flow models at K=1 / K=2 | 122–265 / 96–162 ms | group A |
| one 10-flight variant, `dpcc-r` | **67,033 s = 18.6 h** | trial 10/10 stamp |
| one 10-flight variant, `dpcc-r-tightened` (2 trials measured) | 6,588 s/trial → **18.3 h** | trial 2/10 stamp |
| `c` and `t` rules relative to `r`, measured at K=5 | ×1.18–1.32 (+6% tightened) | fm/mf K=5 |
| → `dpcc-c`, `dpcc-c-tightened` at K=20, projected | **~24.4 h, ~26 h** | exceeds the cap on their own |
| six per-step variants, ten flights each | **~118–130 h ≈ 5 GPU-days** | sum |
| job `--time` cap | **24 h** | cluster rule |
| endpoint projection for this arm | **unavailable by construction** | `engine_registry`: `supports_hardflow=False` for ddpm (no velocity field) |

So the per-step block for the baseline cannot be completed inside the protocol: one variant per 24 h
job fits only for the `r` rules; the `c` rules do not fit at all without cutting `n_trials` below the
ten flights every other row uses, and the eval has no trial-offset option to shard a variant across
jobs (episode ids are `10_000 + i`, so two half-jobs would overwrite each other). The reason is the
budget itself: projection cost scales with K, and K=20 is the baseline's *training* step count, not a
choice — no cheaper pillars checkpoint exists.

**What the paper says, with the delivered numbers.** Diffusion K=20, `pillars_xl`, seed 6, ten
flights, raw output: `success = 0.000`, `success_relaxed = 1.000`, `safe = 1.000`,
`goal_reached = 0.000`, `track_err = 0.319`. The baseline crosses the goal line on every flight,
contact-free, and never comes within 0.30 m of the goal point. Its S&C is 0 before any constraint is
applied. (For the record only, not for print: the delivered `dpcc-r` cell also reads `success = 0.000`.)

What 25951 actually did, kept for history:

| | measured |
| :-- | --: |
| `dpcc-r` at K=20, per trial | **6809 s** (~1.9 h) |
| × 10 trials, one variant | **~19 h** |
| × 7 variants | **~130 h** |
| job `--time` cap | **24 h** |

At 6 h it had finished variant 1 (`diffuser`) and was on trial 3/10 of variant 2. It will wall out
having produced **two of seven variants**. Projection cost scales hard with K and K=20 is four
times the K=5 budget that already walled the `pillars_hg` jobs.

~~Group D must be restructured as one job per variant~~ — **superseded by the decision above.** The
restructure was costed (one variant per job, ~19–26 h each, ~5 GPU-days) and rejected: the baseline
reads 0.000 unprojected, every delivered projected cell on this scene reads 0.00 (§3e), and the
scene's scoring is itself under review. Compute here would be spent twice.

**Also recorded from 25951, and it matters for the scene's claim:** the diffusion baseline's
*unprojected* row reads `success=0.000`, `goal_reached=0.000`, `success_relaxed=1.000`,
`track_err=0.319`. The baseline gets near the goal but never inside the 0.30 m radius, so its S&C
is 0 before any constraint is applied. This is a property of the baseline on pillars, not of the
`_xl` geometry (unprojected rollouts are geometry-independent) — but it means the
"beat the diffusion baseline" comparison on this scene cannot rest on the unprojected row.

> ⚠️ **DISK — watch this before the wave lands.** `/u/home` was at **2.9 GiB free** at submission,
> down from 4.9 GiB the previous day. The group V cell wrote **9.7 MiB for a single variant**
> (9.1 MiB of it the `diffuser` folder: `diffuser.npz`, per-rollout logs, and a
> `*_mpc_foresight.svg` + `*_stats.json` per rollout under `diagnostics/`). The wave is
> **13 children × 5–7 variants ≈ 80 variant folders ≈ 0.8 GiB**, plus the diffusion baseline at
> K=20. That fits in 2.9 GiB, but not with much room, and a full disk mid-wave corrupts whatever
> job is mid-write. Free space before the K=5 jobs (B/C) start, or be ready to.

### 3e. 🔴 FIRST READ OF THE WAVE, 2026-09-21 — sanity check before any DA

**Status.** 10 of 12 attempt-2 children complete, 2 running and on track, diffusion walled at 2/7
variants. No crash, no disk event, every finished cell has all its variants. **As a run, the wave is
clean.** As a *result*, it is a floor, and that has to be said before anyone builds a table on it.

**The numbers, straight from the eval's per-variant summary lines.** These print `success` (goal
reached within 0.30 m **and** contact-free) and `safe` (contact-free); S&C is not printed, but
S&C ≤ `success` by definition, so **every 0.00 below is an S&C of 0.00**.

*Per-step block — `success`, geometry `pillars_xl`, tag `u7xl`, seed 6, 10 flights per cell:*

| model | $\nfe$ | unproj. | $r$ | $r$ tight | $c$ | $c$ tight | $t$ | $t$ tight |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| MeanFM (mf) | 1 | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MeanFM | 2 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| MeanFM | 5 | **0.90** | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CI-MeanFM (af) | 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CI-MeanFM | 2 | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| CI-MeanFM | 5 | 0.00 | 0.10 | 0.00 | 0.00 | 0.00 | *(running)* | *(running)* |
| FM (fm) | 1 | **0.80** | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FM | 2 | 0.30 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| FM | 5 | **0.90** | 0.00 | 0.00 | 0.00 | 0.00 | *(running)* | *(running)* |
| Diffusion | 20 | 0.00 | 0.00 | *(partial)* | — | — | — | — |

*Endpoint block at $\nfe=5$ (`hardflow_new` single / $r$ / $c$ / $t$):* MeanFM **0.00 / 0.00 / 0.00 / 0.00**.
fm and af endpoint rows are in the two running jobs.

**Read against `pillars_hg`** (the withheld tables, `v3/withheld/20260918_uav_pillars_section.tex`):
at `_hg`, $\nfe=5$ per-step reached S&C 0.80–1.00 (FM $c$ 1.00, MeanFM $c$-tight 0.90, CI-MeanFM
$t$-tight 0.90) and endpoint 0.90–1.00. At `_xl` the same cells read **0.00–0.10**. Across all
finished projected cells — 36 at $\nfe\le2$, 14 at $\nfe=5$, 4 endpoint — **53 of 54 are exactly 0.00**
and the 54th is 0.10. The unprojected rows, which are geometry-independent, reproduce their `_hg`
values (mf/fm 0.90 at $\nfe=5$, as the verify cell already showed).

**So the scene flipped from one degenerate regime to the other.** `_hg`: every plan already feasible,
projector had nothing to repair, models could not be ordered. `_xl`: projector repairs every plan
into one that **never reaches the goal**, models cannot be ordered. Neither version orders the four
models, which is the one thing the section exists for. **This wave, as it stands, cannot restore
`sec:res:uav:pillars`.**

**Two more signals in the logs, both unexplained without the results tree:**

1. At $\nfe\le2$ every projected row is `success=0.00, success_relaxed=1.00, safe=1.00` — the drone
   **crosses the goal line, contact-free, but ends more than 0.30 m from the goal point.** That is a
   *lateral* miss (altitude is locked: `actions[2]` is degenerate, ±3.1e-05 m/step).
2. At $\nfe=5$ `safe` collapses on the per-step rows — fm `dpcc-r` **0.00**, `dpcc-r`-tight 0.00, mf
   `dpcc-c` 0.00, af `dpcc-c` 0.40 — meaning **MuJoCo contact in most or all flights**, which the
   $\nfe\le2$ rows never show. A plan obeying $|y|\ge1.26$ cannot touch an $r=0.12$ pillar at
   $y=\pm0.6$; either the plan is not being obeyed, or the contact is the floor.

**Leading hypothesis — the goal criterion, not the projector.** The goal sits at $(3.2,\ \pm1.11)$, the
end of the demonstrated channel, with radius $0.30$. The constraint forces the flown lane out to
$|y|\ge1.26$. Nothing brings it back: the models were trained only on $|y|=1.11$ straight lines, so a
drone displaced to $1.3$–$1.5$ continues straight. That alone spends $0.15$ of the $0.30$ radius;
the measured tracking error on these cells is $0.34$–$0.45$ m. A displaced plan plus that error lands
outside $0.30$ almost every time, which is exactly the $\nfe\le2$ signature (line crossed, safe, goal
missed). If this is right, the projected flights may well be *constraint-satisfying* — `n_violations`
low, `collision_free` high — and fail only the goal test. That would be a **metric/geometry mismatch**
in the scene design, not a model result, and it is fixable at evaluation time (goal placed on the
detour lane, or radius sized to the detour + tracking error, as U16 did explicitly for the corridor
slide: "after the exit the goal needs only $y\ge-0.30$").

**Alternative — the projector's output is not being flown.** Both DPCC per-step and HardFlow solve
with SLSQP; 25993 logs `[hardflow][NLP-FAILURE] first non-converged SLSQP solve at tau=0.600 …
keeping scipy's last iterate, which may be INFEASIBLE` on every HardFlow variant. If the per-step
projector is likewise returning non-converged iterates, the plan is neither the demonstration nor a
legal detour, which would also explain the $\nfe=5$ contacts.

**What separates the two — four `results.json` files, nothing else:**

| cell | why |
| :-- | :-- |
| fm K1 `dpcc-c` (job 25988) | the cleanest $\nfe\le2$ floor row |
| mf K5 `dpcc-r` (25991) | the only non-zero projected cell (0.10) |
| fm K5 `dpcc-r` (25989) | `safe=0.00` — the contact case |
| mf K5 `hardflow_new-r` (25993) | endpoint, with the NLP-FAILURE line |

Read, per rollout: `goal.dist` (hypothesis 1 predicts a cluster just above 0.30, lateral),
`constraint.n_violations` / `collision_free` (hypothesis 1 predicts *low* violations on projected rows —
the opposite of what a broken projector gives), `physical.min_z` / `contact_frac` (floor vs pillar),
`homotopy_flown`, and `projection_health` (`cb_tripped`, `backstop_hits`). The `constraint_overview.png`
of one cell and `diffuser.png` vs `dpcc-c.png` top-down plots settle the lateral question by eye.

**What is usable from this wave regardless of the verdict:** the timing column. Per control step,
`proj_ms`: $\nfe=1$ 122–265 ms, $\nfe=2$ 96–162 ms, $\nfe=5$ 2.3–3.4 s (per-step), HardFlow at
$\nfe=5$ 77–313 ms, diffusion $\nfe=20$ **10.4 s**. Those are clean, and the diffusion number is why
group D cannot fit in 24 h.

**Recommendation.** Do **not** submit the diffusion remainder (5 × ~19 h) or the $\nfe=3$ rung of
`PENDING_20260922` §16 until the four files above have been read. If hypothesis 1 holds, the fix is a
goal/radius change at evaluation time and the whole wave has to be re-scored or re-run anyway, so
compute spent now is compute spent twice. `pillars_xxl` is off the table in either case — the scene is
already too hard for the metric, not too easy.

## 4. What "done" looks like

A batch in which, for geometry `pillars_xl`, tag `u7xl`, seed 6:

- every one of the 10 model–budget cells exists with all eleven variants;
- the **unprojected** (`diffuser`) rows show real failure — that is the whole point of the wave, and if
  they do not, stop and raise the radius rather than reporting the result;
- `n_violations` is non-zero wherever S&C is below 1;
- ~~the diffusion baseline has `dpcc-r` and `dpcc-r-tightened` this time~~ — **withdrawn 2026-09-21
  (§3d): the baseline is reported unprojected only, with the K=20 time argument.**

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
