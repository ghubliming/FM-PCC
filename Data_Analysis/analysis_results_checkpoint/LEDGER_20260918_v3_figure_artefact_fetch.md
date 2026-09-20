# LEDGER — 2026-09-18 · fetching the artefacts behind v3's missing figures

**Status (2026-09-19): F1 and F3 built and usable. F2 (pillars) BUILT BUT WITHDRAWN — the scene is
withheld. F4 and F5 are staged for wave 2; F5 turned out to be a download, not a run.**
**Read the 2026-09-19 section at the foot of this file before acting on anything above it** — it
supersedes the F5 verdict and records two defects in what was built on 09-18.
This entry records what was resolved, where the fetcher lives, and what the next step is, so the path
resolution is not redone.

## What this covers

The figure-side half of [`PENDING_20260916_missing_data_and_analyses.md`](../../logs_in_develop/Writing/Working_Space/data_status/PENDING_20260916_missing_data_and_analyses.md)
— rows **R5** and **D10**, i.e. the `\hole`s and `\todofigure`s in v3 Chapter 6 that are blocked on
artefacts living only on the cluster. It does **not** touch the numeric gaps; those belong to
[`SLURM_RUNBOOK_20260918_pending_runs.md`](../../logs_in_develop/Writing/Working_Space/data_status/SLURM_RUNBOOK_20260918_pending_runs.md)
(jobs 25887–25898), which is a separate, already-submitted wave.

Specifications come from the two request MDs, unchanged:
[`REQUEST_20260917_trajectory_figures.md`](../DA_in_Paper/plotting/REQUEST_20260917_trajectory_figures.md) (groups F1–F4)
and [`REQUEST_20260916_cluster_fm_plan_panels.md`](../DA_in_Paper/plotting/REQUEST_20260916_cluster_fm_plan_panels.md) (group F5).

## The fetcher

`Slurm_Codes/temp_bash/fetch_20260918_v3_figure_artefacts.sh` — gitignored, so it must be copied to
the remote by hand, like the 18-09 submission driver.

```bash
# on i6-gpu-1, from the repo root
bash Slurm_Codes/temp_bash/fetch_20260918_v3_figure_artefacts.sh            # PLAN — finds, prints, copies nothing
bash Slurm_Codes/temp_bash/fetch_20260918_v3_figure_artefacts.sh stage      # copy into export_tmp/v3_figs_20260918/
bash Slurm_Codes/temp_bash/fetch_20260918_v3_figure_artefacts.sh stage tar  # copy + pack one tar.gz
```

Knobs: `REPO`, `STAGE`, `WAVE="F1 F3"`, `KEEP_GIF=1`, `KEEP_LOG=1`.
`WAVE` — not `GROUPS`, which is a bash built-in that silently refuses assignment (the trap the
`temp_bash/README.md` already documents; it was hit while writing this script).

It reads `logs/` and copies. **It submits nothing and deletes nothing.**

## How the cells were resolved

Run folders were resolved offline against [`16_09_logs_tree.txt`](16_09_logs_tree.txt) (capture of
2026-09-16, root `/u/home/llim/FMPCC/FM-PCC/logs` — note `/u/home`, not the `/data/home` written in
REQUEST_20260917). Every F1–F3 folder in that request was confirmed to exist in the capture; F4's
was confirmed as the `_tslogit_normal` → `H8_K20_Meuler_T0.2_…Emf` pair.

The capture stops at depth 8, so leaf files are not listed in it. The script therefore matches on the
cluster itself, and matches **by exact npz basename**: the projection-variant name is the npz name,
sitting in a leaf folder called `<variant>` or `<variant>_<geometry>` (aligning nests geometry as a
parent dir instead — see the F4 note) next to `results.json` and the
variant `.png`. That is geometry-agnostic and cannot confuse `hardflow_sls` with `hardflow_sls-r`,
which a folder-prefix glob would. Each match pulls its leaf's `*.npz`, `results.json`,
`run_provenance*.json` and `*.png`; `.gif` and `rollout_*.log` are excluded by default, `.gif` being
35 GiB of the tree.

| group | figure | scene / run | variants pulled |
| :-- | :-- | :-- | :-- |
| F1 | `fig:uav-scurve-paths` | UAV-s-curve, MeanFM K10, `mjpc` **and** `pid_stopgo`, tag `u7hg` | `diffuser` (unprojected, as the request asks) |
| F2 | UAV-pillars flown paths | pillars K5 `u7hg`: MeanFM, FM, CI-MeanFM, + diffusion K20 | `hardflow_sls-r` (MeanFM), `hardflow_sls` (FM), `dpcc-t-tightened`, `diffuser` |
| F3 | UAV-corridor flown paths | corridor tag `u17cv2`, K1/K3/K5 × three flow models, + diffusion K20 | `dpcc-t-bounds_free-pdes-tightened`, `diffuser` |
| F4 | D3IL-aligning box paths | aligning-visual MeanFM K20, geometry pinned to `combined_5-tightened` | `dpcc-r_train_set`, `hardflow_sls-r_train_set`, `diffuser_train_set` (see F4 note) |
| F5 | `fig:raw-plans` panels | avoiding-d3il, `…FlowMatchingODE_a1.5_b1.0_aw10`, tag `*planpanel*` | `diffuser` |

F4 is the one cell with a geometry filter, because that run carries several geometries and only the
tightened one is wanted; the other cells take every geometry the run holds.

## PLAN result, 2026-09-18 (copied to the remote as `Slurm_Codes/temp_bash/19-09-Fetch.sh`)

```
TOTAL  16 cells found, 2 missing/empty   96 files   138.6 MiB
```

**F1, F2 and F3 are complete and ready to stage** — every run folder resolved on the first attempt, so
the offline resolution against the 16-09 capture held. 138.6 MiB is a single-archive download.

| group | cells | files | size |
| :-- | :-- | :-- | :-- |
| F1 | 2/2 ✅ | 6 | 4.8 MiB |
| F2 | 4/4 ✅ | 30 | 63.3 MiB |
| F3 | 10/10 ✅ | 60 | 70.4 MiB |
| F4 | 0/1 ❌ EMPTY → fixed, re-PLAN to size it | — | — |
| F5 | 0/1 ❌ MISSING | — | — |

### F4 — was EMPTY, cause found and fixed

The run folder resolved, but no `.npz` matched. The EMPTY diagnostic added after the first PLAN named
the cause on the second run: **every variant carries a `_train_set` suffix on disk**.

`eval_mix_visual_aligning.py:3256` appends `_train_set` to the **variant name itself** — not merely to
the results root — when the eval runs with `--eval-on-train`, and it does so *before*
`artifact_variant_label` resolves the artefact name. So the real leaf is

```
results_train_set/combined_5-tightened/dpcc-r_train_set/dpcc-r_train_set.npz
```

`combined_5-tightened` does exist, and the geometry filter was right all along. Both of the causes
guessed before the diagnostic ran were wrong, which is why the script now asks the cluster instead.

Two structural facts about this scene, confirmed while chasing it, worth keeping:

- `eval_mix_visual_aligning.py:3264` writes `{savepath}/results_train_set/{geo_name}/{variant_out}/`, so
  **geometry is a parent directory here**, unlike the UAV scenes where `<variant>_<geometry>` is one
  folder. `config/visual_aligning_eval.yaml:423` warns that in Gen14 `-tightened` is a geo-level flag and
  must **not** be appended to a variant name (that was the Gen3v6/v7 convention).
- `artifact_variant_label` renames only `hardflow*` stems, and only under the slsqp backend, so `dpcc-r`
  and `diffuser` are never rewritten and `hardflow_sls-r` is already the post-rename form.

The F4 cell now asks for `dpcc-r_train_set`, `hardflow_sls-r_train_set`, `diffuser_train_set`, and was
re-verified against a fixture carrying both geometries: 9 files staged, `combined_5` correctly excluded.
F4 is the only cell with a path filter, so no other cell was affected.

## Expect F5 to come back MISSING

`REQUEST_20260916` established that no flow-matching plan file or dashboard exists at K1/K2 anywhere —
those panels need an eval run (`FMPCC_RUN_MSG=planpanel`, `n_trials: 2`) that has **not** been
submitted. F5 is in the fetcher anyway so the PLAN output states presence or absence as a fact rather
than leaving it to memory. A `MISSING` line for F5 confirms the request is still open; it is not a
fault in the script.

Note also that the 18-09 wave is running with `n_trials` as the working tree has it, and the plan-panel
eval needs `n_trials: 2` — so that run has to be sequenced against the wave, not dropped alongside it.

## Progress

| # | step | state |
| :-- | :-- | :-- |
| 1 | resolve every run folder from the 16-09 capture | ✅ done |
| 2 | write the fetcher, exercised against a synthetic tree (exact-name matching, geometry filter, manifest, `WAVE` selection) | ✅ done |
| 3 | copy the script to the remote and run it in PLAN mode | ✅ done 2026-09-18 — F1–F3 clean, F4/F5 open |
| 3b | re-run PLAN with the EMPTY diagnostic to pin F4's real names | ✅ done 2026-09-18 — `_train_set` suffix; cell corrected |
| 4 | run `stage tar`, download the archive | ✅ done — 70 MiB gz / 140 MiB, 96 files, into `temp/18-09-2026/`. **Staged before the F4 fix, so it holds F1–F3 only** |
| 5 | place the artefacts and declare them | ✅ done — `sources.UAV_PATHS`; extract at `plotting/extract/uav_paths.py` → `data/uav_paths.json` (534 KiB, committed) |
| 6 | write the builders, rebuild the store | ✅ done — `plotting/builders/paths.py`, three figures in `figures/da/` |
| 7 | re-stage F4 after the fix, then build the aligning-paths figure | ⏳ manual, next |
| 8 | v3 `.tex`: replace the `\hole`s, then `export_to_draft.py v3` | ⏳ **v3 draft's agent** — handed off, see below |

## What was built from it (2026-09-18)

| figure (in `DA_in_Paper/figures/da/`) | fills | panels | flights |
| :-- | :-- | :-- | :-- |
| `fig_uav_scurve_paths` | `fig:uav-scurve-paths` | 2 | 13 |
| `fig_uav_pillars_paths` | `sec:res:uav:pillars` flown paths | 4 | 40 |
| `fig_uav_corridor_paths` | `sec:res:uav:corridor` flown paths | 10 | 120 |

New code: `plotting/extract/uav_paths.py` (numpy, reads the drop once) and
`plotting/builders/paths.py` (standard library, reads only the JSON). The scene under every path is
drawn by `scenes._uav_constraint_panel` — the same function as `fig_constraints_uav`, from the same
`sources.UAV_CONSTRAINTS` — so a path is never shown against a constraint set the projection did not
see. Method notes are in `plotting/NOTEBOOK_20260915_figure_data_sources.md` (2026-09-18 section).

**The drop is not the corpus.** `temp/` is gitignored, so `data/uav_paths.json` is the committed record
and this ledger is how to recreate the drop behind it.

The v3 `.tex` was deliberately **not** touched — that draft belongs to another agent. The handoff is
[`cross_draft/to_v3/FROM_DA_20260918_uav_path_figures_ready.md`](../../logs_in_develop/Writing/Working_Space/cross_draft/to_v3/FROM_DA_20260918_uav_path_figures_ready.md),
registered in `cross_draft/INBOX.md`.

The staged tree keeps its original `logs/<run folder>/…` layout, since `sources.py` declares figure
sources by run folder — do not flatten it on the way in.

## What was checked and is not a gap

`REQUEST_20260917`'s claim that the positions are absent from the committed 15-09 batches still holds:
those carry `per_rollout_detail.csv` scalars only. The executed positions are in the rollout `npz`
under each run folder, which is exactly what F1–F4 pull.

---

# 2026-09-19 — wave 2, and two defects the new data_status MDs exposed

Triggered by re-reading `data_status/` after the 09-18/09-19 waves landed. Three of the MDs there had
moved under this ledger's feet.

## 1. `fig:raw-plans` is a DOWNLOAD after all — this ledger was wrong

The section "Expect F5 to come back MISSING" above is **superseded**. It said the panels need a cluster
eval with `FMPCC_RUN_MSG=planpanel`. The PLAN's `MISSING` was correct about that tag and wrong about the
conclusion: **four of the five panels already exist**, produced by the 20-trials evaluations of August,
under different run tags entirely.

The correction is not mine — it is `REQUEST_20260916_cluster_fm_plan_panels.md`'s own "CORRECTION
2026-09-18 (v3.39)", restated in `PENDING_20260918_pillars_geometry_redesign.md` §6. The wave-1 script was
written against the pre-correction text, so it looked in a folder that was never going to exist. Confirmed
against `16_09_logs_tree.txt`:

| panel | run folder (under `logs/avoiding-d3il/plans/`) | tree line |
| :-- | :-- | --: |
| FM $K{=}1$ | `flow_matching_v3_ode_selectable/H8_…FlowMatchingODE_a1.5_b1.0_aw10/H8_K1_…_msg20trials` | 22980 |
| FM $K{=}2$ | same model folder, `H8_K2_…_msg20trials` | 23106 |
| CI-MeanFM $K{=}1$ | `…AlphaFlowODE_aw10_bbunet_tslogit_normal_…/H8_K1_…_msgafon02_s6` | 10165 |
| CI-MeanFM $K{=}2$ | same, `H8_K2_…` | 10260 |

**A trap in that first row.** Line 22957 is a folder with the *same* run name under
`…FlowMatchingODE_a1.5_b1.0_aw1` — 6 files, 153 KiB, a stub — sitting beside the real `aw10` run of 4706
files. The cell pins `aw10` literally instead of globbing the model folder, or the stager would find the
stub and report OK on nothing.

The fifth panel (diffusion $K{=}2$) stays impossible: every `GaussianDiffusion` folder in the batch is
`K20`, because a diffusion model's step count is fixed when its noise schedule is discretised at training
time. The thesis leaves that cell empty.

## 2. The wave-2 fetcher

`Slurm_Codes/temp_bash/fetch_20260919_v3_figure_artefacts_wave2.sh` — gitignored, copy to the remote by
hand. Same shape and knobs as wave 1; `WAVE="F4 F5"`.

```bash
bash Slurm_Codes/temp_bash/fetch_20260919_v3_figure_artefacts_wave2.sh            # PLAN
bash Slurm_Codes/temp_bash/fetch_20260919_v3_figure_artefacts_wave2.sh stage tar  # copy + pack
```

Five cells: F4 (aligning box paths, the corrected `_train_set` cell wave 1 never delivered) and four F5
raw-plan panels. Two changes from wave 1, both forced by F5:

- **the path filter takes several substrings, all of which must match.** F5 needs two — geometry
  `both-hard` and seed `/6/` — because those runs carry five seeds and three geometries and exactly one
  cell of that grid is the figure. Wave 1 allowed a single substring.
- **`<variant>.png` is a fallback when `<variant>.npz` is absent.** The wanted F5 artefact is the
  dashboard image the eval writes as it runs, and whether a plan `.npz` sits beside it is not something
  the tree capture can answer (it truncates at depth 8). npz is still tried first, and a leaf is never
  counted twice.

Exercised against a fixture carrying every decoy above — the `aw1` stub, seeds 7/10, geometries
`one-hard`/`no-hard`, and `combined_5` beside `combined_5-tightened`. 5/5 cells found, 13 files staged,
**every decoy excluded**: the four staged PNGs are exactly `aw10 · seed 6 · both-hard` at $K{=}1,2$ for
both models.

## 3. Defect — `fig_uav_pillars_paths` must not enter the draft

`PENDING_20260918_pillars_geometry_redesign.md` withdraws **UAV-pillars entirely**: its constraint set is
the one the demonstration generator was built to satisfy, so the plans are feasible before projection and
the scene measures how little each method disturbs an already-correct plan. `sec:res:uav:pillars` is
withheld as of v3.40, and the 09-18 wave's pillars jobs are marked ⚰️ DEAD with `u7hg`/`u7hga1` excluded
from the DA.

`fig_uav_pillars_paths` — built here on 2026-09-18, four panels, 40 flights — is drawn **entirely from
`u7hg`**. It is not wrong as a drawing; it is a drawing of a measurement the thesis no longer makes. It
stays in the store as a build artefact and must not be cited. It will be rebuilt on `pillars_xl` once
that wave runs, and will then need its own fetch cell — the run folders do not exist yet, so no cell is
written for it here.

## 4. Defect — the s-curve caption I handed to v3 states the wrong mechanism

The handoff note told the v3 author that proportional stop-go "stalls at the second passage". It does not.
Read straight out of the run's own `results.json` in the 18-09 drop:

```
Emf_K10_mpc4_pid_stopgo_T0.5_u7hg/6/s_curve_hg_…/diffuser/results.json
  divergence: n_aborted_trials 10, reasons {0..9: "inverted"}
```

**All ten flights were aborted by the divergence guard because the vehicle flipped.** The paths stop where
the guard cut the episode. The MPC panel of the same figure has zero aborts. So the figure's contrast is
real and is in fact stronger than claimed — it is stability, not tracking — but the stated reason was my
inference from the drawn shapes, not a fact from the data.

This matches the 09-18 wave independently: 34/55/52 `inverted` aborts across the s-curve at three budgets,
hitting the unprojected arm too. It is a property of the scene and the controller; the guard predates the
wave.

**No re-fetch is needed.** The post-fix re-run (tag `u18sc`, job 25909) reproduces this cell exactly —
0.0 reached, 871 steps, 10 flights, same as `u7hg` — so the switched-wall fix did not move it. Scanning
the whole 18-09 drop, this is the **only** cell with aborts; corridor and pillars are clean, so no other
figure is affected.

The handoff note is corrected in place. What is **not** done: the figures do not *draw* the abort — an
aborted flight currently looks like any other flight that did not reach the goal. Carrying
`divergence.aborted_trials` through `extract/uav_paths.py` into a third stroke state is a code change and
is not made without a go-ahead; until then the caption is the only carrier, and the note says so.

## 5. Noted in passing — the `pillars_xl` verify cell has already run, and it passes

`SLURM_RUNBOOK_20260919_pillars_enlarged.md` §2 still reads "verify cell not yet run", but its output is
sitting in `temp/19-09/Emf_K5_mpc4_pid_stopgo_T0.5_u7xlchk/`. Both gates it defines are met:

| §2 gate | required | measured |
| :-- | :-- | :-- |
| the unprojected plan must now violate | well below S&C $0.90$ | **S&C 0.000** (`strict_and_constraints_rate`), collision-free rate 0.000 |
| the violation scorer must use the enlarged radius | $n_\text{violations} > 0$ | **68.7 mean**, total 9.80 |

Goal-reaching is intact (`reached_rate` 0.90, `strict_rate` 0.90) and `phys_safe` is 1.0 — the flights
still fly, they now violate the enlarged constraint, which is the whole design. Zero divergence aborts,
zero projection trips. That is the §3 wave unblocked; it is the runbook owner's call to submit, not this
ledger's.

## Progress, wave 2

| # | step | state |
| :-- | :-- | :-- |
| 9 | re-read `data_status/` after the 09-18/09-19 waves | ✅ done 2026-09-19 |
| 10 | resolve the four raw-plan run folders against the 16-09 capture | ✅ done — all four present, `aw1` stub trap identified |
| 11 | write and exercise the wave-2 fetcher (F4 + F5) | ✅ done — 5/5 cells on the fixture, all decoys excluded |
| 12 | copy to the remote, PLAN, then `stage tar`, download | ⏳ **manual, next** |
| 13 | crop the four panels (`prep/crop_vendored.py`), move `fig_raw_plans_*` PLANNED → VENDORED | ⏳ after 12 |
| 14 | aligning-paths builder from the F4 drop | ⏳ after 12 |
| 15 | withdraw `fig_uav_pillars_paths` from the draft; rebuild on `pillars_xl` when that wave lands | ⏳ blocked on the run |
| 16 | draw the divergence abort as its own stroke state in the path figures | ⏳ needs a go-ahead (code change) |

---

## Wave 2 landed — 2026-09-19, `fig:raw-plans` closed (R5)

PLAN on the cluster returned **5 cells found, 0 missing, 90 files, 86.5 MiB** — F4 and all four F5
panels, first attempt. Archive `v3_figs_20260919.tar.gz` (69.4 MiB gz) unpacked to
`temp/19-09/v3_figs_20260919/`, 91 files, gzip and tar both verified.

The `aw1` stub trap was live and was avoided: the real `aw10` cell came back at 25.6 MiB, where the
6-file stub would have been ~150 KiB.

### What the dashboards turned out to be

**The npz is scalars only.** Last turn I raised the possibility of *drawing* these panels from plan
data instead of cropping the dashboard image, since the leaves carry `diffuser.npz`. They do not carry
plans: the keys are `n_success`, `n_success_and_constraints`, `n_steps`, `n_violations`,
`total_violations`, `avg_time`, `collision_free_completed`, `args`. **The plan fan exists only as the
rendered image**, so the crop route was the only route. Recorded because it is the question anyone
would ask next.

**Two dashboard layouts, and they do not match.** The 08-19 sources are 3000×1000 — two episodes, the
`n_trials: 2` default the crop box was calibrated for. The four new ones are 3000×5000: ten episodes,
because they come from the 20-trials campaign. Same six columns at the same x (the last column band
measures 2349–2701, exactly the figure the old comment records), same 500 px row pitch — but a title
band shifts row 1 down and squeezes its axes:

| source | plot frame | over data range |
| :-- | :-- | :-- |
| 08-19 (2 episodes) | x 2368–2700, y 620–970 — **332 × 350** | x 0.2–0.8, y −0.3–0.4 |
| 09-19 (10 episodes) | x 2368–2700, y 600–926 — **332 × 326** | identical |

Measured, not assumed. Untreated, the four new panels would sit 7 % flatter in the same matrix and
their obstacles would read as ellipses beside circles. The new crop boxes are therefore chosen so that
scaling the cut to 403×400 lands the frame on exactly (56, 28)–(388, 378), where the old panels have
it. Verified afterwards: **all seven panels report size (403, 400) and frame (56, 28, 388, 378).**

### Changes made

| file | change |
| :-- | :-- |
| `DA_Result_Curated_MD/Report_20260903_AF_UNet/fig8{a,b,e,f}_*.png` | the four dashboards, under the names that report's §8 already reserved — so the vendored source is committed like the other three and the gitignored drop is not in the build path |
| same report's `README.md` §8 | four rows placeholder → landed; 8g/8h annotated (8h marked impossible); the scalars-only warning added |
| `plotting/sources.py` | the four moved `PLANNED` → `VENDORED`; four `VENDORED_CROP` entries with the new box and a resize; the two-layout explanation written into the block comment |
| `plotting/prep/crop_vendored.py` | `VENDORED_CROP` entries may carry an optional third field, a resize target; staleness stamps include it, so changing it rebuilds |
| `plotting/make_figs.py` | unpacks the 3-field entry and records the stretch in the manifest provenance |

`PLANNED` now holds **one** entry, `fig_raw_plans_diffusion_K2`, and its text says the cell is
impossible rather than pending: every `GaussianDiffusion` folder is K20 because a diffusion model's
step count is fixed when its noise schedule is discretised at training time.

Store rebuild clean: **19 generated + 17 vendored; 1 planned.** The assembled matrix was rendered and
looked at — seven consistent panels, and the argument is legible in the drawing: average-velocity's fan
is tightest and stays in the corridor, consistency-interpolated is noisy at K=1 and tighter at K=2,
naive FM leans into the left forbidden wedge at both budgets, and the one-step diffusion baseline is
near-scribble.

### One caveat carried into the handoff

The seven panels come from **two campaigns** — episode 1 of a 2-episode run and episode 1 of a
10-episode run. Same scene, same seed 6, same `both-hard`, same axes. But the saved npz carries no
state, so it **cannot be asserted from the artefacts** that episode 1 is the same initial context in
both. The v3 note says to write "same scene and seed" and not "the same episode".

### Progress

| # | step | state |
| :-- | :-- | :-- |
| 12 | PLAN, `stage tar`, download | ✅ done 2026-09-19 — 5/5 cells, 90 files, 86.5 MiB |
| 13 | crop the four panels, `PLANNED` → `VENDORED` | ✅ done — aspect normalised, all seven verified identical |
| 14 | aligning-paths builder from the F4 drop | ⏳ **next** — artefacts on disk, no builder yet |
| 15 | rebuild `fig_uav_pillars_paths` on `pillars_xl` | ⏳ blocked on that wave |
| 16 | draw the divergence abort as its own stroke state | ⏳ needs a go-ahead |

Handoff: [`cross_draft/to_v3/FROM_DA_20260919_raw_plans_matrix_complete.md`](../../logs_in_develop/Writing/Working_Space/cross_draft/to_v3/FROM_DA_20260919_raw_plans_matrix_complete.md),
registered in `INBOX.md`. The v3 `.tex` was not touched.

---

## 2026-09-20 — wave 3, staged for the cluster

`PENDING_20260920_all_lacking_runs.md` replaced the older gap lists with one current checklist.
Its **section 4** is the only part that is a transfer; sections 1–3 are evaluations and trainings and
belong in a `pipeline_*.sh`, not in a fetcher. Reading section 4 item by item against the artefacts
actually on disk changes what is left to fetch:

| item | PENDING says | what is true on disk | in wave 3? |
| :-- | :-- | :-- | :-- |
| **D10d** diffusion K20 raw-plans panel | download one dashboard `.png` | not on disk; run folder present in the 09-19 tree (562 files, 284.4 MiB) | ✅ **F6** — this is the one open `\todofigure` |
| **D10a** avoiding executed paths | download `<run>/<variant>.npz` | `dpcc-t-tightened.npz` carries `obs_all`; 4 of 24 leaves already local from wave 2 | ✅ **F7** |
| **D10b** aligning end-effector paths | download existing `obs_all` | **already satisfied** by the 09-19 drop — `obs_all` is `(10, 400, 6)` at `combined_5-tightened`, seed 6, for all three requested variants | ⚪ **F8**, opt-in, only to widen geometry coverage |
| **D10c** aligning box paths | logging change then re-evaluate | **confirmed not fetchable**: the six columns are `[desired xyz, actual xyz]` of the end effector. The box pose is not in the observation at all | ❌ correctly excluded |
| **D8/D9** compute + provenance | `sacct`, job ids, git revisions | `run_provenance*.json` exist under `logs/`; `sacct` retention is bounded | ⚪ **F9**, opt-in |

### Two facts the script is built around

**`obs_all` is variant-dependent in the avoiding runs.** The projected variants
(`dpcc-t-tightened.npz`, 750 KiB) carry `obs_all` as a `(20,)` object array of per-episode `(T, d)`
paths. The **unprojected `diffuser.npz` of the same leaf carries scalars only** — that is the same
finding that forced the raw-plan panels to be cropped images rather than drawn vectors, and it means
D10a's figure can only be built from a projected variant. The request already asks for
`dpcc-t-tightened`, so this costs nothing; it is recorded because the opposite assumption has already
been made once in this ledger.

**A whole-leaf copy would have been ~600 MiB.** One avoiding geometry leaf holds about fourteen
variants × (`.npz` + `.png`) ≈ 25 MiB, and F7 matches 24 leaves. Wave 3 therefore adds a sixth cell
field, `pick`: `leaf` keeps the wave-1/2 behaviour, `exact` takes only `<variant>.npz`,
`<variant>.png` and the leaf's `results.json` / `run_provenance*.json`. Every wave-3 cell is `exact`,
which puts the whole fetch at roughly 40 MiB.

### Geometry folder names

The avoiding geometry folder is `halfspace_both-hard`, not `both-hard` — the wave-2 filter matched it
only because the filter is a substring test. F7 spells the prefix out; F6 keeps the looser spelling
because it also has to survive the `/6/` seed filter beside it.

### Script

`Slurm_Codes/temp_bash/fetch_20260920_v3_figure_artefacts_wave3.sh` (gitignored, like its two
predecessors). `WAVE` defaults to `"F6 F7"`; `WAVE="F6 F7 F8 F9"` turns on the two opt-in groups.
The runner was exercised against the local 09-19 drop as a partial mirror: the four leaves that exist
there resolve and pick exactly 2 files / 1.8 MiB each, `stage` reproduces the run-folder paths, and
every absent cell reports `MISSING` or `EMPTY` with the diagnostic listing rather than silently
counting zero.

### Progress

| # | step | state |
| :-- | :-- | :-- |
| 17 | wave-3 stager written and dry-run | ✅ done 2026-09-20 — awaiting a cluster PLAN |
| 18 | F6 → crop → `fig_raw_plans_diffusion_K20` | ⏳ closes the last `\todofigure` of `fig:raw-plans` |
| 19 | F7 → extract → avoiding executed-path builder | ⏳ new figure, no builder yet |

### Wave 3 PLAN result — 20/22, and the 2 failures are not fetchable

Cluster PLAN, 2026-09-20: **20 cells found, 2 missing/empty, 40 files, 35.0 MiB.** Both failures are
the same leaf, and the `ls` settles what the diagnostic could only bound:

```
logs/avoiding-d3il/plans/diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10/
    H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials/6/results/halfspace_both-hard/
```

holds five variants and stops:

| written | at | |
| :-- | :-- | :-- |
| `dpcc-r` | 11:40 | ✅ |
| `dpcc-r-tightened` | 11:54 | ✅ |
| `dpcc-c` | 12:06 | ✅ |
| `dpcc-c-tightened` | 12:19 | ✅ |
| `dpcc-t` | 12:32 | ✅ |
| `dpcc-t-tightened` | 12:32 | ❌ `eval_dpcc-t-tightened.log` is **109 bytes**, no `.npz`, no `.png` |
| `gradient` … `post_processing`, `diffuser` | — | ❌ never reached |

12:32 on 2026-08-18 is also the newest mtime of the **entire run folder**, so the job ended there.
A complete leaf carries thirteen variants (checked against the FM 20-trials leaf in the 09-19 drop).
Neither wanted file exists: `diffuser.png` (D10d) and `dpcc-t-tightened.npz` (D10a).

**The batch CSV reproduces the truncation exactly and independently** — for this run,
`(seed 6, both-hard)` carries five variants and every other cell carries thirteen. So this is not a
staging or filter artefact; the evaluation stopped.

### What that truncation costs, measured

Coverage of the 20-trials corpus, `(geometry → number of seeds)`:

| run | both-hard | top-left | top-right |
| :-- | --: | --: | --: |
| Diffusion K20 | **1** (partial) | 5 | 5 |
| FM K20 | **3** | 5 | 5 |
| MeanFM K20 | **0** | 3 | 5 |
| everything else at K1/K2/K5/K10 | 5 | 5 | 5 |

(The `msgafon02_s6` CI-MeanFM runs show 1/1/1 by design — the `_s6` tag *is* seed 6, and seeds 7–10
are already tracked as **R25**. Not a new gap.)

`avoiding_table_spread.py` already tolerates short geometries — it averages each seed over the
geometries that seed covers, and prints `seeds=` / `geos=` per cell. Running it makes the cost visible
in one line. Of the 48 published cells it prints, **exactly one** is not `geos=3`:

```
Diffusion  K=20  t  S&C=0.960 +/- 0.045  steps=82.8 +/- 14.7  ms=564.3 +/- 29.5  seeds=5 geos=2
```

The extended-protocol diffusion K20 **per-step-tightened** row is averaged over **two of three
geometries**, because `dpcc-t-tightened` is the variant the job died on. The `r` and `c` rows survive
at `geos=3` only because the job got through `dpcc-c-tightened` nineteen minutes earlier.

Not a bias claim: on `n_success_and_constraints` both-hard is not the harder geometry — every model
scores 1.000 there under the tightened-c rule. The issue is a coverage asymmetry in one published cell,
not a flattered baseline.

### Consequence

D10d and D10a's diffusion/both-hard cell move **out of section 4** of the pending list: they are not
transfers. One evaluation re-run — diffusion K20, 20-episode protocol, seed 6, `both-hard`, the eight
variants from `dpcc-t-tightened` onward — closes the last `\todofigure` of `fig:raw-plans`, fills the
diffusion row of the new avoiding path figure, **and** repairs the `geos=2` table cell. Completing
seeds 7–10 at `both-hard` as well would make the diffusion K20 row a true 5x3 like every other row;
that is a larger job and a separate decision.

| # | step | state |
| :-- | :-- | :-- |
| 17 | wave-3 stager written and dry-run | ✅ done 2026-09-20 |
| 18 | wave-3 PLAN on the cluster | ✅ done — 20/22, 35.0 MiB; stage pending |
| 19 | diffusion K20 seed 6 `both-hard` re-run | ⏳ **blocks D10d, D10a-diffusion and the `geos=2` cell** |

### Wave 3 landed, 2026-09-20 — and the ninth panel came from a sibling campaign

Archive `v3_figs_20260920.tar.gz` verified (`gzip -t`, 94 entries, 36 MB extracted) plus two loose files
the user pulled by hand. Extracted to the gitignored `temp/20-09/`.

**The ninth `fig:raw-plans` panel is in the store.** It is *not* from
`H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials` — that run is the truncated one. It comes from the
complete sibling campaign `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5`, seed 6, `both-hard`. The
substitution is sound and the reason is worth keeping: **`thres0.5` is a projection threshold, and
`diffuser` is the unprojected arm** — the projector never runs, so it cannot have shaped these plans.
Same checkpoint, same K=20, same scene, same seed, same geometry.

**The layout was measured, not assumed, and the spec was wrong about it.** The 2026-09-20 addition to
`REQUEST_20260916` says to use the twenty-episode box `(2312, 574, 2715, 947)` with the `(403, 400)`
resize. The file that arrived is **3000x1000** — the *two*-episode layout. Grid bands measured on it:
columns `[(370,709) … (2349,2702)]`, rows `[(98,476), (518,896)]`, identical to the 08-19 sources. So it
takes the 08-19 box `(2312, 92, 2715, 492)` and **no resize**. Verified after cropping: size `(403,400)`,
frame `(56,28)-(388,378)` — the same as the other eight.

Store rebuild clean: **23 generated + 19 vendored; 0 planned.** `PLANNED` is now empty — every figure
any draft asks for exists. All nine raw-plan panels verified programmatically to share one canvas and
one frame, and the assembled matrix was rendered and looked at: the K20 baseline is tight and threads
the obstacle field, which is the expected contrast against its near-scribble K=1 panel.

### The 20 executed-path cells, validated

All 20 `dpcc-t-tightened.npz` load; 0 failures. Every cell carries 20 episodes; episode lengths run
35–84 steps; per-cell S&C matches the published tables (1.000 for most, 0.850–0.950 for the
`top-right-hard` cells that the tables also report short).

**Column convention, so the builder does not guess.** `config/projection_eval.yaml:20` declares
`'avoiding': {'x_des': 0, 'y_des': 1, 'x': 2, 'y': 3}`, and the eval prepends the commanded position at
`eval_flow_matching_v3_ode_selectable.py:377`. So `obs_all[ep][:, 2:4]` is the **executed** end-effector
`(x, y)` and `[:, 0:2]` is the commanded one — the same des-then-actual layout as the UAV and aligning
artefacts.

Still missing, and still the same single cell: **diffusion K20 at `both-hard`** for the executed-path
figure. `dpcc-t-tightened.npz` has no image substitute, so that row of the new figure stays empty until
the re-run. The `geos=2` table cell is unchanged by today's work.

| # | step | state |
| :-- | :-- | :-- |
| 20 | ninth raw-plans panel wired in and verified | ✅ done 2026-09-20 — `PLANNED` is empty |
| 21 | 20 executed-path cells staged and validated | ✅ done — builder not written |
| 22 | diffusion K20 `both-hard` re-run | ⏳ blocks the path figure's diffusion row + the `geos=2` cell |

### v3.50, 2026-09-20 — the executed-path builders are written

Step 21 said "builder not written". It is written: `plotting/extract/exec_paths.py` (numpy, reads the
gitignored drops) writes `data/exec_paths.json`, and `plotting/builders/exec_paths.py` (standard
library) draws two figures from it.

* **`fig_avoiding_paths`** — four panels, `top-right-hard`, `dpcc-t-tightened`, seed 6, 20 episodes
  each: MeanFM / CI-MeanFM / FM at one network evaluation and the diffusion baseline at twenty.
  All 80 episodes are violation-free; 19 / 17 / 17 / 20 reach the goal. `top-right-hard` rather than
  `both-hard` because the diffusion cell of `both-hard` is the one the truncated job never wrote —
  the same missing cell as everywhere else in this ledger.
* **`fig_aligning_paths`** — three panels, `combined_5-tightened`, seed 6, the ten contexts, under no
  projection / per-step / endpoint. Violation-free counts **2 / 9 / 10**, which reproduce the
  $\nfe=20$ rows of the draft's `tab:va-projection` exactly — an independent check that the artefacts
  and the published table describe the same evaluation.

Two panels were factored out so that a path is never drawn over a redrawn constraint set:
`avoiding.geometry_panel` and `scenes.aligning_constraint_panel`, the manipulator equivalents of
`scenes._uav_constraint_panel`.

Measured while building, worth keeping: on the unprojected alignment arm, **7** of the ten contexts
enter the keep-out region and an eighth leaves the halfspace; under per-step projection neither the
region nor the halfspace is entered in any context, and the single context that is not violation-free
breaches the **action box** instead (27 steps). Under endpoint projection every column is zero.

| # | step | state |
| :-- | :-- | :-- |
| 21 | avoiding + aligning executed-path builders | ✅ done 2026-09-20 (v3.50) |
| 22 | diffusion K20 `both-hard` re-run | ⏳ still the only gap: the `geos=2` table cell and this figure's fourth geometry |
| 23 | alignment **box**-pose logging (D10c) | ⏳ the alignment path figure draws the end effector until then |
