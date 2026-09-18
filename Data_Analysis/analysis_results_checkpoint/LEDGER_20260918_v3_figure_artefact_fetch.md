# LEDGER — 2026-09-18 · fetching the artefacts behind v3's missing figures

**Status: F1-F3 downloaded, extracted and BUILT INTO THE FIGURE STORE on 2026-09-18. F4 and F5 open.**
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
