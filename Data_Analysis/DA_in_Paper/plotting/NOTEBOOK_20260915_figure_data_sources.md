# NOTEBOOK — where every thesis figure's data comes from

**Created:** 2026-09-15 · **Scope:** every figure in `../figures/`, by exact file name.
**Purpose:** tracing and audit. For one figure name, this says which builder drew it, which files it
read, what those files themselves came from, at which protocol, and where the thesis uses it.

Rules this file records rather than repeats:
- a figure name is **unique across groups** (`da`, `demo`, `env`, `schematic`);
- the **only** file holding data paths is [`sources.py`](sources.py); builders take paths from it;
- a figure copied in rather than drawn here is checked against `/workspaces/aux_repo/` first and
  recorded with its provenance (`VENDORED`);
- drafts hold **copies**, listed with their checksums in `<draft>/figures/EXPORTED.md`.

Paths are relative to the repository root `/workspaces/FM-PCC` unless stated otherwise.

---

## 0. Data source of record — the 15-09 checkpoint

**`Data_Analysis/analysis_results_checkpoint/15-09/`** is the corpus to read when a figure needs data.
It arrived 2026-09-15 and supersedes the `temp/` batches for every new or regenerated figure. Use
`temp/` only to reproduce a figure exactly as it stands today.

| batch | environment | coverage |
| :-- | :-- | :-- |
| `batch_avoiding_combined_20260915_100757` | obstacle avoidance | 5 seeds (6–10) × 3 geometries, 185 candidates over 117 evaluation folders, 30 projection methods, $K \in \{1,2,3,5,10,20,40,50,100\}$ |
| `batch_va2_20260915_100754` | vision-conditioned alignment | 7 seeds, train and test splits, 29 candidates; carries box angle and distance metrics |
| `batch_uav_20260915_100816` | quadrotor | seed 6, 44 scene variants including corridor, pillars and s-curve; carries `fm_ms`, `track_err_mean`, `n_fm_steps` |

Two reasons this is the better source: it is **committed to the repository** (the avoiding raw table is
gzipped to stay under GitHub's file limit — commit `b6cb1870`), so a figure stays reproducible on a
fresh clone, which is exactly what `temp/` cannot promise; and it is the first corpus that covers all
three environments, including the two whose chapters still carry holes.

**It is not drop-in.** The checkpoint's `candidates_multidimensional_raw.csv` is **long** — one row per
`(seed, variant, constraint_type, halfspace_variant, metric, value)` — whereas the builders in
[`sources.py`](sources.py) read the **wide** tables of the `temp/` batches, taking
`n_success_and_constraints`, `n_steps` and `avg_time` as columns. Re-pointing a builder therefore means
a loader that pivots long to wide, not a changed path. Column order also differs between the three
batches, and the avoiding table is gzipped, so read them with `csv.DictReader` (plus `gzip.open`), never
by column index.

**Status.** No figure has been re-pointed yet: every figure below still names the `temp/` batch it was
built from. Re-pointing is a code change to `sources.py` and the builders, to be made on request.

## 1. Obstacle avoidance — environment figures

Both read one extract, `../data/avoiding_scene.json`, written by
[`extract/avoiding_scene.py`](extract/avoiding_scene.py) (needs `python3.14`: numpy + PyYAML).

| what | source file | what is taken from it |
| :-- | :-- | :-- |
| constraint geometry | `config/projection_eval.yaml` | `halfspace_constraints['avoiding-d3il']` (4 lines), `obstacle_constraints['avoiding-d3il']` (keep-out disks), `ax_limits` $x\in[0.2,0.8]$, $y\in[-0.3,0.4]$, `enlarge_constraints['avoiding']` $=0.025$ |
| which constraints each geometry uses | `scripts/eval.py:93–102` | top-left-hard → halfspace 0, disk 3 · top-right-hard → halfspace 1, disk 4 · both-hard → halfspaces 2 and 3, disk 5 |
| obstacles and goal line | `diffuser/utils/constraints_helpers.py`, `plot_environment_constraints` | six obstacle centres, radius $0.025$, goal line $y=0.35$ |
| demonstrations | `/workspaces/aux_repo/HardFlow/d3il/environments/dataset/data/avoiding/data` — D3IL obstacle-avoidance dataset, 96 `.pkl` episodes | `robot['c_pos'][:, :2]`, the measured end-effector position, all steps but the last |

**The extract asserts the mapping.** It re-reads `scripts/eval.py` and fails if the geometry-to-constraint
mapping there stops matching the table above, so a change in the evaluation code cannot silently
invalidate the figures.

**Dataset caveat.** The 96 episodes are read from the copy inside `aux_repo/HardFlow`, because this
repository does not carry the D3IL data. It is the same public D3IL dataset the training loads through
`diffuser/datasets/d4rl.py:137` (`environments/dataset/data/avoiding/data`). If the figures are ever
regenerated on the cluster, that copy is the one to use.

| figure | builder | shows |
| :-- | :-- | :-- |
| `env/fig_env_avoiding` | `builders/avoiding.py:186` | the scene from above: all 96 demonstrations, six obstacles, goal line, start points |
| `env/fig_constraints_avoiding` | `builders/avoiding.py:204` | the three geometries: excluded region, tightened boundary, keep-out disk with its tightened ring, demonstrations in grey |

Chapter 5 uses two authentic MuJoCo stills side by side: `env/fig_render_avoiding_start.png`
is frame 0 (start pose) and `env/fig_render_avoiding.png` is frame 200 (end effector beyond the
green goal line). Both crop `(0, 0, 320, 180)` from the top-left tile of the tracked montage
`d3il/figures/github_readme.gif`. The selections are declared in `sources.ENV_RENDER_FRAMES` and
reproduced by `prep/extract_env_frames.py`.

## 1a. Alignment and quadrotor environment figures

The environment figures deliberately pair two kinds of evidence. A `fig_render_*` PNG is a frame from
the simulator or an evaluation rollout. A `fig_scene_*` SVG is a clean orthographic reconstruction for
reading the complete geometry; it is not labelled as a simulator screenshot.

In the table, **alignment expert GIF** is the exact path registered in
`sources.ENV_RENDER_FRAMES`: `temp/0408/mix_visual_aligning_mf/`
`H8_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_a1.5_b1.0_aw1_VTrue_steps1000_bs64_filmv1_Emf_tslogit_normal/`
`H8_K100_Meuler_T0.5_Dmix_visual_aligning.models.visual_mf_diffusion.VisualMeanFlow_VTrue_mpc4_filmv1_Emf/6/`
`results_train_set/expert_references/expert_rollout_0.gif`.

| figure | exact source | selection or builder |
| :-- | :-- | :-- |
| `env/fig_render_aligning` | `d3il/figures/github_readme.gif` | frame 0, crop `(640, 180, 960, 360)` |
| `demo/fig_aligning_camera_overhead` | alignment expert GIF | expert demonstration frame 60, crop `(0, 0, 96, 96)`; the recorded `bp-cam` observation |
| `demo/fig_aligning_camera_wrist` | the same expert demonstration and instant | crop `(96, 0, 192, 96)`; the recorded `inhand-cam` observation |
| `env/fig_scene_aligning` | D3IL aligning scene primitives and `d3il/environments/dataset/data/aligning/test_contexts.pkl` | `builders/scenes.py`; first held-out context: box `(0.58057404, -0.20366790, -43.513584 deg)`, target `(0.49818864, 0.33333877, -58.283190 deg)` |
| `env/fig_render_uav_corridor` | MuJoCo render of `d3il/.../quadrotor/scenes/scene_corridor_v2.xml` + Skydio X2 mesh; path from `uav_expert_data_collect/trajectories.py` `corridor_path('C', 1.1, 8.0)` | `prep/render_mujoco_scenes.py`, `sources.MUJOCO_RENDERS` |
| `env/fig_scene_uav_corridor` | `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_corridor_v2.xml` | `builders/scenes.py`, oblique orthographic view |
| `env/fig_render_uav_pillars` | MuJoCo render of `d3il/.../quadrotor/scenes/scene_pillars.xml` + Skydio X2 mesh; path from `uav_expert_data_collect/trajectories.py` `pillar_path(('L','R','L'), 1.1, 13.0)` | `prep/render_mujoco_scenes.py`, `sources.MUJOCO_RENDERS` |
| `env/fig_scene_uav_pillars` | `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_pillars.xml` | `builders/scenes.py`, oblique orthographic view |
| `env/fig_render_uav_scurve` | MuJoCo render of `d3il/.../quadrotor/scenes/scene_s_curve.xml` + Skydio X2 mesh; path from `uav_expert_data_collect/trajectories.py` `s_curve_scene_path(1.1, 19.0)` | `prep/render_mujoco_scenes.py`, `sources.MUJOCO_RENDERS` |
| `env/fig_scene_uav_scurve` | `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_s_curve.xml` | `builders/scenes.py`, oblique orthographic view |

**2026-09-16 — the three UAV stills are rendered, no longer cut from rollout GIFs.** The GIF frames
were 140–320 px, the pillars one mostly black, and all carried the diagnostic step counter.
`prep/render_mujoco_scenes.py` loads the scene MJCF in MuJoCo, places the X2 level at the first sample
of the demonstration generator's reference path, draws that path as a thin tube, and renders
offscreen at 1600×1100. Nothing is simulated and no model runs (`mj_forward` only places bodies).
Camera and path arguments are declared in `sources.MUJOCO_RENDERS` and stamped in
`data/prepared/MUJOCO_RENDERS.json`; `--check` reports a changed declaration as stale. Needs the
`mujoco` wheel and OSMesa (`MUJOCO_GL=osmesa`), which the AI container now has in a scratch venv.
The GIF declarations were removed from `ENV_RENDER_FRAMES`, so `extract_env_frames.py` cannot
overwrite the renders.

The two D3IL stills (`fig_render_avoiding`, `fig_render_aligning`) are still frames of
`d3il/figures/github_readme.gif`, which is byte-identical to the upstream D3IL checkout — the D3IL
authors' own README asset. ⚠️ Their captions must credit D3IL, or they must be replaced.

The two `fig_aligning_camera_*` panels are not crops of the README montage. They are the two raw
$96\times96$ views concatenated by the alignment evaluator while replaying an expert demonstration.
They use the same instant and therefore show exactly the observation pair consumed by the visual
policy. The ignored `temp/` GIF remains the declared source; the prepared PNGs are the portable record.

The orthographic `fig_scene_uav_*` views carry geometry only: no path and no vehicle, since the render
above each one shows the real path (an illustrative straight line through the pillars had contradicted
it). Wall and pillar dimensions come from the MJCF.

The per-panel counts — **0, 1 and 2 of 96** demonstrations satisfy top-left-hard, top-right-hard and
both-hard — are computed in the extract with the check of `scripts/visualize_data_constraints.py`:
every step must be on the feasible side of each halfspace and outside the disk enlarged by the
obstacle radius $0.025$. The same numbers appear in `tab:avoiding-geometries`.

## 2. Obstacle avoidance — data figures

All three read `candidates_multidimensional_raw.csv` of a batch directory registered in `sources.py`,
and use the columns `n_success_and_constraints`, `n_steps` and `avg_time`.

| figure | builder | batch | protocol |
| :-- | :-- | :-- | :-- |
| `da/fig_avoiding_tradeoff` | `builders/avoiding.py:131` | `temp/2508/batch_avoiding_combined_20260825_143212` | 5 training seeds (6–10) × 20 episodes |
| `da/fig_avoiding_k_ladder` | `builders/avoiding.py:262` | same, **plus** the baseline's smaller-sample rows from the same batch | solid series 5 × 20; dashed series 5 × 2 |
| `da/fig_avoiding_projector_cost` | `builders/avoiding.py:337` | `temp/0609/I/batch_avoiding_combined_20260906_125724` (job 25444) | 4 training seeds (7–10) × 3 geometries × 2 episodes |

**Which rows are read.** A model is selected by the exact evaluation folder name, listed in
`sources.py`:

| model | folder pattern (`%d` is the step budget) |
| :-- | :-- |
| diffusion (DPCC) | `H8_K%d_T0.5_Dmodels.GaussianDiffusion_msg20trials` |
| flow matching | `H8_K%d_Meuler_T0.5_Dmodels.diffusion.FlowMatchingODE_msg20trials` |
| MeanFlow | `H8_K%d_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE_msg20trials` |
| consistency training | `H8_K%d_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_alphaflow.models.AlphaFlowODE_msg20trials` (SiT backbone in this batch — a confounded comparison, not drawn by default) |

The k-ladder's dashed series comes from three separate folders, because those runs were produced by
different jobs: `H8_K1_T0.5_Dmodels.GaussianDiffusion`,
`H8_K10_Dmodels.GaussianDiffusion_aw10_thres0.5`, `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5`.
They form a smaller five-seed by two-episode sample and are drawn dashed and hollow so they cannot be
mistaken for the powered comparison. They are a mechanism check, not the DPCC paper's published
evaluation protocol (five seeds by ten episodes per geometry).
`fig_avoiding_projector_cost` selects its rows by the run tag `hfmink_A1_mfunet` instead.

**Aggregation.** Per geometry first, then across geometries (`geometry_mean`). A flat mean over
(seed × geometry) cells gives different numbers, because the baseline has one training seed on
both-hard; only the per-geometry mean reproduces the analyses of record.

## 3. Vendored figures — produced elsewhere, copied in

These come from the evaluation's own diagnostics and cannot be rebuilt from a CSV. Each was checked
against `/workspaces/aux_repo/` and is ours.

| figure | copied from | what it is |
| :-- | :-- | :-- |
| `demo/fig_raw_plans_meanflow_K1` | `Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/fig6a_plans_mfunet_K1_seed6_both-hard.png` | every plan of one episode, no projection, MeanFlow at $K=1$ |
| `demo/fig_raw_plans_diffusion_K1` | same report, `fig6b_plans_dpcc_K1_seed6_both-hard.png` | the same for the diffusion baseline at $K=1$ |
| `demo/fig_raw_plans_meanflow_K2` | same report, `fig6c_plans_mfunet_K2_seed6_both-hard.png` | the same at $K=2$ |
| `da/fig_raw_goal_reached_K1` | `Report_20260903_AF_UNet/fig7_raw_diffuser_K1.svg` | goal reached without projection at $K=1$, top-right-hard (not currently used by the draft) |

The plan panels are per-episode diagnostics of `batch_avoiding_combined_20260818_152911`, seed 6,
both-hard, from the runs `H8_K1_Meuler_T0.5_A0.5_B1_D…MeanFlowODE` and
`H8_K1_T0.5_Dmodels.GaussianDiffusion` (report §7, candidate index rows 138 and 8).

**The three `fig_raw_plans_*` sources are cut before use.** Each is a $3000\times1000$ diagnostic
dashboard — two episodes × six columns: `x`, `y`, `x_des`, `y_des` time series, the executed path, and
the overlaid plans. Only the last is a thesis figure. The box `(2312, 92, 2715, 492)` keeps the top row's
last column, $403\times400$; it is declared in `VENDORED_CROP` in [`sources.py`](sources.py), applied by
[`prep/crop_vendored.py`](prep/crop_vendored.py) into `../data/prepared/`, and `make_figs.py` copies that
file and refuses to fall back to the uncut source. Boxes were read off the white gutters between panels;
the column is `ax[i, 5]` in `scripts/eval.py:411–415`.

What that panel actually contains: the renderer draws the plan fan every
`plot_samples_every = max(1, horizon//2)` $= 4$ control steps, and at most four candidates per drawn
step (`scripts/eval.py:410–411`) — **not** every plan of the episode. The caption of `fig:raw-plans` says
so; it previously claimed "every plan".

At $403\times400$ the cut is about 146 dpi at the printed width. That is the ceiling of the available
sources — the report holds no vector version, and the panel is inherently ~400 px inside a 2×6 grid.
A print-resolution version needs the plans panel re-rendered alone from the saved rollouts (the comment
at `scripts/eval.py:279` notes the npz keeps every step), which is a cluster job.

## 4. Where the thesis uses each figure

From `Working_Space/v3/figures/EXPORTED.md`, written by `export_to_draft.py`:

| figure | used in v3 |
| :-- | :-- |
| `fig_render_avoiding_start`, `fig_render_avoiding` | `chapters/05_setup.tex` — `fig:env-avoiding`, §5.2.1; authentic MuJoCo start and goal-passed stills. The 96-demonstration `fig_env_avoiding` remains in the DA store but is not in the thesis figure. |
| `fig_constraints_avoiding` | `chapters/05_setup.tex` — `fig:constraints-avoiding`, §5.1.1 |
| `fig_render_aligning`, `fig_scene_aligning` | `chapters/05_setup.tex` — `fig:env-aligning`, §5.1.2 |
| `fig_aligning_camera_overhead`, `fig_aligning_camera_wrist` | `chapters/05_setup.tex` — `fig:aligning-cameras`, alignment camera observations |
| `fig_render_uav_corridor`, `fig_scene_uav_corridor`, `fig_render_uav_pillars`, `fig_scene_uav_pillars`, `fig_render_uav_scurve`, `fig_scene_uav_scurve` | `chapters/05_setup.tex` — `fig:env-uav`, §5.1.3 |
| `fig_avoiding_tradeoff` | `chapters/06_results.tex` — `fig:avoiding-tradeoff`, §6.1.1 |
| `fig_avoiding_k_ladder` | `chapters/06_results.tex` — `fig:k-ladder`, §6.1.2 |
| `fig_raw_plans_meanflow_K1`, `fig_raw_plans_meanflow_K2`, `fig_raw_plans_diffusion_K1` | `chapters/06_results.tex` — `fig:raw-plans`, §6.1.3 |
| `fig_avoiding_projector_cost` | `chapters/06_results.tex` — `fig:projector-cost`, §6.1.4 |

Figures still to be made are listed as `PLANNED` in `sources.py` and in `../figures/MANIFEST.md`.
For `fig:raw-plans`, these include flow matching at $K=1$ and $K=2$. Aggregate outcome rows exist,
but neither diagnostic images nor saved per-control-step plan states are present in this checkout, so
those panels cannot be reconstructed honestly here. `sources.PLANNED` records the corresponding
cluster plan directories and rendering requirement.

## 5. Formats

Every figure is drawn as SVG. PNGs next to them were rendered with `svg/preview_png.py --scale 3`, so
that `\includegraphics` has something it can read — pdfLaTeX cannot read SVG and this container has no
SVG converter. **The PNGs are raster and approximate the fonts;** for the final document produce PDFs
with `svg/svg2pdf.sh` on a machine that has `rsvg-convert`, `inkscape` or `cairosvg`, or build the
bundle with `--svg-package`. A `.pdf` is preferred over a `.png` automatically, so no source changes.

## 6. Re-verifying any of this

```bash
python3 plotting/make_figs.py --list        # corpora on disk, builders, vendored, planned
python3 plotting/make_figs.py               # rebuild; writes figures/MANIFEST.md
python3.14 plotting/extract/avoiding_scene.py   # re-extract geometry + demonstrations, re-check eval.py
python3 plotting/export_to_draft.py v3      # refresh a draft's copies and its EXPORTED.md
python3.14 plotting/svg/preview_png.py ../figures/env/fig_env_avoiding.svg /tmp/x.png
python3.14 plotting/prep/crop_vendored.py --check   # are the cut dashboards current?
MUJOCO_GL=osmesa <python with mujoco> plotting/prep/render_mujoco_scenes.py --check
python3.14 plotting/prep/extract_env_frames.py --check # are selected simulator stills current?
```

`temp/` is a local drop directory and is not version-controlled: on a fresh machine the batch
directories are absent, the affected builders return nothing and say so, and the environment figures
still build from `../data/avoiding_scene.json`, which lives in the repository and is not ignored
(it is untracked only until `DA_in_Paper/` is first committed).

## 2026-09-16 · attribution of the alignment camera panels (`fig:aligning-cameras`)

`fig_aligning_camera_overhead` / `fig_aligning_camera_wrist` are frames of our own expert-replay GIF, but the
camera set-up and the two views are D3IL's (front view and in-hand view, D3IL appendix Fig. 8, 96×96). The
v3 caption now reads "Adapted from D3IL \parencite[appendix, Fig.~8]{jia2024towards}", the panel labels
follow D3IL's names, and the text says "front view" instead of "overhead". The file names are unchanged.

## 2026-09-18 · the flown-path figures (`fig_uav_{scurve,pillars,corridor}_paths`)

Three new `da/` figures, and the first corpus here that is **not** a batch directory.

The executed positions are in none of the committed batches: the 15-09 batches carry per-rollout
scalars (`per_rollout_detail.csv`), and a scalar cannot be drawn as a path. The positions live in the
rollout `npz` under each run folder on the cluster, which were staged and downloaded on 2026-09-18 by
`Slurm_Codes/temp_bash/fetch_20260918_v3_figure_artefacts.sh` (groups F1–F3) into `temp/18-09-2026/`.

`temp/` is not version-controlled, so **the drop is not the corpus — `data/uav_paths.json` is.**
`extract/uav_paths.py` reads the drop once with numpy and writes that JSON; the builders (standard
library, like all of them) read only the JSON. On a fresh machine the drop is absent and the figures
still build. To recreate the drop, the ledger
`Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md` has the stager
and every run folder.

What is drawn, and what was decided in the extract rather than the builder:

| field | source | drawn as |
| :-- | :-- | :-- |
| path | `obs_all[:, 3:5]` — the **executed** position; `0:3` is the reference the controller tracked | polyline |
| `passed` | `success_relaxed` — crossing the finish line | solid vs dashed, filled vs hollow end mark |
| `clean` | `constraint_collision_free` / `rollouts[i].constraint.collision_free` | green vs red |
| `homotopy` | `rollouts[i].homotopy`, corridor only (L/C/R) | carried, not yet drawn |

**Corrected 2026-09-18 (v3.35).** The first extract took `passed` from `success_strict`, the
0.30 m goal-point criterion the author replaced with crossing the finish line in v3.28. The panel
subtitles therefore disagreed with every table in §6.3 --- the corridor figure read 8/12 reached where
`tab:uav-corridor` reads 12/12 passed. The extract now reads `success_relaxed`, which is the field
`analysis/uav_results.py` reports, and every panel count matches its table cell. The wording in the
legend and the subtitles is "passed", not "reached", for the same reason.

The two outcomes are kept independent because a flight can reach the goal *through* a pillar, and the
pillars figure has exactly that case. Paths are decimated to every 2nd control step with both endpoints
kept (`STRIDE`), which is what keeps the ten-panel corridor figure at ~330 KiB.

The scene under every path is drawn by `scenes._uav_constraint_panel`, the same function as
`fig_constraints_uav`, from the same `sources.UAV_CONSTRAINTS` — so a path can never be shown against a
constraint set the projection did not see. Panel width is 560 rather than 470 there: these panels carry
an outcome count in the subtitle, which clips at 470.

**Still open.** The D3IL-aligning box paths (`sec:res:aligning:projection`) were requested in the same
batch but the cell was staged before a variant-naming bug in the stager was fixed (`_train_set` suffix),
so they are not in this drop and that figure does not exist yet.
