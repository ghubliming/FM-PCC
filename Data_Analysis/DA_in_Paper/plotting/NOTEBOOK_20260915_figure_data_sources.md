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
| `da/fig_avoiding_k_ladder` | `builders/avoiding.py:262` | same, **plus** the baseline's published-protocol rows from the same batch | solid series 5 × 20; dashed series 5 × 2 |
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
They are the published 10-episode protocol and are drawn dashed and hollow for that reason.
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
| `demo/fig_raw_plans_meanflow_K2` | same report, `fig6c_plans_mfunet_K2_seed6_both-hard.png` | the same at $K=2$ (not currently used by the draft) |
| `da/fig_raw_goal_reached_K1` | `Report_20260903_AF_UNet/fig7_raw_diffuser_K1.svg` | goal reached without projection at $K=1$, top-right-hard (not currently used by the draft) |

The plan panels are per-episode diagnostics of `batch_avoiding_combined_20260818_152911`, seed 6,
both-hard, from the runs `H8_K1_Meuler_T0.5_A0.5_B1_D…MeanFlowODE` and
`H8_K1_T0.5_Dmodels.GaussianDiffusion` (report §7, candidate index rows 138 and 8).

## 4. Where the thesis uses each figure

From `Working_Space/v3/figures/EXPORTED.md`, written by `export_to_draft.py`:

| figure | used in v3 |
| :-- | :-- |
| `fig_env_avoiding` | `chapters/05_setup.tex` — `fig:env-avoiding`, §5.1.1 |
| `fig_constraints_avoiding` | `chapters/05_setup.tex` — `fig:constraints-avoiding`, §5.1.1 |
| `fig_avoiding_tradeoff` | `chapters/06_results.tex` — `fig:avoiding-tradeoff`, §6.1.1 |
| `fig_avoiding_k_ladder` | `chapters/06_results.tex` — `fig:k-ladder`, §6.1.2 |
| `fig_raw_plans_meanflow_K1`, `fig_raw_plans_diffusion_K1` | `chapters/06_results.tex` — `fig:raw-plans`, §6.1.3 |
| `fig_avoiding_projector_cost` | `chapters/06_results.tex` — `fig:projector-cost`, §6.1.4 |

Figures still to be made are listed as `PLANNED` in `sources.py` and in `../figures/MANIFEST.md`:
`fig_env_aligning` and `fig_env_uav`.

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
```

`temp/` is a local drop directory and is not version-controlled: on a fresh machine the batch
directories are absent, the affected builders return nothing and say so, and the environment figures
still build from `../data/avoiding_scene.json`, which lives in the repository and is not ignored
(it is untracked only until `DA_in_Paper/` is first committed).
