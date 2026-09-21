# REQUEST — executed trajectories for the result figures (download from the cluster)

**2026-09-17 · needed for** four figures in v3 Chapter 6 that are currently `\hole`s. All of them draw the
same kind of picture as `fig_constraints_uav` / `fig_constraints_avoiding`: the scene from above, the
constraint set, the flown paths over it, and a mark where each run ends.

None of this data is in the committed 15-09 batches: those carry per-rollout **scalars**
(`per_rollout_detail.csv`), not the executed positions. The positions live in the rollout artefacts on the
cluster (`rollouts*.npz` / `results.json` / the recorded GIFs) under each run folder below.

## What to download

| # | figure (label) | run folder on the cluster (`/data/home/llim/FMPCC/FM-PCC/logs/…`) | flights |
| :-- | :-- | :-- | :-- |
| 1 | `fig:uav-scurve-paths` — s-curve under two controllers (companion of `tab:uav-controller`) | `UAV_MIX/uav-s_curve/plans/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/Emf_K10_mpc4_mjpc_T0.5_u7hg` **and** the same path with `pid_stopgo` in place of `mjpc` | 3 + 10, unprojected (`diffuser`) |
| 2 | UAV-pillars flown paths, one panel per model (`tab:uav-pillars-best`) | `UAV_MIX/uav-pillars/plans/mix_uav_mf/H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet/Emf_K5_mpc4_pid_stopgo_T0.5_u7hg`; `…/mix_uav_fm/H8_Dmodels.diffusion.FlowMatchingODE_9D/Efm_K5_mpc4_pid_stopgo_T0.5_u7hg`; `…/mix_uav_af/H8_Dmodels.af_diffusion.AlphaFlowODE_9D_as1_ae0.2_bbunet/Eaf_K5_mpc4_pid_stopgo_T0.5_EPlatest_u7hg`; plus the diffusion reference run | variants `hardflow_sls-r` (MeanFM), `hardflow_sls` (FM), `dpcc-t-tightened` (CI-MeanFM, Diffusion); 10 each |
| 3 | UAV-corridor flown paths through the slide | `UAV_MIX/uav-corridor/plans/…` , tag `u17cv2`, variant `dpcc-t-bounds_free-pdes-tightened`, K1/K3/K5 per model + diffusion K20 | 12 per cell (routes L/C/R × 4) |
| 4 | D3IL-aligning box paths over the ten contexts (`tab:va-projection`) | `aligning-d3il-visual/plans/mix_visual_aligning_mf/H8_…_filmv1_Emf_tslogit_normal/H8_K20_Meuler_T0.2_…_VTrue_mpc4_filmv1_Emf` , geometry `combined_5-tightened`, variants `dpcc-r` and `hardflow_sls-r` | 10 contexts each |
| 5 | **D3IL-avoiding executed paths** (added 2026-09-20) — the counterpart of `fig:uav-corridor-paths` for the foundation benchmark; the executed end-effector path of a **projected** episode over the geometry it was projected against | `avoiding-d3il/plans/…`, the cells of `batch_avoiding_combined_20260919_132703`, variant `dpcc-t-tightened`, seed 6 | 1 episode per geometry per model |

## What each artefact must contain

- the executed position at every control step (end-effector `x,y` for alignment; vehicle `x,y,z` for the
  quadrotor) — enough to draw the path and its end point;
- the episode outcome per flight/context, so a path can be drawn as passed / not passed and
  collision-free / violating;
- for #4 additionally the box pose per step and the target pose;
- for #5 the executed end-effector `x,y` per control step, and which steps violated the constraint set, so a violating stretch can be drawn apart from a clean one.

`Slurm_Codes/download_remote_logs/export_to_laptop.sh` is the existing export path. Please keep the run
folder names, since the figure sources are declared by folder in `sources.py`.

## Then, locally

1. Put the downloaded artefacts under `Data_Analysis/DA_in_Paper/data/` (or `temp/` and declare the path).
2. Add one entry per figure to `sources.py` (the scene geometry is already there:
   `UAV_CONSTRAINTS`, `D3IL_SCENES`).
3. Write the builders next to `builders/scenes.py::fig_constraints_uav`, which already draws the scenes and
   their constraints; the paths are an overlay on those panels.
4. `make_figs.py`, then `export_to_draft.py v3`, then replace the `\hole`s and the `\todofigure` in
   `06_results.tex`.

## Size

Rollout artefacts are the large files in this project. Download only the cells listed above (≈ 70 flights, 20 alignment contexts and 9 avoiding
episodes), not whole campaign trees.

---

## STATUS 2026-09-20 (v3.50) — items 3 and 4 are drawn

Both manipulator path figures now exist and are in the draft:

* **`fig_avoiding_paths`** (`fig:avoiding-paths`, §6.1.2.2) — four cells × 20 episodes on
  `halfspace_top-right-hard`, variant `dpcc-t-tightened`, seed 6.
* **`fig_aligning_paths`** (`fig:aligning-paths`, §6.2) — three cells × 10 contexts at
  `combined_5-tightened`, seed 6, variants `diffuser`, `dpcc-r`, `hardflow_sls-r`.

Pipeline: `extract/exec_paths.py` (numpy, reads the gitignored `temp/20-09` and `temp/19-09` drops)
→ `data/exec_paths.json` → `builders/exec_paths.py` (standard library).

**Item 4 as originally written is not satisfied and cannot be by a download.** It asks for the *box*
paths; the evaluation logs only the end effector (`obs_all` is `[des_xyz | xyz]` of the end effector).
What is drawn is the end-effector path, which is the quantity the constraint set applies to. The box
path needs an evaluation re-run with the pose recorded — `PENDING_20260922_all_lacking_runs.md`, D10c —
and the figure's `\guard` says so in the thesis.

## STATUS 2026-09-20 (v3.53) — `fig_avoiding_paths` re-scoped by the author

The four panels are no longer one per model. The author asked for the models the chapter carries
forward instead: **MeanFM and CI-MeanFM at $K=1$ and $K=2$**, still on `halfspace_top-right-hard`,
variant `dpcc-t-tightened`, seed 6, twenty episodes per panel. All four cells were already in the
20-09 drop, so this was an `AVOIDING_PANELS` edit in `extract/exec_paths.py` and a re-run — no fetch.

Two consequences:

* the figure's four cells are now **exactly the four panels of the two average-velocity columns of
  `fig:raw-plans`**, same checkpoints and budgets, with the projector switched on — the two figures
  are a matched before/after of the projector rather than two different selections;
* the diffusion `both-hard` artefact this figure was waiting on is **no longer needed**
  (`PENDING_20260922_all_lacking_runs.md` §11).

Counts: 19, 19, 17 and 20 of 20 episodes reach the goal; **all 80 are violation-free**.
