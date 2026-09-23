# HANDOVER — R39 (UAV-pillars v2) work done in the U19 corridor chat, 2026-09-23

For the U18 / pillars chat. The author asked for these changes in the corridor chat; this note lists everything that chat
touched on the pillars side so nothing is a surprise. The corridor chat does no further pillars work.

## 1 · Author decisions recorded (23-09, in order)

1. Pillars paper set: one seed, tightened only, turbo only → **reverted** the same hour to **5 seeds × 2 episodes**.
2. **Final:** real evaluation only, **no turbo** (turbo re-flies the Panda's stored setpoints open loop; the planner and
   projector never see the vehicle, so it gives no new planning number). R39 = the four Chapter 6 cells of
   `tab:uav-pillars-raw`: MeanFM K1, K2 and CI-MeanFM α_end 0.2 K1, K2, variants `diffuser` + `dpcc-t-tightened`, seeds 6–10 ×
   3 geometries × 2 episodes, evaluated live (Mode L). `fig:uav-pillars-paths` = one panel picked from the appendix
   `fig:avoiding-paths` (MeanFM K1, seed 6, *top-right-hard*, episode 2): arm path beside the flown path, pre and post
   projection; no extra run.

## 2 · Files

| file | state | what |
| :-- | :-- | :-- |
| `Slurm_Codes/sbatch/uav_avoiding_bridge/live_p23_pillars.sh` | **new**, committed by the author with the run | one GPU job per `<mf|af> <K>`; plant env as the 26077 check (scale 36, clock 1 Hz, ff off, v_max 1.0, fan 4, `MUJOCO_GL=disable`); mf one call per `--seed`, af via `AF_SEEDS`; `MF_BACKBONE=unet` / `AF_BONE=unet AF_ALPHA_END=0.2 AF_EPOCH=latest`; tag `p23uavpv2live`, refuses a tag without `uav`; sidecars in `_live/p23uavpv2live/<engine>_K<k>/` |
| `config/alphaflow_projection_eval_u18_live.yaml` | **new** | `alphaflow_projection_eval.yaml` with `projection_variants: [diffuser, dpcc-t-tightened]` only (checked: the only differing key) |
| `Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh` | **new**, `git add -f` | `plan` / `submit` of the four jobs, pre-flight + checkpoint check |
| `Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh` | edited, net: seeds now env-overridable | `P23_SEEDS` (default `6 7 8 9 10`), `P23_T5_SEEDS` (default `7 8 9 10`); defaults = the original behaviour. Turbo is no longer read by the thesis |
| `logs_in_develop/Gen15/U18/CHANGELOG_20260923_U18_fix7_p23_live_real_eval.md` | new | changelog of the above |
| `data_status/PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md` | edited | §0 principles 3–4 and **§2 rewritten** (what Ch 6 reads, the four live jobs, tag rule, DA/figure spec, superseded turbo + first-draft live cells); cost table row |
| `data_status/SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md` | edited | amendment item 5, **§3 rewritten** (commands, job table, outputs), completion row, §5 job ids |
| `data_status/PENDING_20260922_all_lacking_runs.md` | edited | banner + both R39 rows |
| `Slurm_Codes/logs/important_runs/important_runs.md` | appended | the four job ids |
| memory `u18-pillars-v2-avoiding-bridge.md` | edited | paper-runs paragraph: live only, tag must contain `uav` |

Not touched: Chapter 5 / 6 / appendix (no thesis edit), `factory.py`, `plant.py`, the eval scripts, `live_p2_*.sh`.

## 3 · Bug found (not fixed, superseded)

`uav_avoiding_bridge/factory.py` raises unless `FMPCC_RUN_MSG` contains `uav`. `live_p2_meanflow.sh` / `live_p2_dpcc.sh`
default to `TAG=p23pv2live`, which does not → they would stop at the plant switch. The runbook's old line "`p23pv2live`
does [contain uav]" was wrong. The new driver uses `p23uavpv2live`.

## 4 · Submitted (by the author, 23-09)

> **U18 note (23-09):** these four were cancelled while pending and resubmitted after U18 fix8 (sidecar overwrite) as
> **26151** MeanFM K1 · **26152** MeanFM K2 · **26153** CI-MeanFM K1 · **26154** CI-MeanFM K2 — see `CHANGELOG_20260923_U18_fix8_sidecar_overwrite.md`.

| job | cell |
| :-- | :-- |
| 26147 | MeanFM K1 |
| 26148 | MeanFM K2 |
| 26149 | CI-MeanFM K1 |
| 26150 | CI-MeanFM K2 |

Results: `logs/avoiding-d3il/plans/flow_matching_v3_{meanflow,alphaflow}/<train bbunet…>/H8_K<k>_…_msgp23uavpv2live/<seed>/results/halfspace_<geo>/`.

## 5 · Not verified — check first in the logs

- MeanFM and CI-MeanFM have **never run live before** (26077 was FM K20). Watch for a GL / render call in the CI-MeanFM eval
  under `MUJOCO_GL=disable`, and that each log prints the per-geometry `Success rate / Constraints satisfied` blocks and
  the `[ uav-plant ]` sidecar lines.
- MeanFM runs one python call per seed: the eval's all-seeds aggregate plots are per call; the DA reads per-seed npz.
- The CI-MeanFM result folder's `B` token follows `HFFM_BATCH` (HardFlow fan), not the DPCC fan; it may differ from the
  table cell's folder name (`B4`). Irrelevant to the two variants run, but match folders by `_msgp23uavpv2live`, not by `B`.
- Still open for the pillars chat: the DA (table: air columns + contacts from sidecars) and the `fig_uav_pillars_paths`
  builder (arm path from `exec_paths.py`, flown path from the live npz `obs_all`, world frame via `frame.py`).

Claude (Opus 5.5, Claude Code, U19 corridor chat) · 2026-09-23
