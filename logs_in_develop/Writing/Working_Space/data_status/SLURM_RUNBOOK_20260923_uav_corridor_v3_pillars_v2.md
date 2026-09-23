# SLURM RUNBOOK 23-09 — UAV-corridor v3 (R33) and UAV-pillars v2 (R39), paper-only

**Companion of [`PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md`](PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md).**
Cluster i6-gpu-1, submit through `Slurm_Codes/submit.sh`. Every command is `plan` (dry) by default. Nothing here was
run from the container. Group C (R30) and the s-curve corridor-style notes of the 22-09 runbook are superseded by this file.

> **🔴 Run-side amendments, 2026-09-23 (U19 session, author-approved) — read before §0–§2.**
> 1. **Corridor v3 is TWO scenes, not one.** Author: *"two scenes as piloted."* `corridor_v3_tilt` (the v2 slide leaned
>    −60° about the launch altitude; one x–y–z plane; tag **`p23cv3t`**, folder `corridor_cv3t_…`) is *the* v3;
>    `corridor_v3_ablation_hump` (roof only, ceiling 2.80; tag **`p23cv3ah`**, folder `corridor_cv3ah_…`) is an ablation.
>    The combined `corridor_v3` / `p23cv3` / `hs=5` of §0.2 and §1 is **not run**. Step 0.2's coding landed differently
>    (`plane: xz` and `z_lean` keys, helpers `_hs_plane` / `_hs_lean`, no third return value) — see
>    `Gen15/U19/CHANGELOG_20260922_U19_corridor_v3_coding1.md`. Cross-draft note: `cross_draft/to_v3/FROM_U19_20260923_corridor_v3_is_two_scenes.md`.
> 2. **C0 was run on 22-09** (jobs 26071 hump / 26072 tilt, tags `u19smokecv3ah` / `u19smokecv3t`): G1 PASS, **G3 PASS**
>    (the projector moves z on demand), **G2 FAIL** on every projected arm — 6–11 steps of ≤ 2–6 cm at the window end /
>    apex with the setpoint clean (plant lag). **Author overrode the stop rule**: *"if it is the model / projector, no need
>    to fix the env"* → waves run on the unchanged scenes. `Gen15/U19/PILOT_20260922_U19_gates_G1-G3.md`.
> 3. **Driver** = `Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh` (`plan` / `smoke` / `submit C1..C5|all`), both
>    scenes, 68 cells each; launched **wave-by-wave** through `…_p23_corridor_v3_master.sh start` (one 10-min CPU link per
>    wave, next link queued with `--dependency=afterany` on the wave's job ids; squeue holds one wave + one pending master).
> 4. **HardFlow spelling:** the eval accepts only the registered input names `hardflow_new{,-r,-c,-t}-…` and writes the
>    folders as `hardflow_sls-…`; §1's `HF_*` names are the folder names. **C3 composition:** the eval refuses a job with
>    HardFlow variants and no `dpcc-*` row, so the HF jobs carry `dpcc-t` and C2 runs `dpcc-t` only at K = 1, 2 (§2's
>    C2/C3/C4 rows are re-cut accordingly; same 68 cells, each once). Diffusion K20 `diffuser` is in C1 as written.
> 5. **Pillars v2 (author, 23-09, final): real evaluation, no turbo.** R39 = the four Chapter 6 cells (MeanFM and CI-MeanFM
>    at K1, K2; `diffuser` + `dpcc-t-tightened`) evaluated live with the quadrotor as the plant, seeds 6–10 × 3 geometries
>    × 2 episodes, tag `p23uavpv2live` (must contain `uav`). Submitter `eval_20260923_p23_pillars_live.sh`, job
>    `live_p23_pillars.sh` (§3). Turbo P1 and the first-draft P2 (`live_p2_*.sh`, tag without `uav`) are dropped.
>    Corridor runs seed 6 only (`SEEDS=6` default in the driver).

## 0 · Before the first job

| step | what | check |
| :-- | :-- | :-- |
| 0.1 | `git pull`; `python -m py_compile mix_uav_test/eval_mix_uav.py uav_avoiding_bridge/*.py` | clean |
| 0.2 | **done (U19, 22-09):** two entries `corridor_v3_tilt` (`z_lean`) and `corridor_v3_ablation_hump` (`plane: xz`, `ub[2]: 2.80`); pilots 26071/26072 passed G0/G1/G3, G2 overridden by the author | `grep -n "corridor_v3" config/uav_projection.yaml` shows both entries |
| 0.3 | G0 preview — **done** for both scenes (`Gen15/U19/figs/`) | — |
| 0.4 | driver **exists**: `Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh` (`plan` / `smoke` / `submit C1..C5|all`), both scenes, tags `p23cv3t` / `p23cv3ah` | `bash … plan` lists 68 cells per scene and no other variant |

## 1 · Corridor v3 — variant lists (exhaustive; anything else is NOT run)

```
PCC_R=dpcc-r-bounds_free-pdes-tightened   PCC_C=dpcc-c-bounds_free-pdes-tightened   PCC_T=dpcc-t-bounds_free-pdes-tightened
HF_S=hardflow_new-bounds_free-pdes-tightened  HF_R=hardflow_new-r-…  HF_C=hardflow_new-c-…  HF_T=hardflow_new-t-…
```
(`hardflow_new` is the registered **input** name; the eval writes the folders as `hardflow_sls-…`, which the DA reads.)
Two scenes, run one after the other with the same lists: `UAV_MIX_GEO_VARIANTS=corridor_v3_tilt FMPCC_UAV_EVAL_TAG=p23cv3t`
and `UAV_MIX_GEO_VARIANTS=corridor_v3_ablation_hump FMPCC_UAV_EVAL_TAG=p23cv3ah`. Common env on every child: `FMPCC_SAFE_EPS_FRAC=1.0
FMPCC_SAFE_EPS_MODE=scaled UAV_EVAL_HOURS=24`; af children add `UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2
UAV_MIX_EPOCH=latest`. Trials 12 (4 per route). `-tightened` is always the **last** token (DA reads it with `endswith`).

## 2 · Corridor v3 — waves and commands

| wave | command (mode) | children | env / variants | wall |
| :-- | :-- | --: | :-- | --: |
| **C0 smoke** | **done 22-09** (jobs 26071/26072) | — | — | — |
| **C1** | `… submit C1` | 4 | `UAV_MIX_VARIANTS=diffuser`: mf K "1 2 3", af K "1 2 3", fm K "1 2 3 5 20", diffusion K20 (`eval_mix_uav.sh diffusion corridor 6 12 …`) | ~1 h |
| **C2** | `… submit C2` | 3 | `UAV_MIX_VARIANTS=$PCC_R,$PCC_C,$PCC_T` at K "1 2"; at K 3 `$PCC_R,$PCC_C` (the `$PCC_T` row of K 3 rides in C3); mf, af, fm | ~3 h |
| **C3** | `… submit C3` | 4 | `UAV_MIX_VARIANTS=$PCC_T,$HF_S,$HF_R,$HF_C,$HF_T` (the eval refuses HF without a `dpcc-*` row): mf K3, af K3, fm K "3 5"; plus fm K5 `$PCC_R,$PCC_C` | ~3 h |
| **C4** | `… submit C4` | 2 | fm K20: `$PCC_R,$PCC_C,$PCC_T` and `$HF_S,$HF_R,$HF_C,$HF_T` | ~4 h |
| **C5** | `… submit C5` | 3 | diffusion K20, **one variant per job**: `$PCC_T` first, then `$PCC_R`, then `$PCC_C` | up to 24 h each |

The C0 gates were read on 22-09 (`Gen15/U19/PILOT_20260922_U19_gates_G1-G3.md`): G1 and G3 pass on both scenes, G2 fails by a
plant-lag residue of 2–6 cm with the setpoint clean; **the author decided to run C1–C5 as they are.** Result paths:
`…/E<engine>_K<k>_mpc4_pid_stopgo_T0.5_p23cv3t/6/corridor_cv3t_bounds+dynamics+geo_bounds+halfspace+obstacles/<variant>/`
and the `p23cv3ah` / `corridor_cv3ah_…` twin.

HardFlow rule: never submit an HF variant without its `dpcc` row in the same run family (the eval's own guard). K20 flow
cells and diffusion K20 go last (C4, C5) so the fast waves are never behind them.

## 3 · Pillars v2 — commands (rev 3, 2026-09-23: REAL evaluation of the four Chapter 6 cells; turbo dropped)

Spec: PENDING_20260923 §2 (rev 23-09). Gates G0 and the live check (26077) are done; nothing to re-gate.

```bash
bash Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh            # plan: pre-flight + checkpoint check + the 4 jobs
bash Slurm_Codes/temp_bash/eval_20260923_p23_pillars_live.sh submit     # 4 GPU jobs: MeanFM K1, K2 · CI-MeanFM K1, K2
# one job by hand:  GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_p23_pillars.sh mf 1
```

| job | engine · K | checkpoint knobs | config | variants | protocol |
| :-- | :-- | :-- | :-- | :-- | :-- |
| 1 | MeanFM · 1 | `MF_BACKBONE=unet MF_HORIZON=8` | `config/meanflow_projection_eval_u18_live.yaml` (one call per seed, `--seed 6..10`) | `diffuser`, `dpcc-t-tightened` | seeds 6–10 × 3 geos × 2 ep |
| 2 | MeanFM · 2 | same | same | same | same |
| 3 | CI-MeanFM · 1 | `AF_BONE=unet AF_ALPHA_END=0.2 AF_EPOCH=latest AF_SEEDS="6 7 8 9 10" AF_NTRIALS=2` | `config/alphaflow_projection_eval_u18_live.yaml` | same | same |
| 4 | CI-MeanFM · 2 | same | same | same | same |

Plant on every job: `FMPCC_AVOIDING_PLANT=uav`, `FMPCC_RUN_MSG=p23uavpv2live`, scale 36, clock replay 1 Hz, feed-forward
off, v_max 1.0, `FMPCC_MPC_BATCH=4`, `MUJOCO_GL=disable`. 🔴 The tag must contain `uav` (factory refuses otherwise);
`live_p2_meanflow.sh` / `live_p2_dpcc.sh` default to `p23pv2live` and would stop there — **do not use them**.
`turbo.sh MODE=paper|papergif` is **not** run (author, 23-09: real eval only).

Outputs: `logs/avoiding-d3il/plans/flow_matching_v3_{meanflow,alphaflow}/<train bbunet…>/H8_K<k>_…_msgp23uavpv2live/<seed>/results/halfspace_<geo>/{diffuser,dpcc-t-tightened}.npz`
(+ the eval's usual pngs and logs); plant sidecars (world paths, track_err, contacts) under
`logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/_live/p23uavpv2live/<engine>_K<k>/`.
Read in each log: the eval's `Success rate / Constraints satisfied / … ` blocks per geometry and the `[ uav-plant ]` sidecar lines.
Download both trees (`export_to_laptop.sh`) — the DA needs the npz and the sidecars; Figure `fig_uav_pillars_paths` takes
seed 6, *top-right-hard*, episode 2 of the MeanFM K1 cell.

## 4 · Completion checks

| for | check |
| :-- | :-- |
| corridor | **68 result folders per scene**: `…_p23cv3t/6/corridor_cv3t_…/` and `…_p23cv3ah/6/corridor_cv3ah_…/`; `projection_health.n_tripped_trials = 0`; `divergence.n_aborted_trials` reported per cell; DA_UAV_v1 batch runs **per tag** (`p23cv3t`, then `p23cv3ah`), never pooled; `data_quality.csv` clean |
| pillars v2 | four live jobs → per engine × K × seed × geometry: `diffuser.npz` + `dpcc-t-tightened.npz` under `…_msgp23uavpv2live/`, 5 seeds × 3 geometries each; plant sidecars under `_live/p23uavpv2live/<engine>_K<k>/`; 0 divergence aborts expected (26077); contacts reported, not a failure of the job |
| both | download with `Slurm_Codes/download_remote_logs/export_to_laptop.sh` into `temp/2309/`; nothing pooled with older tags |

## 5 · Submission record (fill in)

| date | wave | job ids | outcome |
| :-- | :-- | :-- | :-- |
| 2026-09-22 | C0 pilots (as `u19smoke*`, U16 layout) | 26071 (hump), 26072 (tilt) | done; G1/G3 pass, G2 fail by plant-lag residue; author: run as is |
| 2026-09-23 | chain start | master 26078 (`p23_master_C1`) | ran; submitted C1, queued C2 |
| 2026-09-23 | **C1** `diffuser` (both scenes: mf/af K1,2,3, fm K1,2,3,5,20, diffusion K20) | 26079–26102 (24 eval jobs) | queued (`AssocGrpGRES` at submit time) |
| 2026-09-23 | C2 master | 26103 (`p23_master_C2`, Dependency: afterany C1) | pending |
| | C2 / C3 / C4 / C5 | (queued by each link in turn — read the `p23_master_*` logs in `Slurm_Codes/logs/<date>/`) | |
| 2026-09-23 | **R39 pillars v2 live** (`p23uavpv2live`, 5 seeds × 3 geos × 2 ep, `diffuser` + `dpcc-t-tightened`) | 26147 MeanFM K1 · 26148 MeanFM K2 · 26149 CI-MeanFM K1 · 26150 CI-MeanFM K2 | submitted |

Claude (Fable 5.1, Claude Code) · 2026-09-23 · not run from the container.
