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

## 0 · Before the first job

| step | what | check |
| :-- | :-- | :-- |
| 0.1 | `git pull`; `python -m py_compile mix_uav_test/eval_mix_uav.py uav_avoiding_bridge/*.py` | clean |
| 0.2 | **U19 coding landed** (`config/uav_projection.yaml` entry `corridor_v3` with the slide **and** the two `plane: xz` hump halfspaces, `ub[2]: 2.80`, `geo_tag_suffix: '_cv3'`; `_normalize_halfspace` returns `plane`; projector maps `y→z` for `plane: xz`; scorer uses `p[2]`; drawing guards) | `grep -n "corridor_v3" config/uav_projection.yaml`; a one-cell dry-run prints `hs=5, obs=4` |
| 0.3 | G0 preview (`temp/geo_demo` style script, CPU): hump roof inflated by 0.31 (+0.025) against z ∈ [0.86, 1.24]; slide against y | roof at x = 0 ≥ 1.49 m; slot to 2.49 m ≥ 0.6 m |
| 0.4 | copy `Slurm_Codes/temp_bash/eval_20260913_u16_corridor_v2_paper.sh` → `eval_20260923_p23_corridor_v3.sh`; set `GEO=corridor_v3`, `TAG=p23cv3`, keep `plan/smoke/submit` modes; **variant lists exactly as §1**; `git add -f` | `bash … plan` lists 68 cells and no other variant |

## 1 · Corridor v3 — variant lists (exhaustive; anything else is NOT run)

```
PCC_R=dpcc-r-bounds_free-pdes-tightened   PCC_C=dpcc-c-bounds_free-pdes-tightened   PCC_T=dpcc-t-bounds_free-pdes-tightened
HF_S=hardflow_sls-bounds_free-pdes-tightened  HF_R=hardflow_sls-r-…  HF_C=hardflow_sls-c-…  HF_T=hardflow_sls-t-…
```
Common env on every child: `UAV_MIX_GEO_VARIANTS=corridor_v3 FMPCC_UAV_EVAL_TAG=p23cv3 FMPCC_SAFE_EPS_FRAC=1.0
FMPCC_SAFE_EPS_MODE=scaled UAV_EVAL_HOURS=24`; af children add `UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2
UAV_MIX_EPOCH=latest`. Trials 12 (4 per route). `-tightened` is always the **last** token (DA reads it with `endswith`).

## 2 · Corridor v3 — waves and commands

| wave | command (mode) | children | env / variants | wall |
| :-- | :-- | --: | :-- | --: |
| **C0 smoke** | `bash …p23_corridor_v3.sh smoke` | 1 | mf K3, 3 trials, `diffuser,$PCC_T,$HF_T`, `record=gif` 320 px | 20 min |
| **C1** | `… submit C1` | 4 | `UAV_MIX_VARIANTS=diffuser`: mf K "1 2 3", af K "1 2 3", fm K "1 2 3 5 20", diffusion K20 (`eval_mix_uav.sh diffusion corridor 6 12 …`) | ~1 h |
| **C2** | `… submit C2` | 3 | `UAV_MIX_VARIANTS=$PCC_R,$PCC_C,$PCC_T`: mf, af, fm at K "1 2 3" | ~3 h |
| **C3** | `… submit C3` | 4 | `UAV_MIX_VARIANTS=$HF_S,$HF_R,$HF_C,$HF_T`: mf K3, af K3, fm K "3 5"; plus fm K5 `$PCC_R,$PCC_C,$PCC_T` | ~3 h |
| **C4** | `… submit C4` | 2 | fm K20: `$PCC_R,$PCC_C,$PCC_T` and `$HF_S,$HF_R,$HF_C,$HF_T` | ~4 h |
| **C5** | `… submit C5` | 3 | diffusion K20, **one variant per job**: `$PCC_T` first, then `$PCC_R`, then `$PCC_C` | up to 24 h each |

Gates on the C0 log before C1: `[ U11 ] geo variants for 'corridor': ['corridor_v3']`, `E9 geo … (hs=5, obs=4)`, results
path ending `…_p23cv3/6/corridor_cv3_…/`, exactly three variant folders, `diffuser` violates 3/3, a projected arm
collision-free ≥ 2/3 and success ≥ 2/3, and the paired z at x ∈ [−0.5, 0.5] differs by > 0.15 m (read `obs_all[:,5]`
from the two npz). **If G2 or G3 fails, stop and report; do not submit C1–C5.**

HardFlow rule: never submit an HF variant without its `dpcc` row in the same run family (the eval's own guard). K20 flow
cells and diffusion K20 go last (C4, C5) so the fast waves are never behind them.

## 3 · Pillars v2 — commands (rev 2, 2026-09-23: implemented as tracked sbatch, no hand-copied temp bash)

Everything below arrives with `git pull` (the U17 lesson: gitignored `temp_bash/` had to be copied by hand). Every
driver is **plan-by-default**: without `GO=1` it prints what it would run and exits. Repo root, cluster.

**Gates G0/G1 — done.** Pilot 26076 (36×) and the live check 26077 passed (`Gen15/U18/CHANGELOG_…fix4…`,
`Gen15/U18/L1_20260922_live_vs_turbo_result.md`): agreement 17–20/20, live within 0.05 of the Panda on every cell,
0 contacts / 0 divergence live. Gap p95 0.007 units vs the Panda's 0.03–0.13. Nothing to re-gate.

```bash
# ── P1a: the two pictured cells WITH GIFs (T1 MeanFM K1, T4 diffusion K20), GPU, 2 GIFs per cell — run FIRST
GO=1 MODE=papergif GIF=2 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo_gif.sh
# ── P1b: the whole representative set T1–T5 (CPU, < 15 min); T1/T4 are skipped as done, T2/T3/T5 are flown
MODE=paper      ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh      # dry-run: must list 10 cells (T1–T4 × 2 + T5 × 2)
GO=1 MODE=paper ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh
# ── P2: live closed loop (GPU, minutes each)
GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_p2_meanflow.sh    # L1: MeanFM K1 bbunet, diffuser + dpcc-t-tightened, seed 6, 3 geos × 2
GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/live_p2_dpcc.sh        # L2: diffusion K20, dpcc-c-tightened, seed 6, 3 geos × 2
```

What `MODE=paper` selects (exact folder names from the 19-09 batch CSV `Full_Path`; tag `p23pv2turbo`, `clock` replay,
DPCC protocol seeds 6–10 × 3 geometries × 2 episodes unless stated):

| # | `--engine` | `--train-glob` / `--eval-glob` | variants | seeds |
| :-- | :-- | :-- | :-- | :-- |
| T1 | `flow_matching_v3_meanflow` | `*bbunet*` / `H8_K1_Meuler_T0.5_A0.5_B1_Dflow_matcher_v3_meanflow.models.MeanFlowODE` | `diffuser`, `dpcc-t-tightened` | 6–10 |
| T2 | `flow_matching_v3_alphaflow` | `*bbunet*ae0.2*` / `*K1_*msgdpccproto` | same | 6–10 |
| T3 | `flow_matching_v3_ode_selectable` | — / `H8_K1_*msgdpccproto` | same | 6–10 |
| T4 | `diffusion` | — / `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5` | `diffuser`, `dpcc-c-tightened` | 6–10 |
| T5 | `flow_matching_v3_meanflow` | `*bbunet*` / `H8_K3_*A1_B4*msghfmink*` | `dpcc-t-tightened`, `hardflow_sls-t-tightened` | 7–10 |

Outputs: `logs/UAV_MIX/uav-pillars/plans/avoiding_bridge/<engine>/<train>/<eval>_msgp23pv2turbo/<seed>/results/halfspace_<geo>/`
(`<eval>-p23pv2turbo` where the source already carries a `_msg` tag) — per cell npz, cell png, `eval_<variant>.log`,
world png, sidecar, 3 MPC-foresight SVGs; GIFs only from P1a under `diagnostics/<variant>/rollout_<i>.gif`.
Live results stay where the avoiding evals write: `…/H8_K1_…_msgp23pv2live/6/results/` (MeanFlow tree) and
`plans/diffusion/H8_K20_…_thres0.5_msgp23pv2live/6/results/`; plant sidecars under `avoiding_bridge/_live/p23pv2live/`.
The live runs use `config/meanflow_projection_eval_u18_live.yaml` (`--config`) and `config/projection_eval_u18_live_dpcc.yaml`
(`FMPCC_PROJ_CFG`, a new one-line hook in `scripts/eval.py`) — copies of the shared yamls with only seeds / trials / variants
changed. `MF_BACKBONE=unet` selects the T1 checkpoints.

Read in the P1 log, per cell: `panda …` vs `drone …`, `agree k/n`, `ended {…}`, `plant track_err / gap_a p95 / contact eps`.
Read in the P2 logs: the eval's own `Success rate / Constraints satisfied / … / Average computation time` blocks and the
`[ uav-plant ] sidecar ->` lines. Comparison: `python uav_avoiding_bridge/compare_live_turbo.py --live <…_msgp23pv2live/6/results>
--turbo <…p23pv2turbo/6/results>` on the downloaded folders.

## 4 · Completion checks

| for | check |
| :-- | :-- |
| corridor | **68 result folders per scene**: `…_p23cv3t/6/corridor_cv3t_…/` and `…_p23cv3ah/6/corridor_cv3ah_…/`; `projection_health.n_tripped_trials = 0`; `divergence.n_aborted_trials` reported per cell; DA_UAV_v1 batch runs **per tag** (`p23cv3t`, then `p23cv3ah`), never pooled; `data_quality.csv` clean |
| pillars v2 | ten `p23pv2turbo` cells (each with `agree` line, npz, png, eval log, sidecar, 3 SVGs; T1/T4 with 2 GIFs); two `p23pv2live` runs (3 geometries each); the T-vs-L agreement from `compare_live_turbo.py` |
| both | download with `Slurm_Codes/download_remote_logs/export_to_laptop.sh` into `temp/2309/`; nothing pooled with older tags |

## 5 · Submission record (fill in)

| date | wave | job ids | outcome |
| :-- | :-- | :-- | :-- |
| 2026-09-22 | C0 pilots (as `u19smoke*`, U16 layout) | 26071 (hump), 26072 (tilt) | done; G1/G3 pass, G2 fail by plant-lag residue; author: run as is |
| 2026-09-23 | chain start | master 26078 (`p23_master_C1`) | ran; submitted C1, queued C2 |
| 2026-09-23 | **C1** `diffuser` (both scenes: mf/af K1,2,3, fm K1,2,3,5,20, diffusion K20) | 26079–26102 (24 eval jobs) | queued (`AssocGrpGRES` at submit time) |
| 2026-09-23 | C2 master | 26103 (`p23_master_C2`, Dependency: afterany C1) | pending |
| | C2 / C3 / C4 / C5 | (queued by each link in turn — read the `p23_master_*` logs in `Slurm_Codes/logs/<date>/`) | |

Claude (Fable 5.1, Claude Code) · 2026-09-23 · not run from the container.
