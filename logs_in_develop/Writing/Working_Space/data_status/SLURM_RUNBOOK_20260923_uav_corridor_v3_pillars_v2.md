# SLURM RUNBOOK 23-09 — UAV-corridor v3 (R33) and UAV-pillars v2 (R39), paper-only

**Companion of [`PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md`](PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md).**
Cluster i6-gpu-1, submit through `Slurm_Codes/submit.sh`. Every command is `plan` (dry) by default. Nothing here was
run from the container. Group C (R30) and the s-curve corridor-style notes of the 22-09 runbook are superseded by this file.

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

## 3 · Pillars v2 — commands

```
# gates G0/G1 (CPU, minutes): the pilot as shipped
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh                  # dry-run: lists the pilot cells
GO=1 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_avoiding_bridge/turbo.sh             # G1
```
Read: `agree k/20` ≥ 19 per cell, `contact eps 0` on legal episodes, `gap_a p95` within the Panda's. If not: `HZ=3`, then
`EXTRA="--gain pid_high_gain"`, then `REPLAYS=settle`, then `SCALE` up — one change at a time, re-run the pilot.

```
# P1 — the representative replay (CPU): five turbo.py calls, tag p23pv2turbo, clock replay
COMMON="--tag p23pv2turbo --replay clock --seeds 6 7 8 9 10 --geos top-left-hard top-right-hard both-hard"
python uav_avoiding_bridge/turbo.py $COMMON --engine meanflow   --eval-glob 'H8_K1_*'  --variants diffuser dpcc-t-tightened
python uav_avoiding_bridge/turbo.py $COMMON --engine alphaflow  --eval-glob 'H8_K1_*'  --variants diffuser dpcc-t-tightened
python uav_avoiding_bridge/turbo.py $COMMON --engine fmv3       --eval-glob 'K1_*'     --variants diffuser dpcc-t-tightened
python uav_avoiding_bridge/turbo.py $COMMON --engine dpcc       --eval-glob '*K20*'    --variants diffuser dpcc-c-tightened
python uav_avoiding_bridge/turbo.py --tag p23pv2turbo --replay clock --seeds 7 8 9 10 --engine meanflow \
       --eval-glob 'H8_K3_*A1_B4*msghfmink*' --variants dpcc-t-tightened hardflow_sls-t-tightened
```
Wrap the five calls in a copy of `turbo.sh` (`MODE=paper`) so they run as one CPU job; `--dry-run` first and confirm
**exactly ten cells** are listed (engine folder names: check `ls logs/avoiding-d3il/plans/` and adjust the `--engine`
substrings; the DPCC-protocol cells are the ones the 19-09 batch aggregated, `_msgdpccproto` where that tag exists).
GIFs: `GIF=2` through `turbo_gif.sh` (GPU) for the MeanFM K1 and diffusion K20 cells only.

```
# P2 — live closed loop (GPU): the avoiding eval sbatch with the plant switched in
FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=p23pv2live  <MeanFlow eval sbatch>  K=1  seed 6  variants diffuser,dpcc-t-tightened  2 episodes × 3 geometries
FMPCC_AVOIDING_PLANT=uav FMPCC_RUN_MSG=p23pv2live  <DPCC eval sbatch>      K=20 seed 6  variants dpcc-c-tightened           2 episodes × 3 geometries
```
`FMPCC_RUN_MSG` must contain `uav` (the factory refuses otherwise) — `p23pv2live` does. Log must show the plant's
`track_err / gap_a p95 / contacts` lines. Compare with T1/T4 per (geometry, episode).

## 4 · Completion checks

| for | check |
| :-- | :-- |
| corridor | 68 result folders under `…_p23cv3/6/corridor_cv3_…/`; `projection_health.n_tripped_trials = 0`; `divergence.n_aborted_trials` reported per cell; DA_UAV_v1 batch runs with tag `p23cv3` only; `data_quality.csv` clean |
| pillars v2 | ten `p23pv2turbo` cells with `agree` lines and sidecars; two `p23pv2live` cells; the T-vs-L agreement table |
| both | download with `Slurm_Codes/download_remote_logs/export_to_laptop.sh` into `temp/2309/`; nothing pooled with older tags |

## 5 · Submission record (fill in)

| date | wave | job ids | outcome |
| :-- | :-- | :-- | :-- |
| | | | |

Claude (Fable 5.1, Claude Code) · 2026-09-23 · not run from the container.
