# Gen11 E6 — U2: Per-Scene FM models + scene-keyed run structure — CHANGELOG

**Date:** 2026-06-22
**Implements:** [PLAN.md](PLAN.md). One FM **per scene** (state-only universal is underdetermined —
same state ⇒ contradictory actions across scenes; per-scene is well-defined because geometry is fixed
within a scene). Added a thin **outer loop** over the existing per-scene scripts; the FM model/dataset/
training code is **unchanged** (D1 = Option A from PLAN §5).

## Changes

**Config**
- `config/uav.py`: `max_path_length` **600 → 750** (s_curve reaches 22s×33Hz ≈ 726 steps; 600 silently
  truncated its tail).

**Eval — projection nesting (small edit)**
- `FM_v3_uav_test/eval_fm_uav.py`: new `--projection` arg (default `fm_only`); results now written to
  `<savepath>/eval/<projection>/results.json` (the **scene→…→seed→projection** level from PLAN §2).
  `fm_only` = state-only FM, no DPCC; the DPCC variants slot in here later with **zero** restructuring.
- `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`: `$4=projection` passthrough (default `fm_only`).

**New — roll-up**
- `FM_v3_uav_test/aggregate_scene_summaries.py` (pure stdlib): reads every
  `logs/uav-<scene>/*/<seed>/eval/<proj>/results.json` → per-scene `SCENE_SUMMARY.json` (mean±std across
  seeds) + cross-scene `logs/fm_uav_ALL_SCENES_SUMMARY.json`.

**New — outer-loop orchestrators (`Slurm_Codes/sbatch/uav_fm/`)**
- `train_all_scenes.sh "<scenes>" "<seeds>"` — submits one `train_fm_uav.sh <scene> <seed>` per combo.
- `eval_all_scenes.sh  "<scenes>" "<seeds>" <n_trials> <proj>` — submits one eval per combo, then a final
  `aggregate_summaries.sh` chained `afterok` on all evals.
- `fm_uav_all_pipeline.sh "<scenes>" "<seeds>" <n_trials> <proj>` — **one-shot**: per (scene,seed)
  train→eval chained, then a single aggregate after all evals. Fully batched (no login-session steps).
- `aggregate_summaries.sh` — CPU job wrapping `aggregate_scene_summaries.py`.

## Run structure produced
```
logs/uav-<scene>/<exp_name>/<seed>/        ← scene (top) → train line (weights/configs)
                              /eval/<proj>/results.json   ← eval line → seed → projection
logs/uav-<scene>/SCENE_SUMMARY.json        ← per scene, across seeds
logs/fm_uav_ALL_SCENES_SUMMARY.json        ← cross-scene roll-up
```
(D1=A: kept the existing `logs/uav-<scene>/…` Parser path — already scene-top — instead of restructuring
to `logs/fm_uav/<scene>/…`; zero FM-code change.)

## How to run
```bash
# one-shot, per-scene train→eval→aggregate (needs curated data/uav_fm/v1/ from Aux-1):
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh "empty corridor s_curve pillars" "5 6 7" 20
# or in phases:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh "empty corridor s_curve pillars" "5 6 7"
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh  "empty corridor s_curve pillars" "5 6 7"
```

## Verified
- `py_compile`: `config/uav.py`, `eval_fm_uav.py`, `aggregate_scene_summaries.py` — OK.
- `aggregate_scene_summaries.py` **functionally tested** on synthetic results (2 seeds → correct
  `success_rate_mean` + cross-scene file written).
- `bash -n` clean on all 7 `uav_fm/` sbatch scripts.

## Notes / not done
- **`--scene all` (pooled) is NOT in the loop** — kept only as an experimental flag on the underlying
  scripts; it stays meaningless for a state-only FM until Gen7 vision / scene-conditioning.
- DPCC projection variants: the `<projection>` level exists (`fm_only`); wiring real DPCC is Phase-3.
- No execution of train/eval here (Docker has no torch/GPU/MuJoCo) — cluster-only.
- Local only; no commit/push.
