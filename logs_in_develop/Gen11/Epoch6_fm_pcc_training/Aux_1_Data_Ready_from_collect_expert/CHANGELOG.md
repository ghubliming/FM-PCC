# Gen11 E6 — Aux-1: batched "collect → data-ready" SLURM jobs — CHANGELOG

**Date:** 2026-06-22
**Why:** the steps between the E4 expert collect and a train-ready UAV FM dataset (curate raw→flat, verify)
were login-node manual steps — a disconnect would lose them. Wrapped them as **batch SLURM jobs** so the
whole path runs unattended and can't be killed by getting kicked out.

## What was built — `Slurm_Codes/sbatch/uav_fm_data_ready/` (new folder)

| Script | Resources | Does |
|---|---|---|
| `prepare_uav_fm_data.sh` | CPU, 30 min | Curate raw `logs/uav_expert_data/<scene>/<homotopy>/*.pkl` → **flat** `data/uav_fm/v1/<scene>/*.pkl` (+ `manifest.json`), then **verify** every scene has >0 episodes — fails loudly here (cheap) instead of mid-train. Prints `DATASET READY ✓`. |
| `mini_fm_gate.sh` | GPU, 2 h | **Optional** — trains the tiny `mini_fm_sanity.py` FM on the curated data, checks held-out RMS (action-convention / shape catch). |
| `collect_to_ready_pipeline.sh` | orchestrator, 10 min | **One-shot, fully batched:** submits the 4 E4 collect jobs + `prepare` (chained `--dependency=afterok` on all 4); `… <n_trials> gate` also chains the mini-FM gate after prepare. Thin submitter — exits immediately, children run on their own. |

## The flow

```
collect_to_ready_pipeline.sh
  ├─ collect.sh empty 500 ─┐
  ├─ collect.sh corridor 500 ─┤
  ├─ collect.sh s_curve 500 ─┤ afterok (all 4)
  └─ collect.sh pillars 500 ─┘
        └─ prepare_uav_fm_data.sh  (curate → flat data/uav_fm/v1/ + verify)  → "DATASET READY ✓"
              └─ [optional] mini_fm_gate.sh
                    → then: train_fm_uav.sh all 5
```

## How to run

```bash
# A) full one-shot (collect everything + curate + verify), then train:
rm -rf logs/uav_expert_data/pillars     # clear old 274 first (U9 "do not mix")
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/collect_to_ready_pipeline.sh 500
# (add 'gate' as 2nd arg to also run the mini-FM gate)

# B) collects already done → just make it train-ready:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm_data_ready/prepare_uav_fm_data.sh

# then, once prepare logs "DATASET READY ✓":
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh all 5
```

## Key correctness points
- **Curate is required, not optional:** raw pkls live in homotopy *subdirs*; the trainer's loader lists a
  scene dir **non-recursively**, so it needs the **flat** `data/uav_fm/v1/<scene>/*.pkl` that curate
  produces (it also drops `run_summary.json`). Manual move must flatten too.
- **Pillars:** `BLEND_RADIUS=0.45` (U9 fix) is already in `trajectories.py`; clear the old pillars dir
  before a fresh collect so they don't mix.
- Repo-root resolution + conda-activate mirror `collect.sh`; partition `gpu-1-student`; logs go through
  `submit.sh`'s dated tree.

## Verified
- `bash -n` clean on all 3 scripts; the embedded verify-Python parses (`ast.parse`).
- No execution here (Docker has no SLURM/MuJoCo/torch) — runs are cluster-only.

## Not done / notes
- `collect.sh` writes to the fixed `logs/uav_expert_data/<scene>/` (no per-run subdir); fresh pillars assumes
  a cleared dir (the pipeline echoes the reminder, doesn't auto-`rm`).
- Local only; no commit/push.
