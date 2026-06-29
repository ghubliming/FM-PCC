# UAV FM SLURM Scripts

All submitted via `bash Slurm_Codes/submit.sh <script> [args]` from repo root.
Seeds always loop **inside** one job — adding seeds adds time, not jobs.

---

## Scripts

| Script | What it does | Args |
|---|---|---|
| `train_fm_uav.sh` | Train one scene, N seeds sequentially | `$1=scene` `$2=seeds` |
| `eval_fm_uav.sh` | Eval one scene, N seeds sequentially | `$1=scene` `$2=seeds` `$3=n_trials` `$4=projection` |
| `aggregate_summaries.sh` | Roll up all `results.json` → `SCENE_SUMMARY.json` + `ALL_SCENES_SUMMARY.json` | `$1=scenes` `$2=projection` |
| `train_all_scenes.sh` | Submit one `train_fm_uav` job **per scene** in parallel | `$1=scenes` `$2=seeds` |
| `eval_all_scenes.sh` | Submit one `eval_fm_uav` job per scene in parallel, then auto-aggregate | `$1=scenes` `$2=seeds` `$3=n_trials` `$4=projection` |
| `fm_uav_pipeline.sh` | Single scene: train → eval chained (`afterok`) | `$1=scene` `$2=seed` `$3=n_trials` |
| `fm_uav_all_pipeline.sh` | All scenes: train→eval per scene in parallel, then aggregate | `$1=scenes` `$2=seeds` `$3=n_trials` `$4=projection` |

---

## Common usage patterns

**Quick single-scene test (one seed):**
```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_pipeline.sh pillars 6
```

**Full run — all scenes, multi-seed, auto-aggregate:**
```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh \
    "empty corridor s_curve pillars" "6 7 8 9 10"
```

**Train only (if eval will run later):**
```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh \
    "empty corridor s_curve pillars" "6 7 8 9 10"
```

**Eval only (after train done) + aggregate:**
```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh \
    "empty corridor s_curve pillars" "6 7 8 9 10"
```

**Manual aggregate (after partial eval):**
```bash
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/aggregate_summaries.sh \
    "empty corridor s_curve pillars" fm_only
```

---

## E8 (MJPC) — projection arg

E8 eval writes to `fm_only_ctrlmjpc` instead of `fm_only`. Pass it explicitly to aggregate:
```bash
# eval
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh \
    "empty corridor s_curve pillars" "6" "" fm_only_ctrlmjpc

# aggregate
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/aggregate_summaries.sh \
    "empty corridor s_curve pillars" fm_only_ctrlmjpc
```

---

## Outputs

```
logs/UAV_FM/
  uav-<scene>/
    flow_matching_v3_uav/<exp_name>/<seed>/   ← train checkpoints
    plans/<exp_name>/<seed>/fm_only[_tag]/    ← eval results.json, diagnostics/
    plans/SCENE_SUMMARY.json                  ← per-scene mean±std (aggregate)
  fm_uav_ALL_SCENES_SUMMARY.json              ← cross-scene roll-up (aggregate)
```

`<exp_name>` = `H8_D…ODE` (E7 default) or `H8_D…ODE_cmpos_only_ctrlmjpc` (E8). Set by `config/uav.py`.
