# F1 — Fix defective SLURM submit logic (sbatch-storm risk)

## Problem

The U2 orchestrator scripts (`train_all_scenes.sh`, `eval_all_scenes.sh`,
`fm_uav_all_pipeline.sh`) submitted **one `sbatch` job per (scene, seed)
combination** via nested bash loops. With the default 4 scenes × 3 seeds:

- `train_all_scenes.sh`: 12 train jobs
- `eval_all_scenes.sh`: 12 eval jobs + 1 aggregate = 13 jobs
- `fm_uav_all_pipeline.sh`: 12 train + 12 eval (chained) + 1 aggregate = 25 jobs

Job count scaled with `num_scenes * num_seeds`, with no cap — adding seeds
silently multiplied the number of `sbatch` calls fired in one shot. This risks
hitting the cluster's job-submission policy and getting the account flagged.

## Fix

Moved the seed loop **inside** the job scripts themselves, so seeds run
sequentially within a single job allocation instead of spawning one job each:

- `train_fm_uav.sh` / `eval_fm_uav.sh`: `$2` is now a **quoted, space-separated
  list of seeds** (default `"5 6 7"`), looped with a plain `for seed in $SEEDS`
  inside the one SLURM allocation.
- `train_all_scenes.sh` / `eval_all_scenes.sh` / `fm_uav_all_pipeline.sh`: the
  scene loop stays (scenes get separate per-scene FM models, so they
  legitimately need separate jobs), but the seed loop was removed — each
  scene now gets exactly **one** train job and **one** eval job, regardless of
  how many seeds are requested.

Total `sbatch` calls is now `2 * num_scenes + 1`, independent of seed count:

| scenario                  | before (jobs) | after (jobs) |
|----------------------------|--------------:|-------------:|
| 4 scenes × 3 seeds (default) | 25          | 9             |
| 4 scenes × 10 seeds           | 81          | 9             |
| 1 scene × 3 seeds              | 7            | 3             |

Adding seeds now only changes `--time`, never the job count.

## Time-budget tradeoff

Since seeds run sequentially in one allocation, the per-job `--time` ceiling
must scale with seed count. The orchestrators now compute this and pass it as
an `sbatch --time=...` override (the in-script `#SBATCH --time` is just the
single-seed fallback for direct/manual submission):

- train: `seed_count * 24h`
- eval: `seed_count * 8h`

**Caveat**: these per-seed ceilings (24h train / 8h eval) were already
generous upper bounds for a single (scene, seed) run, not measured real
runtimes. With many seeds the computed `--time` can exceed the
`gpu-1-student` partition's max walltime — check the partition limit
(`sinfo -p gpu-1-student -o "%l"`) before submitting a large seed list, and
lower the per-seed assumption in the orchestrator if real runtimes are much
shorter than the ceiling.

## Files changed

- `Slurm_Codes/sbatch/uav_fm/train_fm_uav.sh`
- `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`
- `Slurm_Codes/sbatch/uav_fm/train_all_scenes.sh`
- `Slurm_Codes/sbatch/uav_fm/eval_all_scenes.sh`
- `Slurm_Codes/sbatch/uav_fm/fm_uav_all_pipeline.sh`
- `aggregate_summaries.sh` unchanged (already a single roll-up job).
