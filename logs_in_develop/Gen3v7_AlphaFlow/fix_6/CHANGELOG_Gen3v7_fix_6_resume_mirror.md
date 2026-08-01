# CHANGELOG — Gen3v7 fix_6: resume, mirrored from Gen3v6

**Date:** 2026-08-01 · **Full write-up:** [`../../Gen3v6_MeanFlow/fix_6/CHANGELOG_Gen3v6_fix_6_resume.md`](../../Gen3v6_MeanFlow/fix_6/CHANGELOG_Gen3v6_fix_6_resume.md)

Gen3v7 was **not** the generation that broke — job **24070** completed all four seeds
(`13_22_43_train_alphaflow_24070.log:1277`). Gen3v6's job 24069 died at seed 9 / step 80000
on a full disk with no way to continue, and Gen3v7 inherited the same missing resume path by
copy-modify. This mirror is preventative.

**Files changed (identical in substance to Gen3v6):**

- `FM_v3_alphaflow_test/train_flow_matching_v3_alphaflow.py` — `--auto-resume`,
  `--resume-step`, `--resume-seed`, `--force-restart`, plus `find_latest_checkpoint()`,
  `training_already_complete()`, `resolve_resume()`.
- `flow_matcher_v3_alphaflow/utils/training.py` — `optimizer` / `lr_scheduler` /
  `best_test_loss` added to `_checkpoint_payload()`; new `_restore_optimizer_state()` called
  from `load()`, with an LR-schedule fast-forward for pre-fix_6 checkpoints.
- `Slurm_Codes/sbatch/AlphaFlow/train_alphaflow.sh` — `TRAIN_SEEDS` / `AUTO_RESUME` env
  vars, `"$@"` forwarded, disk pre-flight.

**Gen3v7-specific note.** The α anneal needs no restoring: `train_epoch` recomputes it from
`self.step` through `set_train_step` on every step (`training.py:179-180`), so a resumed run
picks the schedule up exactly where it stopped. That also makes `train/alpha` the fastest
resume check there is — **after any resume, look at `train/alpha` first**: if it jumps back
toward 1.0, the step counter was not restored and the run is contaminated (PLAN §11 trap 1).

Usage:

```bash
TRAIN_SEEDS="6 7 8 9 10" ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/AlphaFlow/train_alphaflow.sh
```
