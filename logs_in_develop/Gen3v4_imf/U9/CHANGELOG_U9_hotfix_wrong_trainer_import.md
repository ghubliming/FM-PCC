# U9 Hotfix: `train(on_epoch_end=)` crash — wrong Trainer imported

**Date:** July 8, 2026
**Fixes:** regression from `CHANGELOG_U9_validation_loss_fix.md` (same folder).
**Symptom:** cluster job 23163 (git `da47e09`, seed 6) crashed at training start after ~24s:
`TypeError: Trainer.train() got an unexpected keyword argument 'on_epoch_end'` (`temp/slurm_debug.txt`).

## Root cause

The U9 Trainer edits (on_epoch_end callback, 3-tuple `test()` with `raw_mse`, seeded split) were
added to `flow_matcher_v3_imeanflow/utils/training.py` — but the train script imports
`diffuser.utils` and instantiated `diffuser.utils.Trainer` (the shared DPCC Trainer, which has none
of those edits). The script's `trainer.train(on_epoch_end=...)` call therefore hit a Trainer that
does not accept the kwarg. Confirmed by the run log: `Config: <class 'diffuser.utils.training.Trainer'>`.

## Changes

1. **`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`**: import the iMF package's own
   Trainer (`from flow_matcher_v3_imeanflow.utils.training import Trainer as iMFTrainer`) and use it
   in the `utils.Config(...)` call instead of `utils.Trainer`. Everything else (dataset/model/
   diffusion `Config`, `Parser`) stays on `diffuser.utils` — those ran fine before the crash and are
   untouched. This aligns the train script with the eval script, which already imports
   `flow_matcher_v3_imeanflow.utils`.

2. **`flow_matcher_v3_imeanflow/utils/training.py`**: `load()` now restores `test_raw_mse_losses`
   on resume (both the checkpoint-dict and losses.pkl fallback branches) — consistency with the
   U9 save paths; without it a resumed run reset the in-memory raw-mse list to empty.

## Deliberately NOT changed

- The shared `diffuser/utils/training.py` was left untouched. It is used by Gen0/Gen6v3/other gens,
  and the U9 3-tuple `test()` change would break their 2-tuple `test_loss, test_a0_loss = self.test()`
  callers. The iMF sibling package is the correct home for these edits (copy-modify convention).

## Verify on cluster

Re-submit `./Slurm_Codes/submit.sh Slurm_Codes/sbatch/iMF/train_imf.sh`. Expect: training proceeds
past step 0 (no TypeError), and W&B shows `train/loss` + `test/loss` + `val/raw_mse` updating per
epoch with `train_test_split=0.9` in the run config.
