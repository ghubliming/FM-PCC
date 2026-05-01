# W&B Artifact Upload Hotfix Report

## Issue Description
A critical issue was identified in the training scripts where multi-seed jobs would crash after completing the first seed. The crash was caused by an `AttributeError` in the `upload_wandb_artifact` function.

### Root Cause
The code attempted to call `run.Artifact(...)`, but `Artifact` is a class within the `wandb` module, not a method of the `wandb.Run` object. Additionally, the `wandb` module was imported locally within the training loop, making it unavailable to the top-level function `upload_wandb_artifact` unless explicitly passed or imported globally.

### Impact
- Training jobs were terminated prematurely after the first seed (e.g., job 19765 stopped after seed 6).
- Subsequent seeds (7, 8, 9, 10) were never executed.
- Storage on W&B was being consumed by large model weights (`state_best.pt`) which are not always required for remote logging.

## Applied Hotfix

### 1. Fixed AttributeError and Scoping
The following scripts were updated to move `import wandb` to the global scope and use `wandb.Artifact` instead of the incorrect `run.Artifact`:
- `scripts/train.py`
- `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py`
- `FM_Unet_v2_test/train_FM_Unet_v2.py`
- `FM_v3_test/train_FM_v3.py`
- `FM_v2_test/train_FM_v2.py`
- `FM_test/train_FM.py`
- `FM_hp_tune_test/train_FM_hp_tune.py`

### 2. Disabled Model Weight Uploads
In all the above scripts, the upload of `state_best.pt` has been commented out to save W&B storage space. The code remains in the scripts for future reference but is currently inactive. The following files are still uploaded:
- `losses.pkl`
- `args.json`

## Verification
The changes ensure that the artifact preparation logic is mathematically and programmatically correct according to the W&B API. Jobs should now proceed through all seeds without interruption from this specific error.
