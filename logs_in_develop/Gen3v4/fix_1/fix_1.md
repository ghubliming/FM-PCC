# iMeanFlow (iMF) Pipeline Alignment Hotfix

This hotfix standardizes the iMeanFlow (iMF) training and evaluation infrastructure to match the established **FMv3-ODE** project standards. It transitions the iMF engine from a standalone synthetic test script to a production-ready pipeline integrated with the core D3IL dataset.

## Changes

### 1. Structural Alignment
- **New Wrapper**: Created `flow_matcher_v3_imeanflow.models.ImfDiffusion`, a `nn.Module` wrapper that provides a `.loss()` method compatible with the project's standard `Trainer` class.
- **Config Integration**: Updated `config/avoiding-d3il.py` to define the `flow_matching_v3_imeanflow` experiment block, including model, diffusion wrapper, and training hyperparameters.
- **Boilerplate Standardization**: Refactored `train_flow_matching_v3_imeanflow.py` and `eval_flow_matching_v3_imeanflow.py` to use the standard `utils.Parser`, `utils.Config`, and `utils.load_diffusion` patterns.

### 2. Training Improvements
- **Real Data**: The training script now loads the actual `avoiding-d3il` dataset instead of generating synthetic trajectories.
- **Dual Velocity Loss**: Integrated the `ImfTrainingWrapper` which handles the global ($u$) and local ($v$) velocity loss components with an automated curriculum scheduler.
- **Logging Cleanup**: Removed verbose `tqdm` progress bars that were bloating SLURM logs. Replaced with clean `[ train ]` and `[ eval ]` status reports.
- **CLI Argument Support**: Fixed the `Parser` to accept and process standard cluster-side arguments (`--batch-size`, `--learning-rate`, `--num-epochs`, `--device`). Implemented an automated mapping from `--num-epochs` to project-standard `n_train_steps`.

### 3. Checkpoint Management
- **Deterministic Paths**: Paths are now derived from the config, ensuring consistency: `logs/avoiding-d3il/flow_matching_v3_imeanflow/.../seed_X`.
- **Standard Naming**: Models are saved as `state_best.pt` and periodic `state_<epoch>.pt` files.

## Files Modified
- `flow_matcher_v3_imeanflow/models/imf_diffusion.py` (New)
- `config/avoiding-d3il.py`
- `FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`
- `FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py`

## Verification
To run the standardized training:
```bash
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10 --use-wandb
```
To run the evaluation:
```bash
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10
```
