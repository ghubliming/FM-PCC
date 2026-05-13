# iMeanFlow (iMF) SLURM Batch Scripts

SLURM batch submission scripts for training, evaluation, and results analysis of Improved Mean Flows (iMeanFlow).

## Scripts

### 1. `train_imf.sh`

**Purpose**: Train iMeanFlow models on synthetic trajectory data

**Resources**:
- GPU: 1x GPU
- CPU: 8 cores
- Memory: 32 GB
- Time: 24 hours
- Partition: gpu-1-student

**What it does**:
- Sets up Python environment and conda activation
- Configures PYTHONPATH for FM-PCC, d3il, and gym_avoiding_env
- Sets up MuJoCo rendering for headless nodes
- Runs synthetic data training example: `example_imf_training.py`
- Trains DualVelocityField models with u_first curriculum schedule
- Logs training progress (loss, metrics) every batch
- Saves best model checkpoint at end

**Usage**:
```bash
sbatch train_imf.sh
```

**Expected Output**:
```
Epoch 1/20
  Batch   0: loss=0.8234, L_u=0.4120, L_v=0.4114, w_u=0.80, w_v=0.00
  Train loss: 0.7234, Val loss: 0.7198
  ✓ New best model (val_loss=0.7198)
```

---

### 2. `eval_imf.sh`

**Purpose**: Evaluate and demonstrate iMeanFlow inference capabilities

**Resources**:
- GPU: 1x GPU
- CPU: 8 cores
- Memory: 32 GB
- Time: 4 hours
- Partition: gpu-1-student

**What it does**:
- Sets up Python environment and CUDA/rendering variables
- Configures PYTHONPATH for all required modules
- Runs comprehensive inference demonstrations: `example_imf_inference.py`
- Executes 5 inference scenarios:
  1. **Basic Sampling** - Single-step (NFE=1) vs Dual-step (NFE=2)
  2. **Multi-Phase Sampling** - 4-phase alternating u/v integration
  3. **Goal-Guided Sampling** - Steering trajectories toward goals
  4. **Obstacle Avoidance** - Collision-free trajectory generation
  5. **Velocity Decomposition** - Analyzing u vs v contributions
- Generates performance metrics and quality analysis

**Usage**:
```bash
sbatch eval_imf.sh
```

**Expected Output**:
```
Demo 1: Basic Sampling (Single-step vs Dual-step)
  Single-step output: torch.Size([4, 28])
  Change magnitude: 0.1234

Demo 3: Goal-Guided Sampling
  Initial distance to goal: 5.2341
  Distance after goal-guided sampling: 4.1256
  Goal guidance improvement: 21.3%
```

---

### 3. `load_results_imf.sh`

**Purpose**: Load results, run tests, and analyze iMeanFlow metrics

**Resources**:
- GPU: None (CPU only)
- CPU: 4 cores
- Memory: 16 GB
- Time: 30 minutes
- Partition: gpu-1-student

**What it does**:
- Sets up Python environment without GPU requirements
- Configures headless plotting (agg backend)
- Runs comprehensive unit test suite: `test_imf_core.py`
- Executes 65+ test cases covering:
  - Velocity field models (6 tests)
  - JVP guidance (5 tests)
  - ODE solvers (5 tests)
  - Training infrastructure (8 tests)
  - Metrics tracking (6 tests)
  - DiT transformer (3 tests)
  - Sampling API (7+ tests)
- Verifies model integrity and numerical correctness
- Generates test report with pass/fail summary

**Usage**:
```bash
sbatch load_results_imf.sh
```

**Expected Output**:
```
test_imf_core.py::TestDualVelocityField::test_dual_velocity_forward_shape PASSED
test_imf_core.py::TestImfODESolver::test_euler_step PASSED
test_imf_core.py::TestImfTrainingWrapper::test_compute_training_loss PASSED
...

==================== 65 passed in 23.45s ====================

==========================================
iMF Test Suite Completed
==========================================
Check test results above for any failures
```

---

## Quick Start

### Submit all jobs in sequence:
```bash
# Train model
JOB_TRAIN=$(sbatch train_imf.sh | awk '{print $NF}')

# Evaluate (after training completes)
sbatch --dependency=afterok:$JOB_TRAIN eval_imf.sh

# Load results (after eval completes)
sbatch --dependency=afterok:$JOB_EVAL load_results_imf.sh
```

### Submit with W&B logging:
Ensure W&B API key is stored in `$HOME/FMPCC/.wandb_api_key`:
```bash
echo "your_wandb_api_key_here" > $HOME/FMPCC/.wandb_api_key
chmod 600 $HOME/FMPCC/.wandb_api_key
sbatch train_imf.sh
```

---

## Environment Variables

All scripts automatically configure:

| Variable | Purpose |
|----------|---------|
| `FMPCC` | FM-PCC project root |
| `D3IL_ROOT` | D3IL submodule path |
| `PYTHONPATH` | Python module search paths |
| `MUJOCO_GL` | MuJoCo rendering backend (egl for headless) |
| `MPLBACKEND` | Matplotlib backend (agg for headless) |
| `WANDB_API_KEY` | Weights & Biases authentication |
| `WANDB_MODE` | W&B logging mode (online/offline) |

---

## Monitoring Jobs

### Check job status:
```bash
squeue -u $USER --format="%.18i %.20j %.8T %.10M %.10l %N"
```

### View live logs:
```bash
tail -f Slurm_Codes/logs/latest.log
```

### Check job details:
```bash
scontrol show job <JOB_ID>
```

### Cancel job:
```bash
scancel <JOB_ID>
```

---

## Customization

### Modify training hyperparameters:

Edit the training script to pass custom arguments:
```bash
# Instead of:
python flow_matcher_v3_imeanflow/examples/example_imf_training.py

# Use your custom training script:
python custom_train.py \
    --state-dim 28 \
    --hidden-dim 256 \
    --batch-size 32 \
    --learning-rate 1e-3 \
    --epochs 100 \
    --loss-schedule u_first
```

### Change GPU/memory allocation:

Edit the SBATCH headers at top of script:
```bash
#SBATCH --mem=64G          # Increase memory
#SBATCH --gres=gpu:2       # Use 2 GPUs
#SBATCH --time=48:00:00    # Extend time limit
```

### Disable W&B logging:

Remove or comment out the W&B section:
```bash
# if [ -f "$HOME/FMPCC/.wandb_api_key" ]; then
#     export WANDB_API_KEY=$(cat $HOME/FMPCC/.wandb_api_key)
#     export WANDB_MODE="online"
# fi
```

---

## Troubleshooting

### GPU out of memory:
- Reduce `batch_size` in training script
- Reduce `hidden_dim` in model initialization
- Use `eval_imf.sh` (evaluation doesn't require backprop)

### CUDA/MuJoCo errors:
- Ensure conda environment is activated
- Check PYTHONPATH is properly set
- Verify GPU drivers: `nvidia-smi`

### Job timeout:
- Check computation complexity
- Increase `--time=` limit
- Use `eval_imf.sh` (faster than training)

### Tests fail with import errors:
- Verify PYTHONPATH includes all required paths
- Check conda environment has torch, pytest installed
- Try running test locally first: `pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -v`

---

## Related Documentation

- [Phase 1 Completion Report](../../logs_in_develop/Gen3v4/Gen3v4_iMeanFlow_Phase1_Completion.md)
- [How to Run Guide](../../logs_in_develop/Gen3v4/HOW_TO_RUN.md)
- [iMF Configs](../../flow_matcher_v3_imeanflow/configs/)
- [Example Scripts](../../flow_matcher_v3_imeanflow/examples/)

---

**Last Updated**: May 2026  
**Status**: Phase 1 Complete ✅  
**Next**: Phase 2 (Training Integration with d3il)
