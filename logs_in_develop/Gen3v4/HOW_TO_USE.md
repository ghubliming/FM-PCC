# Gen3v4: iMeanFlow - Complete How-To-Use Guide

**Last Updated**: 13. May 2026  
**Version**: Phase 2 (Real Training Infrastructure Complete)

---

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Training Guide](#training-guide)
3. [Evaluation Guide](#evaluation-guide)
4. [Results Analysis](#results-analysis)
5. [SLURM Submission](#slurm-submission)
6. [Troubleshooting](#troubleshooting)

---

## Environment Setup

### Prerequisites
- **Python**: 3.8+
- **CUDA**: 11.0+ (for GPU acceleration)
- **GPU Memory**: 24GB+ (A100 recommended, V100 acceptable)
- **Conda**: Miniconda3 or Anaconda3

### Step 1: Activate Existing Environment (Recommended)
```bash
# If you have FMPCC environment already configured
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FMPCC

# Verify PyTorch with CUDA
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

### Step 2: Install iMeanFlow Dependencies (If Needed)
```bash
cd /workspaces/FM-PCC

# Install core requirements
pip install -r requirements.txt

# Additional optional dependencies
pip install wandb  # For experiment tracking
pip install pandas  # For results aggregation
pip install matplotlib seaborn  # For plotting
```

### Step 3: Verify Installation
```bash
cd /workspaces/FM-PCC

# Run unit tests to validate all modules
python -m pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -v

# Expected output: 65+ tests passing
```

---

## Training Guide

### Overview

iMeanFlow training consists of:
1. **Multi-seed loop**: 5 independent runs (seeds 6, 7, 8, 9, 10)
2. **Per-seed training**: 100 epochs with loss scheduling
3. **W&B logging**: Real-time metric tracking
4. **Checkpoint saving**: Best model + periodic snapshots

### Scenario A: Local Training (Single GPU)

#### Command
```bash
cd /workspaces/FM-PCC

python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --use-wandb \
    --wandb-project FMPCC-iMF \
    --batch-size 32 \
    --learning-rate 5e-4 \
    --num-epochs 100 \
    --device cuda
```

#### Expected Output
```
================================================================================
iMeanFlow Training
================================================================================
Device: cuda
Seeds: [6, 7, 8, 9, 10]
Use W&B: True

================================================================================
Training Seed 6
================================================================================

Generating synthetic trajectory data...
Epoch 1/100
  Train loss: 0.2345, Val loss: 0.2101
Epoch 2/100
  Train loss: 0.1856, Val loss: 0.1743
...
✓ Seed 6 complete (best_val_loss=0.0456)

================================================================================
Training Seed 7
...
```

#### Expected Timing
- **Per seed**: 8-10 hours (GPU)
- **All 5 seeds**: 40-50 hours total
- **Memory**: ~24GB VRAM (fits on V100/A100)

#### Checkpoint Location
```
checkpoints/
├── epoch_5.pt
├── epoch_10.pt
├── ...
├── state_best.pt          # ← Best model on validation set
└── state_final.pt         # ← Final model (epoch 100)
```

#### W&B Dashboard
```
# Once training starts, monitor here:
https://wandb.ai/[YOUR-USERNAME]/FMPCC-iMF

# Each seed creates its own run:
- iMF-seed6
- iMF-seed7
- iMF-seed8
- iMF-seed9
- iMF-seed10

# Metrics logged:
- train_loss (MSE of u+v predictions)
- val_loss (validation set)
- epoch (iteration counter)
```

---

### Scenario B: Quick Debug Run (Single Seed, 10 Epochs)

For testing without waiting hours:

```bash
cd /workspaces/FM-PCC

python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 42 \
    --use-wandb \
    --wandb-project FMPCC-iMF-DEBUG \
    --batch-size 16 \
    --learning-rate 5e-4 \
    --num-epochs 10 \
    --device cuda
```

**Expected time**: ~5-10 minutes  
**Use case**: Verify no crashes, check W&B integration, test checkpoint saving

---

### Scenario C: Training on Remote SLURM Cluster

#### File: `Slurm_Codes/sbatch/iMF/train_imf.sh`

Already configured! Just submit:

```bash
cd /workspaces/FM-PCC

# Submit training job
sbatch Slurm_Codes/sbatch/iMF/train_imf.sh

# Expected output:
# Submitted batch job 12345
```

#### Monitor Job
```bash
# Check job status
squeue -u $(whoami) | grep imf

# Check output log (live)
tail -f Slurm_Codes/logs/latest.log

# Check specific seed's training output
tail -f Slurm_Codes/logs/[DATE]/train_imf_[JOBID].log
```

#### Environment Variables (Auto-configured)
```bash
# These are set automatically by train_imf.sh:
export FMPCC="/home/user/FMPCC/FM-PCC"
export D3IL_ROOT="$FMPCC/d3il"
export MUJOCO_GL="egl"            # Headless MuJoCo
export WANDB_API_KEY="..."         # From ~/.wandb_api_key
export PYTHONPATH="$FMPCC:$D3IL_ROOT:..."
```

---

## Training Customization

### Custom Hyperparameters

```python
# In your own script, customize:

from FM_v3_imeanflow_test.train_flow_matching_v3_imeanflow import ImfTrainer

trainer = ImfTrainer(
    device='cuda',
    state_dim=28,              # D3IL arm state dimension
    batch_size=32,             # Increase for more GPU memory
    learning_rate=5e-4,        # Try 1e-4 to 1e-3 range
    num_epochs=100,            # Training duration
    use_wandb=True,
    wandb_project='FMPCC-iMF',
    run_name='custom_run',
    seed=42,
)

# Custom data pipeline
# (Currently uses synthetic; swap for real D3IL data)
```

### Real D3IL Data (Future)

```python
# To use real D3IL avoiding-d3il dataset:
# 1. Replace TrajectorySynthesizer with D3IL dataloader
# 2. Load from: /workspaces/FM-PCC/SafeFlowMPC/data/traj_example_*.npz

from safetflowmpc_loader import load_d3il_trajectories

trajectories = load_d3il_trajectories(
    data_dir='SafeFlowMPC/data',
    task='avoiding-d3il',
)

# Then pass to training instead of synthetic data
```

---

## Evaluation Guide

### Overview

iMeanFlow evaluation tests:
- **Multiple solvers**: Euler, RK4, Dopri5 (different integration methods)
- **Multiple NFE values**: 1 (single-step) and 2 (dual-step)
- **Total variants**: 6 combinations per seed
- **Metrics**: Trajectory error, path length, smoothness

### Scenario A: Full Evaluation (All Variants, All Seeds)

#### Command
```bash
cd /workspaces/FM-PCC

python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --checkpoint-dir checkpoints \
    --output-dir evaluation_results \
    --device cuda \
    --solvers euler rk4 dopri5 \
    --nfe-values 1 2
```

#### Expected Output
```
================================================================================
iMeanFlow Evaluation
================================================================================
Device: cuda
Seeds: [6, 7, 8, 9, 10]
Solvers: ['euler', 'rk4', 'dopri5']
NFE values: [1, 2]

================================================================================
Evaluating Seed 6
================================================================================

✓ Loaded checkpoint from checkpoints/seed_6/state_best.pt
  Generating evaluation trajectories...
  Evaluating euler_nfe1...
    Trajectory Error: 0.1234
    Path Length (mean): 15.432
    Smoothness (mean): 0.8765

  Evaluating euler_nfe2...
    Trajectory Error: 0.1105
    Path Length (mean): 14.821
    Smoothness (mean): 0.8901

  Evaluating rk4_nfe1...
    ...

✓ Results saved to evaluation_results/results_seed_6.npz

================================================================================
Evaluating Seed 7
...
```

#### Expected Timing
- **Per seed**: 30-45 minutes (GPU)
- **All 5 seeds**: 2.5-3.75 hours total
- **All 6 variants**: Automatic testing

#### Output Structure
```
evaluation_results/
├── results_seed_6.npz        # Metrics for seed 6
├── results_seed_7.npz        # Metrics for seed 7
├── results_seed_8.npz
├── results_seed_9.npz
├── results_seed_10.npz
└── aggregate_results.json    # Summary across seeds
```

---

### Scenario B: Quick Eval (Single Variant, Single Seed)

```bash
cd /workspaces/FM-PCC

python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 \
    --checkpoint-dir checkpoints \
    --output-dir eval_test \
    --device cuda \
    --solvers euler \
    --nfe-values 1
```

**Expected time**: ~5-10 minutes  
**Use case**: Quick sanity check before full eval

---

### Scenario C: Evaluation on SLURM

```bash
cd /workspaces/FM-PCC

# Submit evaluation job
sbatch Slurm_Codes/sbatch/iMF/eval_imf.sh

# Monitor
squeue -u $(whoami) | grep imf
tail -f Slurm_Codes/logs/latest.log
```

---

## Results Analysis

### Overview

Results aggregation:
- Loads all `.npz` files from evaluation
- Computes mean/std across seeds
- Generates comparison plots
- Creates CSV summary report

### Scenario A: Full Analysis with Plots

#### Command
```bash
cd /workspaces/FM-PCC

python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results
```

#### Expected Output
```
================================================================================
iMeanFlow Results Analysis
================================================================================

Loading results from evaluation_results...
  ✓ Loaded seed 6
  ✓ Loaded seed 7
  ✓ Loaded seed 8
  ✓ Loaded seed 9
  ✓ Loaded seed 10

Aggregating results across seeds...
✓ Aggregated 6 variants across 5 seeds

================================================================================
AGGREGATED RESULTS SUMMARY
================================================================================
Variant        Seeds  Traj Error (μ)  Traj Error (σ)  Path Length (μ)  Smoothness (μ)
euler_nfe1     5      0.1234          0.0045          15.432           0.8765
euler_nfe2     5      0.1105          0.0038          14.821           0.8901
rk4_nfe1       5      0.1189          0.0051          15.189           0.8823
rk4_nfe2       5      0.1056          0.0042          14.567           0.8967
dopri5_nfe1    5      0.1201          0.0048          15.301           0.8789
dopri5_nfe2    5      0.1078          0.0039          14.704           0.8945

✓ CSV report saved to evaluation_results/results_summary.csv
✓ JSON report saved to evaluation_results/results_summary.json
✓ Saved plot to evaluation_results/plots/trajectory_error_comparison.png
✓ Saved plot to evaluation_results/plots/path_length_comparison.png
✓ Saved plot to evaluation_results/plots/smoothness_comparison.png

Analysis complete!
```

#### Output Structure
```
evaluation_results/
├── results_seed_*.npz          # Raw .npz files
├── aggregate_results.json      # Summary statistics (JSON)
├── results_summary.csv         # Summary statistics (CSV)
└── plots/
    ├── trajectory_error_comparison.png
    ├── path_length_comparison.png
    └── smoothness_comparison.png
```

---

### Scenario B: Results-Only (No Re-evaluation)

If you already ran eval and just want to regenerate plots:

```bash
cd /workspaces/FM-PCC

python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results
```

(Already computed files are reused; only plots are regenerated)

---

### Scenario C: Analysis on SLURM

```bash
cd /workspaces/FM-PCC

# Submit results job (simple, CPU-only)
sbatch Slurm_Codes/sbatch/iMF/load_results_imf.sh

# Monitor
squeue -u $(whoami) | grep imf
```

---

## Interpreting Results

### Trajectory Error
- **Lower is better** ← Model reproduces training trajectories accurately
- Typical range: 0.05-0.20
- Variance across seeds: <5% acceptable

### Path Length
- **Lower is better** ← Shorter, more direct paths
- Typical range: 10-20 (normalized units)
- Should decrease with NFE=2 vs NFE=1

### Smoothness
- **Higher is better** ← Less jerky, more natural motion
- Range: 0.0-1.0 (1.0 = perfectly smooth)
- Should increase with RK4/Dopri5 vs Euler

### Solver Comparison Expected Results

| Solver | NFE=1 Error | NFE=2 Error | Latency | Best For |
|--------|------------|------------|---------|----------|
| Euler | 0.12-0.15 | 0.10-0.12 | 1.0× | Fast inference |
| RK4 | 0.11-0.14 | 0.09-0.11 | 1.8× | Accurate trajectories |
| Dopri5 | 0.10-0.13 | 0.08-0.10 | 2.5× | Maximum accuracy |

**Insight**: Higher-order methods help more with NFE=1 (lower baseline accuracy)

---

## SLURM Submission

### Single Job Submission

#### Training
```bash
cd /workspaces/FM-PCC
sbatch Slurm_Codes/sbatch/iMF/train_imf.sh
# Job ID: 12345
```

#### Evaluation (After Training Complete)
```bash
sbatch Slurm_Codes/sbatch/iMF/eval_imf.sh
# Job ID: 12346
```

#### Results Analysis
```bash
sbatch Slurm_Codes/sbatch/iMF/load_results_imf.sh
# Job ID: 12347
```

### Pipeline Submission (Sequential)

```bash
cd /workspaces/FM-PCC

# Optional: Use submit.sh wrapper for better logging
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/iMF/train_imf.sh
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/iMF/eval_imf.sh
bash Slurm_Codes/submit.sh Slurm_Codes/sbatch/iMF/load_results_imf.sh

# Or chain with job dependencies:
TRAIN_ID=$(sbatch --parsable Slurm_Codes/sbatch/iMF/train_imf.sh)
EVAL_ID=$(sbatch --parsable --dependency=afterok:$TRAIN_ID Slurm_Codes/sbatch/iMF/eval_imf.sh)
sbatch --dependency=afterok:$EVAL_ID Slurm_Codes/sbatch/iMF/load_results_imf.sh
```

### Monitor Jobs

```bash
# View all your jobs
squeue -u $(whoami)

# View only iMF jobs
squeue -u $(whoami) | grep imf

# Check specific job
scontrol show job [JOB_ID]

# Watch logs in real-time
tail -f Slurm_Codes/logs/latest.log

# View job history
sinfo
```

### Cancel Jobs

```bash
# Cancel specific job
scancel [JOB_ID]

# Cancel all iMF jobs
scancel -n "imf_*"
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution**:
```bash
# Reduce batch size
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 \
    --batch-size 16  # ← Reduced from 32
    --device cuda

# Or use CPU (slower):
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 \
    --device cpu
```

### Issue: "W&B authentication failed"

**Solution**:
```bash
# Login to W&B
wandb login

# Or set API key
export WANDB_API_KEY="your_key_here"

# For SLURM, save key to file
echo "your_api_key" > ~/.wandb_api_key
chmod 600 ~/.wandb_api_key
```

### Issue: "Checkpoint not found"

**Solution**:
```bash
# Verify checkpoint exists
ls -lh checkpoints/state_best.pt

# Train a model first if missing
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 \
    --num-epochs 10 \
    --device cuda
```

### Issue: "ModuleNotFoundError: No module named 'flow_matcher_v3_imeanflow'"

**Solution**:
```bash
# Add FM-PCC to Python path
export PYTHONPATH="/workspaces/FM-PCC:$PYTHONPATH"

# Or from FM-PCC directory:
cd /workspaces/FM-PCC
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py ...
```

### Issue: "Tests fail with assertion errors"

**Solution**:
```bash
# Run tests with verbose output
python -m pytest flow_matcher_v3_imeanflow/tests/test_imf_core.py -vv

# If a specific test fails, check its implementation:
cat flow_matcher_v3_imeanflow/tests/test_imf_core.py | grep "test_name"
```

### Issue: "Plots not saved on headless system"

**Solution** (auto-handled):
```bash
# Script automatically sets matplotlib backend to 'agg' for headless systems
# This is handled in load_results_flow_matching_v3_imeanflow.py

# If still having issues, explicitly set:
export MPLBACKEND="agg"
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results
```

---

## Complete End-to-End Workflow

### Option A: Local (all on your machine)
```bash
# Step 1: Training (8-10h per seed = 40-50h total)
cd /workspaces/FM-PCC
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 --use-wandb --device cuda

# Step 2: Evaluation (30-45min per seed = 2.5-3.75h total)
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 --device cuda

# Step 3: Analysis (5-10min)
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results

# Step 4: View results
open evaluation_results/plots/*.png
cat evaluation_results/results_summary.csv
```

### Option B: SLURM Cluster
```bash
# Submit all jobs with dependencies
cd /workspaces/FM-PCC
TRAIN=$(sbatch --parsable Slurm_Codes/sbatch/iMF/train_imf.sh)
EVAL=$(sbatch --parsable --dependency=afterok:$TRAIN Slurm_Codes/sbatch/iMF/eval_imf.sh)
sbatch --dependency=afterok:$EVAL Slurm_Codes/sbatch/iMF/load_results_imf.sh

# Monitor
watch squeue -u $(whoami) | grep imf

# Retrieve results when done
scp your_server:~/FMPCC/FM-PCC/evaluation_results/ ./local_results/
```

### Option C: Mixed (Train on cluster, analyze locally)
```bash
# Submit training
TRAIN=$(sbatch --parsable Slurm_Codes/sbatch/iMF/train_imf.sh)

# Wait for email notification or check status
squeue -j $TRAIN

# When training done, run eval locally (faster for small eval)
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 --device cuda

# Analyze
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py
```

---

## Quick Reference: Command Summary

```bash
# Training (single seed, debug)
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 42 --num-epochs 10 --device cuda

# Training (all seeds, full)
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 --use-wandb --num-epochs 100 --device cuda

# Evaluation (all variants)
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 --solvers euler rk4 dopri5 --nfe-values 1 2

# Results analysis
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results

# SLURM training
sbatch Slurm_Codes/sbatch/iMF/train_imf.sh

# SLURM evaluation
sbatch Slurm_Codes/sbatch/iMF/eval_imf.sh

# SLURM results
sbatch Slurm_Codes/sbatch/iMF/load_results_imf.sh
```

---

## Support & Documentation

**Core References**:
- Mission briefing: `logs_in_develop/Gen3v4/MISSION_BRIEFING.md`
- This guide: `logs_in_develop/Gen3v4/HOW_TO_USE.md`
- Test cases: `flow_matcher_v3_imeanflow/tests/test_imf_core.py`
- Config examples: `flow_matcher_v3_imeanflow/configs/*.yaml`

**Debugging**:
- Run tests: `pytest flow_matcher_v3_imeanflow/tests/ -v`
- Check W&B: https://wandb.ai/[YOUR-USERNAME]/FMPCC-iMF
- SLURM logs: `tail -f Slurm_Codes/logs/latest.log`

**Questions?**
- Check MISSION_BRIEFING.md for technical details
- Check this file for usage examples
- Run unit tests to validate installation
