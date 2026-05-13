# Quick Reference: iMF-PCC Operations

**Location**: `/workspaces/FM-PCC/logs_in_develop/Gen3v4/fix_3/`  
**Last Updated**: May 13, 2026

---

## In This Directory

| Document | Purpose | Audience |
|----------|---------|----------|
| `REAL_IMF_IMPLEMENTATION.md` | Implementation report (problem → solution) | Project leads, researchers |
| `ARCHITECTURE_OVERVIEW.md` | Technical architecture (4 layers + flows) | Developers, ML engineers |
| `INTEGRATION_GUIDE.md` | How iMF integrates with FM-PCC | Integration engineers, DevOps |
| `FILES_CHANGED.md` | Complete file manifest + changes | Code reviewers, documentation |
| `OPERATIONS_CHECKLIST.md` → **YOU ARE HERE** | Quick operations reference | Operators, data scientists |

---

## Quick Commands

### Train Multi-Seed iMF
```bash
cd /workspaces/FM-PCC

# Standard: 5 seeds
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --use-wandb \
    --wandb-project FMPCC-iMF

# Or via SLURM
sbatch Slurm_Codes/sbatch/iMF/train_imf.sh
```

**Checkpoints created**:
```
logs/avoiding-d3il/flow_matching_v3_imeanflow/{exp_name}/
├── seed_6/
│   ├── state_0.pt
│   ├── state_5000.pt
│   ├── state_best.pt
│   ├── losses.pkl
│   └── args.json
├── seed_7/
├── seed_8/
├── seed_9/
└── seed_10/
```

### Evaluate Training
```bash
# Evaluate all seeds
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --results-dir evaluation_results/imf

# Or via SLURM
sbatch Slurm_Codes/sbatch/iMF/eval_imf.sh
```

**Results file**:
```
evaluation_results/imf/eval_results.json
```

### Display Results
```bash
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results/imf

# Output:
# Per-Seed Results:
# ------
# Seed      MSE Error    Std Dev   Samples
# ------
#    6         0.0234       0.0012       50
#    7         0.0231       0.0011       50
#    8         0.0235       0.0013       50
#    9         0.0229       0.0010       50
#   10         0.0232       0.0012       50
# ------
# MEAN         0.0232       0.0012       50
# ------
```

---

## Training Parameters

### iMF-Specific (3 LOCKED params in config)
```python
'u_loss_weight': 0.5,           # Mean velocity loss weight
'v_loss_weight': 0.5,           # Instantaneous velocity loss weight
'loss_schedule': 'u_first',     # Curriculum: 'u_first' or 'balanced'
```

### Curriculum Phases (auto, no tuning needed)
```python
'warmup_epochs': 30,            # Epochs 0-30: u-only training
'transition_epochs': 30,        # Epochs 30-60: blend to u+v
# Epochs 60+: balanced u+v (0.5 * u_loss + 0.5 * v_loss)
```

### Architecture (inherited from FMv3ODE)
```python
'freq_dim': 256,                # Feature dimension
'depth': 8,                     # U-Net depth
'num_heads': 4,                 # Attention heads
'time_dim': 256,                # Time embedding dimension
```

### Training (standard FM-PCC)
```python
'n_train_steps': 100000,        # Total training steps
'batch_size': 32,               # Batch size
'learning_rate': 5e-4,          # Learning rate
'ema_decay': 0.995,             # EMA for model averaging
```

### ODE Inference (fast single-step)
```python
'ode_inference_steps_v3': 1,    # Number of ODE steps (1=fast, 8=slow)
```

---

## W&B Dashboard

### Expected Metrics
```
train/u_loss         → MSE loss for u_pred
train/v_loss         → MSE loss for v_pred
train/loss           → Total: u_weight*u_loss + v_weight*v_loss
train/u_weight       → Current curriculum scale for u (1.0 → 0.5)
train/v_weight       → Current curriculum scale for v (0.0 → 0.5)
test/loss            → Validation loss (if applicable)
```

### Curriculum Progression
```
Phase 1 (epochs 0-30):
  train/u_weight:  1.0 (constant)
  train/v_weight:  0.0 (constant)
  → Only u is trained

Phase 2 (epochs 30-60):
  train/u_weight:  1.0 → 0.5 (decreasing)
  train/v_weight:  0.0 → 0.5 (increasing)
  → Gradual transition to dual-loss

Phase 3 (epochs 60+):
  train/u_weight:  0.5 (constant)
  train/v_weight:  0.5 (constant)
  → Both u and v equally weighted
```

---

## Troubleshooting

### ❌ "ModuleNotFoundError: No module named 'flow_matcher_v3_imeanflow.models.iMeanFlowEngine'"

**Check**:
```bash
cat flow_matcher_v3_imeanflow/models/__init__.py
# Should include:
# from .imf_engine import iMeanFlowEngine
# from .imf_diffusion import iMFDiffusion
```

**Fix**: Re-export in `__init__.py` if missing.

---

### ❌ Training gets stuck or loss explodes

**Check curriculum**:
```python
# Increase warmup if v-head is too strong
'warmup_epochs': 40,        # from 30
'transition_epochs': 40     # from 30
```

**Check learning rate**:
```python
'learning_rate': 1e-4,      # try lower than 5e-4
```

---

### ❌ Checkpoints are missing

**Check config**:
```bash
grep -E "(logbase|prefix|exp_name)" config/avoiding-d3il.py | grep -A3 flow_matching_v3_imeanflow
```

Should have:
```python
'logbase': 'logs',
'prefix': 'flow_matching_v3_imeanflow/',
'exp_name': watch(args_to_watch_fmv3_ode_train),
```

---

### ❌ W&B not logging

**Check WANDB_SERVICE**:
```bash
unset WANDB_SERVICE
unset WANDB__SERVICE
python train_flow_matching_v3_imeanflow.py --seed=6 --use-wandb
```

---

## Checkpoint Management

### Save Location
```
logs/avoiding-d3il/flow_matching_v3_imeanflow/
├── H8_D.../               ← Experiment name (from config)
│   ├── seed_6/
│   │   ├── state_0.pt      (epoch 0 checkpoint)
│   │   ├── state_5000.pt   (epoch 5000 checkpoint)
│   │   ├── state_best.pt   (best validation checkpoint)
│   │   └── losses.pkl      (training history for W&B)
```

### Manual Checkpoint Inspection
```python
import torch

ckpt = torch.load('logs/avoiding-d3il/flow_matching_v3_imeanflow/H8_D.../seed_6/state_best.pt')

# Keys
print(ckpt.keys())
# dict_keys(['model', 'diffusion', 'args', 'optimizer_state_dict', ...])

# Model weights
model_state = ckpt['model']
print(model_state.keys())
# Should include: 'model.u_net.*', 'model.v_head.*'
```

---

## Performance Expectations

### Training Time (1 seed, H100 GPU)
- Per epoch: ~30-60 seconds
- Total (100K steps at 32 batch): ~12-24 hours
- 5 seeds parallel: ~24 hours (if sufficient GPUs)

### Memory Requirements
- Model: ~2.5 GB
- Batch size 32: ~8-10 GB total
- Recommended: 1x H100 per seed (40GB memory)

### Disk Space
- Checkpoint per seed: ~200 MB (state_best.pt + backups)
- 5 seeds: ~1 GB
- Plus logs: minimal

---

## Dataset & Configuration

### Data Loading (automatic)
```
D3IL avoiding-d3il dataset (pre-loaded via config)
├── Training split: 85% (default)
├── Validation split: 15%
├── Horizon: 8 timesteps
├── State dim: 28
└── Action dim: 7
```

### Normalizer
```python
'normalizer': 'LimitsNormalizer'  # Scales to [-1, 1]
```

### Preprocessing
```python
'preprocess_fns': [],  # No additional preprocessing
'use_padding': True    # Pad short trajectories
```

---

## Multi-Seed Workflow

### Step 1: Train
```bash
python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --use-wandb
# 5 independent training runs
```

### Step 2: Monitor W&B
```
https://wandb.ai/[entity]/FMPCC-iMF
# Watch curriculum progression per seed
# Compare u_loss vs v_loss curves
```

### Step 3: Evaluate
```bash
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --results-dir evaluation_results/imf
# ~5-10 minutes total (depends on GPU)
```

### Step 4: Display Results
```bash
python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py \
    --results-dir evaluation_results/imf
# Shows mean ± std across seeds
```

---

## Key Differences: iMF vs FMv3ODE

| Aspect | FMv3ODE | iMF |
|--------|---------|-----|
| **Velocity prediction** | Single field `u` | Dual fields `(u, v)` |
| **Model architecture** | U-Net backbone | U-Net + v-head |
| **Loss function** | Single MSE | Dual MSE with curriculum |
| **Training phases** | Constant | 3-phase curriculum |
| **Checkpoint size** | ~200 MB | ~200 MB (same) |
| **Training time** | ~20 hours/seed | ~20 hours/seed (same) |
| **ODE steps** | Configurable | Default 1 (fast) |
| **Inference speed** | Similar | Similar |

---

## Documentation Structure

```
logs_in_develop/Gen3v4/fix_3/
├── REAL_IMF_IMPLEMENTATION.md      ← Full implementation report
├── ARCHITECTURE_OVERVIEW.md        ← Technical deep dive
├── INTEGRATION_GUIDE.md            ← How it works with FM-PCC
├── FILES_CHANGED.md                ← Complete file manifest
└── OPERATIONS_CHECKLIST.md         ← THIS FILE
```

---

## Next: Advanced Topics (Optional)

See `INTEGRATION_GUIDE.md` for:
- Custom curriculum timing
- Multi-NFE sampling (qual vs speed)
- Constraint guidance (collision avoidance)
- Comparative analysis (iMF vs FMv3ODE)

---

## Support

**Questions about**:
- **Implementation**: See `REAL_IMF_IMPLEMENTATION.md`
- **Architecture**: See `ARCHITECTURE_OVERVIEW.md`
- **Integration**: See `INTEGRATION_GUIDE.md`
- **Changes**: See `FILES_CHANGED.md`
- **Operations**: See this file

**Code Issues**:
1. Check checkpoint directory exists
2. Verify config block is correct
3. Ensure torch/CUDA available
4. Check W&B credentials

**Lost/Confused**:
→ Start with `REAL_IMF_IMPLEMENTATION.md` for full context

