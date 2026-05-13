# iMF-PCC Integration Guide

**How iMF replaces FMv3ODE in FM-PCC**

---

## Configuration Entry Point

### Change 1: `config/avoiding-d3il.py`

Replace the old theoretical iMF config with real architecture:

```python
'flow_matching_v3_imeanflow': {
    # Model & engine (REAL iMF)
    'model': 'flow_matcher_v3_imeanflow.models.iMeanFlowEngine',
    'diffusion': 'flow_matcher_v3_imeanflow.models.iMFDiffusion',
    
    # iMF-specific architecture
    'freq_dim': 256,
    'depth': 8,
    'num_heads': 4,
    'mlp_dim': 256,
    'time_dim': 256,
    'dropout_rate': 0.1,
    
    # Dual-velocity training (core iMF)
    'u_loss_weight': 0.5,
    'v_loss_weight': 0.5,
    'loss_schedule': 'u_first',  # Curriculum: u → u+v
    'warmup_epochs': 30,
    'transition_epochs': 30,
    
    # ODE inference (fast single-step)
    'ode_inference_steps_v3': 1,
    
    # Everything else inherited from FMv3ODE
    'loader': 'datasets.SequenceDataset',
    'normalizer': 'LimitsNormalizer',
    'batch_size': 32,
    'learning_rate': 5e-4,
    'n_train_steps': 100000,
    ...
}
```

---

## Module Instantiation Path

### Step 1: Parser reads config
```python
parser = Parser([], exe_name='train')
args = parser.parse_args([
    '--seed=6',
    '--diffusion=flow_matching_v3_imeanflow',  # ← Selects config block
])
```

### Step 2: Config values mapped to classes
```python
# args.model → dynamically loads:
#   'flow_matcher_v3_imeanflow.models.iMeanFlowEngine'
model = args.model  # Instance of iMeanFlowEngine

# args.diffusion → dynamically loads:
#   'flow_matcher_v3_imeanflow.models.iMFDiffusion'
diffusion = args.diffusion  # Instance of iMFDiffusion

# args.trainer → standard FM-PCC Trainer
#   Uses above model + diffusion
trainer = args.trainer
```

### Step 3: Trainer instantiation
```python
class Trainer:
    def __init__(self, model, diffusion, ...):
        self.model = model          # iMeanFlowEngine
        self.diffusion = diffusion  # iMFDiffusion
        ...
    
    def train(self):
        for epoch in range(n_epochs):
            for batch in dataloader:
                x_start, cond, mask = batch
                
                # 1. Sample timestep
                t = torch.rand(batch_size)
                
                # 2. Call diffusion (where iMF magic happens)
                loss, metrics = self.diffusion.p_losses(
                    x_start, t, cond, epoch  # ← epoch for curriculum
                )
                
                # 3. Backprop + optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 4. Log metrics
                wandb.log(metrics)
```

---

## Data Flow: From Parser to Training

```
1. Command Line
   python train_flow_matching_v3_imeanflow.py --seeds 6 --use-wandb
        ↓
2. Parse Args (train_flow_matching_v3_imeanflow.py)
   args = parser.parse_args(['--seed=6', '--diffusion=flow_matching_v3_imeanflow'])
        ↓
3. Config Block Selected (config/avoiding-d3il.py)
   config = avoiding_d3il.config['flow_matching_v3_imeanflow']
   # Returns dict with 33 keys: 'model', 'diffusion', 'u_loss_weight', ...
        ↓
4. Dynamic Class Loading (diffuser.utils.Parser)
   model_class = iMeanFlowEngine (from config)
   diffusion_class = iMFDiffusion (from config)
   trainer_class = Trainer (standard FM-PCC)
        ↓
5. Instantiation
   model = iMeanFlowEngine(
       freq_dim=256,
       depth=8,
       num_heads=4,
       ...
   )
   
   diffusion = iMFDiffusion(
       model=model,
       u_loss_weight=0.5,
       v_loss_weight=0.5,
       loss_schedule='u_first',
       warmup_epochs=30,
       transition_epochs=30,
       ...
   )
   
   trainer = Trainer(
       model=model,
       diffusion=diffusion,
       dataset=SequenceDataset(...),
       ...
   )
        ↓
6. Training Loop (trainer.train())
   for epoch in range(n_epochs):
       for x_start, cond in dataloader:
           t ~ U(0, 1)
           loss, metrics = diffusion.p_losses(x_start, t, cond, epoch)
           
           # iMFDiffusion.p_losses() does:
           # 1. x_noisy = (1-t)*x_start + t*noise
           # 2. u_pred, v_pred = model.forward_train(x_noisy, t, cond)
           # 3. loss_dict = imf_loss.forward(u_pred, v_pred, ..., epoch)
           #    └─ curriculum scaling inside: u_weight, v_weight = get_loss_weights(epoch)
           # 4. total_loss = u_weight*u_loss + v_weight*v_loss
           # 5. return total_loss, metrics_dict
           
           loss.backward()
           optimizer.step()
           wandb.log(metrics)
        ↓
7. Checkpoint Saving
   state_dict = model.state_dict()
   torch.save(state_dict, 'state_best.pt')
        ↓
8. Results: logs/avoiding-d3il/flow_matching_v3_imeanflow/.../{seed}/
```

---

## Module Dependencies

```
iMFDiffusion (wrapper)
   ├─ depends on: iMeanFlowEngine (model)
   ├─ depends on: iMFTrainingLoss (curriculum)
   └─ provides: FM-PCC Trainer interface

iMeanFlowEngine (inference engine)
   └─ depends on: iMFTrajectoryModel (architecture)

iMFTrajectoryModel (architecture)
   ├─ depends on: Flow_matcher_U_Net_v2 (u-head backbone)
   └─ implements: v-head (MLP auxiliary)

iMFTrainingLoss (curriculum training)
   └─ implements: loss weighting schedule
```

**Import Chain** (config → models):
```python
# In config/avoiding-d3il.py
'model': 'flow_matcher_v3_imeanflow.models.iMeanFlowEngine',
'diffusion': 'flow_matcher_v3_imeanflow.models.iMFDiffusion',

# Parser dynamically does (roughly):
from flow_matcher_v3_imeanflow.models import iMeanFlowEngine, iMFDiffusion
model = iMeanFlowEngine(...)
diffusion = iMFDiffusion(model=model, ...)
```

---

## Checkpoint & Loading

### Saving
```python
# Trainer.save_checkpoint()
torch.save({
    'model': model.state_dict(),  # iMeanFlowEngine weights
    'diffusion': diffusion.state_dict(),  # iMFDiffusion weights
    'args': args,
    'losses': {"training_losses": [...], "test_losses": [...]}
}, f'logs/avoiding-d3il/flow_matching_v3_imeanflow/H8_D.../seed_6/state_best.pt')
```

### Loading (in eval script)
```python
diffusion_experiment = utils.load_diffusion(
    savepath='logs/avoiding-d3il/flow_matching_v3_imeanflow/H8_D.../seed_6',
    epoch='best',
    device='cuda'
)

model = diffusion_experiment.model  # Restored iMeanFlowEngine
diffusion = diffusion_experiment.diffusion  # Restored iMFDiffusion

# Now ready for inference
sampled = diffusion.sample(batch_size=64)
```

---

## Comparison: FMv3ODE vs iMF

### FMv3ODE Training
```python
# Single velocity field
u_pred = u_net(x_noisy, t, cond)  # One prediction
loss = MSE(u_pred, u_target)
```

### iMF Training
```python
# Dual velocity fields with curriculum
u_pred, v_pred = model.forward_train(x_noisy, t, cond)  # Two predictions

u_scale, v_scale = imf_loss.get_loss_weights(epoch)     # Curriculum schedule

loss = u_scale * MSE(u_pred, u_target) + v_scale * MSE(v_pred, v_target)
#      └─ Phase 1: (1.0, 0.0) - u only
#      └─ Phase 2: blend from (1.0, 0.0) to (0.5, 0.5)
#      └─ Phase 3: (0.5, 0.5) - balanced
```

---

## Validation Checklist

- [ ] Config block in `config/avoiding-d3il.py` has `u_loss_weight`, `v_loss_weight`, `loss_schedule`
- [ ] iMF model files exist: `models/imf_*.py` (4 files)
- [ ] Training script imports from `diffuser.utils.Parser`
- [ ] Eval script uses `utils.load_diffusion()`
- [ ] W&B logs include `u_loss`, `v_loss`, `u_weight`, `v_weight` metrics
- [ ] Checkpoints save to `logs/avoiding-d3il/flow_matching_v3_imeanflow/{exp}/{seed}/`
- [ ] Multi-seed training: `--seeds 6 7 8 9 10` produces 5 independent checkpoints

---

## Troubleshooting

### Issue: "No module named 'flow_matcher_v3_imeanflow.models.iMeanFlowEngine'"
**Solution**: Ensure `flow_matcher_v3_imeanflow/models/__init__.py` exports iMF classes:
```python
from .imf_trajectory_model import iMFTrajectoryModel
from .imf_engine import iMeanFlowEngine
from .imf_losses import iMFTrainingLoss
from .imf_diffusion import iMFDiffusion
```

### Issue: Training gets stuck or loss explodes
**Solution**: Check curriculum parameters:
- `warmup_epochs`: Should be 20-40 (default 30)
- `transition_epochs`: Should be 20-40 (default 30)
- Increase if v_head is too strong too early
- Decrease if u-only phase is too long

### Issue: Checkpoints are missing
**Solution**: Verify config has `logbase` and `prefix`:
```python
'logbase': 'logs',
'prefix': 'flow_matching_v3_imeanflow/',
'exp_name': watch(args_to_watch_fmv3_ode_train),  # Naming convention
```

---

## Quick Start

1. **Ensure config is updated**:
   ```bash
   grep -A5 "flow_matching_v3_imeanflow" config/avoiding-d3il.py
   ```

2. **Check module imports**:
   ```bash
   python -c "from flow_matcher_v3_imeanflow.models import iMeanFlowEngine"
   ```

3. **Train single seed**:
   ```bash
   python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py --seed=6 --use-wandb
   ```

4. **Train multi-seed**:
   ```bash
   python FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10 --use-wandb
   ```

5. **Evaluate**:
   ```bash
   python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10
   ```

6. **View results**:
   ```bash
   python FM_v3_imeanflow_test/load_results_flow_matching_v3_imeanflow.py
   ```

