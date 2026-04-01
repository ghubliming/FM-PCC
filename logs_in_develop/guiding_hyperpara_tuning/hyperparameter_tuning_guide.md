# Hyperparameter Tuning Guide

> [!IMPORTANT]
> **Golden Rule:** Every tuning experiment needs **two blocks** in `config/avoiding-d3il.py` — one for training and one for evaluation (planning). Both blocks must have a **unique `prefix`** to guarantee data isolation.

## How It Works

The codebase constructs save paths automatically using:
```
logs/{dataset}/{prefix}{exp_name}/{seed}/
```
The `prefix` field in each config block controls the root subfolder. If two experiments share the same prefix and the same `args_to_watch` parameters, **one will silently overwrite the other**.

## Step-by-Step: Create a New Tuning Experiment

### Step 1: Duplicate the Config Block Pair
In `config/avoiding-d3il.py`, copy-paste the example pair:
- `'flow_matching_hp_tune1'` → rename to e.g., `'flow_matching_hp_tune2'`
- `'plan_fm_hp_tune1'` → rename to e.g., `'plan_fm_hp_tune2'`

### Step 2: Set a Unique Prefix (CRITICAL)
In the **training** block:
```python
'prefix': 'flow_matching_hp_tune2/',
```
In the **planning** block:
```python
'prefix': 'plans/flow_matching_hp_tune2/',
```

### Step 3: Update the Planning Load Path
In the planning block, ensure `diffusion_loadpath` points to your new training folder:
```python
'diffusion_loadpath': 'f:flow_matching_hp_tune2/H{horizon}_K{n_diffusion_steps}_D{diffusion}',
```

### Step 4: Change Your Hyperparameters
Modify whatever you want in the training block (e.g., `learning_rate`, `dim`, `batch_size`, `n_train_steps`). The model class stays as `'models.UNet1DTemporalCondModel'` — you are tuning, not changing architecture.

### Step 5: Run Training
In `FM_test/train_FM.py` (or your test script), pass the new experiment name:
```python
args = Parser().parse_args(experiment='flow_matching_hp_tune2', seed=seed)
```

### Step 6: Run Evaluation
In `FM_test/eval_FM.py`, pass the matching plan name:
```python
args = Parser().parse_args(experiment='plan_fm_hp_tune2', seed=seed)
```

---

## Checklist for Each New Tuning Run

| Item | What to check |
|------|---------------|
| Training block key | Unique name, e.g., `flow_matching_hp_tune2` |
| Training `prefix` | Unique path, e.g., `flow_matching_hp_tune2/` |
| Plan block key | Matching name, e.g., `plan_fm_hp_tune2` |
| Plan `prefix` | `plans/` + same unique name, e.g., `plans/flow_matching_hp_tune2/` |
| Plan `diffusion_loadpath` | Points to the training prefix folder |
| Script `experiment=` | Uses the correct block key name |

## Output Folder Structure
After following this guide, your logs will look like:
```
logs/avoiding-d3il/
├── flow_matching/                   # Original baseline
├── flow_matching_hp_tune1/          # Tuning experiment 1
├── flow_matching_hp_tune2/          # Tuning experiment 2
├── flow_matching_unet_v2/           # U-Net V2 architecture
├── plans/
│   ├── diffusion/                   # Diffuser eval results
│   ├── flow_matching/               # FM baseline eval results
│   ├── flow_matching_hp_tune1/      # Tune 1 eval results
│   ├── flow_matching_hp_tune2/      # Tune 2 eval results
│   └── flow_matching_unet_v2/       # U-Net V2 eval results
```

> [!TIP]
> **Naming convention:** Use descriptive suffixes like `_lr1e3`, `_dim64`, `_bigbatch` instead of just `_tune2` so you can easily tell experiments apart months later.
