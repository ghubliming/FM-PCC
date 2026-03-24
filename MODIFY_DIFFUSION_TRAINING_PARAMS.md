# Guide: Modifying Training Parameters for the Diffusion Model

This guide provides **code-verified instructions** for changing the training and evaluation parameters of the diffusion model in this repository.

---

## 1. Architecture Overview: How Parameters Are Set

This project uses a **two-layer parameter system**:

1. **Top-level CLI arguments** (`scripts/train.py`): Control seeds, resumption, and W&B logging. These are parsed via `argparse` in the `parse_top_level_args()` function.
2. **Config-file parameters** (`config/avoiding-d3il.py`): Control model architecture, dataset, and training hyperparameters. These are loaded via `utils.Parser.read_config()` from a Python config module.

> [!IMPORTANT]
> Training hyperparameters (batch size, learning rate, etc.) are **not** standard CLI flags. They are defined in `config/avoiding-d3il.py` under the `base['diffusion']` dictionary and loaded at runtime by `utils.Parser`.

```
scripts/train.py
├── parse_top_level_args()     → --seed, --seeds, --use-wandb, --auto-resume, etc.
├── utils.Parser.parse_args()  → loads config/avoiding-d3il.py → sets args.*
└── utils.Trainer(...)         → receives args.batch_size, args.learning_rate, etc.
```

---

## 2. Training Hyperparameters (from `config/avoiding-d3il.py`)

These parameters are defined in `config/avoiding-d3il.py` → `base['diffusion']`. To change them, **edit the config file directly**.

### 2.1 Model Architecture

| Parameter              | Config Default                    | Description                                        |
|------------------------|-----------------------------------|----------------------------------------------------|
| `model`                | `'models.UNet1DTemporalCondModel'` | Model class                                        |
| `diffusion`            | `'models.GaussianDiffusion'`      | Diffusion model class                              |
| `horizon`              | `8`                               | Planning horizon length                            |
| `n_diffusion_steps`    | `20`                              | Number of diffusion timesteps                      |
| `dim`                  | `32`                              | Base channel dimension of UNet                     |
| `dim_mults`            | `(1, 2, 4, 8)`                    | Channel multipliers per UNet level                 |
| `hidden_dim`           | `256`                             | Hidden dimension for inverse dynamics model        |
| `predict_epsilon`      | `True`                            | Predict noise (epsilon) vs. direct prediction      |
| `loss_type`            | `'l2'`                            | Loss function type                                 |
| `loss_discount`        | `1.0`                             | Discount factor for loss weighting over horizon     |
| `action_weight`        | `10`                              | Weight for action prediction loss                  |
| `condition_dropout`    | `0.25`                            | Dropout rate for classifier-free guidance          |
| `condition_guidance_w` | `1.2`                             | Guidance weight for conditional generation         |
| `returns_condition`    | `False`                           | Whether to condition on returns                    |
| `clip_denoised`        | `False`                           | Whether to clip denoised samples                   |
| `attention`            | `False`                           | ⚠️ **DEAD** — set on `args` but never passed to any constructor in training scripts |
| `dynamic_loss`         | `False`                           | ⚠️ **DEAD** — set on `args` but never passed to any constructor in training scripts |
| `test_ret`             | `0.9`                             | ⚠️ **DEAD in training** — only used in the `plan` config for evaluation, not consumed by training scripts |

### 2.2 Training

| Parameter                    | Config Default | Trainer Param Name           | Description                                       |
|------------------------------|----------------|------------------------------|---------------------------------------------------|
| `n_train_steps`              | `1e5` (100k)   | `n_train_steps`              | Total training steps                              |
| `n_steps_per_epoch`          | `1000`         | `n_steps_per_epoch`          | Steps per epoch (controls logging/checkpoint frequency) |
| `batch_size`                 | `8`            | `train_batch_size`           | Training batch size                               |
| `learning_rate`              | `1e-4`         | `train_lr`                   | Learning rate for Adam optimizer                  |
| `gradient_accumulate_every`  | `2`            | `gradient_accumulate_every`  | Gradient accumulation steps                       |
| `ema_decay`                  | `0.995`        | `ema_decay`                  | Exponential moving average decay                  |
| `train_test_split`           | `0.9`          | `train_test_split`           | Fraction of data for training (rest for test)     |
| `device`                     | `'cuda'`       | `train_device`               | Device to use                                     |
| `seed`                       | `0`            | —                            | ⚠️ **DEAD** — always overwritten by `parse_args(seed=seed)` in the seed loop (`setup.py` line 63) |

> [!NOTE]
> The config parameter names (left column) are what you edit in the config file. They get mapped to the Trainer constructor parameter names (middle column) in `scripts/train.py` lines 328–339.

### 2.3 Trainer-Internal Parameters (hardcoded defaults in `Trainer.__init__`)

These are **not** exposed in the config file but have defaults in `diffuser/utils/training.py`:

| Trainer Parameter    | Default  | Description                                           |
|----------------------|----------|-------------------------------------------------------|
| `lr_warmup_steps`    | `1000`   | Cosine LR warmup steps                                |
| `step_start_ema`     | `2000`   | Step to start EMA updates                             |
| `update_ema_every`   | `10`     | Update EMA model every N steps                        |
| `log_freq`           | `1000`   | Log losses every N steps                              |
| `save_freq`          | `n_train_steps // 5` | Checkpoint save frequency (auto-computed) |

### 2.4 Dataset

| Parameter        | Config Default              | Description                              |
|------------------|-----------------------------|------------------------------------------|
| `loader`         | `'datasets.SequenceDataset'` | Dataset class                            |
| `normalizer`     | `'LimitsNormalizer'`        | Data normalizer class                    |
| `preprocess_fns` | `[]`                        | Preprocessing functions                  |
| `use_padding`    | `True`                      | Pad shorter trajectories                 |
| `max_path_length`| `150`                       | Maximum trajectory length                |
| `include_returns`| `True`                      | Include returns in dataset               |
| `returns_scale`  | `400`                       | ⚠️ **DEAD** — config value is ignored; `args.max_path_length` is hardcoded instead (`train.py` line 247) |
| `discount`       | `0.99`                      | Discount factor                          |

---

## 3. How to Change Training Parameters

### 3.1 Edit the Config File (recommended)

Open `config/avoiding-d3il.py` and modify values in the `base['diffusion']` dictionary:

```python
# config/avoiding-d3il.py
base = {
    'diffusion': {
        # ...
        'n_train_steps': 2e5,        # Train for 200k steps instead of 100k
        'batch_size': 16,            # Increase batch size
        'learning_rate': 5e-5,       # Change learning rate
        'train_test_split': 0.8,     # 80/20 train/test split
        # ...
    },
}
```

### 3.2 To Expose Trainer-Internal Parameters

To change `lr_warmup_steps`, `step_start_ema`, etc., you must either:

1. Add them to the config dict in `config/avoiding-d3il.py`, **and** pass them through in the `trainer_config` in `scripts/train.py`:

```python
# scripts/train.py, line ~328
trainer_config = utils.Config(
    utils.Trainer,
    savepath=(args.savepath, 'trainer_config.pkl'),
    train_test_split=args.train_test_split,
    ema_decay=args.ema_decay,
    n_train_steps=args.n_train_steps,
    n_steps_per_epoch=args.n_steps_per_epoch,
    train_batch_size=args.batch_size,
    train_lr=args.learning_rate,
    gradient_accumulate_every=args.gradient_accumulate_every,
    lr_warmup_steps=args.lr_warmup_steps,       # ← add this
    results_folder=args.savepath,
)
```

2. Or modify the defaults directly in `diffuser/utils/training.py` → `Trainer.__init__`.

---

## 4. Top-Level CLI Arguments (`scripts/train.py`)

These are standard CLI flags handled by `parse_top_level_args()`.

### 4.1 Seed Management

| Argument              | Type       | Default          | Description                                        |
|-----------------------|------------|------------------|----------------------------------------------------|
| `--seed`              | `int`      | —                | Train a single seed                                |
| `--seeds`             | `int` list | —                | Train a list of seeds, e.g. `--seeds 5 6 7`        |
| `--seeds-from-config` | `str`      | —                | Path to JSON file with `seed_list` or `seeds`      |
| `--num-seeds`         | `int`      | —                | Use only the first N seeds from the resolved list  |

If none of the above are provided, the default seed list `[5, 6, 7, 8, 9]` is used (defined as `DEFAULT_SEEDS` in `train.py`).

### 4.2 Checkpoint Resume

| Argument         | Type   | Default | Description                                          |
|------------------|--------|---------|------------------------------------------------------|
| `--resume-step`  | `int`  | —       | Resume from a specific checkpoint step               |
| `--resume-seed`  | `int`  | —       | Which seed to apply the manual resume to             |
| `--auto-resume`  | flag   | `False` | Auto-resume each seed from latest local checkpoint   |

### 4.3 Weights & Biases Logging

| Argument           | Type  | Default                | Description                          |
|--------------------|-------|------------------------|--------------------------------------|
| `--use-wandb`      | flag  | `False`                | Enable W&B logging per seed          |
| `--wandb-project`  | `str` | `'fm-pcc-diffusion'`   | W&B project name                     |
| `--wandb-entity`   | `str` | `None`                 | W&B entity/team name                 |
| `--wandb-group`    | `str` | `None`                 | W&B group name (default: auto-generated) |
| `--wandb-mode`     | `str` | `'online'`             | `online`, `offline`, or `disabled`   |

### 4.4 Example CLI Usage

```bash
# Train with default config, seeds 5-9
python scripts/train.py

# Train a single seed with W&B logging
python scripts/train.py --seed 42 --use-wandb --wandb-project my-project

# Resume training from step 80000, auto-detect latest checkpoint
python scripts/train.py --seeds 5 6 7 --auto-resume

# Resume a specific seed from a specific step
python scripts/train.py --seeds 5 6 7 --resume-seed 5 --resume-step 80000
```

---

## 5. Evaluation Parameters (`scripts/eval.py`)

The evaluation script does **not** use CLI arguments. It reads all configuration from `config/projection_eval.yaml`.

### 5.1 Key Evaluation Settings (from `projection_eval.yaml`)

| Parameter                     | Value/Type | Description                                     |
|-------------------------------|------------|-------------------------------------------------|
| `exps`                        | list       | Experiment names (e.g. `['avoiding-d3il']`)     |
| `seeds`                       | list       | Seeds to evaluate (e.g. `[5]`)                  |
| `n_trials`                    | `int`      | Number of evaluation trials per variant         |
| `projection_variants`         | list       | Projection method variants to test              |
| `constraint_types`            | list       | Types of constraints (halfspace, obstacles, etc.) |
| `avoiding_halfspace_variants` | list       | Halfspace constraint configurations             |
| `plot_how_many`               | `int`      | Max number of trials to plot                    |

### 5.2 Model/Policy Parameters (loaded from `config/avoiding-d3il.py` → `base['plan']`)

| Parameter              | Default          | Description                                 |
|------------------------|------------------|---------------------------------------------|
| `max_episode_length`   | `200`            | Maximum episode length for evaluation       |
| `batch_size`           | `4`              | Number of parallel trajectory samples       |
| `diffusion_epoch`      | `'best'`         | Which checkpoint to load (`'best'`/`'latest'`) |
| `test_ret`             | `0`              | Target return for test-time conditioning    |
| `horizon`              | `8`              | Planning horizon                            |

### 5.3 Running Evaluation

```bash
python scripts/eval.py
```

Edit `config/projection_eval.yaml` to change evaluation settings (seeds, variants, constraints, etc.).

---

## 6. File Reference

| File                           | Purpose                                                 |
|--------------------------------|---------------------------------------------------------|
| `scripts/train.py`            | Training entry point; CLI arg parsing; seed/W&B/resume logic |
| `scripts/eval.py`             | Evaluation entry point; reads from YAML config          |
| `config/avoiding-d3il.py`     | Python config defining model/training/planning defaults |
| `config/projection_eval.yaml` | YAML config for evaluation scenarios                    |
| `diffuser/utils/setup.py`     | `Parser` class: loads config modules, sets up save paths |
| `diffuser/utils/training.py`  | `Trainer` class: training loop, EMA, checkpointing     |
| `diffuser/utils/config.py`    | `Config` class: deferred object construction            |

---

## 7. Dead / Unused Parameters in Config

These parameters exist in `config/avoiding-d3il.py` under both `diffusion` and `flow_matching` blocks but have **no effect** during training:

| Parameter        | Why Dead | Evidence |
|------------------|----------|---------|
| `seed: 0`        | Always overwritten by `parse_args(seed=seed)` in the seed loop | `setup.py` line 63: `args.seed = seed if seed is not None else args.seed` |
| `returns_scale: 400` | Hardcoded to `args.max_path_length` instead | `train.py` line 247, `train_FM.py` line 154 |
| `test_ret: 0.9`  | Only meaningful for evaluation (`plan` config); never consumed during training | Not passed to `model_config`, `diffusion_config`, or `trainer_config` |
| `dynamic_loss: False` | Set on `args` but never passed to any constructor | Not in `model_config`, `diffusion_config`, or `trainer_config` kwargs |
| `attention: False` | Set on `args` but never passed to any constructor | Not in `model_config`, `diffusion_config`, or `trainer_config` kwargs |

> [!CAUTION]
> Changing any of these 5 parameters in the config will have **zero effect** on training. If you want them to actually work, you must also wire them into the corresponding `utils.Config(...)` constructor calls in `scripts/train.py` / `FM_test/train_FM.py`.

---

## 8. Common Pitfalls

- **"I passed `--batch_size 64` on the CLI but nothing changed"**: Training hyperparameters are not CLI flags. Edit `config/avoiding-d3il.py` instead.
- **Config param vs. Trainer param name mismatch**: `batch_size` in config → `train_batch_size` in Trainer; `learning_rate` in config → `train_lr` in Trainer; `device` in config → `train_device` in Trainer. The mapping happens in `scripts/train.py` lines 328–339.
