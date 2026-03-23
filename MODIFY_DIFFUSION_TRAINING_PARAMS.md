
# Guide: Modifying Training Parameters for the Diffusion Model

This guide provides **exact, code-verified instructions** for changing the training and evaluation parameters of the original diffusion model (not Flow Matcher) in this repository.

---

## 1. Where to Set Parameters

- **Training parameters** are set via command-line arguments to `scripts/train.py`.
- Some parameters can also be set in config files (YAML/JSON), but CLI always overrides config values.

---

## 2. Main Training Parameters (as in code)

All of these can be set via CLI or config. **Parameter names must match exactly.**

| Argument                        | Type    | Default   | Description                                      |
|----------------------------------|---------|-----------|--------------------------------------------------|
| `--n_train_steps`                | int     | 100000    | Total number of training steps                   |
| `--n_steps_per_epoch`            | int     | 1000      | Number of steps per epoch                        |
| `--batch_size`                   | int     | 32        | Batch size for training                          |
| `--learning_rate`                | float   | 2e-5      | Learning rate for Adam optimizer                 |
| `--gradient_accumulate_every`    | int     | 2         | Gradient accumulation steps                      |
| `--ema_decay`                    | float   | 0.995     | Exponential moving average decay                 |
| `--train_test_split`             | float   | 1.0       | Fraction of data for training (rest for test)    |
| `--device`                       | str     | 'cuda'    | Device to use ('cuda' or 'cpu')                  |

### Example CLI usage

```bash
python scripts/train.py --n_train_steps 200000 --n_steps_per_epoch 500 --batch_size 64 --learning_rate 0.0001 --gradient_accumulate_every 4 --ema_decay 0.99 --train_test_split 0.9 --device cuda
```

---

## 3. Seed and Resume Parameters

| Argument                | Type   | Description                                                        |
|-------------------------|--------|--------------------------------------------------------------------|
| `--seed`                | int    | Train a single seed                                                |
| `--seeds`               | ints   | Train a list of seeds                                              |
| `--seeds-from-config`   | path   | Load seeds from a JSON file                                        |
| `--num-seeds`           | int    | Use only the first N seeds from the list                           |
| `--resume-step`         | int    | Resume from a specific training step                               |
| `--resume-seed`         | int    | Seed for manual resume step loading                                |
| `--auto-resume`         | flag   | Auto-resume each seed from latest local checkpoint if present      |

---

## 4. Weights & Biases (W&B) Logging

| Argument            | Type   | Description                                  |
|---------------------|--------|----------------------------------------------|
| `--use-wandb`       | flag   | Enable W&B runs per seed                     |
| `--wandb-project`   | str    | W&B project name (default: fm-pcc-diffusion) |
| `--wandb-entity`    | str    | W&B entity/team name                         |
| `--wandb-group`     | str    | W&B group name for per-seed runs             |
| `--wandb-mode`      | str    | W&B mode: online, offline, or disabled       |

---

## 5. Setting Parameters in Config Files

Some parameters can be set in YAML or JSON config files (see `config/` folder). Use the `--config` option if supported by your script:

```bash
python scripts/train.py --config config/projection_eval.yaml
```

**Note:** CLI arguments always override config file values.

---

## 6. To See All Available Options

Run:

```bash
python scripts/train.py --help
```

---

## 7. Reference: Trainer Class (diffuser/utils/training.py)

The following parameters are passed to the `Trainer` class:

- `n_train_steps`, `n_steps_per_epoch`, `train_batch_size`, `train_lr`, `gradient_accumulate_every`, `ema_decay`, `train_test_split`, `train_device`

---

**All parameter names and defaults above are verified from the codebase.**

- CLI arguments always override config file values.
- Edit the config file directly to change defaults.

---

## 4. Evaluation Parameters

Run evaluation with:

```bash
python scripts/eval.py [OPTIONS]
```

**Common options:**

- `--batch-size N` — Batch size for evaluation
- `--checkpoint PATH` — Path to model checkpoint
- `--config CONFIG_PATH` — Config file for evaluation
- `--seed S` — Evaluation seed

**Example:**

```bash
python scripts/eval.py --checkpoint logs/model/state_best.pt --batch-size 128
```

---

## 5. Finding All Available Parameters

- Run `python scripts/train.py --help` or `python scripts/eval.py --help` to see all available options and their descriptions.
- Check the top of `scripts/train.py` and `scripts/eval.py` for argument parser definitions.

---

## 6. Notes

- **Seeds**: You can set seeds easily via CLI (`--seed` or `--seeds`).
- **Epochs**: If not available as `--epochs`, look for `--n-epochs`, `--max-epochs`, or similar.
- **Config files**: For advanced setups, edit YAML/JSON in `config/` and pass with `--config`.
- **Defaults**: If you omit a parameter, the script uses its default value (see `--help`).

---

For more details, see the comments and docstrings in `scripts/train.py` and `diffuser/utils/training.py`.
