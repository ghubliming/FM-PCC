# Training Usage Guide

This guide documents the current diffusion training workflow implemented in scripts/train.py.

## 1. Basic Command

Run from the repository root:

```bash
python scripts/train.py [OPTIONS]
```

Notebook/Colab style:

```bash
!python scripts/train.py [OPTIONS]
```

Your explicit interpreter path example:

```bash
!/content/miniconda3/envs/dpcc/bin/python scripts/train.py [OPTIONS]
```

## 2. Seed Selection

Default behavior (unchanged):

```bash
python scripts/train.py
```

This trains default seeds: 5, 6, 7, 8, 9.

Train one seed:

```bash
python scripts/train.py --seed 5
```

Train an explicit list:

```bash
python scripts/train.py --seeds 5 6 7
```

Train only first N from the selected list:

```bash
python scripts/train.py --seeds 0 1 2 3 4 5 --num-seeds 3
```

Load seed list from JSON file:

```bash
python scripts/train.py --seeds-from-config config/seeds_config.json
```

Accepted JSON formats:

```json
{"seed_list": [5, 6, 7, 8, 9]}
```

```json
{"seeds": [5, 6, 7, 8, 9]}
```

```json
[5, 6, 7, 8, 9]
```

## 3. Resume Training

Manual resume for a specific seed and step:

```bash
python scripts/train.py --seeds 6 7 8 9 --resume-seed 6 --resume-step 80000
```

Manual resume for first selected seed:

```bash
python scripts/train.py --seeds 6 7 8 9 --resume-step 80000
```

Auto-resume from latest checkpoint for each seed:

```bash
python scripts/train.py --seeds 6 7 8 9 --auto-resume
```

Checkpoint naming expected by resume logic:

- state_20000.pt
- state_40000.pt
- state_80000.pt
- state_best.pt is not used for resume step loading

## 4. Weights & Biases (Optional)

Enable W&B logging per seed:

```bash
python scripts/train.py --seeds 5 6 7 --use-wandb
```

Specify project/entity/group:

```bash
python scripts/train.py --seeds 5 6 7 --use-wandb --wandb-project fm-pcc-diffusion --wandb-entity <your-entity> --wandb-group avoiding-run-1
```

Control W&B mode:

```bash
python scripts/train.py --seeds 5 --use-wandb --wandb-mode online
python scripts/train.py --seeds 5 --use-wandb --wandb-mode offline
python scripts/train.py --seeds 5 --wandb-mode disabled
```

Files uploaded as artifact per seed when available:

- state_best.pt
- losses.pkl
- args.json

## 5. Reproducibility Output

The script writes a seed manifest once per launch:

- seeds_config.json

The manifest contains:

- selected seed list
- seed source (default, cli, or config file)
- run timestamp
- resume settings used

## 6. Common Workflows

Quick single-seed debug:

```bash
python scripts/train.py --seed 5
```

Two-seed validation:

```bash
python scripts/train.py --seeds 5 6
```

Full run with W&B:

```bash
python scripts/train.py --seeds 5 6 7 8 9 --use-wandb --wandb-project fm-pcc-diffusion
```

Crash recovery with auto-resume:

```bash
python scripts/train.py --seeds 6 7 8 9 --auto-resume --use-wandb
```

## 7. CLI Rules

- Do not combine --seed and --seeds.
- --num-seeds must be greater than 0.
- --resume-step must be greater than or equal to 0.
- If requested checkpoint is missing, the seed starts fresh and prints a warning.

## 8. Notes

- This guide reflects the current implementation in scripts/train.py and diffuser/utils/training.py.
- No runtime command is required to read or use this guide.
