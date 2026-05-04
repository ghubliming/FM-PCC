# Hotfix Report: Gen3v2u5 — Advanced W&B Identity & Path-Based Grouping

**Date:** 2026-05-04
**Category:** Infrastructure / Observability
**Status:** ✅ COMPLETED
**Impact:** High (Permanent fix for experiment tracking ambiguity)

## 1. Problem: The "Identity Crisis" in W&B

Previously, all training runs for a specific dataset (e.g., `avoiding-d3il`) shared the same flat naming structure in Weights & Biases:
- **Run Name:** `avoiding-d3il-seed-5`
- **Group:** `avoiding-d3il-diffusion` (or similar)

### Why this was failing:
1. **Model Blindness:** Looking at the W&B dashboard, it was impossible to distinguish between a standard `diffuser` run and a `flow_matching_v3` run.
2. **Parameter Hiddenness:** Hyperparameters like Horizon (`H`) and Step Count (`K`) were buried inside the `config` tab, preventing "at-a-glance" performance comparisons between different architectures.
3. **Seed Fragmentation:** Seeds were grouped by broad categories rather than specific experiment configurations, making aggregated performance analysis tedious.

---

## 2. Solution: Path-Based Semantic Naming

We shifted the source of truth for run identity from the dataset name to the **Filesystem Path**. Since the project already has a robust directory structure that encodes all experiment parameters, we now project that path directly into W&B.

### The New Naming Formula:
1. **Relative Path Extraction:** Calculate the path from `logbase` (e.g., `logs/`) to the specific seed folder.
2. **Sanitization:** 
    - Replace `/` with `-` to create a valid W&B identifier.
    - Strip redundant class prefixes (e.g., `models.diffusion.GaussianDiffusion` -> `GaussianDiffusion`) for brevity.
3. **Seed Labeling:** Prepend `S` to the seed (e.g., `5` -> `S5`) to clearly demarcate the trial ID.
4. **Automatic Grouping:** Extract the parent directory of the seed folder to serve as the W&B Group name.

---

## 3. Code Implementation Details

The following changes were applied to `scripts/train.py` and `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py`.

### Before (Cryptic)
```python
wandb_group = cli_args.wandb_group if cli_args.wandb_group is not None else f'{args.dataset}-{args.exp_name}'
run = wandb.init(
    project=cli_args.wandb_project,
    group=wandb_group,
    name=f'{args.dataset}-seed-{seed}',
    ...
)
```

### After (Descriptive)
```python
# Derive descriptive name from the weight save path
savepath_rel = os.path.relpath(args.savepath, args.logbase)
wandb_name = savepath_rel.replace('/', '-').replace('models.diffusion.', '').replace('models.', '')

# Label seed part clearly as S<seed>
name_parts = wandb_name.split('-')
if name_parts[-1].isdigit():
    name_parts[-1] = f'S{name_parts[-1]}'
wandb_name = '-'.join(name_parts)

# Group name clusters seeds of the same experiment together
default_group = '-'.join(name_parts[:-1]) if len(name_parts) > 1 else wandb_name
wandb_group = cli_args.wandb_group if cli_args.wandb_group is not None else default_group

run = wandb.init(
    project=cli_args.wandb_project,
    group=wandb_group,
    name=wandb_name,
    ...
)
```

---

## 4. Verification & Examples

### Scenario: FMv3 ODE-Selectable Training
- **Filesystem Path:** `logs/avoiding-d3il/flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.GaussianDiffusion/5`
- **W&B Run Name:** `avoiding-d3il-flow_matching_v3_ode_selectable-H8_DGaussianDiffusion-S5`
- **W&B Group:** `avoiding-d3il-flow_matching_v3_ode_selectable-H8_DGaussianDiffusion`

### Scenario: Standard Diffuser Training
- **Filesystem Path:** `logs/avoiding-d3il/diffusion/H8_K20_Dmodels.GaussianDiffusion/7`
- **W&B Run Name:** `avoiding-d3il-diffusion-H8_K20_DGaussianDiffusion-S7`
- **W&B Group:** `avoiding-d3il-diffusion-H8_K20_DGaussianDiffusion`

---

## 5. Summary of Impact

| Metric | Before | After |
| :--- | :--- | :--- |
| **Searchability** | Poor (Requires config lookup) | **Excellent** (Search by H/K in name) |
| **Grouping** | Coarse (Task-level) | **Fine-grained** (Config-level) |
| **Consistency** | Manual / Ad-hoc | **Automatic** (Path-driven) |
| **Visual Clarity** | Overlapping names | **Unique identity per config** |

This update ensures that the W&B dashboard acts as a digital twin of the local `logs/` directory, making high-throughput experimentation significantly more manageable.
