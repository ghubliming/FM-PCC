# New Parameters in `avoiding-d3il.py` Configuration

This document explains the meaning and purpose of new or notable parameters added to the `avoiding-d3il.py` configuration file for training and evaluating diffusion and flow matching models.

---

## Parameter Explanations

### Common Model Parameters

- **model**: Specifies the neural network architecture used (e.g., `'models.UNet1DTemporalCondModel'`).
- **diffusion**: The diffusion process implementation. For Flow Matching, this may be `'models.diffusion.GaussianDiffusion'`.
- **horizon**: The planning horizon (number of steps the model predicts).
- **n_diffusion_steps**: Number of diffusion steps (discretization of the diffusion process).
- **loss_type**: Type of loss function used (e.g., `'l2'` for mean squared error).
- **loss_discount**: Discount factor applied to the loss.
- **returns_condition**: Whether to condition the model on returns (reward-to-go).
- **action_weight**: Weighting factor for the action loss term.
- **dim**: Model dimensionality (hidden size).
- **dim_mults**: Multipliers for the model's hidden dimensions at each layer.
- **predict_epsilon**: If `True`, the model predicts noise (epsilon) instead of the denoised value.
- **dynamic_loss**: Whether to use a dynamic loss schedule.
- **hidden_dim**: Size of hidden layers.
- **attention**: Whether to use attention mechanisms.
- **condition_dropout**: Dropout rate for conditional inputs.
- **condition_guidance_w**: Guidance weight for conditional sampling.
- **test_ret**: Target return for test episodes.

### Dataset and Preprocessing

- **loader**: Dataset loader class (e.g., `'datasets.SequenceDataset'`).
- **normalizer**: Normalization method for data.
- **preprocess_fns**: List of preprocessing functions to apply.
- **clip_denoised**: Whether to clip denoised outputs.
- **use_padding**: Whether to pad sequences to a fixed length.
- **max_path_length**: Maximum trajectory length in the dataset.
- **include_returns**: Whether to include returns in the dataset.
- **returns_scale**: Scaling factor for returns.
- **discount**: Discount factor for future rewards.

### Serialization and Logging

- **logbase**: Base directory for logs.
- **prefix**: Subdirectory prefix for experiment logs.
- **exp_name**: Experiment name, often auto-generated.

### Training Parameters

- **n_steps_per_epoch**: Number of training steps per epoch.
- **n_train_steps**: Total number of training steps.
- **batch_size**: Training batch size.
- **learning_rate**: Learning rate for the optimizer.
- **gradient_accumulate_every**: Number of steps to accumulate gradients before updating.
- **ema_decay**: Exponential moving average decay for model weights.
- **train_test_split**: Fraction of data used for training.
- **device**: Device to use (`'cuda'` or `'cpu'`).
- **seed**: Random seed for reproducibility.

---

## Notable Additions for Flow Matching

- **flow_matching** section: Mirrors the `'diffusion'` section but uses the Flow Matching implementation for the diffusion process.
- **prefix**: Set to `'flow_matching/'` for Flow Matching experiments, distinguishing logs and checkpoints.
- **diffusion**: For Flow Matching, may reference `'models.diffusion.GaussianDiffusion'` or a similar FM-specific class.

---

## Usage

These parameters allow fine-grained control over model architecture, training, and evaluation. Adjust them in `avoiding-d3il.py` to customize experiments for both standard diffusion and flow matching models.

---

**For further details, see comments in the config file or refer to the main documentation.**
