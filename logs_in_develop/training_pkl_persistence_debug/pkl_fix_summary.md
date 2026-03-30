# Mission Briefing: Training Persistence & Debugging Infrastructure

## 🎯 Strategic Objective
Establish a bulletproof persistence layer for the Flow Matching and Diffusion training pipelines. The goal is to ensure that no training progress is ever lost and that every run is perfectly reconstructible for debugging and post-analysis.

## 🛠 Tactical Implementation

### 1. Robust Loss Persistence (State vs. History)
We now maintain TWO parallel records of training losses to satisfy different debugging needs:
- **`losses.pkl` (The Cleaned State)**:
  - **Logic**: Merging existing and new data. For any overlapping steps (common during a "rewind" resume), the **NEW** data overwrites the old.
  - **Purpose**: Provides a single, logical training curve for the current "best" version of the run.
- **`losses.json` (The Exhaustive History)**:
  - **Logic**: Cumulative exhaustive log. Every loss point ever reported across all sessions is preserved chronologically.
  - **Purpose**: Unfiltered visibility into "all we done." Ideal for debugging divergence or identifying exactly when a model crashed or was restarted.

### 2. Metadata Versioning & Protection
To prevent the "original" run configuration from being polluted or lost during resumes:
- **`*_config.pkl` & `args.json`**: The original versions are preserved as the ground truth.
- **Resume Contexts**: Subsequent resumes automatically save their own metadata as versioned files (e.g., `args_resume_1.json`, `model_config_resume_1.json`).

### 3. Redundant Safety Nets
- **Checkpoint Backups**: `state_*.pt` files now embed the full loss history. If the `.pkl` or `.json` logs are lost, the `Trainer.load()` method can reconstruct the history from the model checkpoint itself.

## 🔍 Debugging File Map

| File Path | Description | Debug Utility |
| :--- | :--- | :--- |
| `losses.pkl` | Integrated loss state | Standard plotting and success rate evaluation. |
| `losses.json` | Full chronological log | Identifying trajectory shifts between resumes. |
| `args.json` | Original CLI arguments | Verifying initial run conditions. |
| `args_resume_N.json` | Resume CLI arguments | Tracking changes in hyperparameters during resume. |
| `*_config.pkl` | Original class configs | Ensuring architectural consistency. |
| `state_best.pt` | Best model weights | includes `train_losses` for that specific model. |

## 🧪 Console vs. PKL Verification

A meticulous audit of the `Trainer` class was conducted to verify the consistency between console outputs (tqdm) and the persistent files (`losses.pkl` / `losses.json`).

### 1. Value Consistency
The values recorded in the persistent files are derived from the exact same `loss.item()` variable used for the `tqdm` console summary. Therefore:
- **Numerical Equality**: The values are identical at the moment of logging.
- **Divergence in Frequency**: The **Console** updates every single training step, while the **PKL/JSON** files only append new data every `log_freq` steps (defaulting to every 1,000 steps).

### 2. Consistency Across Updates
A comparison between the original codebase (pre-update) and the current version confirms that the core logging mechanism remains unchanged:
- **Before Update**: Console and PKL both used the same `loss.item()` from the final mini-batch.
- **After Update**: Console and PKL still use the same `loss.item()` from the final mini-batch.

The recent infrastructure updates only improved how these values are **summarized**, **merged on resume**, and **exported to JSON**, but did not alter the fundamental values being recorded.

### 3. Gradient Accumulation Nuance
It's important to note how `loss.item()` is captured when `gradient_accumulate_every > 1`:
- **Current Behavior**: The logged loss (both in console and PKL) reflects the contribution of the **last mini-batch** in the accumulation sequence.
- **Implication**: While this provides a consistent sample of the loss magnitude, it is not a mathematically perfect average of all mini-batches in that step. However, both the console and the files share this exact same representation.

## 🚀 Mission Status
The infrastructure is fully deployed in both `flow_matcher` and `diffuser` modules. Training can now be resumed from any checkpoint with complete confidence in data integrity and reporting consistency.
