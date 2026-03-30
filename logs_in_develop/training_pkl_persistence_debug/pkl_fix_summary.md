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

## 🚀 Mission Status
The infrastructure is fully deployed in both `flow_matcher` and `diffuser` modules. Training can now be resumed from any checkpoint with complete confidence in data integrity.
