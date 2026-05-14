# Detailed Technical Report: Visual Aligning Evaluation Pipeline Upgrade

## 1. Objective: Functional Parity and Architectural Integrity

The goal of this upgrade was to resolve the "silent hang" and metric inaccuracies in the Visual Aligning pipeline by establishing functional parity with the **FMv3ODE** blueprint. This involves a surgical integration of D3IL's native visual manipulation logic with FM-PCC's configuration and experiment management system.

## 2. Technical Implementation: The `VisualAgentWrapper`

The core of the upgrade is the `VisualAgentWrapper`, which serves as the translation layer between the FM-PCC model (VisualGaussianDiffusion) and the D3IL simulation environment (`Aligning_Sim`).

### A. Sliding-Window Context Management
Visual diffusion models are trained on sequences of observations to infer motion.
*   **The Problem**: The old script passed single observations, breaking the temporal consistency the model expected.
*   **The Fix**: Implemented three `collections.deque` buffers (maxlen=8) for `agentview_image`, `in_hand_image`, and `robot_pos`.
*   **Data Flow**: On each environment step, the new observation is appended. The wrapper then stacks these into tensors of shape `[1, 8, C, H, W]` and `[1, 8, 3]`, providing the full context required by the `VisualUNet` encoder.

### B. Scaler Integration (Normalization)
Normalization is the most common cause of failed policy transfer.
*   **Mechanism**: The wrapper builds a D3IL `Scaler` instance by reading the original `train_files.pkl` dataset.
*   **Application**: 
    *   **Inputs**: End-effector positions are scaled to the range `[-1, 1]` based on the dataset statistics before being fed to the model.
    *   **Outputs**: The model's raw noise/action predictions are inverse-scaled back to world-space velocity deltas (`dx, dy, dz`).

### C. Action Chunking and Latency Optimization
Inference in DDPM is computationally expensive (stochastic denoising).
*   **Action Chunks**: The model is configured with `action_seq_size=4`. This means one inference call produces 4 consecutive actions.
*   **Optimization**: The wrapper triggers the model only once every 4 steps. This reduces the total denoising overhead by **75%** and drastically speeds up simulation time.
*   **Denoising Steps**: Corrected the `n_timesteps` from 100 down to **16** (using the cosine schedule defined in training), which resolved the perceived "hanging" of the script.

## 3. Model Loading and Configuration Overrides

The pipeline utilizes the FM-PCC `load_diffusion_with_override` utility, which is the gold standard for experiment reproducibility in this repository.

*   **Config Deserialization**: It loads `dataset_config.pkl`, `model_config.pkl`, and `diffusion_config.pkl` from the training directory.
*   **Class Protection**: It compares the pickled class paths with the current project structure. If a mismatch is found (e.g., due to code refactoring), it dynamically updates the `_class` reference while preserving the original hyperparameters.
*   **Checkpoint Resolution**: Automatically identifies the `best` or `latest` checkpoint (`state_*.pt`) and restores the weights into the `VisualGaussianDiffusion` engine.

## 4. Standardized Metrics and Data Persistence

The evaluation now outputs metrics that are both D3IL-native and FM-PCC compatible.

### A. D3IL Standard Metrics (Aligning Task)
| Metric | Calculation | Rationale |
| :--- | :--- | :--- |
| **Success Rate** | Binary check of block pose vs. target pose at $T_{max}$ | Primary performance indicator. |
| **Entropy** | Shannon entropy of mode selection probabilities | Measures multimodality (pushing from different sides). |
| **Mean Distance** | L2 distance between achieved and desired pose | Measures precision beyond simple success. |
| **Score** | Average of Success Rate and Entropy | Standard D3IL leaderboard metric. |

### B. Persistence Layer
*   **`diffuser.npz`**: Stores raw numpy arrays of successes, steps, and mode encodings for cross-seed statistical analysis.
*   **`results_seed_<seed>.pkl`**: A dictionary containing aggregated metrics, timestamps, and the full `args` namespace. This file is specifically formatted for the **Performance Scorecard** (Data Analysis Visualizer).
*   **`eval_diffuser.log`**: A mirrored output of the console (via `Tee` logger), preserving the `tqdm` progress bars and detailed simulation logs.

## 5. Summary of Infrastructure Upgrades

### Configuration (`config/visual_aligning_eval.yaml`)
A new top-level configuration file that decouples evaluation parameters from the training config. It defines:
*   **Seeds**: `[6, 7, 8, 9, 10]` for statistical significance.
*   **Simulation Scope**: `n_contexts=30` (test scenarios) with `n_trajectories_per_context=1`.
*   **Variant Control**: Defines the `projection_variants` (currently restricted to the `diffuser` baseline).

### Execution (`eval_visual_aligning.sh`)
The SLURM entry point now supports:
*   **Auto-Detection**: Automatically resolves the model path via the FM-PCC `Parser`.
*   **Metadata Logging**: Captures GPU memory, driver versions, and Git revisions in the log header for forensic debugging.
*   **Seed Overrides**: Allows single-seed re-runs via CLI arguments (`sbatch ...sh 7`).

---
**Conclusion**: This upgrade transforms the Visual Aligning pipeline from a fragmented experimental script into a production-grade, standardized evaluation engine, ready for comparative benchmarking against the FMv3ODE baselines.
