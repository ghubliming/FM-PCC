# FM-PCC

## Test History (incl. Code tests and Model tests)

Master Overview See [logs_in_develop/MASTER_TEST_HISTORY.md](logs_in_develop/MASTER_TEST_HISTORY.md).

### Results and Data Analysis

- **New SLURM Runs Data & HTML Visualizer**: See [Data_Analysis/analysis_results](Data_Analysis/analysis_results/) and the interactive [Visualizer](Data_Analysis/Visualizer/index.html).
- **Old Colab Runs**: See [Results_and_Data_Analysis_Colab_T4](Results_and_Data_Analysis_Colab_T4/).

> **Note**: The actual `/logs` folders generated during training and evaluation (which contain model weights and raw outputs) are ignored by git and not included in this repository. They only exist on your local machine or remote cluster.

And all the Jupyter notebooks used for Colab training/evaluation/data analysis are in [ipynbs_Colab](ipynbs_Colab/).

---

## Methodology & Implementation Summary

This repository implements **Flow Matching Predictive Control (FM-PCC)**, replacing the stochastic diffusion engine of Diffusion Predictive Control (DPCC) with deterministic Flow Matching to achieve faster inference and smoother physical control. 

The architecture features a dual-path design that decouples generative planning from physical execution:
* **The Generative Brain (Flow Matching):** A U-Net predicts an optimal velocity vector field ($v_\theta$). An ODE solver uses this field to generate an unconstrained reference trajectory (the "Ghost Path") directly toward the target.
* **The Physical Brakes (MPC):** A Model Predictive Control algorithm filters this idealized path, enforcing strict physical and environmental constraints (e.g., maximum torque, obstacle boundaries) to output safe, executable motor commands.

---

## Code Versions Develop History & Notes

See [logs_in_develop](logs_in_develop/).

## How to use
### Training Usage in CLI (new)

For the current training CLI commands and options, see [TRAINING_CLI_USAGE.md](TRAINING_CLI_USAGE.md).

Pending