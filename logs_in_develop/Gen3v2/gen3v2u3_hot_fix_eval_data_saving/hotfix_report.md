# Audit: Evaluation Pipeline Data Persistence and Aggregation Hotfix

**Date**: 2026-05-03
**Status**: Completed
**Target Pipeline**: FMv3 ODE-selectable & Baseline DPCC

## 1. Objective
The goal was to transform the evaluation scripts from "one-shot" batch processors into modular tools that can:
1.  Run individual seeds in parallel (Slurm-compatible).
2.  Persist **all** raw data (not just statistics) for audit and visualization.
3.  Regenerate aggregate summary plots (`all_seeds`) from existing disk results without re-running model inference.

## 2. Problem Statement
Previously, the `all_seeds` summary plots were generated using in-memory buffers during a single script execution. Because raw trajectory coordinates were **not saved** to the `.npz` result files, it was impossible to:
-   Combine results from different Slurm jobs.
-   Re-visualize paths after the script finished.
-   Debug specific failures without re-simulating the entire environment.

## 3. Detailed Comparison: Before vs. After

| Feature | BEFORE (Legacy) | AFTER (Hotfix) | Rationale |
| :--- | :--- | :--- | :--- |
| **Data Persistence** | Only scalar statistics (Success rate, violations) were saved. | **Full trajectory data** (`obs_all`) and actions (`act_all`) are saved for every trial. | Enables deep-dive debugging and path visualization without re-simulation. |
| **Execution Mode** | Monolithic. You had to run all seeds in one script to get an aggregate plot. | **Modular.** You can run one seed at a time or all together. | Critical for Slurm parallelization and cluster resource efficiency. |
| **Aggregation** | Ephemeral. `all_seeds` plots were drawn from in-memory lists during the run. | **Retrospective.** Plots can be reconstructed from disk data at any time. | Allows combining results from different jobs into a single summary plot. |
| **Model/Env Dependency** | Always required a GPU model and MuJoCo simulation. | **Optional.** `--aggregate-only` mode needs neither a model nor MuJoCo. | Fast post-processing on CPU-only machines or login nodes. |
| **Command Line** | Hardcoded list from `projection_eval.yaml` was the only source. | Supports `--seed` and `--aggregate-only` overrides. | Increases flexibility for automated pipelines and manual audits. |

## 4. Implementation Details

### A. Data Persistence (Default)
The evaluation loop was modified to collect full trial data:
-   **`obs_all`**: Array of shape `(n_trials, max_episode_length, obs_dim)`.
-   **`act_all`**: Array of shape `(n_trials, max_episode_length, act_dim)`.

These are now saved in every `.npz` file by default:
```python
np.savez(f'{save_path}/{variant}.npz', 
         ...,
         obs_all=np.array(obs_all, dtype=object),
         act_all=np.array(act_all, dtype=object))
```

### B. New Command-Line Arguments
Added `argparse` to both `eval_flow_matching_v3_ode_selectable.py` and `scripts/eval.py`:
-   `--seed <int>`: Overrides the YAML config to run exactly one seed.
-   `--aggregate-only`: Skips the MuJoCo environment and Model loading. Instead, it iterates through the results folder, loads the `obs_all` data, and populates the `all_seeds` plots.

### C. Decoupled Plotting Logic
The plotting logic was refactored into a "Dual-Source" system:
-   **Live Run**: Uses the `obs_buffer` directly from the simulation.
-   **Aggregation Run**: Uses the `obs_all` array loaded from the existing `.npz` file.

## 4. Modified Files
1.  [eval_flow_matching_v3_ode_selectable.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py)
2.  [eval.py (baseline)](file:///workspaces/FM-PCC/scripts/eval.py)

## 5. Usage Guide

### Parallel Execution (Slurm)
You can now submit individual seeds to the cluster:
```bash
python FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py --seed 0
python FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py --seed 1
...
```

### Post-Hoc Aggregation
Once your seeds are finished, run this once to generate the summary plots:
```bash
python FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py --aggregate-only
```

## 6. Debug Audit
-   **File Size**: A typical `.npz` with 100 trials now consumes ~2-5MB. This is an acceptable trade-off for having a complete coordinate audit trail.
-   **Graceful Failures**: If the script is run in `--aggregate-only` mode but a seed's `.npz` file is missing, it prints a warning and continues to the next seed instead of crashing.
