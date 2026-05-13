# Project Audit: Shifting Checkpoints & Visualizer Fixes

## 1. Who is "shitting" in `checkpoints/`?
The investigation revealed two primary sources responsible for creating the `checkpoints/` directory on remote environments:

### A. The iMF Evaluation Script (Offending Command)
In `Slurm_Codes/sbatch/iMF/eval_imf.sh`, the evaluation command was explicitly passing a legacy argument:
```bash
python FM_v3_imeanflow_test/eval_flow_matching_v3_imeanflow.py \
    --seeds 6 7 8 9 10 \
    --checkpoint-dir checkpoints \  <-- THE CULPRIT
```
Since the Python script's `argparse` did not explicitly define this, it was being swallowed into `extras` and handled by the `Parser` logic, which defaulted to creating that directory.

### B. Robomimic Baseline (Implicit Behavior)
The `robomimic` agents (used as baselines in `d3il`) have a default `BCConfig` that sets the output directory to `../{algo_name}_trained_models` or relative `checkpoints/`. When running these baselines on remote clusters without an explicit `--output-path` override, they create these folders at the repository root or parent.

**Resolution:**
- Removed the `--checkpoint-dir` argument from `Slurm_Codes/sbatch/iMF/eval_imf.sh`.
- Standardized the output to `evaluation_results/imeanflow` to align with the rest of the Gen3v4 pipeline.

---

## 2. Visualizer: Why is QUICK_LIST empty?
The `QUICK_LIST` dropdown relies on `results_manifest.json` being present in `Data_Analysis/analysis_results/`. 

### Root Cause
The previous implementation of `main_da_batch.py` used `os.path.dirname(output_dir)` to find the parent folder. If the user provided a path with a trailing slash or a relative path from a different CWD, this logic would fail to find the correct `analysis_results` root, causing the manifest to either not be created or be saved in the wrong place.

**Resolution:**
- Refactored `main_da_batch.py` to use `os.path.abspath()` for manifest path resolution.
- Added explicit directory listing of the parent results folder (excluding `plots` and `logs` system folders).
- Added error logging to the manifest generation to prevent silent failures.

## 3. Verified Fixes
| Issue | File | Action |
| :--- | :--- | :--- |
| **Path Duplication** | `main_da_batch.py` | Robust `abspath` resolution for manifest. |
| **Folder "Shitting"** | `eval_imf.sh` | Removed legacy `--checkpoint-dir` flag. |
| **Quick List Empty** | `index.html` | (Synced with new manifest logic). |

The pipeline is now **clean and manifest-driven**.
