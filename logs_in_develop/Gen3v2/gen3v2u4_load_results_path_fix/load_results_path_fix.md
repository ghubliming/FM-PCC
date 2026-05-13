# Gen3v2u4: Plot Output Path Standardization

**Date**: 2026-05-04
**Naming**: gen3v2u4

## 🎯 Strategic Objective
Standardize the output location for aggregated result plots to ensure they are stored alongside the experiment data rather than in the script's source directory or the current working directory.

## 🛠 Problem Description
The `load_results` scripts (both for FMv3 and general DPCC) had inconsistent and "weird" output paths for their generated plots:
1.  **FMv3 Selectable**: Hardcoded to a `plots/` subfolder within the script's test directory.
2.  **General Script**: Defaulted to saving directly into the current working directory (CWD).

This made it difficult to locate plots after running multi-seed aggregations and polluted the source/root directories.

## 💻 Code Changes

### 1. Dynamic Path Resolution
I implemented a dynamic `plot_path` resolution logic that triggers upon the first successful data load.

```python
# Extract the base experiment directory from the savepath
# args.savepath is usually: logs/dataset/exp_name/seed
load_path = os.path.dirname(args.savepath) 

# Standardized subfolder for aggregated plots
plot_path = os.path.join(load_path, 'plots', 'load_results_output_all_seeds')
os.makedirs(plot_path, exist_ok=True)
```

### 2. File Updates

#### [FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py](file:///workspaces/FM-PCC/FM_v3_ode_selectable_test/load_results_flow_matching_v3_ode_selectable.py)
- Removed static `plot_path` setup at the top of the script.
- Injected the dynamic resolution block inside the seed loop.
- Updated all `plt.savefig` calls to use the new `plot_path`.

#### [scripts/load_results.py](file:///workspaces/FM-PCC/scripts/load_results.py)
- Added `import os`.
- Implemented the same dynamic `plot_path` logic inside the variant/seed loops.
- Replaced hardcoded filenames in `plt.savefig` with `os.path.join(plot_path, save_name)`.

## 🧠 Explanation of Rationale
By anchoring the `plot_path` to `os.path.dirname(args.savepath)`, we ensure that:
- **Encapsulation**: All results for a specific experiment (logs, models, `.npz` data, and plots) are stored in the same parent folder.
- **Clarity**: The subfolder `load_results_output_all_seeds` explicitly identifies these plots as aggregated results from multiple seeds.
- **Portability**: The script works regardless of where it is executed from, as long as it can find the data.

## ✅ Verification
- The scripts now print the resolved `plot_path` to the console:
  `[ utils ] Set plot_path to: /workspaces/FM-PCC/logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/plots/load_results_output_all_seeds`
