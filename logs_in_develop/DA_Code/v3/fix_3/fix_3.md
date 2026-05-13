# FIX LOG: DA_v3_VERSION_1.0_STABLE

## 1. Release Overview
This version marks the **first fully runnable and stable release** of the FM-PCC Matrix Explorer. It resolves all path discovery, UI layout, and scientific auditing requirements.

## 2. Key Features

### A. Zero-Manifest Discovery Engine
- **Mechanism**: The Visualizer no longer requires a backend manifest file. It directly fetches and parses the HTML directory listing from `python -m http.server`.
- **Regex Logic**: Dynamically identifies `batch_v3_...` folders, making the UI instantly reactive to new Data Analysis runs.

### B. Dual-Axis Plot Controls
- **Physical Layout (FigWidth)**: A manual input box allows users to set the Matplotlib figure width (e.g., 10, 20, 30). This directly controls the fatness of the bars for crowded plots.
- **Visual Scale (Magnify)**: Discrete `SMALLER` and `LARGER` buttons provide instant CSS-based visual zooming without re-rendering the plot.

### C. Scientific Audit Workflow
- **Traceable Filenames**: Exported plots use a standardized naming convention: `plot_[METRIC]_[ENV]_[VARS]v_[CANDS]c_[TIME].png`.
- **Companion Metadata (.txt)**: Every PNG download generates a matching text file containing:
  - Full list of all selected variants.
  - Full list of all selected candidates.
  - **Absolute Source Paths** for every candidate to ensure 100% reproducibility.

### D. Infrastructure Cleanup
- **Slurm Patch**: Neutralized legacy `--checkpoint-dir` flags in evaluation scripts to prevent repository pollution on remote compute nodes.

## 3. Usage Instructions
1.  Launch server from repository root: `python3 -m http.server 8000`.
2.  Open Visualizer: `http://<IP>:8000/Data_Analysis/Visualizer/index.html`.
3.  Select a batch from the `QUICK_LIST` and click `SYNC_SOURCE`.

## 4. Status
**PRODUCTION READY**. Version 1.0 is stable and serves as the baseline for all subsequent experiment analysis.
