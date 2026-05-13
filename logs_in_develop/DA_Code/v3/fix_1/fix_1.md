# DA Pipeline v3: Multidimensional Matrix Analysis & Dashboard

This update expands the Data Analysis (DA) pipeline to handle massive cross-comparison matrices across candidates, test environments, and variants. It introduces an automated hierarchical folder structure and an interactive web-based dashboard for dynamic exploration.

## Core Features

### 1. Hierarchical Plotting Engine
The `BatchVisualizer` now automatically generates a structured tree of visualizations:
*   **By Environment** (`hierarchical_analysis/by_test_env/`): 
    *   Compare all candidates on a specific test type (e.g., `both_hard`).
    *   Separates **Major** variants (DPCC) from **Auxiliary** variants for clarity.
*   **By Candidate** (`hierarchical_analysis/by_candidate/`):
    *   Analyze a single candidate's performance across all environments (robustness profiling).
*   **Global Matrices** (`hierarchical_analysis/matrices/`):
    *   Success density heatmaps (Candidate vs. Env) for every major variant.

### 2. Universal PyScript Visualizer
The system now uses a single, portable, and dynamic viewer at `Data_Analysis/Visualizer/index.html` powered by **PyScript**.
*   **In-Browser Matplotlib**: It executes the exact same Python/Matplotlib plotting logic as the batch scripts but runs it directly inside the browser's engine.
*   **Dynamic Generation**: Plots are generated on-the-fly from the `aggregated_stats.csv`. There is no longer any dependency on static `.png` files.
*   **Naive Scientific Design**: Optimized for readability with Arial fonts and standard Matplotlib axes. No fancy UX overhead.

### 3. Clean Data Pipeline
*   **No Redundant Files**: Removed all HTML generation and app server scripts from the analysis folders.
*   **Single Source of Truth**: The `aggregated_stats.csv` is now the primary driver for both numerical analysis and interactive visualization.

## Implementation Details
*   **File Modified**: `Data_Analysis/DA_Code_v3/batch_visualizer.py` (Focuses purely on CSV/Data generation).
*   **Universal Tool**: `Data_Analysis/Visualizer/index.html` (PyScript + Matplotlib engine).

## Usage
1. Run analysis: `python Data_Analysis/DA_Code_v3/batch_visualizer.py <batch_dir>`
2. Open: `Data_Analysis/Visualizer/index.html` in any browser.
3. Wait for `PYTHON_RUNTIME_READY`, paste CSV path, and plot.
