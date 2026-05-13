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

### 2. Standalone Scientific Visualizer
Instead of generating a one-off HTML file per experiment, the system now provides a centralized, high-utility "cold" viewer at `Data_Analysis/Visualizer/index.html`.
*   **Scientific Design**: Minimalist, monochrome interface with monospace typography for research efficiency.
*   **Dynamic Loading**: Allows loading any analysis result folder via relative path input.
*   **7-Metric Scorecard**: Displays high-fidelity results for Success, Constraints, Time, Steps, and Violations.

### 3. Metric & Variant Separation
To eliminate "catastrophic" visual clutter in plots:
*   **Surgical Separation**: Major variants (DPCC 3+3) and Auxiliary variants (others) are now saved into distinct plot files (e.g., `MAJOR_comp_...png` and `AUX_comp_...png`).
*   **Priority Focus**: The visualizer defaults to the MAJOR perspective to ensure primary benchmarks are analyzed in isolation.

## Implementation Details
*   **File Modified**: `Data_Analysis/DA_Code_v3/batch_visualizer.py`
*   **New Tool**: `Data_Analysis/Visualizer/index.html` (Universal Viewer)
*   **Data Structure**: Uses hierarchical folders under `hierarchical_analysis/` to enable matrix-based navigation.

## Usage
1. Run the batch analysis: `python Data_Analysis/DA_Code_v3/batch_visualizer.py <batch_dir>`
2. Open the scientific viewer: `Data_Analysis/Visualizer/index.html`
3. Load the result path and use dropdowns to traverse the performance matrices.
