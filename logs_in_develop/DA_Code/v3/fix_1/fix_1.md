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

### 2. Premium Interactive Dashboard
A high-end `dashboard.html` is generated in the output directory.
*   **Dynamic View Modes**: Toggle between Environment, Candidate, and Matrix perspectives.
*   **Live Selectors**: Instantly update plots for different metrics (Success Rate, Time, Smoothness).
*   **Design**: Modern dark-mode interface with responsive layout and glassmorphism effects.

### 3. Metric Standardization
*   **Major Variants**: `dpcc-r`, `dpcc-c`, `dpcc-t` and their `tightened` versions are treated as primary benchmarks.
*   **Auxiliary Variants**: All other variants (Diffuser, Gradient, etc.) are relegated to secondary comparison layers to avoid clutter.

## Implementation Details
*   **File Modified**: `Data_Analysis/DA_Code_v3/batch_visualizer.py`
*   **Folder Generation**: Uses `os.makedirs` to create deterministic paths for SLURM compatibility.
*   **Data Aggregation**: Leverages `pandas` pivot tables for multi-dimensional grouping of cross-seed statistics.

## Usage
Run the visualizer with the batch log directory as the first argument:
```bash
python Data_Analysis/DA_Code_v3/batch_visualizer.py <batch_dir> <output_dir>
```
Open the resulting `dashboard.html` in any browser to explore the data.
