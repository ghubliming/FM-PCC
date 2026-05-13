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

### 2. Naive Data-First Visualizer
The system now uses a single, persistent, and "naive" universal viewer at `Data_Analysis/Visualizer/index.html`.
*   **Simple Aesthetics**: Uses standard Arial/Sans-Serif fonts and a basic high-contrast layout for maximum legibility.
*   **CSV-First Loading**: Users paste the relative path to `aggregated_stats.csv`. The viewer parses the CSV locally and displays the raw data table immediately.
*   **Hierarchical Plot Linking**: Automatically resolves paths to the `hierarchical_analysis/` folder to display segmented MAJOR/AUX plots based on the loaded CSV location.

### 3. Clean Data Pipeline
*   **No Auto-HTML**: Removed all HTML generation logic from the Python analysis scripts to prevent file-system clutter.
*   **Surgical Plot Files**: Major (DPCC) and Auxiliary plots are kept in strictly separate files to prevent visual overlap and ensure clear benchmarking.

## Implementation Details
*   **File Modified**: `Data_Analysis/DA_Code_v3/batch_visualizer.py` (Removed dashboard generation).
*   **New Tool**: `Data_Analysis/Visualizer/index.html` (Naive CSV/Plot Viewer).

## Usage
1. Run analysis: `python Data_Analysis/DA_Code_v3/batch_visualizer.py <batch_dir>`
2. Start server: `python3 -m http.server 8000` (from project root).
3. Open: `http://localhost:8000/Data_Analysis/Visualizer/index.html`
4. Paste CSV path and explore raw data and plots.
