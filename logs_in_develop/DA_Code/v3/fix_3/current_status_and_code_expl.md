# DA Code v3 - Current Status, Architecture, and Usage

## 1. Current Status

DA Code v3 is the stable Data Analysis branch for FM-PCC. It is currently usable end to end and has replaced the fragile v2 behavior that depended on broad substring matching and unsafe aggregation access.

The key v3 characteristics are:

- candidate discovery is more precise,
- aggregation exposes safe read-only properties,
- only real constraint data is counted,
- the visualizer discovers results from the server's HTML directory listing,
- batch and single-run Slurm launchers are available,
- the output folders are timestamped and browsable.

The result is a complete analysis stack rather than a single script.

## 2. High-Level Architecture

DA v3 is organized into three layers:

1. Data generation and discovery.
2. Aggregation and reporting.
3. Visualization and user interaction.

The code is designed so each layer can be used on its own, but the normal workflow is to run them together.

### 2.1 Data Flow Overview

The common execution path is:

```text
Slurm sbatch script
	-> main_da_batch.py or main_da.py
	-> data loader
	-> aggregator
	-> reporter
	-> visualizer
	-> analysis_results/<timestamped folder>
	-> Data_Analysis/Visualizer/index.html reads the results through HTTP
```

The important architectural point is that the browser visualizer is not coupled to a custom backend service. It reads the generated files directly from the repository served over HTTP.

## 3. Repository Components

### 3.1 Batch analysis entry point

File: `Data_Analysis/DA_Code_v3/main_da_batch.py`

This is the main cross-candidate pipeline. It:

- discovers candidate folders,
- loads each candidate's result files,
- aggregates candidate statistics,
- produces comparison plots,
- writes reports and CSV exports.

It is the entry point used when comparing multiple experimental candidates.

### 3.2 Single-run analysis entry point

File: `Data_Analysis/DA_Code_v3/main_da.py`

This is the single-directory analysis path. It:

- loads one evaluation result tree,
- aggregates metrics across seeds, variants, and constraints,
- generates plots,
- writes summary reports,
- supports `--no-plots` when the user wants only the tabular outputs.

This is the simplest way to analyze one experiment folder in isolation.

### 3.3 Discovery helpers

File: `Data_Analysis/DA_Code_v3/multi_candidate_discovery.py`

This module finds candidate folders under a parent directory. In practice, a candidate is a folder that contains the required seed subfolders.

The recursive discovery logic is important because experiment folders are not always exactly one directory deep. The search walks the directory tree up to a configurable depth and assigns stable letter codes like `A`, `B`, `C`.

### 3.4 Result loading

Files:

- `Data_Analysis/DA_Code_v3/data_loader.py`
- `Data_Analysis/DA_Code_v3/batch_data_loader.py`

These modules convert on-disk results into in-memory dictionaries.

The single-run loader reads the seed/variant/constraint hierarchy for one candidate. The batch loader simply repeats that process for each discovered candidate and stores the results under the candidate letter.

### 3.5 Aggregation

Files:

- `Data_Analysis/DA_Code_v3/aggregator.py`
- `Data_Analysis/DA_Code_v3/batch_aggregator.py`

The single-run aggregator computes statistics from one candidate's loaded data. The batch aggregator wraps that logic for every candidate and then builds the cross-candidate comparison metrics.

### 3.6 Reporting

Files:

- `Data_Analysis/DA_Code_v3/reporter.py`
- `Data_Analysis/DA_Code_v3/batch_reporter.py`

These modules write human-readable summaries and machine-readable CSV outputs.

The batch reporter is especially important because it produces the comparison tables that the browser visualizer later loads.

### 3.7 Plotting

Files:

- `Data_Analysis/DA_Code_v3/visualizer.py`
- `Data_Analysis/DA_Code_v3/batch_visualizer.py`

These modules create the static plots saved into the output folder. They are separate from the browser visualizer.

The distinction matters:

- `visualizer.py` and `batch_visualizer.py` generate files on disk during analysis,
- `Data_Analysis/Visualizer/index.html` is the interactive browser front-end for browsing the outputs.

### 3.8 Configuration

File: `Data_Analysis/DA_Code_v3/config.py`

This module defines the default seeds, variants, constraint types, halfspace variants, metric names, plot styling, and output folder prefix.

It keeps the behavior consistent across the loaders, aggregators, reporters, and plotting code.

## 4. How the Single-Run Pipeline Works

The single-run pipeline in `main_da.py` is the simplest path through the codebase.

### 4.1 Input arguments

The script accepts:

- `--input-path`: required path to the evaluation results directory,
- `--output-path`: base output folder,
- `--seeds`: optional comma-separated seed list,
- `--variants`: optional comma-separated projection variants,
- `--constraint-types`: optional constraint filter,
- `--verbose`: enable debug logging,
- `--no-plots`: skip plot generation.

### 4.2 Execution phases

The script runs in four phases:

1. Create the output directory.
2. Load raw evaluation data.
3. Aggregate the results and save reports.
4. Generate plots unless `--no-plots` is set.

### 4.3 Data loader behavior

`DataLoader.load_results(...)` expects a directory layout like this:

```text
input-path/
	6/
		results/
			halfspace_top-right-hard/
				dpcc-c.npz
	7/
		results/
			halfspace_top-right-hard/
				dpcc-c.npz
```

The loader reads files from the halfspace directories, prefers exact filename matches, and falls back to more permissive matching only when necessary.

### 4.4 Aggregation behavior

`DataAggregator` produces four views:

- by variant,
- by constraint type,
- by halfspace variant,
- detailed row-level data.

Those outputs are stored in `aggregator.aggregated` and also exposed through properties such as `aggregated_by_variant` and `detailed_df`.

### 4.5 Reporting and plots

The reporter writes summary text and CSV files into the output folder. The visualizer creates static PNG plots for the most important metrics, including the Pareto-style efficiency comparison.

## 5. How the Batch Pipeline Works

The batch pipeline in `main_da_batch.py` is the cross-candidate version of the same idea.

### 5.1 Candidate discovery

The batch pipeline first discovers candidates under a parent directory.

The discovery logic looks for folders that contain the required seed directories. It can also recurse through nested folder structures when the experiment organization is deeper than one level.

### 5.2 Candidate naming

Discovered candidates are assigned letter codes in alphabetical order:

- `A`
- `B`
- `C`
- and so on.

This gives the analysis a stable, compact identifier for ranking and plotting.

### 5.3 Batch loading

`BatchDataLoader` calls the single-run `DataLoader` for each candidate, then stores the result under the candidate letter.

This keeps the per-candidate loading behavior consistent while adding a batch dimension.

### 5.4 Batch aggregation

`BatchAggregator` creates one `DataAggregator` per candidate, then extracts candidate-level metrics.

The batch aggregator is what makes the cross-candidate comparisons possible. It computes:

- accuracy-oriented scores,
- timing scores,
- robustness estimates,
- rankings across candidates.

It also relies on the safe property accessors exposed by `DataAggregator`, which avoids the earlier attribute-access crashes from v2.

### 5.5 Batch reporting and plotting

`BatchReporter` writes:

- a human-readable summary,
- a ranking CSV,
- a detailed comparison CSV,
- a multidimensional aggregated CSV.

`BatchVisualizer` then generates the static comparison figures from the aggregated candidate statistics.

## 6. Data Loading Logic

The loader is the place where most of the v3 robustness improvements matter.

### 6.1 Precise file matching

The loader first tries exact filename matching for files like:

- `variant.npz`
- `variant.log`
- `variant.txt`

If no exact match is found, it falls back to prefix-based matching. That is much safer than broad substring matching because it avoids collisions such as a short name accidentally matching a longer filename.

### 6.2 Prefer `.npz`

If multiple files match, the loader prefers `.npz` files.

That matters because `.npz` files contain the structured metric payloads used for analysis, while logs and text files are only fallback sources.

### 6.3 No data replication across missing constraints

Earlier versions duplicated data into all constraint types even when only one real source existed. v3 does not do that.

Instead, the loader stores only the constraint types that actually came from the file system. This keeps counts honest and prevents false inflation of trial totals.

### 6.4 Metric extraction

For `.npz` files, the loader converts arrays into scalar means and standard deviations where appropriate.

For log or text files, it uses regex patterns to extract common metrics and stores the raw text as `raw_log` for auditability.

## 7. Aggregation Logic

The aggregator converts nested dictionaries into data frames and summary statistics.

### 7.1 What `DataAggregator` stores

`DataAggregator` exposes four main outputs:

- `aggregated_by_variant`
- `aggregated_by_constraint`
- `aggregated_by_halfspace`
- `detailed_df`

These properties are thin read-only accessors over the `aggregated` dictionary.

### 7.2 Variant aggregation

This view groups all metric values by projection variant and computes:

- mean,
- standard deviation,
- minimum,
- maximum,
- count.

This is the primary source for most comparison charts.

### 7.3 Constraint aggregation

This view groups values by constraint type. It is useful for seeing whether the behavior changes under halfspace, obstacle, dynamics, or bounds constraints.

### 7.4 Halfspace aggregation

This view groups values by halfspace variant, which helps compare scenarios like `top-right-hard` and `both-hard`.

### 7.5 Detailed row-level data

`detailed_df` stores each seed/variant/constraint/halfspace/metric/value row individually.

That table is important because:

- the batch reporter uses it for detailed CSV exports,
- the batch visualizer uses it for robustness plots,
- the browser visualizer can use the aggregated CSV derived from it.

## 8. Batch Comparison Logic

The batch pipeline is not just a loop over candidates. It adds candidate-level statistics on top of the single-candidate aggregator.

### 8.1 Candidate statistics

`BatchAggregator` extracts:

- major-variant accuracy,
- major-variant time,
- standard-group accuracy and time,
- tightened-group accuracy and time,
- auxiliary variant metrics,
- robustness estimates.

### 8.2 Rankings

After candidate statistics are extracted, candidates are ranked by accuracy.

That ranking is used in the summary report and the comparison visualizations.

### 8.3 Why the properties matter

The batch layer reads `aggregator.aggregated_by_variant` and `aggregator.detailed_df` rather than reaching into internal storage directly. That gives the batch code a stable interface and prevents runtime errors when the aggregator internals change.

## 9. Static Output Files

The analysis scripts write a consistent set of files into the output folder.

### 9.1 Single-run output

Typical files include:

- loading log,
- analysis log,
- summary report,
- aggregated result CSVs,
- PNG plots.

### 9.2 Batch output

Batch runs additionally write:

- candidate summary text,
- candidate ranking CSV,
- multidimensional aggregated CSV,
- batch loading log,
- comparison plots.

### 9.3 Output folder layout

The batch launcher writes to timestamped folders such as:

```text
Data_Analysis/analysis_results/batch_v3_YYYYMMDD_HHMMSS/
```

The browser visualizer reads these folders directly.

## 10. Visualizer HTML Architecture

File: `Data_Analysis/Visualizer/index.html`

This is the interactive front-end for browsing the generated outputs.

### 10.1 Technology stack

The page uses:

- HTML for layout,
- CSS for the interface styling,
- PyScript for Python execution in the browser,
- pandas for tabular data manipulation,
- Matplotlib for plotting.

### 10.2 Why it is browser-based

The browser model keeps the UI portable. Any user can open the page from a web server without needing a separate desktop application or custom backend.

### 10.3 Discovery mechanism

The page fetches `../analysis_results/` and parses the directory listing HTML for folders matching `batch_v3_...`.

This means the UI discovers new batches automatically when they appear on disk and are exposed by the server.

### 10.4 Fallback behavior

If directory listing is unavailable, the page can fall back to `results_manifest.json`.

That fallback is useful, but the intended primary path is the HTML directory listing.

### 10.5 Main controls

The page provides controls for:

- selecting a discovered batch,
- entering a custom CSV path,
- switching analysis mode,
- selecting a metric,
- selecting an environment focus,
- selecting variants and candidates,
- adjusting figure width,
- changing visual zoom,
- downloading plots and metadata.

### 10.6 Runtime behavior

After loading a CSV, the page:

1. populates dynamic filters,
2. renders the plot,
3. displays the scorecard,
4. builds the path audit map,
5. enables download actions.

### 10.7 Export behavior

The download action saves:

- the current plot as PNG,
- a metadata text file with the chosen variants, candidates, timestamp, and source paths.

This is meant for traceability and later audit.

## 11. Slurm Entry Points

The system is launched from the Slurm scripts under `Slurm_Codes/sbatch/DA/`.

### 11.1 Batch launcher

File: `Slurm_Codes/sbatch/DA/run_da_batch_v3.sh`

This script:

- configures the Conda environment,
- exports the repository path and D3IL paths,
- adds `Data_Analysis/DA_Code_v3` to `PYTHONPATH`,
- sets `MPLBACKEND=agg` for headless plotting,
- runs `python Data_Analysis/DA_Code_v3/main_da_batch.py`,
- writes output to a timestamped `batch_v3_...` folder.

### 11.2 Single-run launcher

File: `Slurm_Codes/sbatch/DA/run_da_single_v3.sh`

This script:

- uses the same environment setup,
- accepts an optional input path,
- defaults to a selectable flow-matching evaluation folder,
- runs `python Data_Analysis/DA_Code_v3/main_da.py`,
- writes output to a timestamped `single_v3_...` folder.

### 11.3 Why the launch scripts matter

These scripts are the operational entry points. They define how the analysis runs on the cluster and ensure the output layout is compatible with the visualizer.

## 12. How to Use DA v3

### 12.1 Batch analysis

Use this when you want to compare multiple candidates.

```bash
sbatch Slurm_Codes/sbatch/DA/run_da_batch_v3.sh
```

If you need a different parent path, call the Python entry point directly with the desired arguments.

### 12.2 Single analysis

Use this when you want to inspect one result tree.

```bash
sbatch Slurm_Codes/sbatch/DA/run_da_single_v3.sh logs/path/to/results
```

If no argument is provided, the script uses its built-in default input path.

### 12.3 Serve the repository

The visualizer expects the repository root to be served over HTTP.

```bash
cd /workspaces/FM-PCC
python3 -m http.server 8000
```

### 12.4 Open the visualizer

Open:

```text
http://<IP>:8000/Data_Analysis/Visualizer/index.html
```

The page should populate the quick list from the `analysis_results` directory listing.

### 12.5 Load data in the browser

1. Choose a batch from `QUICK_LIST` or enter a custom CSV path.
2. Press `SYNC_SOURCE`.
3. Select the analysis mode, metric, environment, variants, and candidates.
4. Refresh the plot if needed.
5. Download the PNG and metadata if you want a traceable export.

## 13. Practical Examples

### 13.1 Analyze a batch of experiments

```bash
sbatch Slurm_Codes/sbatch/DA/run_da_batch_v3.sh
```

Then open the visualizer and select the newest `batch_v3_...` folder.

### 13.2 Analyze one candidate folder

```bash
sbatch Slurm_Codes/sbatch/DA/run_da_single_v3.sh logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable
```

This is useful for debugging or validating a single run.

### 13.3 Use a custom CSV in the browser

If you already know the CSV path, you can switch to `CUSTOM_PATH` and point the visualizer directly at the aggregated file.

## 14. Current Status Summary

DA v3 is in a stable, usable state.

The current design is:

- backend analysis scripts generate real outputs,
- reporters save text and CSV artifacts,
- visualizers save static plots,
- the HTML page discovers batch results from the server listing,
- Slurm scripts provide the standard execution path.

That means the system is not just “fixed”; it is now structured as a complete analysis workflow with discovery, analysis, visualization, and export all tied together.