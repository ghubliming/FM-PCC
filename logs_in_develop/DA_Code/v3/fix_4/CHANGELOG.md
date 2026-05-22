# DA Code v3 - Fix 4 Changelog

## Overview
This fix addresses a critical flaw in the `DA_Batch_v2` candidate discovery mechanism that caused incomplete experimental runs (e.g., candidates that had only completed Seed 6, such as `imeanflow` and `drifting`) to be silently ignored. 

Rather than failing or dropping these candidates, the pipeline now gracefully ingests them, correctly computes partial statistics, and surfaces prominent warnings about the missing seeds across the console, reports, and the visualizer dashboard.

## Modifications
- **`main_da_batch.py`**: 
  - Fixed a logical bug where the `--seeds` argument was being ignored during Phase 1 (Auto-Discovery). The argument is now parsed early and properly passed to `discover_candidates_recursive`.
- **`multi_candidate_discovery.py`**:
  - Replaced strict `has_seeds()` logic with `get_existing_seeds()`. 
  - Candidates are now successfully discovered as long as at least **1 valid seed** exists, instead of strictly requiring all 5 seeds (`[6, 7, 8, 9, 10]`).
  - Added robust console warnings when partial seeds are detected (`logger.warning`).
  - Augmented the candidate metadata dictionary to track `missing_seeds`.
- **`batch_reporter.py`**:
  - Added a `Missing_Seeds` column to `candidates_detailed.csv` and `candidates_multidimensional_aggregated.csv`.
  - Added missing seeds warnings to the `candidates_summary.txt` report.
- **`Visualizer/index.html`**:
  - Upgraded the **Path Audit Map** to dynamically render a **"Warnings"** column if missing seeds are detected in the dataset.
  - Candidates with partial evaluations prominently display **"MISSING: [7, 8, 9, 10]"** in bold orange.
  - Fully evaluated candidates confidently display **"ALL SEEDS"** in bold green.

## Fix 4.2: Per-Seed Scatter Visualization
Based on the realization that the DA v3 pipeline strictly preserves raw data, a powerful frontend visualization upgrade was implemented to allow direct "per-seed" visual inspection.

- **`batch_reporter.py`**:
  - Injected `Folder_Name`, `Full_Path`, and `Missing_Seeds` metadata into the `candidates_multidimensional_raw.csv` so it matches the aggregated schema.
- **`Visualizer/index.html`**:
  - The default data source was switched to `candidates_multidimensional_raw.csv`.
  - The PyScript backend was rewritten to calculate both `mean` and `std` dynamically on the fly from the raw dataframe.
  - Added a new UI Toggle: **2.5 Plot Style** (`Bar Chart (Mean ± Std)` vs `Per-Seed Scatter (Raw)`).
  - Designed custom Matplotlib logic that computes the exact spatial X-axis offsets for grouped pandas bar charts, allowing individual seed values to be plotted perfectly aligned on top of the bars.
  - Color-coded the scatter dots by seed (`6: red`, `7: blue`, `8: green`, `9: purple`, `10: orange`) with a dedicated legend, empowering researchers to instantly spot single-seed anomalies or systemic failures across candidates.
