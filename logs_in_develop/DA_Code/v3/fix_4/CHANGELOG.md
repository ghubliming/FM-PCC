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

## Fix 4.2: Custom Seed Compare Logic (Rebuild)
Based on user feedback, the previous "per-seed scatter plot" experimental feature was reverted in favor of a robust, standard-preserving **Custom Seed Compare** mode.

- **`Visualizer/index.html`**:
  - **Standard Mode (Default):** Restored full backward compatibility with Fix 4.1. The visualizer explicitly fetches `candidates_multidimensional_aggregated.csv` and preserves the clean, high-quality "Path Audit Map" rendering. It calculates the mean/std based on whatever seeds were found during discovery (with warnings displayed for partial seeds).
  - **Custom Seed Compare Mode:** A new UI option that dynamically fetches `candidates_multidimensional_raw.csv`.
  - Added a responsive UI to select exact seeds to compare (e.g., `[6, 7]`).
  - Added strict validation: if a candidate in the current view is missing any of the user-selected seeds, plotting aborts gracefully and an error is displayed (e.g., `Error: CAND_A missing seed(s) [7]!`).
  - If validation passes, it dynamically recalculates the exact Mean and Std strictly constrained to the subset of user-selected seeds.
