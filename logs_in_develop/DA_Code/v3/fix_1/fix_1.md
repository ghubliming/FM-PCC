# DA Code v3 - Fix 1: Robust Log Parsing, Array Alignment, and Multi-Dimensional Analysis

## 1. Context & Problem
While executing the v3 batch data analysis on standard text-based evaluation logs (`.log` or `.txt`), the generated summaries contained empty values (`NaN`) for crucial metrics such as `Accuracy`, `Time_ms`, and `Robustness_Score`. This data omission ultimately led to corrupted and incomplete comparison plots. 

**Root Cause of Empty Metrics (Even for .npz files):**
When parsing `.npz` files that contained arrays (e.g., multiple episodes), the v2 code automatically appended `_mean` and `_std` to the metric key (e.g., `n_success_and_constraints_mean`). However, the batch aggregator explicitly searched for the exact base string (`n_success_and_constraints`). This mismatch caused the aggregator to silently skip these metrics entirely, resulting in empty tables and broken plots even when the raw data existed.

Furthermore, the existing v2/v3 plotting logic strictly averaged over all variants and test types to produce single candidate-level metrics. The thesis requirement demands a **complex multi-dimensional analysis**: slicing performance by Metric (e.g., Success rate) × Variant (e.g., `dpcc-c-tightened`) × Test Type (e.g., `Both hard`) × Candidate (`A`, `B`, `C`...).

## 2. Solutions Implemented

### A. Metric Key Alignment & Array Parsing (`data_loader.py`)
- Fixed the `.npz` parsing logic to assign the mean of arrays directly to the base metric key (e.g., `metrics_dict[key] = np.mean(value)`). This ensures exact string matching succeeds in the aggregator.
- Added a fallback mechanism in `BatchAggregator` to use `n_success` if `n_success_and_constraints` is missing.

### B. Robust Regex Log Parsing (`data_loader.py`)
The `DataLoader` module was updated to fall back on regex parsing when standard `.npz` arrays are unavailable. The following console log metrics are now accurately extracted and mapped:
- `Success rate` -> `n_success`
- `Constraints satisfied` -> `collision_free_completed`
- `Success rate (goal and constraints)` -> `n_success_and_constraints`
- `Avg number of steps` -> `n_steps`
- `Avg number of constraint violations` -> `n_violations`
- `Avg total violation` -> `total_violations`
- `Average computation time per step` -> `avg_time` (automatically converted to milliseconds).

### C. Multi-Dimensional DataFrame Exposure (`batch_aggregator.py`)
Instead of restricting access to solely variant-averaged global metrics, `BatchAggregator` now exposes a full, unaggregated dataset via the new method:
```python
def get_full_detailed_dataframe(self):
    # Returns dimensions: Candidate, Variant, Constraint, Halfspace, Metric, Seed, Value
```

### D. Advanced Multi-Dimensional Plotting (`batch_visualizer.py`)
A new method `plot_multidimensional_comparison` was integrated into the batch visualization suite. When executed, it generates grouped bar charts slicing the data exactly as requested:
- **One discrete plot per Test Type** (e.g., `halfspace_both-hard`).
- **X-axis**: Distinct Candidates (Updated to use shortened names: `A`, `B`, `C` instead of `Candidate A`).
- **Grouped Bars**: Projection Variants representing discrete methodologies (e.g., `dpcc-c`, `dpcc-c-tightened`).
- Contains standard deviation error bars automatically calculated across all constituent random seeds.

### E. Advanced Table Exports (`batch_reporter.py`)
The `BatchReporter` has been expanded to output two separate multi-dimensional files:
1. **`candidates_multidimensional_raw.csv`**: Dumps the entire highly-dimensional `get_full_detailed_dataframe()` object to disk, including individual seed values.
2. **`candidates_multidimensional_aggregated.csv`**: Automatically groups the raw data by `[Candidate, Variant, Constraint_Type, Test_Type, Metric]` and pre-calculates the `mean`, `std`, and `count` across all seeds. This provides a direct, pivot-free table ready for immediate consumption in Excel or other BI tools.

## 3. Impact
- **No More Corrupted Plots or Missing Data**: Complete, non-null datasets actively feed into the plotting mechanisms regardless of whether the source was `.npz` arrays, `.npz` scalars, or raw `.log` text files.
- **Accurate Granularity**: Candidates can now be objectively compared across specific variables (such as constrained vs. tightened configurations on specific hard obstacle layouts) rather than relying exclusively on vague aggregate scores.
- **Improved UX & Exporting**: Plot X-axis titles are cleaner (just letters), and the multi-dimensional dataset is exported both raw and pre-aggregated, bypassing the need for manual aggregation in Excel.

## 5. Prioritized Analysis: Major vs. Auxiliary Metrics
The analysis pipeline has been refactored to prioritize core research results over parameter studies and baseline variations.

**Classification:**
- **Major Variants**: Includes the 3 core DPCC variants (`dpcc-r`, `dpcc-c`, `dpcc-t`) and their respective `tightened` versions. These represent the primary focus of the performance analysis.
- **Auxiliary Variants**: All other variants (e.g., `diffuser`, `gradient`, `post_processing`, `model_free`, and `dt` sensitivity studies).

**Key Improvements:**
- **Refined Aggregation**: Candidate-level rankings and scores are now calculated specifically from the **Major Variants**, preventing noisy auxiliary baselines from skewing the high-level comparison.
- **Separated Visualization**: The multidimensional grouped bar charts are now generated in two distinct sets:
    - `05_multi_comp_..._MAJOR.png`: Focuses exclusively on comparing core DPCC methodologies.
    - `05_multi_comp_..._AUX.png`: Captures auxiliary and baseline performance for comprehensive reporting.
- **Enhanced Summary Report**: The `candidates_summary.txt` now features a "Major Variant Accuracy" ranking and includes a detailed per-variant breakdown for each candidate, showing individual success rates for `dpcc-r`, `dpcc-c`, etc., side-by-side.
