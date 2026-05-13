# v3 Mission Briefing: Robust Cross-Candidate Analysis

## Overview

**v3** addresses fundamental bugs discovered in v2, primarily revolving around incorrect data loading (substring matching) and broken attribute access in the aggregation logic. It provides a more robust and accurate comparison of experimental candidates.

## Major Fixes in v3

### 1. Robust Data Discovery
- **Problem**: v2 used substring matching for variants (e.g., `dpcc-c` matched `dpcc-c-tightened.npz`, `dpcc-c.png`, etc.). This led to incorrect files being loaded and nested dictionary structures that crashed the aggregator.
- **Solution**: v3 implements precise matching. It looks for exact filename matches (e.g., `{variant}.npz`) and only falls back to "starts with" matching if no exact match is found. It also prioritizes `.npz` files over others.

### 2. Fixed Aggregation Logic
- **Problem**: `BatchAggregator` and `BatchVisualizer` tried to access non-existent attributes on `DataAggregator` (e.g., `agg.aggregated_by_variant`), causing runtime crashes.
- **Solution**: Added property accessors to `DataAggregator` to safely expose internal results while maintaining a clean API.

### 3. Honest Data Representation
- **Problem**: v2 replicated data for all 4 constraint types (halfspace, obstacles, dynamics, bounds) even if data was only found for one. This multiplied the trial count in aggregations and gave misleading "per-constraint" results.
- **Solution**: v3 only fills the constraint types actually found in the file system. Aggregations now accurately reflect the available data.

### 4. Improved Recursive Discovery
- **Problem**: Discovery was sometimes too shallow or too strict with seed directory names.
- **Solution**: Enhanced `discover_candidates_recursive` with better depth handling and path normalization.

---

## Technical Details

### Module Updates
- `data_loader.py`: Precise filename matching, removed data replication.
- `aggregator.py`: Added properties for safe result access.
- `batch_aggregator.py`: Optimized global metric extraction.
- `batch_visualizer.py`: Fixed attribute access and improved heatmap reliability.

### New Sbatch Scripts
- `run_da_batch_v3.sh`: Runs the full multi-candidate comparison using v3 code.
- `run_da_single_v3.sh`: Runs single-directory analysis using v3 code.

---

## Usage

### Batch Analysis (Recommended for Comparisons)
```bash
sbatch Slurm_Codes/sbatch/DA/run_da_batch_v3.sh
```

### Single Analysis
```bash
sbatch Slurm_Codes/sbatch/DA/run_da_single_v3.sh logs/path/to/results
```

---

## Verification Plan

1. ✅ Discovery of candidates with nested seeds verified.
2. ✅ Variant matching logic tested against substring collisions.
3. ✅ Aggregation properties verified for compatibility.
4. ✅ Sbatch scripts updated with correct PYTHONPATH for v3 modules.

**Status**: v3 Implementation COMPLETE. Ready for deployment.
