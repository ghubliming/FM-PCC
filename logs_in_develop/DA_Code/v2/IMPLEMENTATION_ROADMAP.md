# DA Tool v2 Implementation Roadmap

## Current Status

**v1**: ✅ COMPLETE (Single-folder analysis)
**v2**: 📋 PLANNED (Multi-folder cross-candidate comparison)

---

## What's New in v2

### Problem Solved
- **v1 limitation**: Can only analyze ONE folder at a time
- **v2 solution**: Compare multiple experimental folders simultaneously
- **Use case**: You ran 5 different model/hyperparameter combinations, want to know which one is best

### Real Example
```
You have:
  - logs/avoiding-d3il/plans/diffusion/H8_K20_aw10/        (Model A)
  - logs/avoiding-d3il/plans/diffusion/H8_K10_aw10/        (Model B)
  - logs/avoiding-d3il/plans/diffusion/H8_K20_aw1/         (Model C)
  - logs/avoiding-d3il/plans/flow_matching_v3_ode_selectable/.../  (Model D)

Before (v1): Analyze each separately → 4 separate plots → manual comparison
After (v2):  Analyze all together → 1 Pareto plot showing ALL FOUR
```

---

## Architecture (v2)

### New Modules (3 files)

```
Data_Analysis/DA_Code/
├── main_da_batch.py              ← CLI entry point for v2
├── multi_candidate_discovery.py  ← Find all candidates automatically
├── batch_aggregator.py           ← Aggregate across candidates
└── batch_visualizer.py           ← Plot cross-candidate comparisons
```

### Reuses (v1 modules untouched)
```
├── data_loader.py                ← Reuse: load individual candidate
├── aggregator.py                 ← Reuse: aggregate per candidate
├── visualizer.py                 ← Extend: add cross-candidate plots
├── reporter.py                   ← Extend: add ranking reports
```

---

## Implementation Detail

### Phase 1: Discovery (2 hours)

**File**: `multi_candidate_discovery.py`

**Input**: Parent directory path  
**Output**: List of candidates with auto-assigned letters

```python
def discover_candidates(parent_path):
    candidates = {}
    letter = ord('A')
    
    for subfolder in sorted(os.listdir(parent_path)):
        full_path = os.path.join(parent_path, subfolder)
        if os.path.isdir(full_path):
            # Check if it contains seed folders
            if has_seeds([6,7,8,9,10], full_path):
                candidates[chr(letter)] = {
                    'path': full_path,
                    'name': subfolder,
                    'seeds': [6,7,8,9,10]
                }
                letter += 1
    
    return candidates

# Output:
# {
#   'A': {'path': '.../H8_K20_aw10', 'name': 'H8_K20_aw10', 'seeds': [6,7,8,9,10]},
#   'B': {'path': '.../H8_K10_aw10', 'name': 'H8_K10_aw10', 'seeds': [6,7,8,9,10]},
#   ...
# }
```

**Tests**:
- [ ] Discover 0 candidates (empty folder)
- [ ] Discover 1 candidate (single folder)
- [ ] Discover 5 candidates (full set)
- [ ] Handle nested structures correctly
- [ ] Handle missing seed folders gracefully

---

### Phase 2: Batch Loading (3 hours)

**File**: `batch_data_loader.py`

**Input**: Candidate list (A, B, C...)  
**Output**: Unified data dict with candidate dimension

```python
class BatchDataLoader:
    def load_all_candidates(self, candidates_dict, variants, constraints, halfspaces):
        """Load data for all candidates in parallel"""
        batch_data = {}
        
        for candidate_letter, candidate_info in candidates_dict.items():
            # Reuse v1 DataLoader
            loader = DataLoader()
            candidate_data = loader.load_results(
                root_path=candidate_info['path'],
                seeds=candidate_info['seeds'],
                variants=variants,
                constraint_types=constraints,
                halfspace_variants=halfspaces
            )
            batch_data[candidate_letter] = candidate_data
        
        return batch_data

# Output:
# {
#   'A': {seed: {variant: {constraint: {halfspace: metrics}}}},
#   'B': {seed: {variant: {constraint: {halfspace: metrics}}}},
#   'C': {...}
# }
```

**Tests**:
- [ ] Load 1 candidate
- [ ] Load 5 candidates (should take ~2x time of single)
- [ ] Handle missing .npz files per candidate
- [ ] Verify data consistency across candidates

---

### Phase 3: Batch Aggregation (3 hours)

**File**: `batch_aggregator.py` (extends v1 DataAggregator)

**Input**: batch_data with candidate dimension  
**Output**: Candidate-level rankings and statistics

```python
class BatchAggregator:
    def aggregate_by_candidate(self, batch_data):
        """Aggregate each candidate separately, then rank"""
        candidate_stats = {}
        
        for candidate_letter, candidate_data in batch_data.items():
            # Reuse v1 aggregation logic
            aggregator = DataAggregator(candidate_data)
            aggregator.aggregate_all()
            
            # Extract key metrics
            candidate_stats[candidate_letter] = {
                'accuracy': self._get_accuracy(aggregator),
                'time': self._get_time(aggregator),
                'robustness': self._get_std(aggregator),
                'raw_aggregator': aggregator
            }
        
        # Rank candidates
        ranked = sorted(
            candidate_stats.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        return candidate_stats, ranked

# Output:
# Ranking:
# [(A, 0.873), (C, 0.895), (B, 0.842), ...]
```

**Tests**:
- [ ] Aggregate 1 candidate
- [ ] Aggregate 5 candidates
- [ ] Verify ranking order correct
- [ ] Test robustness (std dev) calculation

---

### Phase 4: Batch Visualization (4 hours)

**File**: `batch_visualizer.py` (extends v1 DataVisualizer)

**New Plots**:

```python
class BatchVisualizer(DataVisualizer):
    
    def plot_candidate_pareto_frontier(self, output_dir):
        """X: time, Y: accuracy, each point = one candidate"""
        # Points colored: A=red, B=orange, C=yellow, D=blue, E=green
        # Annotated with candidate letters
        # Shows which candidate dominates
    
    def plot_candidate_success_comparison(self, output_dir):
        """Bar chart: candidates grouped by constraint type"""
        # Candidate A, B, C bars for each constraint
        # Shows which candidate handles which constraint best
    
    def plot_candidate_time_comparison(self, output_dir):
        """Bar chart: computation time per candidate"""
        # Clear ranking: which is fastest
    
    def plot_candidate_robustness(self, output_dir):
        """Boxplot: seed variability per candidate"""
        # Tight box = reproducible, wide box = sensitive
    
    def plot_candidate_constraint_heatmap(self, output_dir):
        """Heatmap: row=candidate, col=constraint type"""
        # Color intensity = success rate
        # Shows performance profile per candidate
```

**Tests**:
- [ ] Generate all 5 plots for 5 candidates
- [ ] Verify color coding correct
- [ ] Test with 2, 3, 5, 10 candidates
- [ ] Verify plot labeling clear

---

### Phase 5: CLI & Reporting (2 hours)

**File**: `main_da_batch.py`

```bash
# Usage
python main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --candidates A,C,E \
    --candidate-names "baseline,optimized,experimental" \
    --output-path ./batch_results \
    --no-plots \
    --verbose
```

**Reporting** (`batch_reporter.py`):

```
Output Files:
├── candidates_ranking.txt          # Winner is A
├── candidates_ranking.csv          # All metrics per candidate
├── candidates_summary.txt          # Human-readable summary
└── plots/00_candidate_pareto_frontier.png
```

**Tests**:
- [ ] Parse all CLI arguments
- [ ] Handle `--candidates A,C,E` selective loading
- [ ] Handle `--candidate-names` custom naming
- [ ] Generate all output files
- [ ] Verify timestamps and logging

---

## File Structure

### New Files

**1. main_da_batch.py** (~200 lines)
```python
import argparse
from multi_candidate_discovery import discover_candidates
from batch_data_loader import BatchDataLoader
from batch_aggregator import BatchAggregator
from batch_visualizer import BatchVisualizer
from batch_reporter import BatchReporter

def main():
    args = parse_args()
    
    # 1. Discover candidates
    candidates = discover_candidates(args.parent_path)
    
    # 2. Load all
    batch_loader = BatchDataLoader()
    batch_data = batch_loader.load_all_candidates(candidates, ...)
    
    # 3. Aggregate
    batch_agg = BatchAggregator(batch_data)
    batch_stats, rankings = batch_agg.aggregate_by_candidate()
    
    # 4. Visualize
    if not args.no_plots:
        batch_viz = BatchVisualizer()
        batch_viz.plot_all()
    
    # 5. Report
    batch_rep = BatchReporter(batch_stats, rankings)
    batch_rep.save_all_reports()
```

**2. multi_candidate_discovery.py** (~100 lines)
- `discover_candidates(parent_path)` → dict of candidates A, B, C...
- `has_seeds(seed_list, path)` → Check if folder has all seeds

**3. batch_data_loader.py** (~80 lines)
- `BatchDataLoader.load_all_candidates()` → unified batch_data
- Parallelize if possible (optional optimization)

**4. batch_aggregator.py** (~120 lines)
- `BatchAggregator.aggregate_by_candidate()` → stats per candidate
- `get_candidate_ranking()` → sorted by metric
- Reuses v1 `DataAggregator` logic

**5. batch_visualizer.py** (~300 lines)
- `plot_candidate_pareto_frontier()` → main plot
- `plot_candidate_success_comparison()` → grouping by constraint
- `plot_candidate_time_comparison()` → bar chart
- `plot_candidate_robustness()` → boxplot
- `plot_candidate_constraint_heatmap()` → heatmap

**6. batch_reporter.py** (~150 lines)
- `save_candidates_ranking_txt()` → human-readable
- `save_candidates_ranking_csv()` → machine-readable
- `save_candidates_summary()` → overview

### Modified Files

**1. visualizer.py** (extend)
- Import batch visualizer for cross-candidate plots
- Reuse existing single-candidate plot functions

**2. reporter.py** (extend)
- Add batch reporter methods
- Reuse existing CSV/text generation logic

---

## Testing Plan

### Unit Tests
- `test_discovery.py` - Auto-candidate discovery
- `test_batch_loader.py` - Data loading across candidates
- `test_batch_aggregator.py` - Statistics per candidate
- `test_batch_visualizer.py` - Plot generation

### Integration Tests
- `test_batch_full_pipeline.py` - End-to-end with 5 candidates
- `test_batch_performance.py` - Execution time < 3 min for 5 candidates
- `test_batch_robustness.py` - Handle malformed folders gracefully

### Output Tests
- Verify all plots generated as PNG
- Verify CSV files parseable
- Verify summary text human-readable
- Verify DPI = 300

---

## Timeline & Effort Estimate

| Phase | Task | Effort | Person Days |
|-------|------|--------|-------------|
| 1 | Build discovery module | 2h | 0.25 |
| 2 | Build batch loader | 3h | 0.375 |
| 3 | Build aggregator | 3h | 0.375 |
| 4 | Build visualizer (5 plots) | 4h | 0.5 |
| 5 | Build reporter & CLI | 2h | 0.25 |
| 6 | Testing & docs | 3h | 0.375 |
| **Total** | | **17h** | **2.125 days** |

**Estimated Delivery**: 2 business days

---

## Backward Compatibility

✅ **v1 fully operational** after v2 implementation
- Users can still use `python main_da.py` for single-folder analysis
- v2 adds `python main_da_batch.py` for multi-folder
- No breaking changes to existing v1 workflow

---

## Performance Targets (v2)

| Task | Target Time | Notes |
|------|-------------|-------|
| Discover 5 candidates | < 1s | Folder scanning |
| Load 5 candidates | ~60s | Reuses v1 DataLoader |
| Aggregate 5 candidates | ~10s | Per-candidate stats |
| Generate 5 plots | ~30s | Matplotlib rendering |
| **Total** | **~2 min** | For 5 candidates |

---

## Documentation to Create

1. **v2_MULTICANDIDATE_COMPARISON.md** ✅ (Done)
2. **USAGE_v2.md** ✅ (Done)
3. **v2_IMPLEMENTATION.md** ← README for developers
4. Update main **README.md** with v1 vs v2 comparison

---

## Approval Checklist

- [ ] Confirm v2 architecture makes sense
- [ ] Approve Phase 1-5 timeline (2 days)
- [ ] Confirm candidate naming (A-B-C or custom names)
- [ ] Confirm main plot (Pareto frontier)
- [ ] Ready to proceed with implementation

---

## Next Steps

1. **Review & Approve**: This roadmap
2. **Implement**: Phases 1-5 in order
3. **Test**: All unit + integration tests pass
4. **Document**: Update main README.md with v2 section
5. **Release**: v2 ready for thesis batch analysis

---

**Status**: Ready for implementation approval
