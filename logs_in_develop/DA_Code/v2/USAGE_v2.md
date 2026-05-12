# v2 Usage Guide: How to Use Multi-Candidate Batch Analysis

## Quick Start (30 seconds)

```bash
cd /workspaces/FM-PCC

# Analyze all candidates in a parent directory
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans
```

**Output**: Timestamped folder with:
- `00_candidate_pareto_frontier.png` ← Main thesis figure
- `candidates_ranking.csv` ← Rankings table
- `candidates_summary.txt` ← Human-readable summary

Done! All candidates A, B, C, D, E automatically discovered and compared.

---

## Installation & Setup

### 1. Verify Dependencies
```bash
pip install numpy pandas matplotlib pyyaml scipy
```

### 2. Verify v2 Modules Exist
```bash
ls -la /workspaces/FM-PCC/Data_Analysis/DA_Code/
# Should show:
#   multi_candidate_discovery.py
#   batch_data_loader.py
#   batch_aggregator.py
#   batch_visualizer.py
#   batch_reporter.py
#   main_da_batch.py
```

### 3. Test Discovery (Optional)
```bash
python Data_Analysis/DA_Code/multi_candidate_discovery.py logs/avoiding-d3il/plans
```

Expected output:
```
Candidate A: diffusion/H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5
Candidate B: diffusion/H8_K10_Dmodels.GaussianDiffusion_aw10_thres0.5
Candidate C: diffusion/H8_K20_T1_Dmodels.GaussianDiffusion_aw10
...
```

---

## Usage Examples

### Example 1: Basic Full Analysis (All Candidates)

```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --output-path ./thesis_batch_results
```

**What it does**:
1. Finds all subfolders with seeds [6, 7, 8, 9, 10]
2. Auto-assigns letters A, B, C, D, E...
3. Loads all seeds for each candidate
4. Aggregates metrics across variants
5. Generates 5 comparison plots
6. Creates ranking tables

**Output structure**:
```
thesis_batch_results/
└── 20260512_143022_FM_V3_BATCH/          ← Timestamped
    ├── plots/
    │   ├── 00_candidate_pareto_frontier.png
    │   ├── 01_candidate_success_comparison.png
    │   ├── 02_candidate_time_comparison.png
    │   ├── 03_candidate_robustness_boxplot.png
    │   └── 04_candidate_constraint_heatmap.png
    ├── candidates_summary.txt
    ├── candidates_ranking.csv
    ├── candidates_detailed.csv
    └── logs/
        ├── batch_analysis.log
        └── batch_loading.log
```

**Time**: ~2-3 minutes for 5 candidates

---

### Example 2: Select Specific Candidates (A, C, E Only)

```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --candidates A,C,E \
    --output-path ./selective_batch
```

**Why use this**:
- Faster analysis (skip B and D)
- Focus on specific hyperparameter variants
- Cleaner plots with fewer candidates

**Time**: ~1 minute for 3 candidates

---

### Example 3: Use Custom Candidate Names (For Thesis)

```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --candidate-names "aw=10_K20,aw=10_K10,threshold=1,threshold=0.5" \
    --output-path ./named_batch
```

**Output plots will show**:
```
✓ Candidate aw=10_K20: 87.3% success
✓ Candidate aw=10_K10: 84.2% success
✓ Candidate threshold=1: 89.5% success
✓ Candidate threshold=0.5: 79.8% success
```

Instead of just "A", "B", "C", "D".

**Recommendation**: Use custom names for thesis publication quality

---

### Example 4: Quick Check Without Plots (Fast)

```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --no-plots
```

**Output**: CSVs + summary text only (no matplotlib rendering)

**Time**: ~30 seconds

---

### Example 5: Debug Mode (Verbose Logging)

```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --verbose 2>&1 | tee batch_debug.log
```

**Shows**:
- Which candidates discovered
- Loading progress for each candidate
- Aggregation details
- Plotting progress

**Use when**: Something goes wrong or you want to understand what happened

---

### Example 6: Custom Seeds (If Not 6-10)

```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --seeds 0,1,2,3,4
```

**Use when**: Your evaluation used different seeds [0-4] instead of default [6-10]

---

### Example 7: Select Specific Variants Only

```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --variants "dpcc-c,dpcc-r,diffuser"
```

**Use when**: You only care about specific methods (ignore others like post_processing variants)

---

## Understanding the Output

### File 1: `00_candidate_pareto_frontier.png` (MAIN THESIS FIGURE)

Shows accuracy vs time for all candidates.

**Interpretation**:
```
Accuracy
    │
  90% ─────●C         ← Highest accuracy (89.5%)
    │    /│ \         
  85% A●   │  \      ← Best tradeoff (87.3%, 42ms)
    │  \  │   \
  80% ─B●─┼────\    ← Fastest (38ms) but lower accuracy
    │    \│     \
  75% ────●D,E   ← Dominated (worse than others)
    │
    └────────────────
     35  40  45  50 ms
          (Time)
```

**How to read**:
- Upper-left region = good (high accuracy, low time)
- Lower-right = bad (low accuracy, high time)
- Closer to upper-left = better candidate

**For thesis text**:
```
"Figure 2 shows the cross-candidate Pareto frontier. Candidate C 
achieves the highest success rate (89.5%) but requires longer 
computation time (45.2ms). Candidate A represents the optimal 
tradeoff, balancing accuracy (87.3%) with efficient computation 
time (42.1ms)."
```

---

### File 2: `candidates_ranking.csv` (IMPORT TO EXCEL)

```csv
Rank,Candidate,Folder,Accuracy (%),Accuracy Std (%),Time (ms),Time Std (ms),Robustness
1,C,diffusion_H8_K20_T1,89.5,1.9,45.2,3.1,0.018
2,A,diffusion_H8_K20_aw10,87.3,2.1,42.1,2.3,0.015
3,B,diffusion_H8_K10_aw10,84.2,2.8,38.2,1.9,0.020
4,D,diffusion_H8_K20_T0.5,79.8,4.2,48.5,4.1,0.035
5,E,flow_matching_ode_selector,72.1,5.5,52.3,5.9,0.052
```

**How to use**:
1. Open in Excel
2. Copy into thesis appendix
3. Reference in main text: "See Table X for full rankings"

---

### File 3: `candidates_summary.txt` (QUICK LOOKUP)

```
=== CANDIDATES DISCOVERED ===
Total: 5
  A: diffusion/H8_K20_aw10_thres0.5
  B: diffusion/H8_K10_aw10_thres0.5
  C: diffusion/H8_K20_T1_aw10
  D: diffusion/H8_K20_T0.5_aw1
  E: flow_matching_v3_ode_selectable/...

=== RANKINGS BY ACCURACY ===
1. Candidate C: 89.5% (±1.9%), Time: 45.2ms
2. Candidate A: 87.3% (±2.1%), Time: 42.1ms
3. Candidate B: 84.2% (±2.8%), Time: 38.2ms
...

=== RECOMMENDATIONS ===
Overall Best: Candidate A
Reason: Good accuracy (87.3%) with lowest time overhead (42.1ms)
Fastest: Candidate B (38.2ms)
```

**When to read**: When you need the answer in plain English

---

### File 4: `candidates_detailed.csv` (SUPPLEMENTARY)

More detailed metrics per candidate:
```csv
Candidate_Letter,Folder_Name,Full_Path,Accuracy,Accuracy_Std,Time_ms,Time_Std,Robustness_Score
A,diffusion_H8_K20_aw10,/path/to/A,0.873,0.021,42.1,2.3,0.015
B,diffusion_H8_K10_aw10,/path/to/B,0.842,0.028,38.2,1.9,0.020
C,diffusion_H8_K20_T1,/path/to/C,0.895,0.019,45.2,3.1,0.018
...
```

**Use when**: Creating supplementary tables for technical appendix

---

## Real-World Example: Ablation Study

You varied action weight (aw) and want to see which is best.

### Setup
```bash
# Trained 3 models with different action weights
python train.py --aw 1 → logs/.../H8_K20_aw1/seeds-[6,7,8,9,10]/...
python train.py --aw 5 → logs/.../H8_K20_aw5/seeds-[6,7,8,9,10]/...
python train.py --aw 10 → logs/.../H8_K20_aw10/seeds-[6,7,8,9,10]/...
```

### Run v2 Analysis
```bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path logs/avoiding-d3il/plans \
    --candidate-names "aw=1,aw=5,aw=10" \
    --output-path ./ablation_aw_study
```

### Interpret Results
```
Rankings:
1. Candidate aw=10: 87.3%
2. Candidate aw=5: 85.2%
3. Candidate aw=1: 78.9%

Conclusion: Higher action weight improves accuracy.
Recommendation: Use aw=10 for baseline method setup.
```

### Write for Thesis
```
"Ablation study on action weight parameter (Figure 3) shows that 
increasing action weight from 1 to 10 improves goal success rate 
from 78.9% to 87.3%. This indicates that stronger guidance towards 
goal achievement is beneficial within tested range."
```

---

## Common Use Cases

### Use Case 1: Best Hyperparameter Selection
```bash
# Compare different learning rates / action weights / time horizons
python main_da_batch.py --parent-path logs/... \
    --candidate-names "lr=1e-3,lr=5e-4,lr=1e-4" \
    --output-path ./hyperparam_study
```

**Output**: Which learning rate performs best?

---

### Use Case 2: Method Comparison
```bash
# Compare different control/planning methods
python main_da_batch.py --parent-path logs/... \
    --candidate-names "DPCC-Safe,Diffuser-Baseline,FM-v3" \
    --output-path ./method_comparison
```

**Output**: Which method is most accurate/fast?

---

### Use Case 3: ODE Solver Benchmarking
```bash
# Compare numerical integration methods
python main_da_batch.py --parent-path logs/... \
    --candidate-names "Euler_K10,RK4_K10,Midpoint_K10" \
    --output-path ./solver_benchmark
```

**Output**: Which solver gives best accuracy vs speed tradeoff?

---

### Use Case 4: Constraint Type Analysis
```bash
# Focus only on specific constraints
python main_da_batch.py --parent-path logs/... \
    --constraint-types "halfspace,obstacles" \
    --candidate-names "aw=1,aw=10" \
    --output-path ./constraint_study
```

**Output**: How do candidates perform on spatial constraints?

---

## Troubleshooting

### Issue 1: "No candidates found"

```
✗ Candidate Discovery: 0 candidates found
```

**Check 1**: Do folders have seed subdirectories?
```bash
ls logs/avoiding-d3il/plans/*/[6-9]/
# Should show seed folders with halfspace/results directories
```

**Check 2**: Is parent path correct?
```bash
# Try pointing directly to parent of seed folders
python main_da_batch.py --parent-path logs/avoiding-d3il/plans/diffusion
```

**Check 3**: Use verbose mode to debug
```bash
python main_da_batch.py --parent-path logs/... --verbose
```

---

### Issue 2: Only 1-2 candidates instead of expected 5

**Likely cause**: Experimental folders are nested at different levels

**Solution**: 
```bash
# Adjust path to parent of candidate folders
# Instead of: logs/avoiding-d3il/plans/
# Try:        logs/avoiding-d3il/plans/diffusion/
```

---

### Issue 3: Plots show all same color / hard to distinguish

**Fix**: Use custom names with clear differences
```bash
python main_da_batch.py --parent-path logs/... \
    --candidate-names "Method-A-Fast,Method-A-Accurate,Method-B,Method-C"
```

---

### Issue 4: "ValueError: float() arg is NaN"

**Cause**: Some candidates missing data for certain metrics

**Fix**: Run with `--verbose` to see which candidate failed
```bash
python main_da_batch.py --parent-path logs/... --verbose
```

Or skip problematic candidate:
```bash
python main_da_batch.py --parent-path logs/... --candidates A,C,E
```

---

### Issue 5: Execution takes too long (> 5 min)

**Optimization 1**: Skip plots
```bash
python main_da_batch.py --parent-path logs/... --no-plots
```

**Optimization 2**: Select fewer candidates
```bash
python main_da_batch.py --parent-path logs/... --candidates A,B,C
```

**Optimization 3**: Profile where time is spent
```bash
time python Data_Analysis/DA_Code/main_da_batch.py --parent-path logs/...
```

---

## CLI Reference

```bash
python main_da_batch.py --help

Options:
  --parent-path PARENT_PATH
      Required: Parent directory containing candidate subfolders
      
  --output-path OUTPUT_PATH
      Output directory for results (default: ./fm_v3_batch_analysis_output)
      
  --candidates A,C,E
      Optional: Select specific candidates to analyze (default: all)
      
  --candidate-names "name1,name2,name3"
      Optional: Custom names for candidates (replaces auto A, B, C)
      
  --seeds 6,7,8,9,10
      Optional: Seed numbers to load (default: 6,7,8,9,10)
      
  --variants "dpcc-c,diffuser"
      Optional: Specific variants to include (default: all)
      
  --constraint-types "halfspace,obstacles"
      Optional: Constraint types to include (default: all)
      
  --no-plots
      Skip plot generation (faster, CSVs only)
      
  --verbose
      Enable detailed logging
```

---

## Performance Expectations

| Task | Time | Notes |
|------|------|-------|
| Discover 5 candidates | < 1s | Folder scanning |
| Load 5 candidates (18 variants each) | ~60s | Reuses v1 DataLoader |
| Aggregate 5 candidates | ~10s | Per-candidate stats |
| Generate 5 plots | ~30s | Matplotlib rendering |
| **Total** | **~2 min** | For 5 candidates × 18 variants |

**Pro tip**: First run takes longest due to I/O. Subsequent runs faster.

---

## Integration with v1

**v1 still works!** You can use both:

```bash
# v1: Analyze single folder
python main_da.py --input-path FM_v3_ode_test

# v2: Compare multiple folders
python main_da_batch.py --parent-path logs/...
```

No conflicts. Choose tool based on your need:
- **v1** when analyzing single experimental run
- **v2** when comparing multiple experimental runs

---

## Tips & Tricks

### Tip 1: Save command for reproducibility
```bash
# Save exact command used
cat > reproduce_analysis.sh << 'EOF'
#!/bin/bash
python Data_Analysis/DA_Code/main_da_batch.py \
    --parent-path /workspaces/FM-PCC/logs/avoiding-d3il/plans \
    --candidate-names "aw=1,aw=5,aw=10,dpcc-baseline" \
    --output-path ./thesis_batch_final \
    --verbose
EOF

chmod +x reproduce_analysis.sh
./reproduce_analysis.sh > batch_log.txt 2>&1
```

**Why**: Can reproduce exact analysis for thesis documentation

---

### Tip 2: Compare results across runs
```bash
# Run 1: January 2026
python main_da_batch.py --parent-path logs/jan_2026 \
    --output-path ./results_jan2026

# Run 2: May 2026  
python main_da_batch.py --parent-path logs/may_2026 \
    --output-path ./results_may2026

# Compare CSV files:
diff results_jan2026/candidates_ranking.csv results_may2026/candidates_ranking.csv
```

---

### Tip 3: Create publication-ready figures
```bash
# Use custom names + high-contrast candidates
python main_da_batch.py --parent-path logs/... \
    --candidates A,B,C \
    --candidate-names "Proposed Method,DPCC Baseline,Diffuser" \
    --output-path ./publication_results
    
# Copy PNG to thesis:
cp publication_results/*/plots/*.png ~/thesis/figures/
```

---

## Next Steps

1. ✅ Read this usage guide (done)
2. 🔲 Try Example 1 (basic analysis)
3. 🔲 Check output in timestamped folder
4. 🔲 Open Pareto plot
5. 🔲 Read summary text
6. 🔲 Import CSV to Excel if needed
7. 🔲 Try custom names for your actual use case
8. 🔲 Copy best plots to thesis

---

## Questions?

Check [MISSION_BRIEFING_v2.md](MISSION_BRIEFING_v2.md) for:
- Technical architecture
- What each phase does
- Pareto frontier explanation
- Integration with thesis workflow

---

**Version**: v2.0  
**Status**: Production Ready  
**Last Updated**: 2026-05-12  
**Next**: Thesis batch analysis enabled!
