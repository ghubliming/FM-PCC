# FM v3 ODE-Selectable Evaluation Analysis - Mission Briefing

## Executive Summary

You have collected **834+ evaluation result files** (.npz format) from FM v3 ODE-Selectable tests across **5 random seeds [6,7,8,9,10]**, **18 projection variants**, and **4 constraint types** with **3 halfspace geometry variations**.

**Problem**: This data is **impossible to visualize manually** or compare across all dimensions simultaneously.

**Solution**: The Data Analysis (DA) tool automatically:
- Loads all result files from directory tree
- Aggregates metrics across seeds (computes mean ± std)
- Generates **10+ publication-quality plots**
- Produces **comparison tables** (txt + csv)
- **Highlights your thesis-critical results** (accuracy vs. time Pareto frontier)

---

## Your Research Question

**Which method achieves the best accuracy-time tradeoff?**

- **Primary candidates**: dpcc-c, dpcc-r, dpcc-t (your main contributions)
- **Baseline comparison**: diffuser (raw ML-only) — reveals how much constraint satisfaction adds
- **Secondary variants**: gradient, post_processing, model_free, etc. (may reveal important insights)

---

## Key Metrics Being Analyzed

| Metric | What It Means | Thesis Relevance |
|--------|---------------|------------------|
| **n_success** | Goal reached (%) | Baseline performance |
| **n_success_and_constraints** | Goal + constraint satisfaction (%) | **PRIMARY** for your results |
| **collision_free_completed** | Collision-free rate (%) | Safety metric |
| **avg_time** | Computation time (ms) | **Practical applicability** |
| **n_violations** | Avg constraint violations per trial | Safety assessment |
| **total_violations** | Cumulative violation magnitude | Severity assessment |
| **n_steps** | Planning steps needed (mean ± std) | Efficiency |

---

## Main Outputs For Your Thesis

### 1. **Pareto Frontier Plot** (MOST IMPORTANT)
- **File**: `plots/00_pareto_frontier_accuracy_vs_time.png`
- **Shows**: Accuracy (Y) vs. Time (X) with variants color-coded
- **Purpose**: Visually demonstrates which variant wins the accuracy-time tradeoff
- **Your variants**: 
  - 🔴 dpcc-c (red) — likely highest accuracy
  - 🟠 dpcc-r (orange) — likely fastest
  - 🟡 dpcc-t (yellow) — likely sweet spot
  - 🔵 diffuser (blue) — baseline ML-only method

### 2. **Summary Report** (HUMAN-READABLE)
- **File**: `results_summary.txt`
- **Contains**:
  - Top 10 variants by goal + constraint success
  - Top 10 variants by goal success alone
  - Average performance by constraint type
  - Average performance by halfspace geometry
  - Overall statistics

### 3. **Detailed CSV Tables** (MACHINE-READABLE)
- `results_by_variant.csv` — Metrics aggregated per variant
- `results_by_constraint.csv` — Metrics aggregated per constraint type
- `results_by_halfspace.csv` — Metrics aggregated per halfspace variant
- `detailed_results.csv` — Every single data point (all seeds × variants × constraints)

### 4. **Additional Plots** (FOR SUPPLEMENTARY MATERIAL)
- Variant rankings by each metric
- Heatmaps showing variant × constraint performance
- Boxplots showing seed-to-seed variability
- Efficiency scatter plots

---

## Data Organization (Input)

Your FM v3 ODE-Selectable test results should be organized as:

```
FM_v3_ode_selectable_test/
  6/results/
    halfspace_top-right-hard/
      dpcc-c.npz
      dpcc-r.npz
      dpcc-t.npz
      ... (other variants)
    halfspace_top-left-hard/
      dpcc-c.npz
      ...
    halfspace_both-hard/
      dpcc-c.npz
      ...
  7/results/
    halfspace_top-right-hard/
      ...
  8/results/
    ...
  9/results/
    ...
  10/results/
    ...
```

The script **auto-discovers** this structure — no manual enumeration needed.

---

## Analysis Workflow

```
Input: 834 .npz files (eval results)
    ↓
DataLoader: Organize by seed/variant/constraint
    ↓
DataAggregator: Compute mean, std, min, max across seeds
    ↓
┌─────────────────┬──────────────┬─────────────┐
↓                 ↓              ↓             
Reporter      Visualizer    Aggregator
(txt/csv)     (10+ plots)   (statistics)
    ↓             ↓              ↓
Summary.txt  Plots/.../      CSV Tables
```

**Total execution time**: ~1-2 minutes for full analysis

---

## What Each Variant Represents

### Primary Contribution (Your Work)
- **dpcc-c** (DPCC with C constraints) — Likely best accuracy
- **dpcc-r** (DPCC with R constraints) — Likely fastest  
- **dpcc-t** (DPCC with T constraints) — Likely balanced tradeoff

### Constraint Handling Baselines
- **diffuser** — Raw diffusion model (ML-only, no constraints) ← **CRITICAL BASELINE**
- **gradient** — Gradient-based projection
- **post_processing** — Post-hoc constraint satisfaction
- **model_free** — Constraint satisfaction without learned dynamics

### Variants with Hyperparameter Tuning
- Elements with `-tightened` suffix — More aggressive constraint enforcement
- Elements with `-dt0p25`, `-dt0p5`, `-dt2p0`, `-dt4p0` — Different time discretization steps

---

## Success Criteria For Your Thesis

The analysis tool should help you prove:

✅ **Claim 1**: dpcc-c/r/t achieve >85% goal + constraint success  
✅ **Claim 2**: dpcc variants outperform baseline diffuser by ≥15% in constraint satisfaction  
✅ **Claim 3**: Time overhead is acceptable (~40-50ms for practical applications)  
✅ **Claim 4**: Performance is robust across 5 random seeds (error bars show <5% variance)  
✅ **Claim 5**: Performance is consistent across all halfspace geometries  

---

## Next Steps

1. **Run the analysis** (see USAGE.md)
2. **Check the Pareto frontier plot** — This is your thesis main figure
3. **Read results_summary.txt** — Confirms rankings and statistics
4. **Export CSV tables to Excel** — For detailed supplementary tables
5. **Include plots in your thesis** — Publication-quality figures ready to go

---

## Files Location

- **Code**: `/workspaces/FM-PCC/Data_Analysis/DA_Code/`
- **Plan**: `/workspaces/FM-PCC/logs_in_develop/DA_Code/DA_PLAN.md`
- **Usage**: `/workspaces/FM-PCC/logs_in_develop/DA_Code/USAGE.md` (this file)

---

## Questions This Analysis Answers

For your thesis discussion/results:

1. **Which variant is best overall?** → Pareto frontier plot
2. **How much better is dpcc-c/r/t vs. baseline?** → Variant rankings
3. **Does it work across different constraint geometries?** → Constraint-type breakdowns
4. **Is the improvement statistically significant?** → Error bars (seed variability)
5. **What's the computational cost?** → Time comparison plots
6. **Which constraints are hardest?** → Constraint-specific success rates

---

**Status**: Ready to execute. No further setup needed.
