# U4 Accuracy Audit: Usage & Architecture Guide

File: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_accuracy_v3.py`  
Date: 2026-04-18

This document outlines the usage and engineering architecture of the V3 Accuracy Benchmark. 

---

## 1. The Design: Absolute Parity
The core mandate of this accuracy audit is to guarantee that the physics environment we test for *accuracy* is **100% mathematically identical** to the environment we tested for *speed*.

To achieve this, the accuracy script is built as a **Lightweight Import Wrapper**:
*   It does **not** clone or rewrite the solver logic.
*   It executes `from benchmark_ode_solvers_v3 import p_sample_loop_v3_fair`.
*   It uses the exact same `argparse` configuration. 

Consequently, if the latency tests were fair, the accuracy tests perfectly inherit that fairness.

---

## 2. Benchmark Logic: How it works

When you execute `benchmark_ode_accuracy_v3.py`, the following sequence occurs:

### Phase 1: The Golden Oracle (100-Step Dopri5)
The script generates a static global noise tensor ($x_{noise}$) tied to your chosen `--seed`. 
Instead of trusting any legacy solver, it creates an absolute mathematical **Ground Truth Line**. It feeds the noise into `torchdiffeq:dopri5` evaluated densely over 100 steps with incredibly strict numerical tolerances (`atol=1e-10`, `rtol=1e-10`). This endpoint is saved as the `Oracle State`.

### Phase 2: Candidate Deviation (Drift)
The script iteratively loops over your chosen `--solver-spec`. 
It runs the exact same $x_{noise}$ through the standard evaluation length (e.g., `--steps 10`). At the final integration, it calculates the **L2 Euclidean Distance** between the solver's estimation and the Golden Oracle.

### Phase 3: Plotting
By appending the `--plot` flag, `matplotlib` automatically compares the L2 Drift of all evaluated solvers and generates a Bar Chart.

---

## 3. How to Execute

Because of the Drop-In CLI parity, returning an accuracy matrix is as easy as running your standard latency tests. 

### Basic Accuracy Probe (Terminal Only)
Runs the test and prints the drift matrix to standard output without drawing plots.

```bash
!/content/miniconda3/envs/FMPCC/bin/python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_accuracy_v3.py \
  --mode math \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --device cuda \
  --batch-size 4 \
  --steps 10 \
  --solver-spec legacy:euler,legacy:rk4
```

### Visual Drift Output (With Plotting)
Adding the `--plot` command will instruct the script to render the visual comparisons.

```bash
!/content/miniconda3/envs/FMPCC/bin/python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_accuracy_v3.py \
  --mode math \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --device cuda \
  --batch-size 4 \
  --steps 5 \
  --solver-spec legacy:euler,legacy:midpoint,legacy:rk4 \
  --plot
```

### What to expect in Phase U4:
The output plot (`accuracy_drift_plot.png`) should definitively prove the "Garbage In, Struggle Out" guidance theory: `Euler` will show massive drift at $S=5$, forcing the DPCC Safety Projector to solve broken physics, while `RK4` will register minimal drift, allowing instant collision optimizations.
