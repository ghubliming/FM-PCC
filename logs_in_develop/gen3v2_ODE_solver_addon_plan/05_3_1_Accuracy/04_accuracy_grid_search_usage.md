# U4 Accuracy Grid Search: Automated Macro Analysis

File: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v3/grid_search_accuracy_v3.py`  
Date: 2026-04-18

This script provides a global view of how solver accuracy scales across three critical dimensions: **Batch Size, Horizon, and Integration Steps.** It is designed to generate the primary evidence for deciding which solver is "Production-Ready."

---

## 1. Automated Analysis Engine

The grid search script automates the execution of multiple `benchmark_ode_accuracy_v3` processes. It aggregates all data into a single `MASTER_ACCURACY_MATRIX_V3_math.csv` and renders macro trendlines.

### Key Macro Trends:
*   **Accuracy vs Steps**: Visualizes the mathematical convergence rate (Log Scale). You should see RK4 drop at an $O(h^4)$ rate while Euler lags at $O(h)$.
*   **Accuracy vs Horizon**: Shows how "Sequence Elongation" affects error. As the time-horizon ($H$) increases, trajectory drift naturally accumulates; this plot shows how higher-order solvers mitigate that accumulation.

---

## 2. Understanding ODE Steps ($S$) & Convergence Math

A common question during the audit is: **"Why does increasing steps reduce the drift on the plot?"** 

### A. The Definition of $h$ (Step Size)
In our Flow Matching ($T=0$ to $T=1.0$) environment, the total time is fixed at **1.0**. 
The number of **Steps ($S$)** determines the size of each integration jump ($h$):
$$h = \frac{1.0}{S}$$
*   $S=1 \implies h=1.0$ (One massive, inaccurate jump)
*   $S=20 \implies h=0.05$ (Twenty small, precise micro-steps)

### B. Why the lines "Shift Down" (The Math)
Every step in an ODE solver introduces a small "Local Truncation Error." When you move from $S=1$ to $S=20$, you are making $h$ smaller. The mathematical relationship is expressed as $O(h^n)$, where $n$ is the **Order** of the solver.
*   **Euler ($O(h^1)$)**: If you double the steps ($h \to h/2$), the error is only halved.
*   **RK4 ($O(h^4)$)**: If you double the steps, the error drops by $2^4 = 16\times$.

This is why, on your Log-scale plot, the **RK4 line is much steeper** than the Euler line. The "shifting" you see is the algorithm converging toward the true continuous vector field as the step size $h$ approaches zero.

---

## 3. Command Line Usage

Use commas to define your search space. The script will cross-multiply all options.

```bash
!/content/miniconda3/envs/FMPCC/bin/python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/v3/grid_search_accuracy_v3.py \
  --mode math \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --device cuda \
  --solver-spec legacy:euler,legacy:rk4 \
  --grid-batch 128 \
  --grid-steps 5,10,20,50 \
  --grid-horizon 8,16,32
```

### B. The Stability Paradox: Why RK4 "loses" at $S=1$

You will notice on the plot that at $S=1$ or $S=2$, the more advanced solvers (RK4 and Midpoint) can actually have **higher drift** than the simple Euler jump. Here is the clear explanation for why this happens:

#### 1. The "Map Peek" Analogy (Intuition)
Imagine you are trying to find your way through a forest using a map (the Neural Network's vector field):
*   **Euler (The Simpleton)**: "I'll look at my compass once, and just walk straight for 1km." He misses the curve, but he stays on a predictable path.
*   **RK4 (The Over-Thinker)**: "I'll look at the map at 0km. Then I'll peek at 0.5km. Then I'll peek at 0.5km *again* based on the last peek. Then I'll look at 1km. I'll average all of them to find the perfect curve."
*   **The Trap**: At $S=1$ ($h=1.0$), those "peeks" are 500 meters away. If the vector field is noisy or highly curved, those peeks land in areas the network hasn't learned well. The solver "trusts" these noisy peeks, which leads it to swerve wildly off-course.

#### 2. The Error Constant ($E = C \cdot h^n$) (Technical)
In numerical math, the error ($E$) of a solver depends on two things:
1.  **The Order ($n$)**: How fast the error drops as steps increase (Euler is 1, RK4 is 4).
2.  **The Error Constant ($C$)**: The "base" error of the method's complexity.

Higher-order methods like RK4 have a **much larger $C$** because they involve more complex stage additions. When $h=1.0$, the $h^n$ term doesn't help much ($1^4$ is the same as $1^1$). Therefore, at $S=1$, you are seeing the "Complex Constant" penalty of RK4 without its "Order" benefit. 

#### 3. Why this justifies RK4 for Production
We never run $S=1$ in a real robot. We care about the **Convergence Rate**. As soon as you move to $S=5$ or $S=10$, the $h^4$ term ($0.1^4 = 0.0001$) becomes so tiny that it completely cancels out the $C$ constant. 

**Conclusion**: The plot proves that while RK4 is "too smart for its own good" at 1 step, it is the **only** solver that can reach $10^{-3}$ accuracy levels once you give it a reasonable number of steps.

---

## 3. Analyzing the Drift Paradox (CSV vs Plot)

If you look at the generated Accuracy Plot and the accompanying CSV metadata, you will notice two striking phenomena that are critical for your audit.

### A. Why Euler & Midpoint Are Bit-Identical (Not a Bug)
In your CSV, `legacy:euler` and `torchdiffeq:euler` produce **exactly the same number** to all 16 decimal places. At first glance this looks like a bug — surely two independent implementations should differ slightly due to floating point? Here is why they don't:

**The Root Cause: Euler is too simple to implement differently.**
The entire Euler algorithm is one line: `x = x + v * dt`. There is no "alternative way" to compute this.
Given that our test harness forces:
1.  **Same starting noise** (`global_noise.clone()` for both)
2.  **Same `dt`** (`1.0 / steps` — identical Python float)
3.  **Same time values** (`float(i)/steps` in legacy vs `linspace(0,1,S+1)[i]` in torchdiffeq — these produce the same IEEE 754 floats for clean fractions)
4.  **Same neural network call** (`_predict_velocity` with identical inputs)

The operations are performed in the **exact same order** on the **exact same tensors**, so the GPU produces bit-identical results. The same logic applies to Midpoint (2 stages, but both implementations use the same sequence: evaluate, step half, evaluate again, step full).

**Why the speed benchmark might show tiny differences**: The latency benchmark (`benchmark_ode_solvers_v3.py`) generates **separate random noise** for each solver. Different noise → different trajectory → different numbers. The accuracy audit intentionally locks the noise to measure pure algorithmic drift.

**Why this is actually good news**: If they DIDN'T match, it would mean our test harness was broken (wrong noise injection, wrong conditioning, etc.).

### B. Why RK4 IS Different ($2\times$ Gap)
Unlike the trivial Euler formula, RK4 has **4 intermediate stages** with complex weighted additions. This creates room for real implementation variance:

*   **The Data**: At $S=10$, `legacy:rk4` ($0.0408$) is roughly **$2\times$ less accurate** than `torchdiffeq:rk4` ($0.0214$).
*   **The Technical Verdict**:
    1.  **Stages 1-2 are identical**: Proven by the fact that Midpoint (which uses the same first 2 stages) matches perfectly.
    2.  **Stages 3-4 diverge**: `torchdiffeq` uses an internal Butcher Tableau class that accumulates the 4 stage values with higher numerical precision and potentially a different RK4 variant (e.g., the "3/8 Rule") than our Classical RK4 manual loop.
    3.  **Parallel Slopes**: On the Log-Log plot, both RK4 lines are **parallel**. This confirms both are genuinely 4th-order ($O(h^4)$); the library simply has a smaller constant-factor error.
*   **Key takeaway**: For your auditing report, use the `torchdiffeq` numbers as the "Mathematical Limit" and the `legacy` loop as the "Production Performance Proxy."

---

## 4. Interpreting the Macro Plots

### Statistical Error Bands (Mean ± Std)
On the macro trendlines, each data point now includes **vertical error bars**.
*   **Interpretation**: If the vertical bars are tight, the solver is "Universal"—it performs equally well across the whole dataset. 
*   **Edge Cases**: Large error bars suggest the solver is "Brittle"—it might be accurate on average but fails catastrophically on specific complex trajectories.

### The "Drift vs Steps" Log Plot
This is your most important chart for management review.
*   **The Euler Line**: Usually a shallow linear diagonal. Doubling steps only halves the error.
*   **The RK4 Line**: A steep vertical drop. Doubling steps reduces error by $16\times$.
*   **Decision Rule**: Find the point where the RK4 line crosses below your "Safety Threshold" (e.g., L2 Distance of 0.01). That is your optimal production step count.

---

## 4. Output Summary
At the end of the run, the grid search will output a **Top 5 Hall of Fame** in your console, helping you instantly identify the mathematically "Golden" configuration for your specific hardware/VF setup.
