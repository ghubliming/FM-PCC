# 05.4: Batch Consistency & Noise Refactor Roadmap

Date: 2026-04-18  
Goal: Standardize noise generation and eliminate all stochastic factors to achieve "Pure Algorithmic" metrics.

## 1. The Current Problem: Spatial Variance Contamination

In the current V3 benchmarking suite, the batch initialization (noise) is handled incorrectly for pure numeric auditing:

*   **Loop-Internal Randomness**: Scripts currently call `torch.randn` inside the trial loop. 
    *   *Effect*: Every individual trial solves a different math path. 
    *   *Result*: Latency data is "polluted" because we are mixing hardware speed with the difficulty of the random path.
*   **The Legacy Evidence**: This issue is inherited from V1/V2. In `v1/benchmark_ode_solvers.py`, noise is re-generated inside the trial loop:
    ```python
    414: for trial in range(int(args.n_trials)):
    415:     x0 = np.random.randn(args.batch_size, args.state_dim).astype(np.float32)
    ```
    And in `v2/benchmark_ode_solvers_v2.py`:
    ```python
    329: for trial in range(args.n_trials):
    340:     x = 0.5 * torch.randn(shape, device=args.device)
    ```

---

## 2. Origin: Why was it designed this way?

The V1/V2 "Random per Trial" design (Random Selecting) was originally intended to solve two specific imitation problems:

1.  **Imitating Real Distribution**: By choosing new noise every trial, the benchmark provided an estimate of the "Average Throughput" across the entire state space. It ensured that the final mean timing wasn't skewed by a single "Easy Path" or "Hard Path."
2.  **Imitating Hardware Exhaustion**: There was a concern that re-running the exact same floating-point numbers would allow modern GPUs or compilers to use internal caches to "cheat" the benchmark. Randomizing the input forced the Tensor Cores to perform fresh arithmetic every single trial.

---

## 3. Evolution: Why are we changing it now?

While the random-per-trial approach was good for general throughput estimation, it fails the requirements of the **Safety Projector Accuracy Audit**:

1.  **The "Controlled Experiment" Problem**: To compare Euler vs. RK4 vs. Midpoint, we need a "Clean Room." If the path changes every trial, we cannot prove that one solver is mathematically superior; we can only prove it was "luckier" in that trial.
2.  **High-Resolution Stability Tracking**: We are now tracking per-substep drift. Any spatial jitter (noise change) makes these accumulation lines unreadable.
3.  **Refusal of "Averaged Luck"**: We no longer want the "Average Speed of the Universe." We want the **Deterministic Cost of the Algorithm.** We need to know that if a robot takes a specific corner, the solver will *always* take exactly $X$ ms and have exactly $Y$ drift.

---

## 4. The Split-Logic Strategy: Auditing vs. Production

To maintain both scientific precision and real-world realism, we will implement a "Split Brain" noise protocol:

### A. Math Mode (`--mode math`) -> **LOCKED NOISE**
*   **Goal**: Pure numerical auditing.
*   **Rule**: The batch is generated **once** and used for all trials. This eliminates spatial variance and allows for bit-identical comparison of solver precision.

### B. Production Mode (`--mode production`) -> **RANDOM PER TRIAL**
*   **Goal**: Real-world robustness testing.
*   **Rule**: Every trial generates **new noise**, just like a real robot deployment. This ensures that the production controller is tested across the entire distribution of the state space.

---

## 5. The "No Randomness" Principle (For Math Mode)

To achieve "Clean Room" results, we enforce the following requirements:
*   **Zero Stochasticity**: All potential random factors (Dropout, Jitter, Stochastic Batching) must be locked or disabled.
*   **Locked Batch Basis**: The initial robot coordinates ($x_0$) must be generated **once** per benchmark run and cloned for every trial and every solver.
*   **Bit-Identical Math**: Multiple runs of the same solver on the same machine must produce 100% bit-identical trajectory results.

---

## 6. Fix Plan for 4 Python Files

The refactor targets the following V3 files to ensure consistency:

### A. `benchmark_ode_solvers_v3.py` (Time/Latency)
*   **Task**: Move `torch.randn` outside the trial loop. 
*   **Logic**: Define `global_x_init` before starting the first trial to ensure every trial solves the exact same problem.

### B. `benchmark_ode_accuracy_v3.py` (Accuracy Audit)
*   **Task**: Harmonize the noise injection wrapper. 
*   **Logic**: Force the `production` controller to use the same noise basis as the `math` solver to ensure the L2 drift is calculated on a matched set.

### C. `grid_search_benchmark_for_v3.py` (Time Sweep)
*   **Task**: Standardize seed propagation.
*   **Logic**: Ensure that every coordinate in the grid (e.g. 5 steps vs 10 steps) is solving the same underlying batch problem.

### D. `grid_search_accuracy_v3.py` (Accuracy Sweep)
*   **Task**: Align with the unified `n_trials=1` standard.
*   **Logic**: Use the global noise basis to eliminate the need for cross-trial averaging in math-only audits.

---

## 7. Final Verification
After this refactor, all benchmark versions will produce **bit-identical mathematical results** across all trials/solvers, making the execution time a pure measure of algorithmic complexity.
