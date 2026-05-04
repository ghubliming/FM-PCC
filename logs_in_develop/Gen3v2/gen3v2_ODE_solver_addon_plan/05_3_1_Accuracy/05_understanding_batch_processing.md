# Understanding Batch Processing in ODE Benchmarks

This document clarifies how the `batch_size` parameter is handled across the FM-PCC ODE benchmarking suite and explains the mathematical difference between **Parallel Workload** (Time) and **Statistical Sampling** (Accuracy).

## 1. The Principle of Parallelism
In both the **Time Benchmark** and the **Accuracy Audit**, the batch size (e.g., 128) is treated strictly as a **Parallel Computing** workload.

*   **Logic**: We represent the robot states as a single large Tensor of shape `(Batch, Horizon, Dim)`.
*   **GPU Execution**: PyTorch sends this entire tensor to the GPU. The Vector Field calculation and the ODE update (e.g., $x_{t+1} = x_t + v \cdot dt$) happen for all batch items simultaneously in a single matrix operation.
*   **Boilerplate**: All 128 items share the same "Python tax" (loop overhead, conditioning, etc.), making the benchmark a test of the GPU's throughput.

---

## 2. Batch vs. Accuracy (Statistical Sampling)
While the computing is parallel, the **inputs** to each item in the batch are different. This is where the Accuracy logic diverges from the Time logic.

| Concept | Explanation |
| :--- | :--- |
| **Unique Start Points** | Every robot in the batch starts at a different random noise coordinate ($x_0$). |
| **Field Variance** | The "difficulty" of the ODE integration depends on the local curvature of the vector field. Some robots follow gentle curves; others follow sharp, difficult paths. |
| **The Mean Result** | Because paths vary, we take the **Average (Mean)** of the 128 drift results to find the "Expected Accuracy" of the solver across the entire noise distribution. |

> [!TIP]
> **Batch Size = Reliability.** A larger batch doesn't change the math logic, but it makes the final "Accuracy" number more representative of the model's overall performance.

---

## 3. Comparison: Time vs. Accuracy

| Feature | Time Benchmark (`solvers_v3`) | Accuracy Audit (`accuracy_v3`) |
| :--- | :--- | :--- |
| **Batch Use** | **Workload**: Measures how long the GPU takes to process $B$ paths in parallel. | **Workload + Stability**: Measures accuracy and **variance** across $B$ paths. |
| **Standard Deviation** | Measures **Temporal Jitter** (how much hardware speed changes between trials). | Measures **Stability Variance** (how much error differs between robot paths). |
| **Deterministic?** | No (Hardware speed varies slightly every trial). | Yes (Math is identical if you keep the same noise seed). |

## 4. Why `N_Trials` is handled differently
*   **Time testing** needs 20+ trials to find the stable "average speed" of the hardware.
*   **Accuracy testing** only needs **1 trial** with a large batch. Since the math is deterministic, Trial 1 and Trial 50 will give you the exact same precision results.

---

> [!IMPORTANT]
> **Summary for ODE Testing**:
> - Batching = Parallelizing the math.
> - Mean Drift = The average mistake across the batch.
> - `n_trials = 1` = The fastest way to get an accurate math report.
