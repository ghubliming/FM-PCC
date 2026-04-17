# U2 Guide: Legacy-Standardized ODE Benchmarking

This guide explains how to use the latest `legacy_` solver suite to audit performance and eliminate "Library Call Tax."

## 1. The Legacy Comparison Logic

We have standardized all direct-math solvers under the `legacy_` prefix. However, they use different execution paths to help you prove the "Call Tax" theory:

| Solver | Backend | Logic Source | Description |
| :--- | :--- | :--- | :--- |
| **`legacy_euler`** | `legacy` | `diffusion.py` | **Old Code**: Calls the original model `p_sample_loop`. |
| **`legacy_midpoint`** | `legacy` | `benchmark_v2.py`| **New Code**: Direct math implemented in the benchmark. |
| **`legacy_rk4`** | `legacy` | `benchmark_v2.py`| **New Code**: 4-stage math implemented in the benchmark. |
| **`legacy_dopri5`** | `legacy` | `benchmark_v2.py`| **New Code**: 5-stage math implemented in the benchmark. |

## 2. Running the Audit

To see the "Math Barrier" at Batch 256:

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v2.py \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --batch-size 256 \
  --steps 10 \
  --device cuda \
  --solver-spec legacy_euler,legacy_rk4,torchdiffeq:rk4
```

### What to look for:
- Compare **`legacy_euler`** vs **`legacy_rk4`**: Since `rk4` is implemented in a tight internal benchmark loop, you might even see it beating the "Old Code" Euler because it bypasses the model's high-level Python boilerplate.
- Compare **`legacy_rk4`** vs **`torchdiffeq:rk4`**: This shows you exactly how much time you are wasting in the external package overhead (~10ms per step).

## 3. The Head-to-Head Audit: 4 Legacy vs. 4 Torchdiffeq

The primary goal of this update is to compare every "Direct" implementation against its "Package" counterpart. This proves that the **Fixed Tax** exists for all solver types, not just Euler.

### The 8-Solver Comparative Command:
```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v2.py \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --grid-batch 256 \
  --grid-steps 10 \
  --solver-spec \
legacy_euler,torchdiffeq:euler,\
legacy_midpoint,torchdiffeq:midpoint,\
legacy_rk4,torchdiffeq:rk4,\
legacy_dopri5,torchdiffeq:dopri5 \
  --base-out benchmark_results/U2_HeadToHead
```

### What this Audit Proves:
1.  **Euler vs. Euler**: Shows the baseline Python loop overhead vs. Package Entry overhead.
2.  **Midpoint vs. Midpoint**: Bypasses the package's 2-stage setup.
3.  **RK4 vs. RK4**: This is where the gap should be largest. The `legacy_rk4` will be significantly faster than `torchdiffeq:rk4`.
4.  **Dopri5 vs. Dopri5**: Note that `legacy_dopri5` is a **fixed-step** version designed to measure pure throughput of 6 velocity passes. 

> [!IMPORTANT]
> If the "Package Tax" theory is correct, all four `legacy_` solvers will cluster at a lower latency floor, while all four `torchdiffeq_` solvers will cluster ~200ms higher (for 20 steps).

---

## 4. Utilizing Grid Search (Grid Sweeps)

You can sweep all batch sizes using the automated script:

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/grid_search_benchmark_for_v2.py \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --grid-batch 4,16,64,256 \
  --grid-steps 10 \
  --solver-spec legacy_euler,legacy_rk4,torchdiffeq:rk4 \
  --base-out benchmark_results/U2_Audit
```

## 4. Why this matters for Production
The `legacy_euler` result represents the **Current Reality** (Safe, chunked, but with some Python overhead). The `legacy_rk4` result represents the **Future Speed Limit** (What happens if we move all integration inside a single call).
