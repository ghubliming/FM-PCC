# Benchmark ODE Solvers V3 — Fairness Usage Guide

Date: 2026-04-16
Script: `FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v3.py`

---

## 1) The "Fairness" Revolution in V3
Benchmark V3 was created to resolve the **Performance Paradox** discovered in V2. In the previous version, `legacy:euler` was the only solver forced to run through the complex production boilerplate, making it appear artificially slower than RK4.

In V3, we introduce the **`--mode`** flag to ensure a 100% fair comparison.

### Mode A: `math` (The "Racing Shortcut")
In this mode, **all** legacy solvers (Euler, Midpoint, RK4) bypass the `diffusion.py` boilerplate and use a direct, high-speed mathematical loop.
*   **Purpose**: To prove the **theoretical scaling** of the algorithms (e.g., proving that RK4 takes exactly 4x the math time of Euler).
*   **Use when**: You need to explain the mathematical complexity to your advisor.

### Mode B: `production` (The "Real Robot")
In this mode, **all** legacy solvers are forced through the actual `p_sample_loop` in `diffusion.py`. 
*   **Purpose**: To measure the **true end-to-end latency** on the robot.
*   **Use when**: You need to justify the choice of RK4 for real-world deployment.

---

## 2) How to Run

### Example 1: Proving Theoretical Math Scaling (Mode Math)
Use this to confirm that physics and math are working as expected. Euler should be roughly 4x faster than RK4.

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v3.py \
  --mode math \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --device cuda \
  --batch-size 64 \
  --steps 10 \
  --solver-spec legacy:euler,legacy:rk4 \
  --plot
```

### Example 2: Measuring Real-World Robot Latency (Mode Production)
Use this to show how the "Python Tax" makes the mathematical difference between Euler and RK4 less significant.

```bash
python FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests/benchmark_ode_solvers_v3.py \
  --mode production \
  --vf-mode flow_matcher \
  --loadbase logs \
  --dataset avoiding-d3il \
  --diffusion-loadpath flow_matching_v3/H8_K20_Dmodels.diffusion.GaussianDiffusion \
  --diffusion-seed 6 \
  --device cuda \
  --batch-size 4 \
  --steps 20 \
  --solver-spec legacy:euler,legacy:rk4 \
  --plot
```

---

## 3) New CLI Arguments in V3

| Argument | Choices | Description |
|---|---|---|
| `--mode` | `math`, `production` | **V3 Exclusive**. Controls whether to use the lean math loop or the heavy production pipeline. |
| `--solver-spec` | List | Now uses standardized `backend:method` naming (e.g. `legacy:rk4`, `torchdiffeq:midpoint`). |
| `--n-trials` | Int | Number of repetitions. The warm-up is now automatic (3 cycles) and not counted in trials. |

---

## 4) Interpreting Results

> [!TIP]
> **The Secret of the Paradox**: If Euler is 120ms and RK4 is 130ms in `production` mode, it is NOT a bug. It simply means your system is **Pipeline Bound**. The CPU spends so much time doing boilerplate that the 4x extra math on the GPU becomes "invisible."

**Standard Verdicts:**
*   **Match in scaling?** Check `mode=math`.
*   **Match in production?** Check `mode=production`.
