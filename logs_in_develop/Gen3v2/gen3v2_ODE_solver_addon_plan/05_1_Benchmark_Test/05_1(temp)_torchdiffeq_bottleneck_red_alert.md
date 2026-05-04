# 🚨 RED ALERT: torchdiffeq Bottleneck in ODE Solver Benchmark

>https://github.com/ghubliming/FM-PCC-CPU-WSL/tree/main/dpcc/FM_v3_ode_selectable_test/benchmark_outputs (local CPU inital test, outdated)
> and the follwiing up Colab Tesla T4 tests

**Description:**

Local testing has revealed a significant performance bottleneck when using the `torchdiffeq` package for ODE integration, compared to a raw NumPy explicit Euler implementation. The wall-clock inference time for `torchdiffeq` solvers is noticeably higher, causing lag and reducing overall efficiency in the benchmark.

---

## Action Plan

1. **Verify Bottleneck on Remote GPU (FMPCC):**
   - Run the same ODE solver benchmark (`FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py`) on the FMPCC remote GPU environment.
   - Compare timing results for `torchdiffeq` solvers vs. the raw NumPy Euler integrator.
   - Confirm if the bottleneck persists in the remote setup.

2. **Test Evaluation with Faster Solver:**
   - ODE solvers are only used during evaluation (not training) in this benchmark.
   - Run evaluation/benchmarking using a theoretically faster ODE solver (e.g., raw NumPy Euler or another efficient method).
   - Observe and record the impact on evaluation speed and overall performance.

---

## Why This Matters

- If `torchdiffeq` is the main bottleneck, switching to a more efficient integrator could yield substantial speedups for both benchmarking and real training.
- Results from the remote GPU environment will determine if this is a local-only issue or a fundamental limitation of the current solver setup.

---

**Next Steps:**
- [ ] Run the benchmark on FMPCC remote GPU and collect timing data.
- [ ] Run a real training session with a faster solver and compare results.
- [ ] Decide on solver changes based on findings.

---

**Note:**
- This is a critical performance issue. Addressing it may significantly improve workflow efficiency and model training times.
