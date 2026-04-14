# 05 Benchmark Usage: Gen3v2 ODE Solver Addon

Date: 2026-04-14
Status: Ready to use
Depends on: 03_coding_execution_record_gen3v2_ode_solver_addon.md, 04_validation_record_gen3v2_ode_solver_addon.md

---

## 1) Purpose

Use this standalone benchmark before full FM-PCC evaluation to compare ODE options quickly and consistently.

Benchmark script:
- `FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py`

What it tests:
1. Real rollout episodes in `ObstacleAvoidanceEnv`.
2. Legacy and torchdiffeq solver variants under identical trial settings.
3. Per-trial and aggregated metrics for speed and safety-performance trade-offs.

---

## 2) Quick Start

Run from project root (`FM-PCC/`):

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py
```

Default sweep:
1. `legacy_euler:euler`
2. `torchdiffeq:dopri5`
3. `torchdiffeq:rk4`
4. `torchdiffeq:midpoint`

---

## 3) Common Commands

### 3.1 More trials for a stronger comparison

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --n-trials 50 --max-episode-length 200
```

### 3.2 Custom solver list

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --solver-spec legacy_euler:euler,torchdiffeq:dopri5,torchdiffeq:rk4
```

### 3.3 Per-solver tolerances / step size

Format:
- `backend:method:rtol:atol:step_size`
- Use `none` for optional values.

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --solver-spec torchdiffeq:dopri5:1e-5:1e-6:none,torchdiffeq:rk4:none:none:0.1
```

### 3.4 Enable comparison plots

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --plot
```

### 3.5 Select environment layout variant

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py --halfspace-variant both-hard
```

---

## 4) Output Location and Naming

Outputs are written to:
- `FM_v3_ode_selectable_test/benchmark_outputs/<timestamp>_seed<seed>_<halfspace_variant>/`

Saved files:
1. `run_meta.json` (run configuration snapshot)
2. `summary.json` (aggregate metrics per solver)
3. `summary.csv` (same summary as table)
4. `trials_<backend>_<method>.json` (per-trial records)
5. Optional plots when `--plot` is enabled:
   - `benchmark_summary_plots.png`
   - `benchmark_tradeoff_scatter.png`
   - `benchmark_inference_per_trial.png`

---

## 5) Metrics to Compare First

Primary decision metrics:
1. `success_and_constraints_rate` (goal + safety)
2. `avg_inference_ms` (runtime cost)
3. `avg_n_violations` and `avg_total_violation` (safety pressure)

Recommended selection rule:
1. Keep a legacy reference run (`legacy_euler:euler`).
2. Prefer a torchdiffeq method only if safe-success stays equal or better with acceptable runtime.
3. Reject options with unstable safety metrics even if success-only looks good.

---

## 6) Runtime Notes

1. Solver settings are applied at runtime from benchmark options (plan-time style override behavior).
2. If `torchdiffeq` backend is requested but package is unavailable, run fails fast by design.
3. `step_size` only applies to fixed-step methods in diffusion implementation.
4. This benchmark is a pre-eval filter, not a replacement for full projection-variant evaluation.
