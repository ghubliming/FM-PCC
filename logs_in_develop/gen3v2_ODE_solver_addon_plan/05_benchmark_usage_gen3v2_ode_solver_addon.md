# 05 Benchmark Usage: Direct Synthetic VF ODE Test

Date: 2026-04-14
Status: Simplified to standalone ODE benchmark

---

## 1) Scope

This test is only ODE method performance on a synthetic vector field.

In scope:
1. ODE solver speed comparison.
2. Same synthetic VF dynamics for all methods.

Out of scope:
1. FM model loading.
2. Dataset loading.
3. Env rollout metrics.
4. Projection metrics.

Script:
1. `FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py`

---

## 2) Run

From project root (`FM-PCC/`):

```bash
python FM_v3_ode_selectable_test/benchmark_ode_solvers_v3.py \
  --seed 0 \
  --n-trials 50 \
  --batch-size 128 \
  --state-dim 8 \
  --t0 0.0 \
  --t1 1.0 \
  --steps 20 \
  --solver-spec legacy_euler,torchdiffeq:dopri5,torchdiffeq:rk4,torchdiffeq:midpoint \
  --output-dir FM_v3_ode_selectable_test/benchmark_outputs/synthetic_vf_demo
```

---

## 3) Arguments

1. `--seed`
2. `--n-trials`
3. `--batch-size`
4. `--state-dim` (must be even)
5. `--t0`
6. `--t1`
7. `--steps`
8. `--rtol`
9. `--atol`
10. `--solver-spec`
11. `--output-dir`

Solver entry formats:
1. `legacy_euler`
2. `torchdiffeq:<method>`

---

## 4) Output Files

1. `run_meta.json`
2. `summary.json`
3. `summary.csv`
4. `trials_<backend>_<method>.json`

---

## 5) Compare These Metrics

From `summary.json` / `summary.csv`:
1. `avg_ms`
2. `std_ms`
3. `p50_ms`
4. `p95_ms`
5. `min_ms`
6. `max_ms`

Method selection rule:
1. Use `legacy_euler` as baseline.
2. Prefer lower `avg_ms` with stable `p95_ms`.
