# CHANGELOG — Gen12 solver bench: HardFlow's IPOPT vs DPCC's SLSQP on the same NLP

**2026-08-27** · Type: new test-only script + sbatch. **No production code path modified.** Nothing run (cluster job).

## Why

On `avoiding-d3il` our arm C (HardFlow) costs **1.4–14× more per plan** than arm B (DPCC), while HardFlow's own paper reports the opposite sign. The audit traced that to the **solver**, not the algorithm:

- every row of their D3IL table is IPOPT — theirs is IPOPT-on-endpoint (7.0 ms) vs IPOPT-on-noisy-iterate (22.6 ms);
- ours is IPOPT-on-endpoint (~30 ms) vs **scipy SLSQP**-on-noisy-iterate (2.1–21 ms);
- and **~81 % of an H8 IPOPT solve looks like fixed per-call overhead** rather than optimisation work — 2.09× the variables buys only 1.66× the time, where a dense NLP should scale superlinearly.

Full reasoning: [`AUDIT_20260827_hardflow_paper_timing_and_baselines.md`](../../../Data_Analysis/DA_Result_Curated_MD/AUDIT_20260827_hardflow_paper_timing_and_baselines.md) §0–0.2.

**The user's constraint: do not change the solver permanently.** So this is a standalone bench that builds both projectors side by side in one process and times them. Production `hardflow_projection.py` and `projection.py` are untouched.

## What was added

| file | what it is |
|---|---|
| `FM_v3_hardflow_test/bench_solver_hf_vs_dpcc.py` | the bench — builds `HardFlowNLP` (IPOPT) and `Projector` (scipy SLSQP) on the **same** `constraint_list`, times both on the same references |
| `Slurm_Codes/sbatch/hardflow_fmv3/bench_solver_hf_vs_dpcc.sh` | sbatch wrapper; repeats the whole bench `RUNS` times and appends to one CSV |

## How to run

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/bench_solver_hf_vs_dpcc.sh
```

Knobs (env vars, all optional):

| var | default | meaning |
|---|---|---|
| `HFFM_BENCH_REPS` | 50 | timed solves per regime |
| `HFFM_BENCH_HORIZON` | 8 | planning horizon `H` (use 16 to match HardFlow's paper) |
| `HFFM_BENCH_REF` | `both` | `endpoint` / `iterate` / `both` |
| `HFFM_BENCH_RUNS` | 3 | repeat the whole bench N times, different seed each |
| `HFFM_BENCH_TAG` | `$SLURM_JOB_ID` | output directory label |

Output: `logs/solver_bench/<tag>/solver_bench.csv` + one JSON per run.

The horizon sweep that settles the overhead question:

```bash
HFFM_BENCH_HORIZON=8  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/bench_solver_hf_vs_dpcc.sh
HFFM_BENCH_HORIZON=16 ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow_fmv3/bench_solver_hf_vs_dpcc.sh
```

## What it measures

Per repetition, on the **same** reference trajectory:

1. `HardFlowNLP.solve` — IPOPT/CasADi, arm C's projector
2. `Projector.project` — scipy SLSQP, arm B's projector
3. `‖Π_IPOPT − Π_SLSQP‖` — how far apart the two answers are
4. **max obstacle/bound residual of each output** — is either one *infeasible*?
5. failure counts

**(3)–(5) are the offline gate proposed in [`DEGENERACY_HardFlow_at_low_K.md`](../../HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md) §0.3 and never built.** They are free here, and they outrank the timing: **if a projector returns infeasible output that is a correctness bug**, so the script exits non-zero when any residual exceeds `1e-4`.

### Two reference regimes

The audit's claim is that HardFlow's NLP is cheaper because of *what* it projects, not how often. `--ref` reproduces both cases **on the same solver**, which isolates that effect from the solver choice:

| `--ref` | reference | stands in for |
|---|---|---|
| `endpoint` | small perturbation of a smooth path — near-feasible | HardFlow projecting the predicted **clean terminal** sample |
| `iterate` | heavy noise — far from feasible | Projection-All/Late projecting the **noisy intermediate** ODE iterate |

With `both` (default) the script prints the endpoint→iterate ratio for **each** solver, which is the direct test of audit §0.1.

## Design notes

- **No checkpoint, no dataset, no env.** Geometry is the real `avoiding-d3il` constraint list, imported from `gates_hardflow.build_constraints`, so it is byte-identical to what the eval enforces. Reusing the gates' builder rather than copying it means a geometry change cannot silently desync the bench.
- **Both arms see identical limits.** `StubNormalizer` feeds `Projector` the same `STUB_MINS`/`STUB_MAXS` the HardFlow NLP unnormalizes with. Without this the two would enforce the same *shapes* at different *scales* and the comparison would be void.
- **Warm-up is excluded.** The first solve of each backend pays CasADi codegen, scipy import and BLAS thread spin-up. `--warmup 3` untimed solves run first — the difference between measuring a solver and measuring an import, and it is exactly the effect that makes IPOPT look overhead-bound.
- **`time.perf_counter`, per solve**, not per batch — the quantity in dispute is per-solve cost.
- **The residual check covers obstacles and box bounds only.** Halfspace (`ineq`) and `deriv` rows are linear and are enforced as hard rows by both solvers; the nonconvex obstacle set is where a local solver can legitimately land somewhere different, and the bounds are where an unconverged IPOPT iterate would show up. Those are the two that can actually fail.

## What the outcomes mean

| result | reading |
|---|---|
| IPOPT ≫ SLSQP on **both** regimes, and the gap barely shrinks from H8 → H16 | confirms audit §0.1 — we are timing per-call overhead, not solver quality. HardFlow's cost deficit in our harness is an **engineering tax**, not the algorithm. |
| IPOPT's `iterate`/`endpoint` ratio ≫ SLSQP's | confirms the paper's mechanism — projecting a near-feasible endpoint really is the cheaper NLP, and their speed claim is earned. |
| `‖Π_I − Π_S‖` small and both residuals ≤ 0 | the two projectors agree; the arm B/C differences in our DA corpus are **not** projector disagreement. |
| either residual > 0 | 🔴 **a bug**, and it outranks the whole cost question. IPOPT keeping an infeasible last iterate is the known suspect (`hardflow_projection.py:346-360`). |

## Status

**✅ Run 2026-08-27 — job 25121, i6-gpu-1, git `1897f4f`, 50 s wall.** Default config (`reps=50 horizon=8 ref=both runs=3`), no code changes needed, no solver permanently changed.

Headline: **IPOPT 47.6 ms vs SLSQP 11.0 ms = 4.33×** on the reference HardFlow actually solves; IPOPT is overhead-dominated (1.14× from easy to hard problem, against SLSQP's 3.09×); the two projectors agree to ~1e-3 in the degenerate regime. The feasibility gate fired (non-zero exit) on the synthetic noisy regime, where **both** solvers return infeasible output — read `RESULTS_*` §5 before treating that as a bug.

- Results: `RESULTS_20260827_solver_bench_ipopt_vs_slsqp.md` (this folder)
- Fed back into `Data_Analysis/DA_Result_Curated_MD/AUDIT_20260827_*` §0.1 / §0.3 and `RESPONSE_20260826_*` 4j
- Raw log: `temp/2808/2026-08-27/12_38_52_bench_solver_hf_vs_dpcc_25121.log`; CSV/JSON left on the cluster at `logs/solver_bench/25121/`

**Still open:** the horizon sweep (`HFFM_BENCH_HORIZON=8` then `16`) was not run — now optional confirmation rather than the deciding test, since the difficulty axis already established overhead domination. And the `iterate` regime's σ = 0.60 is uncalibrated against real mid-ODE iterates; calibrate it before using the feasibility gate as a regression check.
