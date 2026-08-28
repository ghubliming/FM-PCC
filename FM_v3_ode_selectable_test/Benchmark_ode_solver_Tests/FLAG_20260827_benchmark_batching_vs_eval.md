# ⚠️ FLAG — benchmark timings are NOT eval timings. Batching differs, in both directions.

**Raised:** 2026-08-27 · **Scope:** every `benchmark_ode_solvers_v*.py` in this folder, **and** the NLP
bench at `FM_v3_hardflow_test/bench_solver_hf_vs_dpcc.py`
**Source:** `logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md` §2.1

Read this before quoting any ms number from a benchmark script in a comparison against eval results.

## The two mismatches

| | benchmark runs at | eval runs at | direction of error |
|---|---|---|---|
| **ODE / generator** (this folder) | `--batch-size` default **128**, one GPU forward | `FMPCC_MPC_BATCH` = **4** (or 1), also one GPU forward | ✅ **structurally correct** — parallel in both, only the batch *size* differs, so per-trajectory cost is amortised differently |
| **NLP / projector** (`bench_solver_hf_vs_dpcc.py`) | **hardcoded batch 1** — `ref.reshape(1, horizon, transition_dim)`, one timed `projector.project()` | `FMPCC_MPC_BATCH` = **4**, solved **serially** | benchmark is **pessimistic by ≈ B×** for DPCC |

## The generator is NOT the problem — it is parallel in both

`v4/benchmark_ode_solvers_v4.py:208` calls `fm_model._predict_velocity(x, cond, t_cont)` with `x` of
shape `(batch_size, H, T)` — **one batched GPU forward over the whole batch**. The eval does exactly the
same, at `FMPCC_MPC_BATCH` instead of 128. Nothing here was "wrongly parallelised".

Confirmed empirically: in the eval the generator is **fan-invariant** — DPCC's `diffuser` arm moves
179 → 175 ms going from B = 4 to B = 1, AlphaFlow's 11.9 → 11.2 ms. A serial generator would cost 4× at
B = 4. It does not, which is only possible if the candidates go through the network together.

The only consequence of 128-vs-4 is amortisation of fixed per-call overhead, so quote per-trajectory
generator ms from this folder only with the batch size attached.

## Why the projector one bites hardest

`diffuser/sampling/projection.py:132` is a plain Python loop — one CPU
`scipy.optimize.minimize(method='SLSQP')` per candidate, one after another:

```python
for i in range(batch_size):
    res = minimize(..., method='SLSQP', ...)
```

**There is no parallel batch path.** `parallelize` is a dead constructor argument
(`projection.py:9`, assigned at `:20`) — never read. The only trace is the comment
`# only implemented for proxsuite and scipy and parallelize=False`.

Cost is therefore `S + B·P` (fixed setup + per-candidate solve). Measured on `avoiding-d3il`:

| model | shared setup `S` | per-candidate `P` | observed scaling 4 → 1 |
|---|---:|---:|---:|
| DPCC Gen0 | 30.0 ms | 86.0 ms | 3.23× |
| AlphaFlow | ≈ 0 | 2.47 ms | 4.38× |
| MeanFlow K2 | ≈ 0 | 2.23 ms | 4.9× |

## The concrete mis-read this prevents

Job 25121 (`bench_solver_hf_vs_dpcc`, 2026-08-27) reports, per solve:

```
IPOPT (HardFlow) :  47.6 ms      SLSQP (DPCC) : 10.8 ms      IPOPT / SLSQP : 4.40x
```

That **4.4× does not carry into the eval.** In the eval, HardFlow's arm C runs at `HFFM_BATCH = 1`
while the DPCC arms run at `FMPCC_MPC_BATCH = 4` — four serial SLSQP solves. At matched work the DPCC
side is ≈ 4 × 10.8 ≈ **43 ms**, essentially level with IPOPT's single 47.6 ms solve, and the
"DPCC's projector is 4.4× cheaper" reading disappears.

**Rule:** state the batch size next to every benchmark number, and never compare a benchmark ratio to
an eval ratio without checking that both arms ran at the same fan (the `B4_PARITY` check).

## Also worth knowing

- The 4 SLSQP solves are embarrassingly parallel and currently serial on one core. Parallelising would
  give **fan-1 latency at fan-4 safety** and would moot the whole fan trade-off — a code change, not an
  experiment.
- Benchmarks in this folder measure the **ODE integrator** (euler / rk4 / …) for the flow sampler; the
  NLP bench lives in a different tree (`FM_v3_hardflow_test/`). They are unrelated code paths that
  happen to share this warning.
