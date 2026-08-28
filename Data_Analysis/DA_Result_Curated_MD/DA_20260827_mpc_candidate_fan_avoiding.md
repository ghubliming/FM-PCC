# MPC candidate fan 4 → 1 — avoiding-d3il, pure state

**Date:** 2026-08-27 · **Task:** `avoiding-d3il` (state) · **Batch:** `temp/2808/batch_avoiding_combined_20260827_224347`
**Jobs:** 25101 (DPCC), 25102 (AlphaFlow), 25104 (MeanFlow), 25105 (FMv3ODE) — all completed, 0 NLP failures

📄 **Full analysis:**
[`logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md`](../../logs_in_develop/HF_Batch_Parity/DA_20260827_mpc1_full_seeds_state_avoiding.md)
(figures, per-arm CSV and LaTeX table in `figs_20260827_mpc_fan/`)

`FMPCC_MPC_BATCH` (arms A/B) drawn at 4 vs 1. 5 seeds × 3 scenarios × 2 trials = **30 episodes per arm**,
paired at the 15 (seed × scenario) blocks; exact Wilcoxon + paired bootstrap. Arm C (`HFFM_BATCH`) is a
separate knob and was 1 in both legs throughout.

## What it shows

| # | Finding | Strength |
|---|---|---|
| **F1** | Fan scales **only the projector** — 3.2–4.4× on the projection stage, generator flat to within 2%. End-to-end gain = projector's budget share: DPCC 1.86–1.89×, AlphaFlow 1.39–1.51×, MeanFlow 1.29–1.33×. | tightly measured |
| **F2** | Safety effect **changes sign by model**. DPCC untightened 20/30 → 7/30 (*p*=0.016); AlphaFlow `-c-tightened` 6/30 → **30/30** (*p*=0.0005) with 113 fewer steps. | resolved, 4 arms |
| **F3** | At fan 1 the `-r`/`-c`/`-t` rules are **bit-identical** (same S&C *and* steps, all 15 blocks, every generation). Running all three = 3× wasted projection compute. | exact |

### Two batch axes — only one of them is parallel

| stage | how the `B` candidates run | cost of B=4 vs 1 | evidence |
|---|---|---|---|
| **Generator** (U-Net / ODE) | **parallel** — one batched GPU forward | **≈ free** | `diffuser` is fan-invariant: DPCC 179→175 ms, AF 11.9→11.2, MF 19.4→19.2 |
| **Projector** (SLSQP) | **serial** — Python `for` loop, one CPU solve each | **≈ B×** | projection scales 3.2–4.9× |

**The fan is free on the generator and linear on the projector.** Every ms the fan costs is a serial CPU
solve — the generator batch is not the problem and is not the thing to change.

### Why the projector is 3.2× and not 4× — and the option it exposes

The projector is **serial**: `diffuser/sampling/projection.py:132` runs one CPU
`scipy.optimize.minimize(method='SLSQP')` per candidate in a Python `for` loop. **No parallel batch path
exists** — `parallelize` (`projection.py:9,20`) is assigned and never read.

Constraints/matrices are built once per call, so cost = `S + B·P`. Solving from the two fan settings:

| model | shared setup `S` | per-candidate solve `P` | observed scaling |
|---|---:|---:|---:|
| DPCC | **30.0 ms** | 86.0 ms | 3.23× |
| AlphaFlow | ≈ 0 | 2.47 ms | 4.38× |
| MeanFlow K2 | ≈ 0 | 2.23 ms | 4.9× |

**Cost is linear in `B`.** It only looks sublinear on DPCC because B=4 is small enough that DPCC's fixed
30 ms is still a quarter of the fan-1 projection. On the flow models, including the flagship, scaling is
already the full 4×.

🔧 **Unexploited:** the four SLSQP solves are independent (same constraints, different `x0`) but run one
after another on one core. Parallelised, B=4 costs `S + P` instead of `S + B·P` — **fan-1 latency at
fan-4 safety**:

| config | generator | proj. B=4 serial | **today** | proj. B=4 parallel | **parallelised** | B=1 |
|---|---:|---:|---:|---:|---:|---:|
| DPCC K20 `-c-tight` | 179 | 374 | **553 ms** | 116 | **≈295 ms** | 291 ms |
| `mf_unet` K1 `-t-tight` | 9.6 | 8.5 | **18.1 ms** | 2.1 | **≈11.7 ms** | ~11.7 ms |
| `mf_unet` K2 `-t-tight` | 18.7 | 8.4 | **27.1 ms** | 2.1 | **≈20.8 ms** | 20.9 ms (meas.) |

**A parallelised B=4 lands within noise of B=1 everywhere.** It matters most where the fan is
*load-bearing* — DPCC loses 13/20 episodes untightened at B=1, and parallelising buys the same 1.87×
without giving up the candidates. Code change, not an experiment; price it before more fan ablations.

⚠️ **DPCC tightened is unresolved, not null:** −2/30, *p*=0.50, CI [−5, 0]. Design MDE is ≈ 8/30 at 80 %
power — do not report it as "no harm". Settling it needs `n_trials=20`.

⚠️ **Only 2 of 4 generations are analysable.** MeanFlow seed 6 = `bbunet` but seeds 7–10 = `bbmf_dit`;
FMv3ODE seed 6 = `act_thr 0.5` but seeds 7–10 = `act_thr 1.0`. Both 08-26 resumes dropped a knob, so the
seed sets describe different experiments. Re-run commands in §8 of the full DA.

## Is it worth it for the flagship `mf_unet` K1 / K2?

> ### ❌ No — keep **B = 4**, and **parallelise the projector** instead.
> The flagship's whole fan cost is serial CPU solves, so a parallel projector gives **B=4 the B=1
> latency** (≈11.7 ms at K1) with the banked 0.993 (298/300) and all four candidates intact — the same
> 1.5×, no safety exposure, no new run. B=1 stays interesting only if that code change is not made, and
> then it needs `mf_unet` K1 fan-1 at 20 trials returning S&C ≥ 0.993.

**Margin, not a requirement — the flagship already wins at fan 4.** Against the DA target, 5 seeds × 20
trials on *both* sides, all 3 scenarios complete:

| | S&C | steps | ms/step | vs target |
|---|---:|---:|---:|---:|
| DA target — DPCC K20/aw10 `dpcc-c-tightened` | 0.983 | 69.0 | 564 | — |
| **`mf_unet` K1** `dpcc-t-tightened` | **0.993** | **61.0** | **18.1** | **31×** |
| **`mf_unet` K2** `dpcc-t-tightened` | **0.993** | **60.4** | **27.1** | **21×** |

**K1 is worth ~2× what K2 is.** The projector costs ≈ 8.4 ms per replan regardless of K (one NLP solve,
not per-flow-step), while the generator halves K2 → K1, so ρ *rises* as K falls:

| config | generator | projector | ρ | predicted fan-1 gain |
|---|---:|---:|---:|---:|
| **K1** | 9.6 ms | 8.5 ms | **0.89** | **1.47–1.60×** → ~11.7 ms, ~48× vs target |
| K2 | 18.7 ms | 8.4 ms | 0.45 | 1.27–1.33× → ~20.9 ms, ~27× vs target |

Model validates twice: ρ=2.02 on the DPCC target predicts 1.85× vs 1.86–1.90× measured on a different run
shape; K2 predicts 1.27–1.33× vs **1.33× measured** (seed 6).

**Why not switch now**
- **Upside:** 1.5–1.6× on the flagship's headline config, plus a likely fix to the `-c` stall — that arm sits
  at **98.0 steps at K2** / 72.0 at K1 (300 episodes) against ~61 for every other tightened arm, and drops to
  63.0 on the seed-6 fan-1 run.
- **Risk:** `-t-tightened` is already 0.993 (298/300) — nothing to gain on safety, a full point to lose. And
  untightened `mf_unet` K2 goes 0.467 → 0.333, same direction as the DPCC loss, far too thin to call.
- **Evidence gap:** fan 4 has 300 episodes at both K; fan 1 has **6 episodes at K2 and nothing at all at K1**.
  Swapping a 300-episode number for a 6-episode one fails on evidence weight alone, whichever way the
  effect actually points.

On the 6 episodes that do exist (K2, seed 6, paired) tightened is flat-or-better — `-t`/`-r-tightened`
6/6 → 6/6, `-c-tightened` 5/6 → 6/6 — while untightened drops (`dpcc-c` 4/6 → 2/6, `dpcc-t` 3/6 → 2/6),
the same direction as DPCC's significant loss. So B=1 is *probably* mildly better at the reported
operating point. "Probably, on six episodes" is not publishable.

⚠️ **Reporting rule.** Quote flagship and baseline at the **same** fan, or state the setting in the table.
Dropping the fan on both gives a **26×** ratio — *lower* than the 31× at matched fan 4 — so there is no
incentive to mix settings.

## Notes for anyone reusing this batch

- Degeneracy still applies at K=2 / A=0.5: arm C is **sample-then-project (Π_S), not HardFlow** — the logs
  say `n_active=1, n_genuine=0`. Don't label those rows HardFlow.
- The DPCC 20-trial target has `dpcc-t-tightened` **missing `both-hard`** — only `-c-tightened` and
  `-r-tightened` are complete across all 3 scenarios. Use `-c-tightened` as the comparator.
- Candidates 68/69 (`H8_K10_D…FlowMatchingODE[_msgmpc1]`) hold **zero metric rows** — plan-scaffold dirs,
  not results. Gen12 results live in the `K2_thres*_mpc1_n2*` siblings.
- Batch reports 11 342 failed file loads across all 177 candidates; concentrated in stale folders, none of
  the nine used here. The batch's own `plots/` is empty — figures were generated from the CSVs.
