# DA — HardFlow's minimum K, run: MeanFlow-UNet at `A = 1.0`, K ∈ {2, 3, 5} on `avoiding-d3il`

**Date:** 2026-09-06 · **Job** `25444` (`eval_meanflow_hardflow`), **COMPLETE** — `JOB START 2026-09-06 08:15:57 UTC`, `JOB END 11:08:24 UTC`, `GIT REV 963faed`
**Drop** `temp/0609/I/` · **Batch** `batch_avoiding_combined_20260906_125724` (candidates **170** K=2, **172** K=3, **178** K=5)
**Answers** [`README.md`](README.md) §4.2 — this is that command, executed verbatim (`FMPCC_RUN_MSG=hfmink_A1_mfunet_s6`).
**Engine** Gen3v6 MeanFlow · `MF_BACKBONE=unet`, `params = 4.0 M`, `MF_HORIZON=8`, EMA weights, step 98 000
**Arms** A `diffuser` · B `dpcc-{r,c,t}[-tightened]` · C `hardflow_sls-{r,c,t}[-tightened]`, one job, shared K/seeds/env resets
**Knobs** `HFFM_ACT_THRESHOLD=1.0` · `HFFM_BATCH=4` **and** `FMPCC_MPC_BATCH=4` (B4 parity held) · `nlp_backend=slsqp` · `dpcc_threshold=0.5` · `replan_steps=1`
**Protocol** seeds **7, 8, 9, 10** · `n_trials=2` · 3 halfspace geometries ⇒ **24 rollouts per row**

---

## 0. Bottom line

1. **The floor predicted in the proposal is confirmed by the shipped code, at run time.** The eval
   printed `[hardflow][THIN] K=2 A=1.0: n_active=2, n_genuine=1` and, at K=3 and K=5, ran with
   `n_genuine = 2` and `4`. **`A = 1.0` is what makes K = 2 the floor; K = 3 is the first citable K.**
2. **The K=3 and K=5 rows are ✅ genuine HardFlow and they are the strongest `avoiding-d3il`
   result in the corpus.** `hardflow_sls-t-tightened` reaches **S&C = 1.000 with 0 violations at
   every one of K = 2, 3, 5**, at 58.6–59.0 steps and 0.047–0.130 s/step.
3. **Against the pinned Target — DPCC K20/aw10/T0.5, `dpcc-c-tightened`, S&C 1.000 / 70.13 steps /
   0.5534 s — the K=3 row is Pareto-dominant:** equal S&C, **16 % fewer steps**, **7.4× lower
   avg_time**. The claim is architecture-matched (4.0 M U-Net vs the U-Net baseline), not a
   backbone confound.
4. **HardFlow beats the DPCC projector on cost, not on safety.** At matched K, in the same job, on
   the same checkpoint, arm C does **more** NLP solves than arm B and still costs **~2× less**
   (K=3: 0.0745 vs 0.1478 s; K=5: 0.1304 vs 0.2268 s). The S&C gap (1.000 vs 0.917) is **2
   rollouts out of 24** and is *not* significant. §4.
5. **⚠️ Power.** 24 rollouts, 4 seeds, `n_trials=2`. Seed 6 is missing and the Target has 5 seeds,
   so the headline comparison is not seed-matched. Treat §3 as a **strong signal that justifies the
   powered run**, not as the final number. §6.

---

## 1. The regime table, as the run actually reported it

Straight from `temp/0609/I/2026-09-05/18_51_26_eval_meanflow_hardflow_25444.log`:

| K | A | `n_active` | `n_genuine` | first lookahead | tier | citable? |
|---|---|---|---|---|---|---|
| 2 | 1.0 | 2 | **1** | 0.500 | ⚠️ **THIN** | supporting only |
| 3 | 1.0 | 3 | **2** | 0.667 | ✅ **OK** | **yes** |
| 5 | 1.0 | 5 | **4** | 0.800 | ✅ **OK** | **yes** |

`nlp_solves_total` scales 954 : 1437 : 2400 across K = 2 : 3 : 5 — ratio 1 : 1.51 : 2.52 against the
predicted 1 : 1.5 : 2.5. **Arm C solved on every ODE step, exactly as `A = 1.0` requires.** No row is
degenerate; `hf_degenerate = 0` everywhere; the guard never fired.

The eval's own THIN banner reproduces the proposal's arithmetic unprompted — *"first non-degenerate:
K>=3 at A=0.5 or K>=2 at A=1.0; for an attributable effect use n_genuine>=2"*. The prediction and the
instrumentation agree.

---

## 2. Full results

Means over 3 halfspace geometries × 4 seeds × 2 trials. `SC` = `n_success_and_constraints`,
`t` = `avg_time` (s/step), `viol` = mean executed violations.

### K = 2 — `A = 1.0`, `n_genuine = 1` ⚠️ THIN (candidate 170)

| variant | tier | SC | success | viol | steps | t (s) |
|---|---|---|---|---|---|---|
| `diffuser` (arm A, no projection) | — | 0.042 | 1.000 | 15.46 | 62.62 | 0.0185 |
| `dpcc-t-tightened` | — | 0.958 | 0.958 | 0.00 | 59.62 | **0.0270** |
| `dpcc-c-tightened` | — | 0.958 | 1.000 | 0.08 | 98.00 | 0.0266 |
| `dpcc-r-tightened` | — | 0.958 | 1.000 | 0.04 | 65.54 | 0.0269 |
| **`hardflow_sls-t-tightened`** | ⚠️ | **1.000** | 1.000 | **0.00** | **58.62** | 0.0468 |
| `hardflow_sls-c-tightened` | ⚠️ 🔴 | 1.000 | 1.000 | 0.00 | 93.12 | 0.0452 |
| `hardflow_sls-r-tightened` | ⚠️ | 0.875 | 1.000 | 0.25 | 65.33 | 0.0475 |
| `hardflow_sls-t` (untightened) | ⚠️ | 0.583 | 1.000 | 10.25 | 67.67 | 0.0470 |

### K = 3 — `A = 1.0`, `n_genuine = 2` ✅ (candidate 172)

| variant | tier | SC | success | viol | steps | t (s) |
|---|---|---|---|---|---|---|
| `diffuser` | — | 0.000 | 1.000 | 17.42 | 62.62 | 0.0277 |
| `dpcc-t-tightened` | — | 0.917 | 0.958 | 0.21 | 59.29 | 0.1478 |
| `dpcc-c-tightened` | — | 0.833 | 1.000 | 0.29 | 64.62 | 0.1280 |
| `dpcc-r-tightened` | — | 0.917 | 0.958 | 0.25 | 66.42 | 0.1526 |
| **`hardflow_sls-t-tightened`** | ✅ | **1.000** | 1.000 | **0.00** | **58.88** | **0.0745** |
| `hardflow_sls-c-tightened` | ✅ 🔴 | 0.875 | 1.000 | 0.17 | 105.17 | 0.0726 |
| `hardflow_sls-r-tightened` | ✅ | 0.875 | 0.958 | 0.17 | 65.88 | 0.0760 |
| `hardflow_sls-t` (untightened) | ✅ | 0.667 | 0.958 | 1.42 | 58.92 | 0.0732 |

### K = 5 — `A = 1.0`, `n_genuine = 4` ✅ (candidate 178)

| variant | tier | SC | success | viol | steps | t (s) |
|---|---|---|---|---|---|---|
| `diffuser` | — | 0.000 | 1.000 | 16.88 | 64.38 | 0.0461 |
| `dpcc-t-tightened` | — | 0.917 | 0.917 | 0.21 | 60.92 | 0.2268 |
| `dpcc-c-tightened` | — | 0.875 | 1.000 | 0.21 | 62.92 | 0.2101 |
| `dpcc-r-tightened` | — | 0.917 | 0.958 | 0.25 | 70.38 | 0.2117 |
| **`hardflow_sls-t-tightened`** | ✅ | **1.000** | 1.000 | **0.00** | **59.00** | **0.1304** |
| `hardflow_sls-c-tightened` | ✅ 🔴 | 0.792 | 0.833 | 0.25 | 114.58 | 0.1304 |
| `hardflow_sls-r-tightened` | ✅ | 0.875 | 0.958 | 0.21 | 66.42 | 0.1339 |
| `hardflow_sls-t` (untightened) | ✅ | 0.750 | 0.958 | 2.00 | 59.96 | 0.1270 |

🔴 `-c` at B=4 is the **known-bad** selection arm (49 % timeouts across 750 B=4 cells,
`logs_in_develop/HF_Batch_Parity/`); the eval re-printed that warning on every `-c` row here.
Reported for completeness, **not cited**. Its 114.6-step blow-up at K=5 is that pathology, not a
HardFlow property.

---

## 3. Against the Target

Target = the pinned baseline's best projection variant: candidate **17**,
`H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5`, 5 seeds, arms A/B fan 4.

| row | backbone | params | K | tier | S&C | steps | t (s) |
|---|---|---|---|---|---|---|---|
| **Target** `dpcc-c-tightened` | U-Net | (DPCC) | 20 | — | 1.000 | 70.13 | 0.5534 |
| Target `dpcc-t-tightened` | U-Net | (DPCC) | 20 | — | 1.000 | 76.13 | 0.5630 |
| **MF-UNet `hardflow_sls-t-tightened`** | U-Net | **4.0 M** | **3** | ✅ | **1.000** | **58.88** | **0.0745** |
| MF-UNet `hardflow_sls-t-tightened` | U-Net | 4.0 M | 5 | ✅ | 1.000 | 59.00 | 0.1304 |
| MF-UNet `hardflow_sls-t-tightened` | U-Net | 4.0 M | 2 | ⚠️ | 1.000 | 58.62 | 0.0468 |
| MF-UNet `dpcc-t-tightened` | U-Net | 4.0 M | 2 | — | 0.958 | 59.62 | 0.0270 |

**The K=3 row Pareto-dominates the Target** — S&C equal at 1.000, steps 58.88 < 70.13, avg_time
0.0745 < 0.5534 (**7.4×**). Both axes improve, neither degrades: this is *dominance*, not a
trade-off. Same statement holds at K=5 (4.2×) and at K=2 (11.8×, ⚠️ THIN).

**The arm-B row at the same K does not clear the Target** (S&C 0.958 < 1.000). Within this job, the
row that dominates the Target is arm C's — so at these budgets the guidance arm is load-bearing,
not decoration. That is the first time in this corpus a `hardflow` row has cleared the Target on a
✅ genuine tier.

Backbone is matched (U-Net vs U-Net), so this is the **strong** form of the claim, not a
SiT/DiT-inflated one.

---

## 4. Why HardFlow is cheaper — more solves, less time

The striking number is not safety, it is cost. At matched K, in one job, on one checkpoint:

| K | arm B solves/plan | arm B t | arm C solves/plan | arm C t | arm C / arm B |
|---|---|---|---|---|---|
| 2 | 1 | 0.0270 | 2 | 0.0468 | 1.73× *slower* |
| 3 | 2 | 0.1478 | 3 | **0.0745** | **0.50×** |
| 5 | 3 | 0.2268 | 5 | **0.1304** | **0.58×** |

(arm B solve count = `K − int((1−T)·K)` at `T = 0.5`; arm C = `K` at `A = 1.0`.)

Arm C's cost is **linear and flat per solve** — fit `t ≈ 0.0279·K − 0.009`, i.e. ~28 ms per solve
regardless of K. Arm B's per-solve cost *rises* with K: its second and third solves cost ~80–120 ms
each, against a first solve of ≤27 ms.

**Leading explanation, consistent with every number here.** DPCC projects the *flow iterate* at
intermediate times (`t = 2/3` at K=3; `t = 0.6, 0.8` at K=5), which is off-manifold and far from
feasible, so SLSQP needs many iterations. HardFlow always projects `x1_ref`, the **predicted clean
endpoint**, which is near-feasible at every step, so SLSQP converges in a few. That is precisely
HardFlow's stated mechanism, and it shows up here as a *cost* effect rather than a safety effect.

**The crossover is K = 3.** At K=2 DPCC does one terminal solve and wins on time; from K=3 on, every
extra DPCC solve is an expensive intermediate one and HardFlow pulls ahead. The floor for a
*citable* HardFlow row and the point where HardFlow starts *paying for itself* are the same K.

**Two confounds excluded, one open:**
- ✅ **B4 parity holds** — `HFFM_BATCH=4` and `FMPCC_MPC_BATCH=4`, so this is not the 2026-08-20
  fan discount that made an earlier "HardFlow speedup" spurious.
- ✅ **Solver held** — both arms are scipy SLSQP. Arm C reaches the same feasible set through
  `_build_slsqp_projector`, which instantiates DPCC's own `Projector`
  (`hardflow_projection.py:436-464`). Same solver, same constraint set, same limits.
- ⚠️ **Open:** per-solve wall time is tracked in code (`self.solve_ms`,
  `hardflow_projection.py:483`) but is **not** exported to the DA CSVs, and arm B reports no NLP
  counters at all (`nlp_solves_total = 0` on every `dpcc-*` row). The mechanism above is inferred
  from totals. Exporting `solve_ms` and an SLSQP iteration count would settle it directly. §7.

**Cross-run consistency check** (not part of the claim, but it corroborates the wiring): at K=5 the
older `A=0.5, B=1` MeanFlow run (candidate 175) shows `dpcc-t-tightened` at 0.2238 s — within 1.3 %
of this run's 0.2268 s. Arms A/B were *always* at fan 4 (the `B` token only ever moved arm C), so
arm B's timing is expected to be identical across the two runs, and it is. Its arm C
(`hardflow_new-c-tightened`, IPOPT, 3 solves, B=1) cost 0.1367 s against this run's SLSQP at 5
solves and B=4 costing 0.1304 s — consistent with the documented 3.88× IPOPT→SLSQP swap offsetting
the 4× fan.

---

## 5. What the low-K PCC question looks like now

The proposal's Q1 was the arm ranking at the budgets MF-UNet actually operates at. At K=2, on this
checkpoint and this fan:

- **Unguided is unusable.** `diffuser` S&C = 0.042 at K=2 and **0.000** at K=3 and K=5, with 15–17
  executed violations per rollout, at 1.000 success. The model reaches the goal and walks through
  the halfspace every time. Projection is not a refinement here, it is the entire constraint story.
- **Tightening is the single biggest lever, larger than the choice of arm.** `dpcc-t` 0.458 →
  `dpcc-t-tightened` 0.958 at K=2; `hardflow_sls-t` 0.583 → `-t-tightened` 1.000. Untightened rows
  never clear 0.75 anywhere in this run. Any arm comparison run on untightened geometry is
  measuring the margin, not the arm.
- **`-t` (temporal_consistency) is the selection rule that works at B=4**, in both arms. `-c` is the
  known-bad arm, `-r` sits in between.
- **K=1 remains structurally out of reach for arm C** at any `A`, so the K=1 cell of this
  comparison is an arms-A/B-only cell, permanently.

---

## 6. Limits — read before citing

1. **24 rollouts per row.** 4 seeds × 3 geometries × `n_trials = 2`. The S&C differences that carry
   §3 (1.000 vs 0.917) are **2 rollouts**. The *time* differences of §4 are systematic and survive
   this; the *safety* differences do not.
2. **Seed 6 was never run.** The yaml `seeds:` list for this job was 7–10; `Missing_Seeds = [6]` on
   all three candidates. The Target (candidate 17) pools 5 seeds. **§3 is therefore not
   seed-matched** — it compares a 4-seed row against a 5-seed row.
3. **The run tag says `s6` and the run is not seed 6.** `FMPCC_RUN_MSG=hfmink_A1_mfunet_s6` was
   carried over from the proposal's smoke command while the yaml seed list was changed. The folder
   names are misleading; no number is affected. Use a corrected tag on the powered run.
4. **No α-Flow arm.** This is MeanFlow only. The Gen3v7 half of the proposal (§4.4) has not run, and
   `eval_alphaflow_hardflow.sh` still has no K loop.
5. **No `A = 0.5` reference at matched K.** Proposal §4.3 was not run, so the `A = 0.5` vs `A = 1.0`
   comparison at equal K — how much of arm C's advantage is the *activation schedule* rather than
   HardFlow itself — is still open.
6. **No smoothness metric.** Unchanged from `Report_20260903_AF_UNet` §8.

---

## 7. Next, in priority order

1. **Power the result.** Re-run §4.2 with all five seeds and `n_trials = 20`, corrected tag. This is
   the one thing standing between §3 and a citable claim.
   ```bash
   MF_BACKBONE=unet MF_HORIZON=8 HFFM_ACT_THRESHOLD=1.0 MF_FLOW_STEPS="2 3 5" \
   HFFM_BATCH=4 FMPCC_MPC_BATCH=4 FMPCC_HF_NLP_BACKEND=slsqp \
   FMPCC_RUN_MSG=hfmink_A1_mfunet_s6to10_nt20 \
     ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh
   ```
   (set `seeds: [6,7,8,9,10]` and `n_trials: 20` in `config/meanflow_projection_eval.yaml` first)
2. **The `A = 0.5` reference ladder** — proposal §4.3, unchanged. Separates the activation schedule
   from HardFlow.
3. **α-Flow-UNet**, proposal §4.4 — after porting the `FLOW_STEPS_GRID` loop into
   `eval_alphaflow_hardflow.sh` (needs a go-ahead; it is a code edit).
4. **Export `solve_ms` and an SLSQP iteration count per arm** into the DA pipeline, and give the
   `dpcc-*` arms the same NLP counters arm C already has. This converts §4's inferred mechanism into
   a measured one, and it is the difference between "HardFlow is faster here" and "HardFlow is
   faster *because* it projects a near-feasible endpoint".

---

## 8. Sources

- `temp/0609/I/2026-09-05/18_51_26_eval_meanflow_hardflow_25444.log` (config echo :19-20, :32; THIN
  banners :120-121; per-variant `[hardflow]` summaries; `JOB END` :final)
- `temp/0609/I/batch_avoiding_combined_20260906_125724/` — `candidates_detailed.csv` (198 rows;
  170 / 172 / 178 are the three new ones), `candidates_multidimensional_aggregated.csv`
- `flow_matcher_v3_meanflow/sampling/hardflow_projection.py:436-496, 584-646`
- `Slurm_Codes/sbatch/MeanFlow/eval_meanflow_hardflow.sh`
- [`README.md`](README.md) — the proposal this answers
- `logs_in_develop/HF_iMF/HF_Study/DEGENERACY_HardFlow_at_low_K.md`
- `logs_in_develop/HF_Batch_Parity/` (the `-c`-at-B4 known-bad arm)
